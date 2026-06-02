# -*- coding: utf-8 -*-
"""
Shared data loading, GPU-side SAR augmentation, evaluation, and visualization
utilities for the LPRINET experiments.
"""
from __future__ import annotations

import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EOC2_CLASSES = ["2s1", "bmp2", "brdm", "btr70", "t72", "d7", "zsu234", "zil131", "btr60", "t62"]
EOC3_CLASSES = ["2s1", "bmp2", "brdm", "btr70", "t72", "zsu234", "zil131", "btr60", "t62", "d7"]
EOC1_CLASSES = ["2s1", "brdm", "zsu234", "t72"]


def normalize_01_to_m11(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] amplitudes to the [-1, 1] scale used by the network."""
    return (x - 0.5) / 0.5


def pil_gray_to_tensor(img: Image.Image, size: int, resample=Image.Resampling.BILINEAR) -> torch.Tensor:
    """Convert a PIL grayscale image to a [1,H,W] float tensor in [0,1] without torchvision."""
    img = img.convert("L").resize((size, size), resample=resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _extract_elevation(path: Path) -> Optional[float]:
    """
    Robust elevation parser used across the five scripts.

    Supports paths that contain "deg15" as well as MSTAR-style folders where
    the depression angle is encoded in the parent-parent directory name.
    """
    m_deg = re.search(r"deg([-\d.]+)", str(path), flags=re.IGNORECASE)
    if m_deg:
        try:
            return float(m_deg.group(1))
        except ValueError:
            pass

    try:
        m_parent = re.search(r"[-]?\d+(?:\.\d+)?", path.parent.parent.name)
        if m_parent:
            return float(m_parent.group())
    except Exception:
        pass

    return None


def _extract_azimuth(path: Path) -> float:
    try:
        return float(path.name.split("_")[0])
    except Exception:
        return 0.0


class SARPairDataset(Dataset):
    """
    Unified dataset for EOC-1/EOC-2/EOC-3/EOC-N/EOC-R experiments.

    Training returns raw [0,1] tensors:
        img_real, img_sim, mask_sim, label, domain_label, sim_el

    Test returns normalized tensors:
        img, dummy, label, domain_label, sim_el

    The training-time random SAR perturbations are intentionally not applied in
    __getitem__; they are applied by gpu_sar_batch_augment after the batch is
    transferred to GPU.
    """

    def __init__(
        self,
        sar_dir: str,
        sim_dir: str,
        target_angles: Sequence[float],
        class_names: Sequence[str],
        mode: str = "train",
        ratio: float = 1.0,
        img_size: int = 64,
        train_sim_angles: Optional[Sequence[float]] = None,
        occlusion_size: int = 0,
        require_sim_for_test: bool = False,
    ):
        self.sar_dir = Path(sar_dir)
        self.sim_dir = Path(sim_dir)
        self.target_angles = list(target_angles)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.mode = mode
        self.is_train = mode == "train"
        self.ratio = ratio
        self.img_size = img_size
        self.train_sim_angles = list(train_sim_angles) if train_sim_angles is not None else None
        self.occlusion_size = int(occlusion_size)
        self.require_sim_for_test = require_sim_for_test

        self.samples = []
        self.sim_index = self._build_sim_index()
        self._build_dataset()

        self.tensor_cache = []
        for data_type, path1, path2, _, _, _, _ in self.samples:
            img_t = pil_gray_to_tensor(Image.open(path1), self.img_size)
            if path2:
                mask_t = pil_gray_to_tensor(Image.open(path2), self.img_size, resample=Image.Resampling.NEAREST)
            else:
                mask_t = torch.zeros_like(img_t)
            self.tensor_cache.append((img_t, mask_t))

    def _build_sim_index(self) -> Dict[str, List[Tuple[float, float, str]]]:
        idx: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
        for p in self.sim_dir.rglob("*.png"):
            m = re.search(r"El([-\d.]+)_Az([-\d.]+)\.png", p.name)
            if m:
                idx[p.parent.name.lower()].append((float(m.group(1)), float(m.group(2)), str(p)))
        return idx

    def _find_closest_sim(self, target: str, el: float, az: float):
        sims = self.sim_index.get(target, [])
        if not sims:
            return None
        return min(sims, key=lambda x: (x[0] - el) ** 2 + (x[1] - az) ** 2)

    def _build_dataset(self) -> None:
        all_files = list(self.sar_dir.rglob("*.jpeg")) + list(self.sar_dir.rglob("*.jpg"))
        candidates = []

        for p in all_files:
            cls = p.parent.name.lower()
            if cls not in self.class_to_idx:
                continue

            el = _extract_elevation(p)
            if el is None:
                continue

            if not any(abs(el - tgt) < 0.5 for tgt in self.target_angles):
                continue

            az = _extract_azimuth(p)
            label = self.class_to_idx[cls]

            if self.is_train or self.require_sim_for_test:
                best_sim = self._find_closest_sim(cls, el, az)
                if best_sim is None:
                    if self.is_train:
                        continue
                    candidates.append(("real", str(p), "", label, el, az, cls))
                else:
                    sim_el, _, sim_path = best_sim
                    candidates.append(("real", str(p), str(sim_path), label, sim_el if self.is_train else el, az, cls))
            else:
                candidates.append(("real", str(p), "", label, el, az, cls))

        if self.is_train:
            if self.ratio < 1.0:
                cls_groups = defaultdict(list)
                for c in candidates:
                    cls_groups[c[-1]].append(c)

                for _, items in cls_groups.items():
                    items.sort(key=lambda x: x[5])
                    step = max(1, int(1.0 / self.ratio))
                    self.samples.extend(items[::step])
            else:
                self.samples.extend(candidates)

            if self.train_sim_angles is not None:
                for cls, sims in self.sim_index.items():
                    if cls not in self.class_to_idx:
                        continue
                    for el, az, sim_path in sims:
                        if any(abs(el - tgt) < 0.5 for tgt in self.train_sim_angles):
                            self.samples.append(("sim", str(sim_path), str(sim_path), self.class_to_idx[cls], el, az, cls))
        else:
            self.samples.extend(candidates)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_eoc_occlusion_64(self, img: torch.Tensor) -> torch.Tensor:
        """
        Direct 64x64 random square occlusion.

        This replaces the earlier virtual 128x128 intersection logic so that
        Mask_5/10/15 actually means a real 5x5/10x10/15x15 block on the final
        64x64 SAR chip.
        """
        if self.occlusion_size > 0:
            block_size = min(int(self.occlusion_size), self.img_size)
            if block_size <= 0:
                return img
            y1 = random.randint(0, self.img_size - block_size)
            x1 = random.randint(0, self.img_size - block_size)
            img[:, y1 : y1 + block_size, x1 : x1 + block_size] = 0.0
        return img

    def __getitem__(self, idx: int):
        data_type, _, _, label, sim_el, _, _ = self.samples[idx]
        img_t, mask_t = self.tensor_cache[idx]
        domain_label = torch.tensor(0.0 if data_type == "real" else 1.0, dtype=torch.float32)
        sim_el_t = torch.tensor(sim_el, dtype=torch.float32)

        if not self.is_train:
            img = self._apply_eoc_occlusion_64(img_t.clone())
            return normalize_01_to_m11(img), torch.zeros(1), label, domain_label, sim_el_t

        img_real = img_t.clone()
        img_sim = mask_t.clone()
        mask_sim = mask_t.clone()

        if data_type != "real":
            img_real = img_sim.clone()

        return img_real, img_sim, mask_sim, label, domain_label, sim_el_t


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def gpu_sar_batch_augment(
    imgs_real: torch.Tensor,
    imgs_sim: torch.Tensor,
    mask_sim: torch.Tensor,
    domain_labels_real: torch.Tensor,
    rotate_prob: float = 0.33,
    max_angle: float = 5.0,
    real_noise_prob: float = 0.20,
    real_noise_amp: Tuple[float, float] = (1.0, 5.0),
    replace_prob: float = 0.33,
    replace_ratio_range: Tuple[float, float] = (0.01, 0.05),
    sim_noise_range: Tuple[float, float] = (0.10, 0.20),
    sim_bg_mean: float = 0.05,
    sim_bg_std: float = 0.05,
    sim_clip_min: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-side SAR batch augmentation.

    All inputs are [0,1].  The returned img tensors are normalized to [-1,1],
    while mask_sim remains [0,1].
    """
    imgs_real = imgs_real.float()
    imgs_sim = imgs_sim.float()
    mask_sim = mask_sim.float()
    domain_labels_real = domain_labels_real.float()

    b = imgs_real.size(0)
    device = imgs_real.device
    dtype = imgs_real.dtype

    real_flat = domain_labels_real.view(-1).eq(0.0)
    real_mask = real_flat.view(b, 1, 1, 1)

    # Synchronous small-angle rotation for real samples and their paired sim/mask.
    do_rot = (torch.rand(b, device=device) < rotate_prob) & real_flat
    angles = torch.empty(b, device=device, dtype=dtype).uniform_(-max_angle, max_angle)
    angles = torch.where(do_rot, angles, torch.zeros_like(angles)) * math.pi / 180.0
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    grid = F.affine_grid(theta, imgs_real.size(), align_corners=False)
    imgs_real = F.grid_sample(imgs_real, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    imgs_sim = F.grid_sample(imgs_sim, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask_sim = F.grid_sample(mask_sim, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    # Real-domain additive perturbation.
    do_noise = (torch.rand(b, 1, 1, 1, device=device) < real_noise_prob) & real_mask
    amp = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(real_noise_amp[0], real_noise_amp[1])
    noise = amp * (torch.randn_like(imgs_real) * 0.1 + 0.1)
    noisy = imgs_real + noise
    noisy_min = noisy.amin(dim=(2, 3), keepdim=True)
    noisy_max = noisy.amax(dim=(2, 3), keepdim=True)
    noisy = (noisy - noisy_min) / (noisy_max - noisy_min + 1e-8)
    imgs_real = torch.where(do_noise, noisy, imgs_real)

    # Random bad-point replacement.
    do_replace = (torch.rand(b, 1, 1, 1, device=device) < replace_prob) & real_mask
    p_replace = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(
        replace_ratio_range[0], replace_ratio_range[1]
    )
    replace_mask = (torch.rand_like(imgs_real) < p_replace) & do_replace
    imgs_real = torch.where(replace_mask, torch.rand_like(imgs_real), imgs_real)

    # Sim branch background/noise randomization.
    bg = torch.randn_like(mask_sim) * sim_bg_std + sim_bg_mean
    imgs_sim = torch.where(mask_sim > 0.05, imgs_sim, bg)
    sim_noise_std = torch.empty(b, 1, 1, 1, device=device, dtype=dtype).uniform_(
        sim_noise_range[0], sim_noise_range[1]
    )
    imgs_sim = torch.clamp(imgs_sim + torch.randn_like(imgs_sim) * sim_noise_std, sim_clip_min, 1.0)

    # Sim-injected samples use the same sim image for both branches.
    sim_only = domain_labels_real.view(-1, 1, 1, 1).eq(1.0)
    imgs_real = torch.where(sim_only, imgs_sim, imgs_real)

    return normalize_01_to_m11(imgs_real.clamp(0.0, 1.0)), normalize_01_to_m11(
        imgs_sim.clamp(0.0, 1.0)
    ), mask_sim.clamp(0.0, 1.0)


def _unpack_eval_batch(batch):
    # Supports either standard 5-field eval batch or older 6-field eval batch.
    if len(batch) == 5:
        imgs, _, labels, domain_labels, sim_el = batch
    elif len(batch) == 6:
        imgs, _, _, labels, domain_labels, sim_el = batch
    else:
        raise ValueError(f"Unexpected eval batch length: {len(batch)}")
    return imgs, labels, domain_labels, sim_el


def evaluate_clean(model, data_loader, device: torch.device) -> float:
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = 0

    with torch.inference_mode():
        for batch in data_loader:
            imgs, labels, _, _ = _unpack_eval_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)[0]
            correct += (logits.argmax(1) == labels).sum()
            total += labels.size(0)

    return 100.0 * correct.item() / total if total > 0 else 0.0


def evaluate_random_zero(model, data_loader, device: torch.device, corruption_prob: float = 0.25) -> float:
    """EOC-R: randomly replace a fraction of normalized pixels by physical zero, i.e. -1 in [-1,1]."""
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = 0

    with torch.inference_mode():
        for batch in data_loader:
            imgs, labels, _, _ = _unpack_eval_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            corruption_mask = torch.rand_like(imgs) < corruption_prob
            imgs_corrupted = torch.where(corruption_mask, torch.full_like(imgs, -1.0), imgs)

            logits = model(imgs_corrupted)[0]
            correct += (logits.argmax(1) == labels).sum()
            total += labels.size(0)

    return 100.0 * correct.item() / total if total > 0 else 0.0


def evaluate_gaussian_snr(model, data_loader, device: torch.device, snr_db: float) -> float:
    """EOC-N: add adaptive Gaussian noise to each SAR chip according to the requested SNR in dB."""
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = 0

    with torch.inference_mode():
        for batch in data_loader:
            imgs, labels, _, _ = _unpack_eval_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs_01 = imgs * 0.5 + 0.5
            signal_power = (imgs_01 ** 2).mean(dim=(1, 2, 3), keepdim=True)
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noisy = imgs_01 + torch.randn_like(imgs_01) * torch.sqrt(noise_power)
            noisy = noisy.clamp(0.0, 1.0)
            noisy_11 = normalize_01_to_m11(noisy)

            logits = model(noisy_11)[0]
            correct += (logits.argmax(1) == labels).sum()
            total += labels.size(0)

    return 100.0 * correct.item() / total if total > 0 else 0.0


def generate_and_save_heatmaps(
    model,
    data_loader,
    output_dir: str,
    class_names: Sequence[str],
    device: torch.device,
    img_size: int = 64,
    samples_per_class: int = 2,
) -> None:
    model.eval()
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    saved_counts = {cls_name: 0 for cls_name in class_names}
    target_count = len(class_names) * samples_per_class
    saved_total = 0

    print(f"🎨 生成热力图：每类 {samples_per_class} 张，共 {target_count} 张。")
    with torch.inference_mode():
        for batch in data_loader:
            if saved_total >= target_count:
                break

            imgs, labels, domain_labels, _ = _unpack_eval_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            _, pred_mask, _ = model(imgs)
            pred_mask = F.interpolate(pred_mask, size=(img_size, img_size), mode="bilinear", align_corners=False)

            for i in range(imgs.size(0)):
                cls_name = class_names[int(labels[i].item())]
                if saved_counts[cls_name] >= samples_per_class:
                    continue

                img_np = imgs[i].cpu().squeeze().numpy() * 0.5 + 0.5
                mask_np = pred_mask[i].cpu().squeeze().numpy()
                dom_name = "Real" if float(domain_labels[i].item()) == 0.0 else "Sim"

                fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
                axes[0].imshow(img_np, cmap="gray")
                axes[0].set_title(f"{dom_name} SAR Input ({cls_name})", fontsize=10)
                axes[0].axis("off")

                axes[1].imshow(img_np, cmap="gray")
                axes[1].imshow(mask_np, cmap="jet", alpha=0.45)
                axes[1].set_title("Aligned Physics Mask", fontsize=10)
                axes[1].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(heatmap_dir, f"class_{cls_name}_sample_{saved_counts[cls_name]}_{dom_name}.png"), dpi=200)
                plt.close(fig)

                saved_counts[cls_name] += 1
                saved_total += 1
                if saved_total >= target_count:
                    break

    print(f"✅ 热力图已保存至: {heatmap_dir}")
