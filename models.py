# -*- coding: utf-8 -*-
"""
LPRINET lite-standard shared network and training utilities.

This file uses the uploaded lite.py network as the standard.  It only fixes the
mask-loss spatial-size mismatch in the training utility; it does NOT enlarge the
backbone, mask head, classifier, or LiteCDAN beyond the lite.py setting.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# -------------------------------
# Reproducibility / speed helpers
# -------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def make_adamw(params, lr: float, weight_decay: float):
    try:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available())
    except TypeError:
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float = 0.99) -> None:
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def get_state_dict_for_save(model: nn.Module):
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()


def print_model_parameters(model: nn.Module) -> None:
    """Print a full parameter audit instead of only selected modules.

    For the standard lite setting used in the paper/code release
    (num_classes=10, dim=128, LiteCDAN proj=96/hidden=128), the expected
    parameter count is 521,866.  If the total is larger, the table below will
    expose which top-level child or direct parameter caused it.
    """
    total_params = sum(p.numel() for p in model.parameters())
    child_sum = 0

    print(f"\n🚀 总参数量 (Total Params): {total_params:,} ({total_params / 1e6:.2f} M)")
    print("   ✅ Lite 标准结构：10 类、dim=128、LiteCDAN 96/128 时应为 521,866 参数。")
    print("   📌 顶层模块参数明细如下，所有 nn.Module 都会打印：")

    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        child_sum += n
        print(f"      {name:<24}: {n:,}")

    direct_params = []
    direct_sum = 0
    for name, param in model.named_parameters(recurse=False):
        direct_params.append((name, param.numel()))
        direct_sum += param.numel()

    if direct_params:
        print("   📌 顶层直接 Parameter：")
        for name, n in direct_params:
            print(f"      {name:<24}: {n:,}")

    reconstructed = child_sum + direct_sum
    print(f"   📌 child_sum + direct_params = {reconstructed:,}")

    # Helpful warning for the exact standard config.  This is a warning only;
    # it does not stop training because 4-class EOC-1 naturally has a slightly
    # different classifier/domain head size.
    if total_params != reconstructed:
        print("   ⚠️ 参数统计不一致：请检查是否存在非标准参数注册方式。")
    if total_params > 600_000:
        print("   ⚠️ 当前模型明显大于 Lite 标准 0.52M。若你期望 Lite 结构，请检查上方是哪一项异常偏大。")
    print()


# -------------------------------
# Training configuration
# -------------------------------
@dataclass
class TrainConfig:
    epochs: int = 400
    learning_rate: float = 1e-3

    lambda_mask: float = 0.50
    lambda_distill: float = 0.30

    lambda_kd_s2r_mid: float = 0.35
    lambda_kd_s2r_late: float = 0.08
    s2r_start_epoch: int = 45
    s2r_ramp_epochs: int = 35
    s2r_decay_epoch: int = 140

    # Keep the original adversarial schedule/weights unless an experiment
    # explicitly changes them.  The LiteCDAN capacity is controlled by model args.
    lambda_adv_max: float = 0.18
    domain_loss_weight: float = 0.75
    domain_focal_gamma: float = 3.0
    domain_focal_alpha: Optional[float] = None
    adv_start_epoch: int = 25
    adv_ramp_epochs: int = 90

    ema_decay: float = 0.99
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1


def sigmoid_rampup(epoch: int, start_epoch: int, ramp_epochs: int, max_value: float) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return max_value
    p = min(1.0, max(0.0, (epoch - start_epoch) / float(ramp_epochs)))
    return max_value * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def cosine_rampup(epoch: int, start_epoch: int, ramp_epochs: int, max_value: float) -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return max_value
    p = min(1.0, max(0.0, (epoch - start_epoch) / float(ramp_epochs)))
    return max_value * 0.5 * (1.0 - math.cos(math.pi * p))


def get_s2r_weight(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.s2r_start_epoch:
        return 0.0
    if epoch < cfg.s2r_decay_epoch:
        return cosine_rampup(epoch, cfg.s2r_start_epoch, cfg.s2r_ramp_epochs, cfg.lambda_kd_s2r_mid)

    decay_range = max(1, cfg.epochs - cfg.s2r_decay_epoch)
    p = min(1.0, max(0.0, (epoch - cfg.s2r_decay_epoch) / float(decay_range)))
    return cfg.lambda_kd_s2r_late + (cfg.lambda_kd_s2r_mid - cfg.lambda_kd_s2r_late) * 0.5 * (
        1.0 + math.cos(math.pi * p)
    )


# -------------------------------
# Losses
# -------------------------------
def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()



def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 11, sigma: float = 1.0) -> torch.Tensor:
    """Torch-only Gaussian blur for NCHW tensors."""
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / (kernel_1d.sum() + 1e-12)

    c = x.size(1)
    kx = kernel_1d.view(1, 1, 1, kernel_size).repeat(c, 1, 1, 1)
    ky = kernel_1d.view(1, 1, kernel_size, 1).repeat(c, 1, 1, 1)

    # Replicate padding is stable for small 16x16 saliency targets.
    out = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    out = F.conv2d(out, ky, groups=c)
    return out


def compute_mask_loss(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """
    BCE + Dice mask loss with automatic spatial alignment.

    Some experiment variants return a predicted mask at 64x64, while the soft
    supervision target is usually pooled to 16x16. To keep all experiments
    compatible, resize the prediction to the target resolution before computing
    the loss. This preserves the original 16x16 mask-loss behavior when the
    model already outputs 16x16.
    """
    true_mask = torch.clamp(true_mask, min=0.0, max=1.0)
    if pred_mask.shape[-2:] != true_mask.shape[-2:]:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[-2:], mode="bilinear", align_corners=False)
    pred_mask = torch.clamp(pred_mask, min=1e-8, max=1.0 - 1e-8)
    return F.binary_cross_entropy(pred_mask, true_mask) + dice_loss(pred_mask, true_mask)


def domain_focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None
) -> torch.Tensor:
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    focal = (1.0 - pt).pow(gamma) * bce
    if alpha is not None:
        alpha_t = targets * float(alpha) + (1.0 - targets) * (1.0 - float(alpha))
        focal = alpha_t * focal
    return focal.mean()


# -------------------------------
# Network blocks
# -------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class CosineMarginClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 15.0, margin: float = 0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        features_f32 = features.float()
        weight_f32 = self.weight.float()
        cosine = F.linear(F.normalize(features_f32, dim=-1), F.normalize(weight_f32, dim=-1))
        raw_logits = cosine * self.scale

        if labels is not None and self.training:
            index = torch.arange(features.size(0), device=features.device)
            logits_m = raw_logits.clone()
            logits_m[index, labels] -= self.margin * self.scale
            return logits_m, raw_logits

        return raw_logits


class LiteConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class LiteConvNeXtBackbone(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, 24),
        )
        self.downsample1 = nn.Sequential(nn.GroupNorm(1, 24), nn.Conv2d(24, 48, kernel_size=2, stride=2))
        self.stage1 = LiteConvNeXtBlock(48, drop_path=0.05)
        self.downsample2 = nn.Sequential(nn.GroupNorm(1, 48), nn.Conv2d(48, dim, kernel_size=3, stride=1, padding=1))
        self.stage2 = nn.Sequential(LiteConvNeXtBlock(dim, drop_path=0.1), LiteConvNeXtBlock(dim, drop_path=0.1))

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.downsample1(x)
        feat_low = self.stage1(x)
        x = self.downsample2(feat_low)
        feat_high = self.stage2(x)
        return feat_low, feat_high


class GeometryAwareMixer(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.2):
        super().__init__()
        self.dim1 = dim // 3
        self.dim2 = dim // 3
        self.dim3 = dim - self.dim1 - self.dim2

        self.conv1 = nn.Conv2d(self.dim1, self.dim1, kernel_size=3, padding=1, dilation=1, groups=self.dim1)
        self.conv2 = nn.Conv2d(self.dim2, self.dim2, kernel_size=3, padding=2, dilation=2, groups=self.dim2)
        self.conv3 = nn.Conv2d(self.dim3, self.dim3, kernel_size=3, padding=3, dilation=3, groups=self.dim3)

        self.calibrator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(1, dim // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(1, dim // 4), dim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(dim * 2, dim))

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_2d.shape
        x1, x2, x3 = torch.split(x_2d, [self.dim1, self.dim2, self.dim3], dim=1)
        x_local = torch.cat([self.conv1(x1), self.conv2(x2), self.conv3(x3)], dim=1)
        x_local = x_local * self.calibrator(x_local)
        x_flat = x_local.flatten(2).transpose(1, 2)
        out = self.norm(x_flat)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x_2d + self.drop_path(out)


class LightweightAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 4), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.norm(self.pw_conv(self.dw_conv(x)))
        out = F.relu(out * self.se(out), inplace=True)
        return out + identity


class HierarchicalMaskHead(nn.Module):
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        self.att_stage1 = LightweightAttentionBlock(32)
        self.att_stage2 = LightweightAttentionBlock(32)
        self.mask_out = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_out(self.att_stage2(self.att_stage1(self.reduce_dim(x))))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float):
    return GradReverse.apply(x, lambd)


class LiteCDANDiscriminator(nn.Module):
    """
    Widened LiteCDAN.

    Standard CDAN constructs p ⊗ f with dimension num_classes * feat_dim.
    LiteCDAN uses low-rank conditional interaction:
        h = LayerNorm(phi_f(GRL(f)) ⊙ phi_p(p))
    The default widened setting (proj_dim=96, hidden=128) strengthens the domain
    discriminator while remaining far lighter than the original outer-product CDAN.
    """

    def __init__(self, feat_dim: int = 128, num_classes: int = 10, proj_dim: int = 96, hidden: int = 128):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, proj_dim, bias=False)
        self.prob_proj = nn.Linear(num_classes, proj_dim, bias=False)
        self.cond_norm = nn.LayerNorm(proj_dim)
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.10),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat: torch.Tensor, cls_prob: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        feat_adv = grad_reverse(feat, grl_lambda)
        prob = cls_prob.detach()
        cond = self.cond_norm(self.feat_proj(feat_adv) * self.prob_proj(prob))
        return self.net(cond).squeeze(1)


class DualStreamPhysicsNet(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        dim: int = 128,
        num_classes: int = 10,
        lite_cdan_proj_dim: int = 96,
        lite_cdan_hidden: int = 128,
    ):
        super().__init__()
        self.grid_size = img_size // 4

        self.backbone = LiteConvNeXtBackbone(dim=dim)
        self.lateral_fusion = nn.Conv2d(48, dim, kernel_size=1)
        self.physics_mask_head = HierarchicalMaskHead(in_dim=dim)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, self.grid_size, self.grid_size) * 0.02)
        self.scanner = GeometryAwareMixer(dim, drop_path_rate=0.2)
        self.global_norm = nn.LayerNorm(dim)

        self.metric_classifier = CosineMarginClassifier(dim, num_classes, margin=0.25)
        self.dropout = nn.Dropout(0.4)
        self.domain_disc = LiteCDANDiscriminator(
            feat_dim=dim,
            num_classes=num_classes,
            proj_dim=lite_cdan_proj_dim,
            hidden=lite_cdan_hidden,
        )

    def extract_feature(self, img: torch.Tensor):
        feat_low, feat_high = self.backbone(img)
        feat_img = feat_high + self.lateral_fusion(feat_low)
        pred_mask = self.physics_mask_head(feat_img)

        clean_feat_map = feat_img * (0.2 + pred_mask) + self.pos_embed
        if self.training:
            clean_feat_map = F.dropout2d(clean_feat_map, p=0.2)

        scanned = self.scanner(clean_feat_map)
        global_feat = self.global_norm(scanned.flatten(2).mean(dim=2))
        return pred_mask, global_feat

    def forward(self, img: torch.Tensor, labels: Optional[torch.Tensor] = None):
        pred_mask, global_feat = self.extract_feature(img)
        if self.training and labels is not None:
            logits_m, raw_logits = self.metric_classifier(self.dropout(global_feat), labels)
            return logits_m, raw_logits, pred_mask, global_feat

        logits = self.metric_classifier(self.dropout(global_feat))
        return logits, pred_mask, global_feat

    def domain_forward(self, feat: torch.Tensor, raw_logits: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        cls_prob = F.softmax(raw_logits, dim=1)
        return self.domain_disc(feat, cls_prob, grl_lambda)


# -------------------------------
# Shared training step
# -------------------------------
def train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    ce_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    augment_fn: Callable,
    augment_kwargs: Optional[Dict] = None,
) -> Dict[str, float]:
    model.train()
    ema_model.eval()

    augment_kwargs = augment_kwargs or {}

    total_loss = total_ce = total_mask = 0.0
    total_distill = total_s2r = total_adv = 0.0
    train_cls_correct = 0
    train_total = 0

    domain_correct = 0
    domain_total = 0

    grl_lambda = sigmoid_rampup(epoch, cfg.adv_start_epoch, cfg.adv_ramp_epochs, cfg.lambda_adv_max)
    lambda_s2r = get_s2r_weight(epoch, cfg)

    for imgs_real, imgs_sim, mask_sim, labels, domain_labels_real, _ in train_loader:
        imgs_real = imgs_real.to(device, non_blocking=True)
        imgs_sim = imgs_sim.to(device, non_blocking=True)
        mask_sim = mask_sim.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        domain_labels_real = domain_labels_real.to(device, non_blocking=True).float()

        imgs_real, imgs_sim, mask_sim = augment_fn(imgs_real, imgs_sim, mask_sim, domain_labels_real, **augment_kwargs)

        imgs_cat = torch.cat([imgs_real, imgs_sim], dim=0)
        masks_cat = torch.cat([mask_sim, mask_sim], dim=0)
        labels_cat = torch.cat([labels, labels], dim=0)

        domain_labels_sim = torch.ones_like(domain_labels_real, device=device)
        domain_labels_cat = torch.cat([domain_labels_real, domain_labels_sim], dim=0).float()

        optimizer.zero_grad(set_to_none=True)

        logits_m, raw_logits, pred_mask, student_feat = model(imgs_cat, labels=labels_cat)

        # Mask supervision is defined on 16x16 saliency maps.
        # Some model variants output 64x64 masks, so explicitly align once here
        # before all mask losses to avoid BCE/L1 size mismatches.
        pred_mask_loss = pred_mask
        if pred_mask_loss.shape[-2:] != (16, 16):
            pred_mask_loss = F.interpolate(
                pred_mask_loss, size=(16, 16), mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            logits_ema, _, feat_ema = ema_model(imgs_cat)

        train_cls_correct += (raw_logits.argmax(1) == labels_cat).sum().item()
        train_total += labels_cat.size(0)

        loss_ce = ce_criterion(logits_m, labels_cat)
        b_half = imgs_real.size(0)

        mask_real = domain_labels_cat == 0.0
        mask_sim_domain = domain_labels_cat == 1.0
        loss_mask_sim_val = torch.tensor(0.0, device=device)
        loss_mask_real_val = torch.tensor(0.0, device=device)

        if mask_sim_domain.sum() > 0:
            true_sal = F.adaptive_max_pool2d(masks_cat[mask_sim_domain], (16, 16))
            true_sal_soft = gaussian_blur2d(true_sal, kernel_size=11, sigma=1.0)
            true_sal_soft = true_sal_soft / (true_sal_soft.amax(dim=(2, 3), keepdim=True) + 1e-8)
            loss_mask_sim_val = compute_mask_loss(pred_mask_loss[mask_sim_domain], true_sal_soft)

        if mask_real.sum() > 0:
            p_m_r = pred_mask_loss[mask_real]
            r_img_p = F.adaptive_max_pool2d(imgs_cat[mask_real], (16, 16))
            s_msk_p = F.adaptive_max_pool2d(masks_cat[mask_real], (16, 16))
            s_msk_soft = gaussian_blur2d(s_msk_p, kernel_size=11, sigma=1.0)
            s_msk_soft = s_msk_soft / (s_msk_soft.amax(dim=(2, 3), keepdim=True) + 1e-8)

            r_norm = (r_img_p - r_img_p.amin(dim=(2, 3), keepdim=True)) / (
                r_img_p.amax(dim=(2, 3), keepdim=True) - r_img_p.amin(dim=(2, 3), keepdim=True) + 1e-8
            )

            soft_target = (r_norm * s_msk_soft).detach()
            if p_m_r.shape[-2:] != soft_target.shape[-2:]:
                p_m_r = F.interpolate(p_m_r, size=soft_target.shape[-2:], mode="bilinear", align_corners=False)
            loss_mask_real_val = F.l1_loss(p_m_r, soft_target)
            loss_mask_real_val = loss_mask_real_val + 0.1 * (
                torch.abs(p_m_r[:, :, :, :-1] - p_m_r[:, :, :, 1:]).mean()
                + torch.abs(p_m_r[:, :, :-1, :] - p_m_r[:, :, 1:, :]).mean()
            )

        loss_mask = loss_mask_sim_val + 0.5 * loss_mask_real_val

        loss_distill = torch.tensor(0.0, device=device)
        loss_kd_s2r = torch.tensor(0.0, device=device)
        real_idx = torch.where(domain_labels_real == 0.0)[0]

        if real_idx.numel() > 0:
            student_real = F.normalize(student_feat[real_idx], p=2, dim=-1)
            teacher_sim = F.normalize(feat_ema[real_idx + b_half].detach(), p=2, dim=-1)
            loss_distill = F.smooth_l1_loss(student_real, teacher_sim) + 0.5 * (
                1.0 - F.cosine_similarity(student_real, teacher_sim, dim=-1).mean()
            )

            if lambda_s2r > 0.0:
                temperature = 2.0
                prob_teacher = F.softmax(logits_ema[real_idx + b_half].detach() / temperature, dim=-1)
                kl_batch = F.kl_div(
                    F.log_softmax(raw_logits[real_idx] / temperature, dim=-1),
                    prob_teacher,
                    reduction="none",
                ).sum(dim=-1) * (temperature**2)
                conf_weight = prob_teacher.max(dim=-1)[0].detach()
                loss_kd_s2r = (kl_batch * conf_weight).mean()

        loss_adv = torch.tensor(0.0, device=device)
        if grl_lambda > 0.0:
            domain_logits = model.domain_forward(student_feat, raw_logits, grl_lambda)
            loss_adv = domain_focal_loss_with_logits(
                domain_logits,
                domain_labels_cat,
                gamma=cfg.domain_focal_gamma,
                alpha=cfg.domain_focal_alpha,
            )

            with torch.no_grad():
                domain_pred = (torch.sigmoid(domain_logits) > 0.5).float()
                domain_correct += (domain_pred == domain_labels_cat).sum().item()
                domain_total += domain_labels_cat.numel()

        loss = (
            loss_ce
            + cfg.lambda_mask * loss_mask
            + cfg.lambda_distill * loss_distill
            + lambda_s2r * loss_kd_s2r
            + cfg.domain_loss_weight * loss_adv
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        update_ema(model, ema_model, decay=cfg.ema_decay)

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_mask += loss_mask.item()
        total_distill += loss_distill.item()
        total_s2r += loss_kd_s2r.item()
        total_adv += loss_adv.item()

    n = max(1, len(train_loader))
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "mask": total_mask / n,
        "feat_kd": total_distill / n,
        "s2r": total_s2r / n,
        "adv": total_adv / n,
        "cls_acc": 100.0 * train_cls_correct / max(1, train_total),
        "dom_acc": 100.0 * domain_correct / domain_total if domain_total > 0 else 0.0,
        "grl": grl_lambda,
        "lambda_s2r": lambda_s2r,
    }


# Backward-compatible aliases for older single-file scripts.
DualStream_Physics_Net = DualStreamPhysicsNet
Geometry_Aware_Mixer = GeometryAwareMixer

