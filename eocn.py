# -*- coding: utf-8 -*-
"""EOC-N: Gaussian SNR stress test at -5 dB and -10 dB."""
from __future__ import annotations

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import EOC2_CLASSES, SARPairDataset, evaluate_gaussian_snr, generate_and_save_heatmaps, gpu_sar_batch_augment, make_loader
from models import DualStreamPhysicsNet, TrainConfig, make_adamw, print_model_parameters, set_seed, train_one_epoch


def _format_complexity_number(x: float) -> str:
    """Human-readable number formatting for Params/MACs/FLOPs."""
    if x is None:
        return "N/A"
    units = ["", "K", "M", "G", "T"]
    value = float(x)
    unit_idx = 0
    while abs(value) >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.3f}{units[unit_idx]}"


def benchmark_latency(model, img_size: int, device: torch.device, warmup: int = 50, repeat: int = 200):
    """Measure single-image inference latency and FPS."""
    was_training = model.training
    model.eval()

    dummy = torch.randn(1, 1, img_size, img_size, device=device)

    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(max(1, repeat)):
            _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()

    if was_training:
        model.train()

    latency_ms = (end - start) * 1000.0 / max(1, repeat)
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    return latency_ms, fps


def profile_macs(model, img_size: int, device: torch.device):
    """Profile MACs with THOP if installed. Falls back gracefully if unavailable."""
    try:
        from thop import profile
    except Exception:
        return None, "thop is not installed. Run: pip install thop"

    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 1, img_size, img_size, device=device)

    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        err = None
    except Exception as exc:
        macs, err = None, str(exc)

    if was_training:
        model.train()

    return macs, err


def print_complexity_report(
    model,
    img_size: int,
    device: torch.device,
    warmup: int = 50,
    repeat: int = 200,
    model_name: str = "LPRINET",
):
    """
    Print Params, MACs/FLOPs, Latency and FPS for inference.
    Note: THOP reports MACs; FLOPs are approximately 2 × MACs.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs, macs_error = profile_macs(model, img_size, device)
    latency_ms, fps = benchmark_latency(model, img_size, device, warmup=warmup, repeat=repeat)

    print(f"\n📊 {model_name} 推理复杂度统计 @ input=1×1×{img_size}×{img_size}")
    print(f"   Params           : {_format_complexity_number(total_params)} ({total_params:,})")
    print(f"   Trainable Params : {_format_complexity_number(trainable_params)} ({trainable_params:,})")
    if macs is not None:
        print(f"   MACs (THOP)      : {_format_complexity_number(macs)}")
        print(f"   FLOPs approx     : {_format_complexity_number(2.0 * macs)}")
    else:
        print(f"   MACs (THOP)      : N/A ({macs_error})")
    print(f"   Latency          : {latency_ms:.3f} ms/image")
    print(f"   FPS              : {fps:.2f} images/s\n")



def parse_args():
    parser = argparse.ArgumentParser(description="Train LPRINET and evaluate Gaussian SNR robustness.")
    parser.add_argument("--sar-dir", default="mstar128")
    parser.add_argument("--sim-dir", default="sim128")
    parser.add_argument("--output-dir", default="output_snr_ours_aligned_data")
    parser.add_argument("--ratios", default="0.2,0.4")
    parser.add_argument("--snr-levels", default="-5,-10")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--skip-complexity", action="store_true", help="Skip Params/MACs/Latency/FPS profiling.")
    parser.add_argument("--complexity-warmup", type=int, default=50, help="Warmup iterations for latency benchmark.")
    parser.add_argument("--complexity-repeat", type=int, default=200, help="Timed iterations for latency benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-heatmaps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    snr_levels = [float(x) for x in args.snr_levels.split(",") if x.strip()]

    cfg = TrainConfig(epochs=args.epochs, learning_rate=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary = {}

    for current_ratio in ratios:
        set_seed(args.seed)
        print("\n" + "=" * 76)
        print(f"🚀 EOC-N | Gaussian SNR stress test | ratio={current_ratio:.2f} | SNR={snr_levels}")
        print("=" * 76)

        current_output_dir = os.path.join(args.output_dir, f"ratio_{current_ratio:.2f}")
        os.makedirs(current_output_dir, exist_ok=True)

        train_ds = SARPairDataset(
            args.sar_dir,
            args.sim_dir,
            target_angles=[17],
            class_names=EOC2_CLASSES,
            mode="train",
            ratio=current_ratio,
            img_size=args.img_size,
            train_sim_angles=[15, 17],
        )
        test_ds = SARPairDataset(
            args.sar_dir,
            args.sim_dir,
            target_angles=[15],
            class_names=EOC2_CLASSES,
            mode="test",
            ratio=1.0,
            img_size=args.img_size,
        )

        train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
        test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

        print(f"👉 train samples: {len(train_ds)} | 15° test samples: {len(test_ds)}")

        model = DualStreamPhysicsNet(num_classes=len(EOC2_CLASSES), img_size=args.img_size, dim=128).to(device)
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        print_model_parameters(model)
        if not args.skip_complexity:
            print_complexity_report(
                model,
                img_size=args.img_size,
                device=device,
                warmup=args.complexity_warmup,
                repeat=args.complexity_repeat,
                model_name="LPRINET-EOCN",
            )

        optimizer = make_adamw(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_loader), epochs=cfg.epochs, pct_start=0.10
        )
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        best_records = {snr: 0.0 for snr in snr_levels}

        for epoch in range(cfg.epochs):
            metrics = train_one_epoch(
                model,
                ema_model,
                train_loader,
                optimizer,
                scheduler,
                ce_criterion,
                device,
                epoch,
                cfg,
                gpu_sar_batch_augment,
                augment_kwargs={
                    "real_noise_amp": (1.0, 5.0),
                    "sim_noise_range": (0.10, 0.20),
                    "sim_clip_min": 0.10,
                },
            )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[Ratio {current_ratio:.2f}][Epoch {epoch+1:03d}/{cfg.epochs}] "
                    f"Loss:{metrics['loss']:.4f} | Cls:{metrics['cls_acc']:.2f}% | "
                    f"DomAcc:{metrics['dom_acc']:.2f}% | GRL:{metrics['grl']:.3f}"
                )
                print(
                    f"   CE:{metrics['ce']:.3f} | Mask:{metrics['mask']:.3f} | "
                    f"FeatKD:{metrics['feat_kd']:.3f} | S2R:{metrics['s2r']:.3f} | Adv:{metrics['adv']:.3f}"
                )

            if (epoch + 1) % args.eval_interval == 0:
                print(f"--- 🌍 [Ratio {current_ratio:.2f}] Gaussian SNR eval ---")
                for snr in snr_levels:
                    acc = evaluate_gaussian_snr(ema_model, test_loader, device, snr_db=snr)
                    best_records[snr] = max(best_records[snr], acc)
                    print(f"   SNR={snr:>5.1f} dB | Acc: {acc:5.2f}% | best: {best_records[snr]:5.2f}%")
                print("-" * 64)

        if not args.no_heatmaps:
            generate_and_save_heatmaps(ema_model, test_loader, current_output_dir, EOC2_CLASSES, device, img_size=args.img_size)

        results_summary[current_ratio] = best_records
        del model, ema_model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🌟 EOC-N summary")
    for r, records in results_summary.items():
        msg = " | ".join([f"SNR={snr:g}dB: {acc:.2f}%" for snr, acc in records.items()])
        print(f"ratio={r:.2f}: {msg}")


if __name__ == "__main__":
    main()
