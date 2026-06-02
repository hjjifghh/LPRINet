# -*- coding: utf-8 -*-
"""EOC-3 + EOC-O: sparse azimuth training and 64x64 square occlusion matrix test."""
from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import EOC3_CLASSES, SARPairDataset, evaluate_clean, generate_and_save_heatmaps, gpu_sar_batch_augment, make_loader
from models import DualStreamPhysicsNet, TrainConfig, make_adamw, print_model_parameters, set_seed, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train LPRINET on EOC-3/EOC-O.")
    parser.add_argument("--sar-dir", default="mstar128")
    parser.add_argument("--sim-dir", default="sim128")
    parser.add_argument("--output-dir", default="output_eoc3_random_sampling_experiment")
    parser.add_argument("--ratios", default="0.1,0.3,0.5,1")
    parser.add_argument("--mask-sizes", default="0,5,10,15")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-heatmaps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    mask_sizes = [int(x) for x in args.mask_sizes.split(",") if x.strip()]

    cfg = TrainConfig(epochs=args.epochs, learning_rate=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary = {}

    for current_ratio in ratios:
        set_seed(args.seed)
        print("\n" + "=" * 80)
        print(f"🎲 EOC-3/EOC-O | 17° train -> 15° masked matrix test | ratio={current_ratio:.2f}")
        print("=" * 80)

        current_output_dir = os.path.join(args.output_dir, f"ratio_{current_ratio:.2f}")
        os.makedirs(current_output_dir, exist_ok=True)

        train_ds = SARPairDataset(
            args.sar_dir,
            args.sim_dir,
            target_angles=[17],
            class_names=EOC3_CLASSES,
            mode="train",
            ratio=current_ratio,
            img_size=args.img_size,
            train_sim_angles=[15, 17],
        )
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
        print(f"👉 train samples: {len(train_ds)}")

        val_loaders = {}
        for m_size in mask_sizes:
            key = f"Mask_{m_size}x{m_size}" if m_size > 0 else "Base_No_Mask"
            val_ds = SARPairDataset(
                args.sar_dir,
                args.sim_dir,
                target_angles=[15],
                class_names=EOC3_CLASSES,
                mode="test",
                ratio=1.0,
                img_size=args.img_size,
                occlusion_size=m_size,
            )
            val_loaders[key] = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

        model = DualStreamPhysicsNet(num_classes=len(EOC3_CLASSES), img_size=args.img_size, dim=128).to(device)
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        print_model_parameters(model)

        optimizer = make_adamw(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_loader), epochs=cfg.epochs, pct_start=0.10
        )
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        best_records = {name: 0.0 for name in val_loaders}

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
                    "real_noise_amp": (1.0, 10.0),
                    "sim_noise_range": (0.10, 0.20),
                    "sim_clip_min": 0.10,
                    "replace_ratio_range": (0.01, 0.05),
                },
            )

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[Ratio {current_ratio:.2f}][Epoch {epoch+1:03d}/{cfg.epochs}] "
                    f"Loss:{metrics['loss']:.4f} | Cls:{metrics['cls_acc']:.2f}% | "
                    f"DomAcc:{metrics['dom_acc']:.2f}% | GRL:{metrics['grl']:.3f}"
                )

            if (epoch + 1) % args.eval_interval == 0:
                print(f"--- 🌍 [Ratio {current_ratio:.2f}] EOC-O matrix eval ---")
                for name, loader in val_loaders.items():
                    acc = evaluate_clean(ema_model, loader, device)
                    best_records[name] = max(best_records[name], acc)
                    print(f"   {name.ljust(15)} | Acc: {acc:5.2f}% | best: {best_records[name]:5.2f}%")
                print("-" * 64)

        if not args.no_heatmaps:
            heatmap_loader = val_loaders.get("Base_No_Mask", next(iter(val_loaders.values())))
            generate_and_save_heatmaps(ema_model, heatmap_loader, current_output_dir, EOC3_CLASSES, device, img_size=args.img_size)

        results_summary[current_ratio] = best_records
        del model, ema_model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🏆 EOC-3/EOC-O summary")
    for r, records in results_summary.items():
        print(f"ratio={r:.2f}")
        for name, acc in records.items():
            print(f"  {name.ljust(15)} -> {acc:.2f}%")


if __name__ == "__main__":
    main()
