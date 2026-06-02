# -*- coding: utf-8 -*-
"""EOC-R: random pixel replacement-to-zero stress test."""
from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import EOC2_CLASSES, SARPairDataset, evaluate_random_zero, generate_and_save_heatmaps, gpu_sar_batch_augment, make_loader
from models import DualStreamPhysicsNet, TrainConfig, make_adamw, print_model_parameters, set_seed, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train LPRINET and evaluate EOC-R random zero corruption.")
    parser.add_argument("--sar-dir", default="mstar128")
    parser.add_argument("--sim-dir", default="sim128")
    parser.add_argument("--output-dir", default="output_eocr_random_zero")
    parser.add_argument("--ratios", default="0.1,0.3,0.5,1,0.4,0.2")
    parser.add_argument("--corruption-prob", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-heatmaps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]

    cfg = TrainConfig(epochs=args.epochs, learning_rate=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary = {}

    for current_ratio in ratios:
        set_seed(args.seed)
        print("\n" + "=" * 76)
        print(f"🚀 EOC-R | random {args.corruption_prob*100:.1f}% zero replacement | ratio={current_ratio:.2f}")
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

        optimizer = make_adamw(model.parameters(), lr=cfg.learning_rate, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_loader), epochs=cfg.epochs, pct_start=0.10
        )
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        best_acc = 0.0
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
                    "replace_ratio_range": (0.01, 0.10),
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
                acc = evaluate_random_zero(ema_model, test_loader, device, corruption_prob=args.corruption_prob)
                best_acc = max(best_acc, acc)
                print(f"🎯 [Ratio {current_ratio:.2f}] EOC-R Acc: {acc:.2f}% | best: {best_acc:.2f}%\n")

        if not args.no_heatmaps:
            generate_and_save_heatmaps(ema_model, test_loader, current_output_dir, EOC2_CLASSES, device, img_size=args.img_size)

        results_summary[current_ratio] = best_acc
        del model, ema_model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🌟 EOC-R summary")
    for r, acc in results_summary.items():
        print(f"ratio={r:.2f}: best EOC-R Acc = {acc:.2f}%")


if __name__ == "__main__":
    main()
