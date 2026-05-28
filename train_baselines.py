from __future__ import annotations
import torch

from fractal_robustness_templates_B import (
    TrainConfig, train_cifar10, AddFBmNoise
)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Базова модель (без фрактальних аугментацій)
    cfg_base = TrainConfig(
        epochs=100,
        lr=0.1,
        amp=True,
        output_dir="checkpoints/base_clean"
    )
    print("=== Training BASELINE model (no fBm aug) ===")
    out_base = train_cifar10(cfg_base, aug_fbm=False, device=device)
    print("Baseline best val acc:", out_base["best_acc"])

    # 2) Модель з fBm‑аугментацією
    fbm_aug = AddFBmNoise(H=0.6, sigma=0.03)
    cfg_fbm = TrainConfig(
        epochs=100,
        lr=0.1,
        amp=True,
        output_dir="checkpoints/fbm_aug"
    )
    print("=== Training FBM-AUG model ===")
    out_fbm = train_cifar10(cfg_fbm, aug_fbm=fbm_aug, device=device)
    print("FBM-aug best val acc:", out_fbm["best_acc"])

    print("Done. Checkpoints:")
    print("  Base :", out_base["checkpoint"])
    print("  FBM  :", out_fbm["checkpoint"])


if __name__ == "__main__":
    main()
