from __future__ import annotations
import matplotlib.pyplot as plt

from fractal_robustness_templates_B import TrainConfig, train_cifar10, AddFBmNoise
import torch


def plot_curves(history: dict, title_prefix: str, out_png: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    # Лосс
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.xlabel("Епоха")
    plt.ylabel("Втрата")
    plt.title(f"{title_prefix}: втрата по епохах")
    plt.grid(True)

    # Точність
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["val_acc"], label="val acc")
    plt.xlabel("Епоха")
    plt.ylabel("Точність")
    plt.title(f"{title_prefix}: точність по епохах")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved plot to {out_png}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Базова модель (для графіка можна взяти менше епох, напр. 30)
    cfg_base = TrainConfig(
        epochs=30,
        lr=0.1,
        amp=True,
        output_dir="checkpoints/base_for_plots",
    )
    print("=== Training BASE model (for curves) ===")
    out_base = train_cifar10(cfg_base, aug_fbm=False, device=device)
    plot_curves(out_base["history"], "Base", "training_curves_base.png")

    # 2) Модель з fBm‑аугментацією (також 30 епох для графіка)
    fbm_aug = AddFBmNoise(H=0.6, sigma=0.03)
    cfg_fbm = TrainConfig(
        epochs=30,
        lr=0.1,
        amp=True,
        output_dir="checkpoints/fbm_for_plots",
    )
    print("=== Training FBM-AUG model (for curves) ===")
    out_fbm = train_cifar10(cfg_fbm, aug_fbm=fbm_aug, device=device)
    plot_curves(out_fbm["history"], "FBM-AUG", "training_curves_fbm.png")


if __name__ == "__main__":
    main()
