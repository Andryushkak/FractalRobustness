from __future__ import annotations
import torch

from train_baselines import main as train_main
from run_cifar10c_eval import eval_model_on_cifar10c
from custom_corruptions import demo_custom_corruptions
from fractal_analysis import demo_fractal_analysis


def main():
    print("=== Fractal Robustness Demo ===")
    print("1) Train baselines (clean + fBm)")
    print("2) Evaluate on CIFAR-10-C (base + fBm)")
    print("3) Evaluate on custom noises (gaussian/speckle/salt-pepper/defocus/jpeg)")
    print("4) Fractal analysis (FD of inputs/activations/gradients)")
    choice = input("Select option (1-4): ").strip()

    if choice == "1":
        # тренування двох моделей (100 епох, як у тебе)
        train_main()

    elif choice == "2":
        cifar_c_path = "data/CIFAR-10-C"
        base_ckpt = "checkpoints/base_clean/best.pt"
        fbm_ckpt  = "checkpoints/fbm_aug/best.pt"
        print("=== BASE model on CIFAR-10-C ===")
        eval_model_on_cifar10c(base_ckpt, cifar_c_path)
        print("\n=== FBM-AUG model on CIFAR-10-C ===")
        eval_model_on_cifar10c(fbm_ckpt, cifar_c_path)

    elif choice == "3":
        # власні шуми з 5 рівнями для базової моделі
        demo_custom_corruptions("checkpoints/base_clean/best.pt")

    elif choice == "4":
        # фрактальний аналіз для базової моделі
        demo_fractal_analysis(
            ckpt_path="checkpoints/base_clean/best.pt",
            layer_names=["net.layer1.0.conv1", "net.layer3.1.conv2"],
            n_batches=3
        )

    else:
        print("Unknown option")


if __name__ == "__main__":
    main()
