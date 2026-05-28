from __future__ import annotations
import torch
from fractal_robustness_templates import CIFARResNet18
from fractal_robustness_templates_BB import evaluate_cifar10c_metrics


def eval_model_on_cifar10c(ckpt_path: str, c_path: str, device: str = 'cuda'):
    device = device if torch.cuda.is_available() else 'cpu'
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    metrics = evaluate_cifar10c_metrics(
        model,
        c_path=c_path,
        device=device
    )

    print("Macro metrics:", metrics["macro"])
    for corr, d in metrics["per_corruption"].items():
        print(f"{corr:>20s}: acc_mean={d['acc_mean']:.3f}, ece_mean={d['ece_mean']:.3f}")
    return metrics


if __name__ == "__main__":
    # ПІДСТАВ СВОЇ ШЛЯХИ
    cifar_c_path = "data/CIFAR-10-C"
    base_ckpt = "checkpoints/base_clean/best.pt"
    fbm_ckpt  = "checkpoints/fbm_aug/best.pt"


    print("=== BASE model on CIFAR-10-C ===")
    eval_model_on_cifar10c(base_ckpt, cifar_c_path)

    print("\n=== FBM-AUG model on CIFAR-10-C ===")
    eval_model_on_cifar10c(fbm_ckpt, cifar_c_path)
