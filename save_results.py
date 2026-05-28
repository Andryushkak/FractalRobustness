from __future__ import annotations
import csv
import torch

from fractal_robustness_templates import CIFARResNet18
from fractal_robustness_templates_BB import evaluate_cifar10c_metrics
from custom_corruptions import demo_custom_corruptions, make_corrupted_loader, eval_corruption


def eval_cifar10c_to_csv(
    ckpt_path: str,
    c_path: str,
    csv_path: str,
    label: str,
    device: str = "cuda",
):
    """Оцінка моделі на CIFAR-10-C + збереження в CSV."""
    device = device if torch.cuda.is_available() else "cpu"
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    metrics = evaluate_cifar10c_metrics(
        model,
        c_path=c_path,
        device=device
    )

    # macro
    macro = metrics["macro"]

    # per-corruption таблиця
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "corruption",
            "sev1_acc",
            "sev2_acc",
            "sev3_acc",
            "sev4_acc",
            "sev5_acc",
            "mean_acc",
            "mean_ece",
        ])
        for corr, d in metrics["per_corruption"].items():
            row = [
                label,
                corr,
                d["acc"][1],
                d["acc"][2],
                d["acc"][3],
                d["acc"][4],
                d["acc"][5],
                d["acc_mean"],
                d["ece_mean"],
            ]
            writer.writerow(row)

    print(f"[CIFAR-10-C] saved per-corruption metrics for {label} to {csv_path}")
    print(f"[CIFAR-10-C] macro for {label}: {macro}")


def eval_custom_noises_to_csv(
    ckpt_path: str,
    csv_path: str,
    device: str = "cuda",
):
    """Оцінка базової моделі на власних шумах (gaussian/speckle/salt_pepper/defocus/jpeg) з 5 рівнями."""
    device = device if torch.cuda.is_available() else "cpu"
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    root = "./data"
    corruptions = ["gaussian", "speckle", "salt_pepper", "defocus", "jpeg"]
    severities = [1, 2, 3, 4, 5]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "corruption",
            "sev1_acc",
            "sev2_acc",
            "sev3_acc",
            "sev4_acc",
            "sev5_acc",
        ])

        for corr in corruptions:
            accs = []
            for s in severities:
                loader = make_corrupted_loader(root, batch_size=256, corruption=corr, severity=s)
                acc = eval_corruption(model, loader, device=device)
                accs.append(acc)
            writer.writerow([corr] + accs)
            print(f"[custom] {corr}: " + "  ".join(f"s{s}={a:.3f}" for s, a in zip(severities, accs)))

    print(f"[custom] saved custom corruption metrics to {csv_path}")


def main():
    cifar_c_path = "data/CIFAR-10-C"
    base_ckpt = "checkpoints/base_clean/best.pt"
    fbm_ckpt  = "checkpoints/fbm_aug/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) CIFAR-10-C для базової моделі
    eval_cifar10c_to_csv(
        ckpt_path=base_ckpt,
        c_path=cifar_c_path,
        csv_path="results_cifar10c_base.csv",
        label="base",
        device=device,
    )

    # 2) CIFAR-10-C для FBM-AUG моделі
    eval_cifar10c_to_csv(
        ckpt_path=fbm_ckpt,
        c_path=cifar_c_path,
        csv_path="results_cifar10c_fbm.csv",
        label="fbm_aug",
        device=device,
    )

    # 3) Власні шуми для базової моделі
    eval_custom_noises_to_csv(
        ckpt_path=base_ckpt,
        csv_path="results_custom_noises_base.csv",
        device=device,
    )


if __name__ == "__main__":
    main()
