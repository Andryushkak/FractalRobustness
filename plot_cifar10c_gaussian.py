from __future__ import annotations
import csv
import matplotlib.pyplot as plt

def load_gaussian_row(csv_path: str) -> dict:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["corruption"] == "gaussian_noise":
                return row
    raise RuntimeError(f"gaussian_noise not found in {csv_path}")

def main():
    base_csv = "results_cifar10c_base.csv"
    fbm_csv  = "results_cifar10c_fbm.csv"

    base_row = load_gaussian_row(base_csv)
    fbm_row  = load_gaussian_row(fbm_csv)

    severities = [1, 2, 3, 4, 5]
    base_acc = [float(base_row[f"sev{s}_acc"]) for s in severities]
    fbm_acc  = [float(fbm_row[f"sev{s}_acc"])  for s in severities]

    plt.figure(figsize=(5, 4))
    plt.plot(severities, base_acc, marker="o", label="Base")
    plt.plot(severities, fbm_acc,  marker="s", label="FBM-AUG")

    plt.xlabel("Рівень інтенсивності (severity)")
    plt.ylabel("Точність")
    plt.title("Gaussian noise (CIFAR-10-C)")
    plt.xticks(severities)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path = "gaussian_severity_base_vs_fbm.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
