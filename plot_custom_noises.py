from __future__ import annotations
import csv
import matplotlib.pyplot as plt

def load_custom_results(csv_path: str) -> dict[str, list[float]]:
    """
    Читає results_custom_noises_base.csv і повертає
    словник: corruption -> [sev1_acc, ..., sev5_acc]
    """
    out: dict[str, list[float]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corr = row["corruption"]
            accs = [float(row[f"sev{i}_acc"]) for i in range(1, 6)]
            out[corr] = accs
    return out

def main():
    csv_path = "results_custom_noises_base.csv"
    results = load_custom_results(csv_path)

    severities = [1, 2, 3, 4, 5]

    # Беремо три шуми: gaussian, speckle, salt_pepper
    gauss_acc  = results.get("gaussian",    [])
    speck_acc  = results.get("speckle",     [])
    salt_acc   = results.get("salt_pepper", [])

    plt.figure(figsize=(6, 4))

    if gauss_acc:
        plt.plot(severities, gauss_acc, marker="o", label="Gaussian")
    if speck_acc:
        plt.plot(severities, speck_acc, marker="s", label="Speckle")
    if salt_acc:
        plt.plot(severities, salt_acc, marker="^", label="Salt-Pepper")

    plt.xlabel("Рівень інтенсивності (s1..s5)")
    plt.ylabel("Точність")
    plt.title("Custom noises (CIFAR-10, Base model)")
    plt.xticks(severities)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_path = "custom_noises_gauss_speck_salt.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
