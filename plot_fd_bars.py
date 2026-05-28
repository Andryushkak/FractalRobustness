from __future__ import annotations
import matplotlib.pyplot as plt

def main():
    # Значення, які ти отримав з fractal_analysis.py
    labels = [
        "Inputs",
        "Acts L1.0.conv1",
        "Grads L1.0.conv1",
        "Acts L3.1.conv2",
        "Grads L3.1.conv2",
    ]
    fd_values = [
        1.788,   # Mean FD(inputs)
        1.907,   # FD(acts) layer1.0.conv1
        1.741,   # FD(grads) layer1.0.conv1
        1.694,   # FD(acts) layer3.1.conv2
        1.411,   # FD(grads) layer3.1.conv2
    ]

    plt.figure(figsize=(7, 4))
    x = range(len(labels))
    plt.bar(x, fd_values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"])

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Фрактальна розмірність (FD)")
    plt.ylim(1.3, 2.0)
    plt.title("FD входів, активацій і градієнтів (ResNet-18)")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = "fd_bars_resnet18.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
