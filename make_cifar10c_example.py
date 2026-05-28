from __future__ import annotations
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as T
import matplotlib.pyplot as plt


CIFAR_C_PATH = "data/CIFAR-10-C"  

def load_cifar10_test_image(idx: int = 0):
    tfm = T.ToTensor()
    testset = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    img, label = testset[idx]
    # перетворимо назад у HxWxC [0,1] для малювання
    img_np = img.permute(1, 2, 0).numpy()
    return img_np, label

def load_cifar10c_image(corruption: str, severity: int, idx: int = 0):
    # CIFAR-10-C: (50000, 32, 32, 3), 5 блоків по 10000 для severity 1..5
    imgs = np.load(f"{CIFAR_C_PATH}/{corruption}.npy")  # uint8
    start = (severity - 1) * 10000
    x = imgs[start:start+10000]      # беремо блок для severity
    img = x[idx].astype(np.float32) / 255.0  # H,W,C в [0,1]
    return img

def main():
    idx = 0
    severity = 3

    clean_img, label = load_cifar10_test_image(idx)

    corr_gauss  = load_cifar10c_image("gaussian_noise", severity, idx)
    corr_defoc  = load_cifar10c_image("defocus_blur",   severity, idx)
    corr_jpeg   = load_cifar10c_image("jpeg_compression", severity, idx)

    # Малюємо колаж 2x2: оригінал + 3 корупції
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    axes[0, 0].imshow(clean_img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(corr_gauss)
    axes[0, 1].set_title("Gaussian noise (sev=3)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(corr_defoc)
    axes[1, 0].set_title("Defocus blur (sev=3)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(corr_jpeg)
    axes[1, 1].set_title("JPEG compression (sev=3)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("cifar10c_examples.png", dpi=300)
    plt.close()
    print("Saved cifar10c_examples.png")

if __name__ == "__main__":
    main()
4