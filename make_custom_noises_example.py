from __future__ import annotations
import torch
import torchvision as tv
import torchvision.transforms as T
import matplotlib.pyplot as plt

from custom_corruptions import (
    gaussian_noise,
    speckle_noise,
    salt_pepper_noise,
    defocus_blur,
    jpeg_compression,
)

def load_clean_cifar10_image(idx: int = 0):
    tfm = T.ToTensor()
    testset = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    img, label = testset[idx]  # img: (C,H,W) in [0,1]
    return img, label

def main():
    idx = 0  # можна змінити на будь-який індекс
    img, label = load_clean_cifar10_image(idx)  # (C,H,W), [0,1]

    # Вибираємо "середній" рівень інтенсивності для ілюстрації (приблизно як s3)
    # Параметри узгоджені з тими, що ти використовуєш у custom_corruptions.py
    img_gauss  = gaussian_noise(img, sigma=0.1645)       # приблизно sigma для sev3
    img_speck  = speckle_noise(img, sigma=0.2302)
    img_salt   = salt_pepper_noise(img, p=0.10)
    img_defoc  = defocus_blur(img, kernel_size=7)
    img_jpeg   = jpeg_compression(img, quality=50)

    # Перетворюємо у HxWxC для matplotlib
    imgs = [
        ("Original",      img),
        ("Gaussian",      img_gauss),
        ("Speckle",       img_speck),
        ("Salt-Pepper",   img_salt),
        ("Defocus",       img_defoc),
        ("JPEG-like",     img_jpeg),
    ]

    plt.figure(figsize=(10, 3))
    for i, (title, im) in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(im.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("custom_noises_examples.png", dpi=300)
    plt.close()
    print("Saved custom_noises_examples.png")

if __name__ == "__main__":
    main()
