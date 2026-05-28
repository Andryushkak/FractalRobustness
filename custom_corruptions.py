from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T
from typing import Callable, Tuple

from fractal_robustness_templates import CIFARResNet18, _normalize_cifar10


def gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0.0, 1.0)


def speckle_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = 1.0 + torch.randn_like(x) * sigma
    return (x * noise).clamp(0.0, 1.0)


def salt_pepper_noise(x: torch.Tensor, p: float) -> torch.Tensor:
    # x in [0,1], (C,H,W)
    mask = torch.rand_like(x[0])  # (H,W)
    x_sp = x.clone()
    x_sp[:, mask < p / 2] = 0.0     # pepper
    x_sp[:, mask > 1 - p / 2] = 1.0 # salt
    return x_sp


def defocus_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # простий бокс-фільтр як наближення дефокусу
    import torch.nn.functional as F
    c, h, w = x.shape
    k = torch.ones((1, 1, kernel_size, kernel_size), device=x.device) / (kernel_size**2)
    x4 = x.unsqueeze(0)  # (1,C,H,W)
    out = []
    for ch in range(c):
        y = F.conv2d(x4[:, ch:ch+1], k, padding=kernel_size // 2)
        out.append(y)
    out = torch.cat(out, dim=1)
    return out.squeeze(0).clamp(0.0, 1.0)


def jpeg_compression(x: torch.Tensor, quality: int) -> torch.Tensor:
    # Спрощена JPEG-подібна корупція через квантизацію рівнів яскравості.
    # quality: 90,70,50,30,10 -> мапимо на крок квантизації.
    step_map = {
        90: 4,
        70: 8,
        50: 16,
        30: 32,
        10: 64,
    }
    step = step_map.get(quality, 16)

    x8 = (x.clamp(0.0, 1.0) * 255.0).round()        # (C,H,W), uint8-подібне
    xq = (x8 / step).round() * step                # груба квантизація
    return (xq / 255.0).clamp(0.0, 1.0)



def make_corrupted_loader(
    root: str,
    batch_size: int,
    corruption: str,
    severity: int,
) -> DataLoader:
    """Loader для CIFAR-10 test з нашими шумами."""
    base_tfms = [T.ToTensor()]

    def apply_corr(img: torch.Tensor) -> torch.Tensor:
        # img: (C,H,W) in [0,1]
        if corruption == "gaussian":
            sigmas = [0.05, 0.1, 0.15, 0.2, 0.25]
            return gaussian_noise(img, sigmas[severity-1])
        if corruption == "speckle":
            sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
            return speckle_noise(img, sigmas[severity-1])
        if corruption == "salt_pepper":
            ps = [0.02, 0.05, 0.1, 0.15, 0.2]
            return salt_pepper_noise(img, ps[severity-1])
        if corruption == "defocus":
            ks = [3, 5, 7, 9, 11]
            return defocus_blur(img, ks[severity-1])
        if corruption == "jpeg":
            qs = [90, 70, 50, 30, 10]
            return jpeg_compression(img, qs[severity-1])
        return img

    tfm = T.Compose(base_tfms + [T.Lambda(apply_corr), T.Lambda(_normalize_cifar10)])
    test_set = tv.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def eval_corruption(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> float:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
    return correct / max(1, total)


def demo_custom_corruptions(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    root = "./data"
    corruptions = ["gaussian", "speckle", "salt_pepper", "defocus", "jpeg"]
    severities = [1, 2, 3, 4, 5]

    for corr in corruptions:
        accs = []
        for s in severities:
            loader = make_corrupted_loader(root, batch_size=256, corruption=corr, severity=s)
            acc = eval_corruption(model, loader, device=device)
            accs.append(acc)
        print(f"{corr:>12s}: " + "  ".join(f"s{s}={a:.3f}" for s, a in zip(severities, accs)))


if __name__ == "__main__":
    demo_custom_corruptions("checkpoints/base_clean/best.pt")
