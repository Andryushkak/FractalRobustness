"""
Fractal Robustness Templates
- fd_boxcount(img): Box-counting fractal dimension for 2D images (NumPy or Torch)
- fbm2d(shape, H, ...): 2D fractional Brownian motion generator via spectral synthesis
- CIFAR-10-C evaluation pipeline in PyTorch

Python >= 3.10, PyTorch >= 2.0, TorchVision >= 0.15
Only depends on: numpy, torch, torchvision (and optionally tqdm)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict, List, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except Exception:
    # graceful fallback if tqdm is not installed
    def tqdm(x, **kwargs):
        return x

# =============================
# Utility: reproducibility
# =============================

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


# =============================
# 1) Fractal Dimension via Box Counting
# =============================

def _to_grayscale01(x: np.ndarray) -> np.ndarray:
    """Convert HxW or HxWxC image array to grayscale float32 in [0,1]."""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[2] in (3, 4):
        # RGB[A] to gray (luminosity method)
        rgb = x[..., :3].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        g = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return g.astype(np.float32)
    elif x.ndim == 2:
        g = x.astype(np.float32)
        if g.max() > 1.0:
            g = g / 255.0
        return g
    else:
        raise ValueError("fd_boxcount expects 2D or 3-channel image array")


def fd_boxcount(img: np.ndarray | torch.Tensor,
                threshold: Optional[float] = None,
                min_box: int = 2,
                max_box: Optional[int] = None,
                box_ratio: int = 2,
                return_fit: bool = False) -> float | Tuple[float, Dict[str, np.ndarray]]:
    """
    Estimate 2D fractal dimension (box-counting) of an image.

    Steps:
      1) Convert to grayscale [0,1].
      2) Binarize using `threshold` (default: image mean).
      3) For box sizes s in geometric progression (min_box .. max_box),
         count non-empty boxes; fit log N(s) vs log (1/s). Slope ≈ FD.

    Args:
      img: HxW or HxWxC array / torch tensor. If tensor, moved to CPU.
      threshold: binarization threshold in [0,1]; default = mean intensity.
      min_box: smallest box size (pixels), >= 2.
      max_box: largest box size; default = min(H,W)//2.
      box_ratio: multiplicative step for box sizes (e.g., 2 => powers of two).
      return_fit: if True, also return dict with fit arrays for plotting.

    Returns:
      fd or (fd, fit_data) where fit_data contains:
        {'log_inv_s': ..., 'log_N': ..., 'sizes': ...}
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # If tensor is CxHxW or HxWxC, attempt to convert to HxWxC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.moveaxis(img, 0, -1)

    g = _to_grayscale01(img)
    H, W = g.shape

    thr = float(g.mean()) if threshold is None else float(threshold)
    bw = (g > thr).astype(np.uint8)

    if max_box is None:
        max_box = max(min(H, W) // 2, min_box)

    # Build list of box sizes: min_box, min_box*box_ratio, ... <= max_box
    sizes: List[int] = []
    s = max(2, int(min_box))
    while s <= max_box:
        sizes.append(s)
        s = max(s * int(box_ratio), s + 1)

    Ns = []
    inv_s = []

    for s in sizes:
        # pad to multiple of s
        pad_h = (s - (H % s)) % s
        pad_w = (s - (W % s)) % s
        if pad_h or pad_w:
            pad = ((0, pad_h), (0, pad_w))
            img_p = np.pad(bw, pad, mode='constant', constant_values=0)
        else:
            img_p = bw
        HH, WW = img_p.shape
        # reshape into blocks s x s and check if any pixel is 1
        blocks = img_p.reshape(HH // s, s, WW // s, s)
        non_empty = (blocks.sum(axis=(1, 3)) > 0)
        N = int(non_empty.sum())
        Ns.append(N)
        inv_s.append(1.0 / s)

    # linear fit: log N = a + FD * log(1/s)
    log_N = np.log(np.maximum(1, np.array(Ns, dtype=np.float64)))
    log_inv_s = np.log(np.array(inv_s, dtype=np.float64))

    # Filter out degenerate points where N==1 across sizes
    mask = np.isfinite(log_N) & np.isfinite(log_inv_s)
    log_N = log_N[mask]
    log_inv_s = log_inv_s[mask]

    if len(log_N) < 2:
        fd = 0.0
    else:
        A = np.vstack([np.ones_like(log_inv_s), log_inv_s]).T
        # least squares fit
        coef, *_ = np.linalg.lstsq(A, log_N, rcond=None)
        # coef = [a, FD]
        fd = float(coef[1])

    if return_fit:
        return fd, {"log_inv_s": log_inv_s, "log_N": log_N, "sizes": np.array(sizes)}
    return fd


# =============================
# 2) 2D Fractional Brownian Motion (spectral synthesis)
# =============================

def fbm2d(shape: Tuple[int, int],
          H: float,
          sigma: float = 1.0,
          seed: Optional[int] = None,
          as_tensor: bool = False,
          device: Optional[torch.device | str] = None,
          normalize: bool = True) -> np.ndarray | torch.Tensor:
    """
    Generate a 2D fBm-like field using spectral synthesis.

    Power spectral density ~ |k|^{-(2H + 2)} for fractional Brownian surfaces.

    Args:
      shape: (H, W)
      H: Hurst parameter in (0, 1). Lower => rougher texture.
      sigma: output std deviation (if normalize=True, scales after standardization).
      seed: RNG seed.
      as_tensor: return torch.Tensor instead of np.ndarray.
      device: device for tensor output.
      normalize: if True, zero-mean and unit-std (then scale by sigma).

    Returns:
      array/tensor of shape (H, W)
    """
    if not (0 < H < 1):
        raise ValueError("H must be in (0,1)")

    rng = np.random.default_rng(seed)
    Hh, Ww = int(shape[0]), int(shape[1])

    # Frequency grids
    ky = np.fft.fftfreq(Hh).reshape(-1, 1)
    kx = np.fft.fftfreq(Ww).reshape(1, -1)
    k2 = kx**2 + ky**2

    # Avoid division by zero at DC; set DC component to zero later
    alpha = (2 * H + 2)
    with np.errstate(divide='ignore'):
        S = np.power(k2, -alpha / 2.0)
    S[0, 0] = 0.0

    # Random complex field with Hermitian symmetry implicitly satisfied
    # by using real ifft of complex spectrum
    phi = rng.normal(size=(Hh, Ww)) + 1j * rng.normal(size=(Hh, Ww))
    F = phi * S

    # Inverse FFT to spatial domain
    f = np.fft.ifft2(F).real

    if normalize:
        m = float(f.mean())
        s = float(f.std() + 1e-8)
        f = (f - m) / s
        f = f * sigma

    if as_tensor:
        t = torch.from_numpy(f.astype(np.float32))
        if device is not None:
            t = t.to(device)
        return t
    return f.astype(np.float32)


class AddFBmNoise:
    """TorchVision-style transform: add 2D fBm noise to an image (Tensor in [0,1]).

    Works with CxHxW float tensors. Noise is generated per-sample.
    """
    def __init__(self, H: float = 0.6, sigma: float = 0.05, seed: Optional[int] = None):
        self.H = H
        self.sigma = sigma
        self.seed = seed
        self._counter = 0

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img):
            raise TypeError("AddFBmNoise expects a torch.Tensor CxHxW in [0,1]")
        if img.ndim != 3:
            raise ValueError("expected CxHxW tensor")
        C, Hh, Ww = img.shape
        # advance seed per call to decorrelate
        seed = None if self.seed is None else (self.seed + self._counter)
        self._counter += 1
        noise = fbm2d((Hh, Ww), H=self.H, sigma=self.sigma, seed=seed, as_tensor=True, device=img.device)
        noise = noise.clamp_(-3 * self.sigma, 3 * self.sigma)
        return (img + noise.unsqueeze(0)).clamp(0.0, 1.0)


# =============================
# 3) CIFAR-10-C Evaluation Pipeline (PyTorch)
# =============================

CIFAR_C_CORRUPTIONS: Tuple[str, ...] = (
    # canonical 15 corruptions from Hendrycks & Dietterich (2019)
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression',
)


def _normalize_cifar10(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor (N,C,H,W) or (C,H,W) to CIFAR-10 stats."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=t.dtype, device=t.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=t.dtype, device=t.device)
    if t.ndim == 3:
        return (t - mean[:, None, None]) / std[:, None, None]
    elif t.ndim == 4:
        return (t - mean[None, :, None, None]) / std[None, :, None, None]
    else:
        raise ValueError("expected (N,C,H,W) or (C,H,W)")


def make_cifar10c_loader(
    c_path: str,
    corruption: str,
    severity: int,
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
    normalize: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for a single corruption & severity from CIFAR-10-C.

    CIFAR-10-C stores shape (50000, 32, 32, 3) for each corruption, with
    severities concatenated in order (1..5), 10000 images each.

    Args:
      c_path: directory with files like '{corruption}.npy' and 'labels.npy'
      corruption: name from CIFAR_C_CORRUPTIONS
      severity: 1..5
    """
    assert 1 <= severity <= 5, "severity must be in 1..5"
    imgs = np.load(f"{c_path}/{corruption}.npy")  # (50000, 32, 32, 3), uint8
    labels = np.load(f"{c_path}/labels.npy")      # (10000,)

    start = (severity - 1) * 10000
    end = severity * 10000
    x = imgs[start:end].astype(np.float32) / 255.0  # (10000, 32, 32, 3)
    y = labels.astype(np.int64)                     # (10000,)

    # to torch tensors
    x = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))  # (N, C, H, W)
    y = torch.from_numpy(y)

    if normalize:
        x = _normalize_cifar10(x)

    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


@dataclass
class EvalStats:
    top1: float
    n: int


def evaluate_loader(model: nn.Module, loader: DataLoader, device: str | torch.device = 'cuda') -> EvalStats:
    model.eval()
    correct = 0
    total = 0
    device = torch.device(device)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return EvalStats(top1=correct / max(1, total), n=total)


def evaluate_cifar10c(
    model: nn.Module,
    c_path: str,
    device: str | torch.device = 'cuda',
    batch_size: int = 256,
    num_workers: int = 2,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    corruptions: Sequence[str] = CIFAR_C_CORRUPTIONS,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model Top-1 accuracy per corruption & severity on CIFAR-10-C.

    Returns a nested dict: {corruption: {severity: acc, ..., 'mean': mean_acc}, 'mean_over_all': ...}
    """
    device = torch.device(device)
    model.to(device)

    results: Dict[str, Dict[int, float]] = {}
    all_acc = []

    for corr in corruptions:
        accs = {}
        for sev in severities:
            loader = make_cifar10c_loader(c_path, corr, sev, batch_size=batch_size, num_workers=num_workers)
            stats = evaluate_loader(model, loader, device)
            accs[sev] = stats.top1
        # mean across severities
        mean_acc = float(np.mean([accs[s] for s in severities]))
        accs['mean'] = mean_acc
        results[corr] = accs
        all_acc.append(mean_acc)

    results['mean_over_all'] = {'mean': float(np.mean(all_acc))}
    return results


# =============================
# 4) Simple CIFAR-10 model (ResNet-18 adapted for 32x32)
# =============================

class CIFARResNet18(nn.Module):
    """ResNet-18 adjusted for CIFAR-10 (no initial maxpool, 3x3 conv)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        from torchvision.models import resnet18
        self.net = resnet18(weights=None)
        # patch for CIFAR: smaller conv and remove maxpool
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================
# 5) Example usage (train stub + CIFAR-10-C eval)
# =============================

def example_usage():
    """
    How to use:
      1) Compute FD of an image or activation map
      2) Generate fBm texture
      3) Evaluate a model on CIFAR-10-C

    NOTE: This is a usage sketch; provide your own training and model weights.
    """
    set_seed(123)

    # 1) Fractal dimension of a random image
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    fd, fit = fd_boxcount(img, return_fit=True)
    print(f"FD ≈ {fd:.3f}, sizes={fit['sizes']}")

    # 2) Generate fBm texture and add to a tensor image
    t = torch.rand(3, 64, 64)
    t2 = AddFBmNoise(H=0.7, sigma=0.05)(t)
    print("fBm-added tensor stats:", t2.mean().item(), t2.std().item())

    # 3) CIFAR-10-C evaluation (requires you to download CIFAR-10-C locally)
    #    c_path should point to a folder containing files like 'gaussian_noise.npy' and 'labels.npy'
    c_path = "/path/to/CIFAR-10-C"  # TODO: set this path

    model = CIFARResNet18(num_classes=10)
    # TODO: load your trained weights here, e.g.:
    # model.load_state_dict(torch.load('cifar_resnet18.pt', map_location='cpu'))

    if False:  # set to True once you have model weights and CIFAR-10-C
        results = evaluate_cifar10c(model, c_path=c_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        print("Mean over all corruptions:", results['mean_over_all']['mean'])
        # print per-corruption means
        for k, v in results.items():
            if k == 'mean_over_all':
                continue
            print(f"{k:>20s}: mean_acc={v['mean']:.4f}")


if __name__ == "__main__":
    example_usage()
