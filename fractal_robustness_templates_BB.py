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
    labels = np.load(f"{c_path}/labels.npy")      # (50000,)

    start = (severity - 1) * 10000
    end = severity * 10000
    x = imgs[start:end].astype(np.float32) / 255.0  # (10000, 32, 32, 3)
    y = labels[start:end].astype(np.int64)          # (10000,)


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


# =============================
# 6) CIFAR-10 training loop (+ checkpoints)
# =============================

from pathlib import Path
import time


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    aug_fbm: bool | AddFBmNoise = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val loaders for CIFAR-10.

    If aug_fbm is True or an AddFBmNoise instance, applies fBm noise during training.
    """
    import torchvision as tv
    from torchvision import transforms as T

    train_tfms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]
    if aug_fbm:
        train_tfms.append(aug_fbm if isinstance(aug_fbm, AddFBmNoise) else AddFBmNoise(H=0.6, sigma=0.03))
    train_tfms.append(T.Lambda(_normalize_cifar10))  # expects (C,H,W) tensor

    test_tfms = [T.ToTensor(), T.Lambda(_normalize_cifar10)]

    train_set = tv.datasets.CIFAR10(root=root, train=True, download=True, transform=T.Compose(train_tfms))
    test_set  = tv.datasets.CIFAR10(root=root, train=False, download=True, transform=T.Compose(test_tfms))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    label_smoothing: float = 0.0
    warmup_epochs: int = 0
    cosine: bool = True
    amp: bool = True
    output_dir: str = "checkpoints"
    seed: int = 42


def save_checkpoint(state: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer,
                    device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler],
                    criterion: nn.Module) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).step(optim)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
        total_loss += float(loss.item()) * yb.size(0)
        pred = logits.argmax(1)
        correct += int((pred == yb).sum().item())
        total += int(yb.size(0))
        pbar.set_postfix(loss=total_loss/max(1,total), acc=correct/max(1,total))
    return total_loss / max(1, total), correct / max(1, total)


def evaluate_top1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return correct / max(1,total)


def train_cifar10(config: TrainConfig = TrainConfig(),
                  aug_fbm: bool | AddFBmNoise = False,
                  device: str | torch.device = 'cuda') -> dict:
    set_seed(config.seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_cifar10_loaders(aug_fbm=aug_fbm)

    model = CIFARResNet18(num_classes=10).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                             weight_decay=config.weight_decay, nesterov=True)
    if config.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.epochs)
    else:
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[60, 80], gamma=0.1)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.cuda.amp.GradScaler() if (config.amp and device.type == 'cuda') else None

    best_acc = 0.0
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device, scaler, criterion)
        va_acc = evaluate_top1(model, val_loader, device)
        sched.step()
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:03d}/{config.epochs} | loss {tr_loss:.3f} | train {tr_acc:.3f} | val {va_acc:.3f} | lr {optim.param_groups[0]['lr']:.3e} | {dt:.1f}s")

        # save last
        save_checkpoint({'model': model.state_dict(), 'epoch': epoch, 'acc': va_acc}, str(outdir / 'last.pt'))
        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint({'model': model.state_dict(), 'epoch': epoch, 'acc': va_acc}, str(outdir / 'best.pt'))

    return {"best_acc": best_acc, "history": history, "checkpoint": str(outdir / 'best.pt')}


# =============================
# 7) ECE (Expected Calibration Error)
# =============================

def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute ECE on a batch or full dataset.

    Args:
      logits: (N, C)
      labels: (N,)
    Returns:
      scalar ECE in [0,1].
    """
    with torch.no_grad():
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()

        bins = torch.linspace(0, 1, n_bins+1, device=logits.device)
        ece = torch.zeros((), device=logits.device)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                acc_bin = correct[mask].mean()
                conf_bin = conf[mask].mean()
                weight = mask.float().mean()
                ece += weight * (conf_bin - acc_bin).abs()
        return float(ece.item())


def evaluate_loader_with_logits(model: nn.Module, loader: DataLoader, device: str | torch.device = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    device = torch.device(device)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# =============================
# 8) CIFAR-10-C metrics: CE / mCE and ECE
# =============================

def evaluate_cifar10c_metrics(
    model: nn.Module,
    c_path: str,
    device: str | torch.device = 'cuda',
    batch_size: int = 256,
    num_workers: int = 2,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    corruptions: Sequence[str] = CIFAR_C_CORRUPTIONS,
    baseline_errors: Optional[dict[str, Sequence[float]]] = None,
    n_bins_ece: int = 15,
) -> dict:
    """
    Compute per-corruption accuracy, error and ECE; also return mean CE and (optionally) normalized mCE.

    baseline_errors: map corruption -> list of 5 baseline error rates in [0,1] (e.g., AlexNet from paper).
    If provided, mCE = mean_over_{c,s} ( CE(c,s) / baseline(c,s) ). Otherwise, returns unnormalized mean CE.
    """
    device = torch.device(device)
    model.to(device)

    out: dict = {"per_corruption": {}, "macro": {}}
    mean_accs = []
    mean_ces = []
    norm_terms = []

    for corr in corruptions:
        accs: dict[int, float] = {}
        ces: dict[int, float] = {}
        eces: dict[int, float] = {}
        for sev in severities:
            loader = make_cifar10c_loader(c_path, corr, sev, batch_size=batch_size, num_workers=num_workers)
            logits, labels = evaluate_loader_with_logits(model, loader, device)
            pred = logits.argmax(1)
            acc = (pred == labels).float().mean().item()
            ce = 1.0 - acc
            ece = compute_ece(logits, labels, n_bins=n_bins_ece)
            accs[sev] = acc
            ces[sev] = ce
            eces[sev] = ece
        acc_mean = float(np.mean([accs[s] for s in severities]))
        ce_mean = 1.0 - acc_mean
        ece_mean = float(np.mean([eces[s] for s in severities]))

        out["per_corruption"][corr] = {"acc": accs, "ce": ces, "ece": eces, "acc_mean": acc_mean, "ece_mean": ece_mean}
        mean_accs.append(acc_mean)
        mean_ces.append(ce_mean)
        if baseline_errors is not None and corr in baseline_errors:
            be = np.array(baseline_errors[corr], dtype=np.float64)
            ce_vec = np.array([ces[s] for s in severities], dtype=np.float64)
            norm_terms.extend(list(ce_vec / np.maximum(1e-12, be)))

    macro_acc = float(np.mean(mean_accs)) if mean_accs else 0.0
    macro_ce = 1.0 - macro_acc
    out["macro"]["mean_acc_over_corruptions"] = macro_acc
    out["macro"]["mean_ce_over_corruptions"] = macro_ce

    if norm_terms:
        out["macro"]["mCE_normalized"] = float(np.mean(norm_terms))
    else:
        out["macro"]["mCE_unnormalized"] = macro_ce

    return out


# =============================
# 9) Plot helpers (optional)
# =============================

def plot_accuracy_vs_severity(results: Dict[str, Dict[int, float]], corruption: str) -> None:
    """Quick matplotlib plot: accuracy vs severity for a given corruption.
    Expects dict like returned by evaluate_cifar10c (per corruption: {1:acc1,...,'mean':acc}).
    """
    import matplotlib.pyplot as plt
    sev = [1,2,3,4,5]
    acc = [results[corruption][s] for s in sev]
    plt.figure()
    plt.plot(sev, acc, marker='o')
    plt.xlabel('Severity')
    plt.ylabel('Top-1 accuracy')
    plt.title(f'{corruption} — accuracy vs severity')
    plt.grid(True)
    plt.show()


# =============================
# 10) End-to-end demo stubs
# =============================

def demo_train_and_eval():
    """Train on CIFAR-10 (few epochs) and evaluate on CIFAR-10-C with metrics.
    Adjust epochs for real training. Requires CIFAR-10-C path.
    """
    cfg = TrainConfig(epochs=5, lr=0.1, amp=True)
    out = train_cifar10(cfg, aug_fbm=True)
    print('Best clean-val acc:', out['best_acc'])

    c_path = "/path/to/CIFAR-10-C"  # set your path
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(out['checkpoint'], map_location='cpu')
    model.load_state_dict(ckpt['model'])

    metrics = evaluate_cifar10c_metrics(model, c_path=c_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    print('Macro mean CE (or mCE):', metrics['macro'])
    # plot example (requires matplotlib)
    # plot_accuracy_vs_severity(results, 'gaussian_noise')


# =============================
# 11) Baselines for mCE: loaders & notes
# =============================
"""
Notes on baselines
------------------
• The normalized mCE popularized in the ImageNet-C paper uses **AlexNet** as a baseline.
• For **CIFAR-10-C**, many works report unnormalized mean CE/accuracy. If you still want a
  normalized mCE on CIFAR-10-C, you must provide a reference baseline error per corruption
  and severity (e.g., your own "reference" model trained without robustness tricks).

Below are helpers to:
  (a) load a CSV baseline file; and
  (b) compute a self-baseline from a given model (turn its CE values into the baseline).
"""

import csv


def load_baseline_csv(csv_path: str) -> dict[str, list[float]]:
    """Load corruption->5 severities baseline **error rates** from CSV.

    Expected header: corruption, sev1, sev2, sev3, sev4, sev5
    Values should be error in [0,1].
    """
    out: dict[str, list[float]] = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corr = row['corruption']
            sev = [float(row[f'sev{i}']) for i in range(1, 6)]
            out[corr] = sev
    return out


def compute_self_baseline_errors(
    model: nn.Module,
    c_path: str,
    device: str | torch.device = 'cuda',
    batch_size: int = 256,
    num_workers: int = 2,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    corruptions: Sequence[str] = CIFAR_C_CORRUPTIONS,
) -> dict[str, list[float]]:
    """
    Compute a **baseline error** dictionary from the given model on CIFAR-10-C.
    You can pass this dictionary back into `evaluate_cifar10c_metrics(..., baseline_errors=...)`
    to obtain normalized mCE w.r.t. this model.
    """
    device = torch.device(device)
    model.to(device)
    baseline: dict[str, list[float]] = {}
    for corr in corruptions:
        errs: list[float] = []
        for sev in severities:
            loader = make_cifar10c_loader(c_path, corr, sev, batch_size=batch_size, num_workers=num_workers)
            logits, labels = evaluate_loader_with_logits(model, loader, device)
            pred = logits.argmax(1)
            acc = (pred == labels).float().mean().item()
            errs.append(1.0 - acc)
        baseline[corr] = errs
    return baseline


# Optional (for reference only): placeholder for ImageNet-C AlexNet baseline
# WARNING: AlexNet baseline below is **NOT** for CIFAR-10-C. If you are working with
# ImageNet-C, populate this dict with the official AlexNet CE values per corruption
# and severity from the paper/repo, then pass it as baseline_errors.
ALEXNET_IMAGENETC_BASELINE_ERRORS: dict[str, list[float]] = {
    # 'gaussian_noise': [e1, e2, e3, e4, e5],
    # 'shot_noise': [...],
    # ...
}


# =============================
# 12) "Notebook" quickstart helpers
# =============================

def notebook_quickstart(cifar_c_path: str,
                        epochs: int = 10,
                        use_fbm_aug: bool = True,
                        device: str | torch.device = 'cuda') -> dict:
    """End-to-end: train a CIFAR-10 model quickly, evaluate on CIFAR-10-C, plot, and dump CSV.

    Returns a dict with: {'ckpt': path, 'metrics': metrics_dict}
    """
    # 1) Train
    cfg = TrainConfig(epochs=epochs, amp=True)
    out = train_cifar10(cfg, aug_fbm=use_fbm_aug, device=device)
    ckpt_path = out['checkpoint']

    # 2) Load best
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    # 3) Evaluate metrics on CIFAR-10-C (unnormalized)
    metrics = evaluate_cifar10c_metrics(model, c_path=cifar_c_path, device=device)

    # 4) Save per-corruption CSV for later analysis
    csv_out = Path('results_cifar10c.csv')
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['corruption', 'sev1_acc', 'sev2_acc', 'sev3_acc', 'sev4_acc', 'sev5_acc', 'mean_acc', 'ece_mean'])
        for corr, d in metrics['per_corruption'].items():
            row = [corr] + [d['acc'][s] for s in (1,2,3,4,5)] + [d['acc_mean'], d['ece_mean']]
            writer.writerow(row)

    # 5) Quick plots (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        # Example: accuracy vs severity for a few corruptions
        for corr in ('gaussian_noise', 'defocus_blur', 'jpeg_compression'):
            if corr in metrics['per_corruption']:
                sev = [1,2,3,4,5]
                acc = [metrics['per_corruption'][corr]['acc'][s] for s in sev]
                plt.figure()
                plt.plot(sev, acc, marker='o')
                plt.xlabel('Severity'); plt.ylabel('Top-1 accuracy'); plt.title(corr)
                plt.grid(True)
        plt.show()
    except Exception as ex:
        print('[plot] skipped:', ex)

    print('Saved CSV:', str(csv_out))
    return {"ckpt": ckpt_path, "metrics": metrics}
