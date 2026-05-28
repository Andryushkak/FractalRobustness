from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fractal_robustness_templates import fd_boxcount, CIFARResNet18
from fractal_robustness_templates_B import get_cifar10_loaders, set_seed


@torch.no_grad()
def compute_fd_inputs(loader: DataLoader, n_samples: int = 512) -> List[float]:
    """FD для вхідних зображень."""
    fds: List[float] = []
    seen = 0
    for xb, _ in loader:
        x = xb.clone()
        # якщо раптом нормалізовано до від'ємних – повернемо в [0,1]
        if x.min() < 0:
            x = x - x.min()
            x = x / (x.max() + 1e-8)
        x = x.permute(0, 2, 3, 1).cpu().numpy()  # (N,H,W,C)
        for img in x:
            fds.append(fd_boxcount(img))
            seen += 1
            if seen >= n_samples:
                return fds
    return fds


def hook_activations(module, inp, out, storage: List[torch.Tensor]):
    storage.append(out.detach().cpu())


def hook_gradients(module, grad_in, grad_out, storage: List[torch.Tensor]):
    # grad_out[0]: градієнти по виходу шару
    storage.append(grad_out[0].detach().cpu())


def compute_fd_activations_and_grads(
    model: nn.Module,
    loader: DataLoader,
    layer_names: List[str],
    device: str = 'cuda',
    n_batches: int = 10,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Для зазначених шарів рахує FD для активацій і градієнт‑карт.
    Повертає: {layer_name: {"acts": [fd...], "grads": [fd...]}}
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()  # потрібно для градієнтів

    # знайти вибрані шари
    layers: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if name in layer_names:
            layers[name] = module

    act_storages = {name: [] for name in layer_names}
    grad_storages = {name: [] for name in layer_names}
    hooks = []

    # реєструємо хуки
    for name, module in layers.items():
        hooks.append(module.register_forward_hook(
            lambda m, inp, out, n=name: hook_activations(m, inp, out, act_storages[n])
        ))
        hooks.append(module.register_full_backward_hook(
            lambda m, gin, gout, n=name: hook_gradients(m, gin, gout, grad_storages[n])
        ))

    criterion = nn.CrossEntropyLoss()
    batches_done = 0

    # прогін кількох батчів із backprop для збору градієнтів
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        batches_done += 1
        if batches_done >= n_batches:
            break

    # знімаємо хуки
    for h in hooks:
        h.remove()

    # обчислення FD
    result: Dict[str, Dict[str, List[float]]] = {}
    for name in layer_names:
        acts_tensors = act_storages[name]
        grads_tensors = grad_storages[name]

        act_fds: List[float] = []
        grad_fds: List[float] = []

        # Активації: (N,C,H,W) -> усереднюємо по каналах -> (H,W)
        for t in acts_tensors:
            arr = t.detach().cpu()           # (N,C,H,W)
            N = arr.shape[0]
            for i in range(N):
                a = arr[i]                  # (C,H,W)
                a = a - a.min()
                a = a / (a.max() + 1e-8)
                a_mean = a.mean(dim=0).numpy()  # (H,W)
                act_fds.append(fd_boxcount(a_mean))

        # Градієнти: (N,C,H,W) -> abs, нормалізація, усереднення по каналах -> (H,W)
        for t in grads_tensors:
            arr = t.detach().cpu()
            N = arr.shape[0]
            for i in range(N):
                g = arr[i].abs()           # (C,H,W)
                g = g - g.min()
                g = g / (g.max() + 1e-8)
                g_mean = g.mean(dim=0).numpy()  # (H,W)
                grad_fds.append(fd_boxcount(g_mean))

        result[name] = {"acts": act_fds, "grads": grad_fds}

    return result


def demo_fractal_analysis(
    ckpt_path: str,
    layer_names: List[str],
    n_batches: int = 5
):
    """Демо: FD для входів, активацій і градієнтів кількох шарів."""
    set_seed(42)
    train_loader, _ = get_cifar10_loaders()
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    print("Computing FD for inputs...")
    input_fds = compute_fd_inputs(train_loader, n_samples=256)
    print(f"Mean FD(inputs) = {np.mean(input_fds):.3f}")

    print("Computing FD for activations and gradients...")
    fd_dict = compute_fd_activations_and_grads(
        model, train_loader, layer_names=layer_names, n_batches=n_batches
    )
    for name, d in fd_dict.items():
        print(f"Layer {name}: FD(acts)={np.mean(d['acts']):.3f}, FD(grads)={np.mean(d['grads']):.3f}")


if __name__ == "__main__":
    demo_fractal_analysis(
        ckpt_path="checkpoints/base_clean/best.pt",
        layer_names=["net.layer1.0.conv1", "net.layer3.1.conv2"],
        n_batches=3
    )
