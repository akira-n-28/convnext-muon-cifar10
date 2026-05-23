"""data.py — Data pipeline per ConvNeXt-CIFAR.

Espone:
    DEVICE                 torch.device auto-detected (cuda > mps > cpu)
    CIFAR_MEAN, CIFAR_STD  statistiche per-canale del training set
    NUM_CLASSES            10
    train_transform        crop + flip + RandAugment + normalize
    val_transform          solo ToTensor + normalize
    build_loaders(...)     loader completo per training (CIFAR-10 full)
    smoke_loaders(...)     loader minimal per smoke test su MPS/CPU
    build_mixup(...)       factory per timm Mixup
"""

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# --- Constants ---------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


# --- Device ------------------------------------------------------------
def get_device() -> torch.device:
    """cuda > mps > cpu, in quest'ordine di preferenza."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()


# --- Transforms --------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=6, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value='random'),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


# --- DataLoaders -------------------------------------------------------
def _loader_kwargs(num_workers):
    """Defaults sensati per (num_workers, pin_memory, persistent_workers)."""
    if num_workers is None:
        num_workers = {'cuda': 4, 'mps': 0, 'cpu': 2}[DEVICE.type]
    pin_memory = (DEVICE.type == 'cuda')
    persistent = (num_workers > 0)
    return num_workers, pin_memory, persistent


def build_loaders(data_root='./data', batch_size=256, num_workers=None):
    """Carica CIFAR-10 completo con augmentation pipeline."""
    num_workers, pin_memory, persistent = _loader_kwargs(num_workers)

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


def smoke_loaders(data_root='./data', batch_size=64,
                  n_train=1000, n_val=200, seed=0):
    """Mini DataLoader per smoke test del training loop."""
    rng = np.random.default_rng(seed)
    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True,  download=True, transform=train_transform)
    full_val = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=val_transform)

    train_idx = rng.choice(len(full_train), n_train, replace=False)
    val_idx   = rng.choice(len(full_val),   n_val,   replace=False)

    train_loader = DataLoader(
        Subset(full_train, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        Subset(full_val, val_idx), batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


# --- Mixup / CutMix ----------------------------------------------------
def build_mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0,
                switch_prob=0.5, label_smoothing=0.1, num_classes=NUM_CLASSES):
    """Factory per timm.data.Mixup. Default tarati per budget 60 epoche."""
    from timm.data import Mixup
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=prob,
        switch_prob=switch_prob,
        mode='batch',
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
