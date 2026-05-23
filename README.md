# ConvNeXt + SE + Muon on CIFAR-10

A CIFAR-adapted **ConvNeXt** with selective **Squeeze-and-Excitation** blocks,
trained with a hybrid **Muon + AdamW** optimizer, reaching **98.10%** test
accuracy with only ~12M parameters.

## Result

| Metric | Value |
|---|---|
| **Test accuracy** | **98.10%** |
| Test error | 1.90% |
| Parameters | 12.37M |
| GPU | NVIDIA T4 (Kaggle) |
| Training time | ~10.7h |
| Epochs | 600 (best @ ep 529 via EMA) |

### Comparison with literature (~10–15M params range)

| Model | Params | Test acc | Notes |
|---|---|---|---|
| WRN-16-8 | 11M | ~97.0% | + Cutout |
| WRN-22-8 | 17M | ~97.3% | + AutoAugment + Cutout |
| WRN-28-10 | 36M | 97.44% | + AutoAugment + Cutout (Cubuk 2019) |
| **This work (ConvNeXt-12M + SE)** | **12M** | **98.10%** | Muon + EMA + SE + mixup |
| WRN-28-10 (AA best) | 36M | 98.52% | best published, 3× params |

## Architecture

CIFAR-adapted **ConvNeXt** (Liu et al. 2022) with **selective Squeeze-and-Excitation**
(Hu et al. 2018):

- `depths = (3, 3, 9, 3)`, `dims = (64, 128, 256, 512)`
- Per-stage kernel sizes: `(7, 5, 3, 3)` (large RF early, semantic late)
- Stem: `Conv3×3, stride=1` (preserves 32×32 input, no patchify)
- 2×2 stride-2 downsamples between stages
- **Squeeze-and-Excitation on stage 2 only** (9 blocks, dim=256, reduction=16) —
  see motivation below
- LayerScale init 1e-6, DropPath 0–0.15 (linear)

### Why SE only on stage 2

Squeeze-and-Excitation adds channel attention via a global-pool → bottleneck →
sigmoid gate, applied to the main branch before the residual merge. Placing SE
in *every* stage would either over-bottleneck early stages (with `C=64` and
`reduction=16` the bottleneck collapses to 4 units, hurting capacity) or add
parameter overhead disproportionate to the gain in the late stage with only 3
blocks. Stage 2 is the sweet spot: `C=256` gives a healthy 16-unit bottleneck,
and the 9 blocks accumulate the most representational depth in the network,
so per-block channel reweighting compounds well.

## Training recipe

| | |
|---|---|
| **Optimizer** | Muon (2D/4D weights) + AdamW (1D + head) |
| Muon LR | 0.02, momentum 0.95, wd 0.02 |
| AdamW LR | 1e-3, wd 0.0 |
| Schedule | Linear warmup → cosine annealing, `min_lr=1e-6` |
| Warmup | 30 epochs (5%) |
| Batch | 256, mixed precision fp16 |
| Augmentation | TrivialAugmentWide + RandomCrop(pad=6) + Flip + RandomErasing(p=0.25) |
| Mixup/CutMix | α=0.2 / 1.0, switch 0.5, prob 1.0, label_smoothing 0.1 |
| EMA | decay 0.9997 (half-life ≈ 12 epochs ≈ 2% of training) |
| Acceleration | `torch.compile(mode="reduce-overhead")` |

## Key engineering insights

**1. EMA decay must be calibrated to the training budget.** A previous 300-epoch
run with `decay=0.9999` underperformed: EMA acc 97.66% < live model 97.85%, i.e.
EMA *worse* than live. With `decay=0.9997` (half-life ≈ 12 epochs, ~2% of 600-ep
budget), EMA stabilized around epoch 100 and consistently beat the live model in
the second half (+0.33pp at best). Rule of thumb: EMA half-life ≤ 5% of total
budget.

**2. Muon on 2D/4D weights, AdamW on the rest.** Muon (Newton-Schulz
orthogonalization of gradient momentum, by Keller Jordan) handles conv and linear
weights; biases, LayerNorm, LayerScale γ, and classification head go to AdamW.
This split works out of the box on CIFAR with minimal LR tuning.

**3. EMA + `torch.compile` requires unwrapping the model.** `torch.compile` wraps
the model in `OptimizedModule`, whose `state_dict()` adds an `_orig_mod.` prefix
to all keys. When updating the EMA, pass `model._orig_mod` to avoid silent
key-order mismatches between EMA and live model state dicts:
```python
if ema is not None:
    ema.update(model._orig_mod if hasattr(model, '_orig_mod') else model)
```
Same applies when saving the live model — strip the compile wrapper before
`state_dict()` so the checkpoint can be reloaded into a non-compiled model.

**4. TTA saturates under aggressive augmentation.** Flip-TTA and 5-crop+flip TTA
do *not* improve the EMA result beyond numerical noise (σ ≈ 0.14pp at p=0.98 on
10k samples): −0.02pp and −0.10pp respectively. The combination of
RandomHorizontalFlip during training + EMA model averaging already saturates the
calibration benefit that TTA usually provides.

**5. Set `min_lr > 0` in cosine schedule.** With `min_lr=0`, the final ~50 epochs
run at LR ≈ 1e-13 and contribute nothing. `min_lr=1e-6` (0.1% of peak) keeps the
EMA gently improving through the very end.

## Repository structure

```
.
├── data.py       # CIFAR-10 pipeline, augmentations, Mixup factory
├── model.py      # ConvNeXt architecture (CIFAR-adapted) + SE block
├── muon.py       # Muon optimizer (Newton-Schulz orthogonalization)
├── train.py      # Training loop, AMP, EMA, param-group splitter
└── notebook/
    └── final-convnext-cifar-10-600ep.ipynb   # Kaggle training notebook
```

## Setup

```bash
git clone https://github.com/akira-n-28/convnext-muon-cifar10.git
cd convnext-muon-cifar10
pip install -r requirements.txt
```

**Note**: requires PyTorch ≥ 2.4 for the new `torch.amp` API used in the
training loop (`GradScaler('cuda')`, `autocast('cuda', ...)`).

## Usage

### Training (from scratch)

The provided `train.py` was authored on Kaggle and uses absolute paths under
`/kaggle/working/`. **On a local machine, edit the `__main__` block at the
bottom of `train.py`** to point to local paths:

```python
# Replace these lines in train.py
train_loader, val_loader = build_loaders(
    data_root="./data",                          # was: "/kaggle/working/data"
    batch_size=256,
    num_workers=4,
)
# ...
fit(
    model, train_loader, val_loader,
    # ... same args ...
    save_path="./best_600ep_compile.pt",         # was: "/kaggle/working/best_600ep_compile.pt"
)
```

Then run:
```bash
python -u train.py
```

Trains 600 epochs on CIFAR-10. ~10h on a single NVIDIA T4, faster on A100 or
RTX 4090 (~3–5h).

### Evaluation

```python
import torch
from data import build_loaders, DEVICE
from model import ConvNeXt
from train import evaluate, evaluate_tta, evaluate_multicrop_tta

_, val_loader = build_loaders(data_root="./data", batch_size=128)

model = ConvNeXt(
    depths=(3, 3, 9, 3), dims=(64, 128, 256, 512),
    kernel_size=(7, 5, 3, 3), drop_path_rate=0.15, layer_scale_init=1e-6,
).to(DEVICE)

ckpt = torch.load("best_600ep_compile.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

_, acc_no_tta = evaluate(model, val_loader)
_, acc_flip   = evaluate_tta(model, val_loader)
_, acc_5crop  = evaluate_multicrop_tta(model, val_loader)
print(f"no TTA:    {acc_no_tta*100:.2f}%")
print(f"flip TTA:  {acc_flip*100:.2f}%")
print(f"5-crop:    {acc_5crop*100:.2f}%")
```

## Pretrained checkpoint

Download from [Hugging Face Hub](https://huggingface.co/AkiraN28/convnext-muon-cifar10)


## Possible future improvements

In approximate value/effort order:

1. **Multi-seed ensemble** (2–3 runs): +0.20/+0.40 pp → 98.30–98.50%
2. **Self-distillation** with this model as teacher: +0.15/+0.30 pp
3. **GRN replacing LayerScale** (ConvNeXt v2 style): +0.05/+0.15 pp
4. **Scale to ~20M params** (`dims=(96, 192, 384, 768)`): +0.20/+0.30 pp

## References

- Liu et al., *A ConvNet for the 2020s*, CVPR 2022 — ConvNeXt architecture
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 — SE block
- Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks* — github.com/KellerJordan/Muon
- Cubuk et al., *AutoAugment: Learning Augmentation Policies*, CVPR 2019
- Müller et al., *TrivialAugment*, ICCV 2021
- Yun et al., *CutMix*, ICCV 2019; Zhang et al., *Mixup*, ICLR 2018
- Zhong et al., *Random Erasing Data Augmentation*, AAAI 2020

## License

MIT (see `LICENSE`).
