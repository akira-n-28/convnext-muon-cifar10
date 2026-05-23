"""train.py - Training loop per ConvNeXt-CIFAR.

Supporta due regimi di optimizer:
    use_muon=False  -> AdamW solo (default ConvNeXt recipe)
    use_muon=True   -> Muon su 2D/4D weights + AdamW su 1D + head

Componenti:
    fit(...)                        main training loop
    train_one_epoch(...)            singolo epoch (forward + backward + step)
    evaluate(...)                   validation con accuracy (no TTA)
    evaluate_tta(...)               validation con horizontal-flip TTA
    param_groups(...)               split per AdamW solo
    param_groups_muon_adamw(...)    split per Muon+AdamW
    cosine_warmup_lr(...)           LR schedule lineare + cosine
"""

import math
import time

import torch
import torch.nn as nn

from data import DEVICE


# --- Parameter groups -------------------------------------------------
def param_groups(model, base_lr, weight_decay):
    """AdamW solo: groups con WD separato (no WD su bias/LN/gamma)."""
    decay, no_decay, gamma_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'gamma' in name:
            gamma_params.append(p)
        elif p.ndim <= 1 or 'norm' in name or 'bias' in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay,        'lr': base_lr,       'weight_decay': weight_decay},
        {'params': no_decay,     'lr': base_lr,       'weight_decay': 0.0},
        {'params': gamma_params, 'lr': base_lr , 'weight_decay': 0.0,},
    ]


def param_groups_muon_adamw(model, muon_wd=0.0, adamw_wd=0.02):
    """Split Muon+AdamW: 2D/4D hidden -> Muon, 1D + head -> AdamW.

    Convenzione (sul tuo ConvNeXt):
        Muon:   pwconv1.weight, pwconv2.weight, dwconv.weight,
                stem conv weight, downsample conv weights
        AdamW:  tutti i bias, LayerNorm weight/bias, LayerScale gamma,
                head.weight, head.bias
    """
    muon_params, adamw_decay, adamw_no_decay, gamma_params = [], [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Output head -> AdamW
        if name.startswith('head.'):
            (adamw_no_decay if 'bias' in name else adamw_decay).append(p)
            continue
        # LayerScale gamma -> AdamW con LR ridotto
        if 'gamma' in name:
            gamma_params.append(p)
            continue
        # 1D params (bias, norm) -> AdamW no decay
        if p.ndim <= 1 or 'norm' in name or 'bias' in name:
            adamw_no_decay.append(p)
            continue
        # 2D+ weights -> Muon
        muon_params.append(p)

    muon_groups = [{'params': muon_params, 'weight_decay': muon_wd}]
    adamw_groups = [
        {'params': adamw_decay,    'weight_decay': adamw_wd},
        {'params': adamw_no_decay, 'weight_decay': 0.0},
        {'params': gamma_params,   'weight_decay': 0.0},
    ]
    return muon_groups, adamw_groups


# --- LR schedule ------------------------------------------------------
def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    """LR per lo step corrente. Lineare durante warmup, poi cosine."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_scale', 1.0)


# --- Training step ----------------------------------------------------
def train_one_epoch(model, loader, optimizers, criterion, mixup_fn,
                    schedule_fn, step_counter,
                    ema=None, scaler=None, grad_clip=1.0):
    """optimizers: lista di (optimizer, base_lr) tuple.

    Per AdamW-solo: optimizers = [(adamw, base_lr)]
    Per Muon+AdamW: optimizers = [(adamw, adamw_base_lr), (muon, muon_base_lr)]
    """
    model.train()
    total_loss = 0.0
    n = 0
    use_amp = scaler is not None

    for x, y in loader:
        # LR per ogni optimizer (ciascuno con il proprio base_lr)
        for opt, base_lr in optimizers:
            set_lr(opt, schedule_fn(step_counter['v'], base_lr))

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        for opt, _ in optimizers:
            opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            for opt, _ in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            for opt, _ in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            for opt, _ in optimizers:
                opt.step()

        if ema is not None:
            ema.update(model._orig_mod if hasattr(model, '_orig_mod') else model)

        step_counter['v'] += 1
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(1, n)


# --- Validation -------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    """Accuracy + loss CE standard, no TTA."""
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = crit(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        n_total += bs
    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate_tta(model, loader):
    """Accuracy + loss con horizontal-flip TTA (media softmax)."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        probs = (model(x).softmax(dim=1) + model(x.flip(-1)).softmax(dim=1)) / 2.0
        loss = nn.functional.nll_loss(probs.clamp(min=1e-12).log(), y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n_correct  += (probs.argmax(dim=1) == y).sum().item()
        n_total    += bs
    return total_loss / n_total, n_correct / n_total

@torch.no_grad()
def evaluate_multicrop_tta(model, loader, padding=4):
    """Multi-crop + horizontal flip TTA.
    
    Per ogni immagine 32x32:
      1. Pad reflect a 40x40 (matching la pipeline di training)
      2. Estrai 5 crop 32x32 (4 angoli + centro)
      3. Applica horizontal flip a ciascuno -> 10 viste totali
      4. Media softmax su tutte le viste
    
    Atteso: +0.1-0.2pp rispetto al solo flip-TTA, ~5x piu' lento in eval.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    crop_offset = padding * 2   # offset per i 4 corner crop
    center_start = padding      # offset per il center crop
    
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        bs = x.size(0)
        
        # Pad reflect (stesso padding del training transform)
        x_pad = nn.functional.pad(x, [padding] * 4, mode='reflect')
        # x_pad shape: (bs, 3, 32+2*padding, 32+2*padding)
        
        # Estrai 5 crop di 32x32
        crops = torch.stack([
            x_pad[:, :, :32, :32],                          # top-left
            x_pad[:, :, :32, crop_offset:crop_offset+32],   # top-right
            x_pad[:, :, crop_offset:crop_offset+32, :32],   # bottom-left
            x_pad[:, :, crop_offset:crop_offset+32, crop_offset:crop_offset+32],  # bottom-right
            x_pad[:, :, center_start:center_start+32, center_start:center_start+32],  # center
        ], dim=0)  # (5, bs, 3, 32, 32)
        
        # Reshape per forward batch: (5*bs, 3, 32, 32)
        crops_flat = crops.view(5 * bs, 3, 32, 32)
        
        # Forward su originale + horizontal flip
        probs = model(crops_flat).softmax(dim=1)
        probs = probs + model(crops_flat.flip(-1)).softmax(dim=1)

        # Media: (5, bs, 10) -> mean su crop axis -> div 2 per le due flip-views
        probs = probs.view(5, bs, 10).mean(dim=0) / 2.0
        
        # NLL loss e accuracy
        loss = nn.functional.nll_loss(probs.clamp(min=1e-12).log(), y)
        total_loss += loss.item() * bs
        correct += (probs.argmax(dim=1) == y).sum().item()
        total += bs
    
    return total_loss / total, correct / total
# --- Main fit ---------------------------------------------------------
def fit(model, train_loader, val_loader,
        epochs=60, base_lr=4e-3, weight_decay=0.02,
        warmup_epochs=5, grad_clip=1.0,
        use_mixup=True, use_ema=True, use_amp=True,
        use_muon=False, muon_lr=0.01, muon_momentum=0.95, muon_wd=0.0,
        mixup_alpha=0.2, cutmix_alpha=1.0,
        label_smoothing=0.1, ema_decay=0.999, use_compile=False,
        save_path="best.pt"):
    """Training loop completo.

    Args principali:
        epochs:           numero totale di epoch
        base_lr:          peak LR per AdamW
        weight_decay:     WD per i weight di Linear/Conv in AdamW (no su bias/LN/gamma)
        warmup_epochs:    durata warmup lineare
        use_muon:         True per Muon (2D/4D) + AdamW (1D/head)
        muon_lr:          peak LR per Muon (tipicamente 2-10x AdamW)
        muon_momentum:    momentum SGD-base per Muon (default 0.95)
        muon_wd:          WD applicato dentro Muon (default 0, raramente serve)
        mixup_alpha:      Beta(alpha, alpha) per Mixup. 0.2 leggero per budget corti
        cutmix_alpha:     idem per CutMix. 1.0 = uniform mixing
        ema_decay:        decay EMA. 0.999 per 60 ep, 0.9995-0.9999 per 100+
    """
    # Criterion + mixup
    if use_mixup:
        from timm.loss import SoftTargetCrossEntropy
        from data import build_mixup
        mixup_fn = build_mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=1.0,
            switch_prob=0.5,
            label_smoothing=label_smoothing,
        )
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer(s)
    if use_muon:
        from muon import Muon
        muon_groups, adamw_groups = param_groups_muon_adamw(
            model, muon_wd=muon_wd, adamw_wd=weight_decay
        )
        muon_opt  = Muon(muon_groups, lr=muon_lr, momentum=muon_momentum)
        adamw_opt = torch.optim.AdamW(adamw_groups, lr=base_lr,
                                       betas=(0.9, 0.999), eps=1e-8)
        optimizers = [(adamw_opt, base_lr), (muon_opt, muon_lr)]
        opt_str = (f"Muon(lr={muon_lr}, mom={muon_momentum}, wd={muon_wd}) + "
                   f"AdamW(lr={base_lr:.0e}, wd={weight_decay})")
    else:
        adamw_opt = torch.optim.AdamW(
            param_groups(model, base_lr, weight_decay),
            betas=(0.9, 0.999), eps=1e-8,
        )
        optimizers = [(adamw_opt, base_lr)]
        opt_str = f"AdamW(lr={base_lr:.0e}, wd={weight_decay})"

    # EMA
    ema = None
    if use_ema:
        from timm.utils import ModelEmaV2
        ema = ModelEmaV2(model, decay=ema_decay, device=DEVICE)

    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")

    # AMP scaler 
    scaler = None
    if use_amp and DEVICE.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')

    # Schedule (per-optimizer, ciascuno con il proprio base_lr)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    schedule_fn = lambda s, base: cosine_warmup_lr(s, total_steps, warmup_steps, base)
    step_counter = {'v': 0}

    # Header
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Device={DEVICE.type}  amp={scaler is not None}  "
          f"ema={ema is not None}  mixup={mixup_fn is not None}  muon={use_muon}")
    print(f"Params={n_params:.2f}M  epochs={epochs}  "
          f"steps/ep={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")
    print(f"Optimizer: {opt_str}")
    print(f"ls={label_smoothing}  clip={grad_clip}  ema_decay={ema_decay}")
    print('-' * 100)

    best_acc = 0.0
    best_kind = "model"

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizers, criterion, mixup_fn,
            schedule_fn, step_counter,
            ema=ema, scaler=scaler, grad_clip=grad_clip,
        )

        val_loss, val_acc = evaluate(model, val_loader)
        ema_loss, ema_acc = None, None
        if ema is not None:
            ema_loss, ema_acc = evaluate(ema.module, val_loader)

        current_lr = optimizers[0][0].param_groups[0]['lr']
        dt = time.time() - t0

        # Resolve underlying model (strip torch.compile wrapper se presente)
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        # Pick best between live model and EMA
        if ema_acc is not None and ema_acc >= val_acc:
            current_best_acc = ema_acc
            current_best_loss = ema_loss
            current_state = ema.module.state_dict()
            current_kind = "ema"
        else:
            current_best_acc = val_acc
            current_best_loss = val_loss
            current_state = base_model.state_dict()        
            current_kind = "model"

        if current_best_acc > best_acc:
            best_acc = current_best_acc
            best_kind = current_kind
            torch.save({
                "epoch": epoch + 1,
                "model_state": current_state,
                "best_acc": best_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "ema_acc": ema_acc,
                "ema_loss": ema_loss,
                "kind": current_kind,
                "config": {
                    "epochs": epochs,
                    "base_lr": base_lr,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "grad_clip": grad_clip,
                    "use_mixup": use_mixup,
                    "use_ema": use_ema,
                    "use_muon": use_muon,
                    "muon_lr": muon_lr if use_muon else None,
                    "muon_momentum": muon_momentum if use_muon else None,
                    "muon_wd": muon_wd if use_muon else None,
                    "mixup_alpha": mixup_alpha,
                    "cutmix_alpha": cutmix_alpha,
                    "label_smoothing": label_smoothing,
                    "ema_decay": ema_decay,
                }
            }, save_path)
            print(f"  -> saved best checkpoint: {save_path} "
                  f"({current_kind}, acc={best_acc:.4f}, loss={current_best_loss:.4f})")

        # Backup periodico ogni 50 epoche (per resume in caso di timeout Kaggle)
        if (epoch + 1) % 50 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state": base_model.state_dict(),
                "ema_state": ema.module.state_dict() if ema is not None else None,
                "best_acc": best_acc,
            }, save_path.replace(".pt", "_last.pt"))

    print('-' * 100)
    print(f"Best validation accuracy: {best_acc:.4f}")
    return best_acc


# --- Self test --------------------------------------------------------
if __name__ == '__main__':
    from data import build_loaders
    from model import ConvNeXt

    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = build_loaders(
        data_root="/kaggle/working/data",
        batch_size=256,
        num_workers=4,
    )

    model = ConvNeXt(
        depths=(3, 3, 9, 3),
        dims=(64, 128, 256, 512),
        kernel_size=(7, 5, 3, 3),
        drop_path_rate=0.15,
        layer_scale_init=1e-6,
    ).to(DEVICE)
    
    fit(
        model, train_loader, val_loader,
        epochs=600,
        warmup_epochs=30,
        base_lr=1e-3,
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_wd=0.02,
        weight_decay=0.0,
        grad_clip=1.0,
        use_mixup=True,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        use_ema=True,
        use_amp=True,
        use_muon=True,
        use_compile=True,  
        label_smoothing=0.1,
        ema_decay=0.9997,
        save_path="/kaggle/working/best_600ep_compile.pt",
    )
        
