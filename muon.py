"""muon.py — Muon optimizer (Keller Jordan et al.).

MomentUm Orthogonalized by Newton-Schulz: ortogonalizza l'update via NS
iteration prima di applicarlo. Solo per parametri 2D/4D (Linear, Conv).
I parametri 1D (bias, norm, gamma) devono usare AdamW separato.

Riferimento: https://github.com/KellerJordan/Muon
"""

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration: approssima G -> G * (G^T G)^(-1/2).
    Lavora in bfloat16 per stabilita' numerica e velocita'.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon per parametri 2D (Linear) e 4D (Conv).

    Update per parametro p:
        v_t = momentum * v_{t-1} + g_t
        g~  = g_t + momentum * v_t        (Nesterov) o solo v_t
        u   = newton_schulz(reshape_2D(g~))   # orthogonalized
        p   = (1 - lr*wd) * p - lr * sqrt(max(1, fan_out/fan_in)) * u
    """
    def __init__(self, params, lr=0.02, weight_decay=0.0,
                 momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']

                # SGD momentum
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf

                # Reshape 4D conv -> 2D matrix
                orig_shape = update.shape
                if update.ndim == 4:
                    update = update.view(update.size(0), -1)

                # Newton-Schulz orthogonalization
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                if len(orig_shape) == 4:
                    update = update.view(orig_shape)

                # Aspect-ratio scaling
                fan_out = p.size(0)
                fan_in = p.numel() // fan_out
                scale = max(1.0, fan_out / max(fan_in, 1)) ** 0.5

                # Decoupled weight decay + apply update
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * scale)
