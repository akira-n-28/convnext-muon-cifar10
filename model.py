"""model.py — ConvNeXt adattato per CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm con supporto channels_last e channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Il collo di bottiglia: riduce i canali e poi li ripristina
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze spaziale (1x1xC)
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()             # Excitation: mappa i pesi tra 0 e 1
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, _, _ = x.size()
        # Calcola i pesi e fa il reshape a (B, C, 1, 1) per il broadcasting
        w = self.fc(x).view(b, c, 1, 1)
        return x * w # Ricalibrazione dinamica dei canali

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, drop_path=0., layer_scale_init=1e-6, use_se=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_last')
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Iniezione condizionale dell'SE Block (lavora in data_format='channels_first')
        self.se = SEBlock(dim, reduction=16) if use_se else nn.Identity()
        
        if layer_scale_init > 0:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.register_parameter('gamma', None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        # Applichiamo l'attenzione SE subito prima del DropPath residuo
        x = self.se(x) 
        
        return shortcut + self.drop_path(x)


class ConvNeXt(nn.Module):
    """ConvNeXt scalato per CIFAR-10.

    kernel_size accetta int (uniforme su tutti gli stage) o tuple di lunghezza 4
    (uno per stage). Es: kernel_size=(7, 5, 3, 3) per kernel decrescente.
    """
    def __init__(self, in_channel=3, num_classes=10,
                 depths=(2, 2, 6, 2), dims=(64, 128, 256, 512),
                 kernel_size=5, drop_path_rate=0.1,
                 layer_scale_init=1e-6, stem_stride=1):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_sizes = (kernel_size,) * 4
        else:
            kernel_sizes = tuple(kernel_size)
            assert len(kernel_sizes) == 4, (
                f"kernel_size deve avere lunghezza 4 (uno per stage), "
                f"ricevuto len={len(kernel_sizes)}"
            )

        # Stem + 3 downsample
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_channel, dims[0], kernel_size=3, stride=stem_stride, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            ))

        # 4 stage di ConvNeXt block
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            # use_se sarà True SOLO per lo stage i == 2 (ovvero lo Stage 3 da 9 blocchi)
            use_se_stage = (i == 2) 
            
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i], kernel_size=kernel_sizes[i],
                              drop_path=dp_rates[cur + j],
                              layer_scale_init=layer_scale_init,
                              use_se=use_se_stage) 
                for j in range(depths[i])
            ]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for downsample_layer, stage in zip(self.downsample_layers, self.stages):
            x = downsample_layer(x)
            x = stage(x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.head(self.forward_features(x))
