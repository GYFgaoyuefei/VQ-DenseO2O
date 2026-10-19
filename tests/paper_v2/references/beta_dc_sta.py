# Frozen numerical reference from beta_select f92edb021fd468e354f5767bb81ac745e0347cfe.
# Original DCSTA class; only unrelated registration/imports and adapter removed.
"""Lightweight degradation-conditioned modulation of the existing STA detail slice."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F



class DCSTA(nn.Module):
    """Global channel and local spatial gates; feature levels are c2/c3/c4."""

    def __init__(self, detail_channels: list[int], options: dict) -> None:
        super().__init__()
        if options.get('descriptor_dim', 8) != 8 or options.get('levels', ['c2', 'c3', 'c4']) != ['c2', 'c3', 'c4']:
            raise ValueError('DC-STA requires the eight-statistic descriptor and native c2/c3/c4 levels')
        if options.get('gate_init', 'identity') != 'identity':
            raise ValueError('Only baseline-compatible initialization is supported')
        self.options = dict(options)
        self.global_on = options.get('use_global_gate', True) and options.get('channel_gate', True)
        self.local_on = options.get('use_local_reliability', True) and options.get('spatial_gate', True)
        active = int(self.global_on) + int(self.local_on)
        gain = float(options.get('init_detail_gain', .999))
        if not 0 < gain < 1 or active == 0:
            raise ValueError('Enable a gate and use an initial detail gain strictly between 0 and 1')
        probability = gain ** (1 / active)
        bias = math.log(probability / (1 - probability))
        self.channel = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, c))
            for c in detail_channels
        ]) if self.global_on else nn.ModuleList()
        self.spatial = nn.ModuleList([nn.Conv2d(10, 1, 3, padding=1) for _ in detail_channels]) if self.local_on else nn.ModuleList()
        for mlp in self.channel:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.constant_(mlp[-1].bias, bias)
        for conv in self.spatial:
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, bias)
        self.capture_diagnostics = False
        self.last_statistics: dict = {}

    def descriptor(self, images: torch.Tensor) -> torch.Tensor:
        """Return [B,8] continuous statistics from inverse-normalized RGB."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('Expected [B,3,H,W] normalized RGB')
        with torch.autocast(device_type=images.device.type, enabled=False):
            x = images.float()
            rgb = (x * x.new_tensor([.229, .224, .225])[None, :, None, None]
                   + x.new_tensor([.485, .456, .406])[None, :, None, None]).clamp(0, 1)
            rgb = F.adaptive_avg_pool2d(rgb, (80, 80))
            y = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
            hf = .5 * ((y[:, 1:] - y[:, :-1]).abs().mean((1, 2))
                       + (y[:, :, 1:] - y[:, :, :-1]).abs().mean((1, 2)))
            lum = torch.stack([y.mean((1, 2)), (y.var((1, 2), unbiased=False) + 1e-8).sqrt(),
                               torch.sigmoid(12 * (.2 - y)).mean((1, 2)),
                               torch.sigmoid(12 * (y - .85)).mean((1, 2))], 1)
            if not self.options.get('use_luminance_stats', True):
                lum = torch.zeros_like(lum)
            color = rgb.mean((2, 3))
            if not self.options.get('use_color_stats', True):
                color = torch.zeros_like(color)
            if not self.options.get('use_high_frequency_proxy', True):
                hf = torch.zeros_like(hf)
            descriptor = torch.cat([lum, color, hf[:, None]], 1)
            if not torch.isfinite(descriptor).all():
                raise FloatingPointError('Nonfinite DC descriptor')
            return descriptor

    def forward(self, images: torch.Tensor, semantic: list[torch.Tensor], detail: tuple) -> list[torch.Tensor]:
        """Modulate only S_l; preserve existing semantic projection and normalization."""
        d = self.descriptor(images)
        result = []
        self.last_statistics = {}
        for index, (sem, sta) in enumerate(zip(semantic, detail)):
            if sem.shape[0] != sta.shape[0] or sem.shape[-2:] != sta.shape[-2:]:
                raise ValueError('Semantic/detail pyramid shapes differ')
            a = torch.sigmoid(self.channel[index](d))[:, :, None, None] if self.global_on else sta.new_ones((sta.shape[0], 1, 1, 1))
            if self.local_on:
                local_input = torch.cat([sem.mean(1, keepdim=True), sta.mean(1, keepdim=True),
                                         d.to(sta.dtype)[:, :, None, None].expand(-1, -1, *sta.shape[-2:])], 1)
                r = torch.sigmoid(self.spatial[index](local_input))
            else:
                r = sta.new_ones((sta.shape[0], 1, *sta.shape[-2:]))
            result.append(sta * a.to(sta.dtype) * r.to(sta.dtype))
            if self.capture_diagnostics:
                self.last_statistics[f'c{index + 2}'] = {
                    'global_mean': a.detach().float().flatten(1).mean(1),
                    'global_std': a.detach().float().flatten(1).std(1, unbiased=False),
                    'local_mean': r.detach().float().flatten(1).mean(1),
                    'local_std': r.detach().float().flatten(1).std(1, unbiased=False),
                }
        return result

