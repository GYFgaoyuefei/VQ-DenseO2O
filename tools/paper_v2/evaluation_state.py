"""Inference-only loading of existing detectors without experiment-version gates.

Training keeps its original checkpoint guards. Here the saved inference options
define the computation, while strict loading checks every actual weight/buffer.
The historical DC bookkeeping marker is not an inference tensor.
"""
import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from engine.backbone.dc_sta import DCSTA


def prepare_inference_config(cfg):
    """Preserve saved DC options and defer its construction to the eval loader.

    Call only in existing-run evaluation mode, before constructing the detector.
    This changes neither the source YAML nor any training configuration on disk.
    """
    if '_evaluation_dc_options' in cfg:
        return cfg
    backbone_name = cfg.get('DEIM', {}).get('backbone')
    legacy_alias = backbone_name == 'DCDINOv3STAs'
    if legacy_alias:
        saved = cfg.get('DCDINOv3STAs')
        if not isinstance(saved, dict):
            raise ValueError('Saved DCDINOv3STAs inference configuration is missing')
        cfg['DINOv3STAs'] = copy.deepcopy(saved)
        cfg['DEIM']['backbone'] = 'DINOv3STAs'
        backbone_name = 'DINOv3STAs'
    if backbone_name != 'DINOv3STAs':
        return cfg
    backbone = cfg['DINOv3STAs']
    options = backbone.get('dc_sta')
    if options is None:
        options = cfg.get('innovation', {}).get('dc_sta', {})
    if not isinstance(options, dict):
        raise ValueError('Saved DC inference options must be a dictionary')
    options = copy.deepcopy(options)
    if not isinstance(options.get('enabled', False), bool):
        raise ValueError('Saved DC enabled option must be boolean')
    if legacy_alias:
        # The historical subclass always constructed its DC module.
        if options.get('enabled', True) is not True:
            raise ValueError('DCDINOv3STAs conflicts with disabled saved DC options')
        options['enabled'] = True
    cfg['_evaluation_dc_options'] = options
    backbone['dc_sta'] = None
    backbone['weights_path'] = None
    return cfg


class _EvaluationDCSTA(DCSTA):
    """Saved sigmoid or residual-tanh computation, with no marker prerequisite."""

    def __init__(self, detail_channels, options):
        options = dict(options)
        self.residual = any(key in options for key in ('alpha', 'input_mean', 'input_std'))
        if self.residual:
            unknown = set(options) - {'enabled', 'alpha', 'input_mean', 'input_std'}
            if unknown:
                raise ValueError(f'Unsupported residual DC inference options: {sorted(unknown)}')
            # Both historical residual gates are present. All initialized weights
            # below must be replaced by the strict checkpoint load before use.
            super().__init__(detail_channels, {'enabled': True})
            self.alpha = float(options.get('alpha', .1))
            if not math.isfinite(self.alpha) or not 0 <= self.alpha < 1:
                raise ValueError('Saved residual DC alpha must be finite and in [0, 1)')
            mean = torch.as_tensor(options.get('input_mean', [.485, .456, .406]), dtype=torch.float32)
            std = torch.as_tensor(options.get('input_std', [.229, .224, .225]), dtype=torch.float32)
            if (mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all()
                    or not torch.isfinite(std).all() or not (std > 0).all()):
                raise ValueError('Saved residual DC normalization must contain finite RGB triples and positive std')
            self.register_buffer('input_mean', mean.view(1, 3, 1, 1), persistent=False)
            self.register_buffer('input_std', std.view(1, 3, 1, 1), persistent=False)
            self.options = options
        else:
            super().__init__(detail_channels, options)
        del self._buffers['_gate_architecture']

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        nn.Module._load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

    def descriptor(self, images):
        if not self.residual:
            return super().descriptor(images)
        # Exact saved residual formula, including std without the sigmoid
        # descriptor's +1e-8 stabilization and its configurable normalization.
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError('DC-STA expects [B,3,H,W] normalized RGB images')
        with torch.autocast(device_type=images.device.type, enabled=False):
            rgb = (images.float() * self.input_std.float() + self.input_mean.float()).clamp(0, 1)
            rgb = F.adaptive_avg_pool2d(rgb, (80, 80))
            luminance = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
            finite_difference = .5 * (
                (luminance[:, 1:, :] - luminance[:, :-1, :]).abs().mean((1, 2))
                + (luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean((1, 2)))
            statistics = torch.stack([
                luminance.mean((1, 2)), luminance.std((1, 2), unbiased=False),
                torch.sigmoid(12 * (.2 - luminance)).mean((1, 2)),
                torch.sigmoid(12 * (luminance - .85)).mean((1, 2)),
            ], dim=1)
            return torch.cat([statistics, rgb.mean((2, 3)), finite_difference[:, None]], dim=1)

    def compute_gates(self, descriptor, semantic, detail, level):
        if not self.residual:
            return super().compute_gates(descriptor, semantic, detail, level)
        mlp = self.channel[level]
        channel = 1 + self.alpha * torch.tanh(mlp(descriptor.to(mlp[0].weight.dtype)))
        local_input = torch.cat([
            semantic.mean(1, keepdim=True), detail.mean(1, keepdim=True),
            descriptor.to(detail.dtype)[:, :, None, None].expand(-1, -1, *detail.shape[-2:]),
        ], dim=1)
        spatial = 1 + self.alpha * torch.tanh(self.spatial[level](local_input))
        return channel[:, :, None, None].to(detail.dtype), spatial.to(detail.dtype)


def load_evaluation_state(model, ema, run_policy='strict', cfg=None):
    """Load all detector tensors strictly; existing mode ignores DC version tags.

    Sigmoid and residual DC have identical learned tensor shapes, so the saved
    options, not weight shapes or a version number, choose their mathematics.
    """
    if run_policy == 'strict':
        return model.load_state_dict(ema, strict=True)
    if run_policy != 'existing-checkpoint':
        raise ValueError(f'Unknown evaluation run policy: {run_policy}')
    if not isinstance(ema, Mapping) or not ema:
        raise ValueError('Evaluation requires a nonempty EMA module state dictionary')
    options = (cfg or {}).get('_evaluation_dc_options')
    backbone = getattr(model, 'backbone', None)
    gate = getattr(backbone, 'dc_sta', None)
    if options is None and gate is not None:
        options = copy.deepcopy(gate.options)
        options['enabled'] = True
    if options and options.get('enabled', False):
        if backbone is None or not getattr(backbone, 'use_sta', False):
            raise ValueError('Saved DC options require the native STA backbone')
        if len(getattr(backbone, 'interaction_indexes', ())) != 3:
            raise ValueError('Saved DC options require three backbone feature levels')
        try:
            channels = [conv.in_channels - backbone.dinov3.embed_dim for conv in backbone.convs]
        except AttributeError as error:
            raise ValueError('Cannot recover native DC detail channel counts from the detector') from error
        reference = next(backbone.parameters())
        backbone.dc_sta = _EvaluationDCSTA(channels, options).to(
            device=reference.device, dtype=reference.dtype).train(backbone.training)
    # Copy without modifying the loaded checkpoint or dropping actual parameters.
    state = copy.copy(ema)
    if hasattr(ema, '_metadata'):
        state._metadata = copy.deepcopy(ema._metadata)
    for name, module in model.named_modules():
        if isinstance(module, _EvaluationDCSTA):
            state.pop((name + '.' if name else '') + '_gate_architecture', None)
    try:
        return model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError('Checkpoint tensors do not match the configured evaluation detector; '
                           'no learned weights were dropped or initialized as a fallback. '
                           f'{error}') from error
