"""Strict T0 loading and frozen, dynamic-resolution feature extraction."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .config import BASE, SCALES, expanded_yaml, read_yaml
from .factories import checkpoint_state, component, sha256_file


def teacher_config(path):
    from engine.core.innovation_config import resolve_innovation
    cfg = resolve_innovation(expanded_yaml(path), expand_env=False)
    spec = read_yaml(BASE / 'teachers/vq_l.yml')
    if (cfg.get('model') != spec['architecture']
            or cfg['DEIM']['backbone'] != 'DINOv3STAs'
            or cfg['DINOv3STAs']['name'] != spec['backbone']
            or not cfg.get('innovation', {}).get('vq_dense_o2o', {}).get('enabled')
            or cfg['num_classes'] != 3):
        raise ValueError('T0 must be the explicitly selected three-class VQ-DEIMv2-L training configuration')
    cfg['DINOv3STAs']['weights_path'] = None
    return cfg


def load_detector(cfg):
    r = cfg['resources']
    path = r['THESIS_KD_TEACHER_CONFIG']
    checkpoint = r['THESIS_KD_TEACHER_CHECKPOINT']
    validation = (r['THESIS_KD_TEACHER_VALIDATION'] or '').strip()
    if not validation or not path or not checkpoint:
        raise ValueError('T0 needs an explicit training config, checkpoint and Validation selection basis')
    native = teacher_config(path)
    # Preserve stored FP32 EMA buffers even where the native backbone's
    # scratch constructor allocates lower-precision RoPE periods.
    model = component(native, native['model']).float()
    state, layout = checkpoint_state(checkpoint, r['THESIS_KD_TEACHER_STATE'] or 'ema')
    model.load_state_dict(state, strict=True)
    model.float().requires_grad_(False).eval()
    digest = sha256_file(checkpoint)
    cfg_digest = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
    identity = {'id': digest + '/' + layout['state_path'], 'checkpoint_sha256': digest,
                'path': str(Path(checkpoint).resolve()), 'config_path': str(Path(path).resolve()),
                'config_digest': cfg_digest, 'validation_basis': validation, **layout}
    return model, identity, native


class Teacher(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.backbone = detector.backbone
        self.encoder = detector.encoder
        # Only this private encoder uses dynamic positions. The full detector's
        # checkpoint (including fixed-resolution decoder anchor buffers) was verified first.
        self.encoder.eval_spatial_size = None
        if list(self.encoder.out_strides) != [8, 16, 32]:
            raise ValueError(f'Unexpected T0 encoder strides: {self.encoder.out_strides}')
        self.channels = dict(zip(SCALES, self.encoder.out_channels))
        self.float().requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, samples):
        from engine.solver.det_engine import fp32_evaluation
        if self.training:
            raise RuntimeError('Frozen T0 must be evaluated in eval mode')
        with fp32_evaluation(self, samples.device, enabled=True):
            features = self.encoder(self.backbone(samples.float()))
        if len(features) != 3:
            raise ValueError('Expected three T0 encoder outputs')
        h, w = samples.shape[-2:]
        for feature, stride, level in zip(features, self.encoder.out_strides, SCALES):
            if feature.shape[1:] != (self.channels[level], h // stride, w // stride):
                raise ValueError(f'Unexpected teacher {level} shape: {tuple(feature.shape)}')
        return dict(zip(SCALES, features))


def build_teacher(cfg):
    model, identity, native = load_detector(cfg)
    effective = deepcopy(native)
    effective['thesis_kd_feature_extraction'] = {
        'encoder_eval_spatial_size': None, 'precision': 'fp32', 'decoder_forward': False}
    return Teacher(model), identity, effective
