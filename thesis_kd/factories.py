"""Construct native objects using private copies of the registered schemas."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import types

import torch

from .config import ROOT, merge


def context(native):
    import engine
    if Path(engine.__file__).resolve() != ROOT / 'engine/__init__.py':
        raise ImportError(f'Wrong engine imported: {engine.__file__}')
    from engine.core import GLOBAL_CONFIG
    # Module objects are immutable references to implementations, not schemas.
    modules = {id(value): value for schema in GLOBAL_CONFIG.values() if isinstance(schema, dict)
               for value in schema.values() if isinstance(value, types.ModuleType)}
    registry = deepcopy(dict(GLOBAL_CONFIG), modules)
    return merge(registry, native)


def component(native, name, **kwargs):
    from engine.core import create
    registry = context(native)
    if isinstance(registry.get(name), str):
        name = registry[name]
    if name in ('train_dataloader', 'val_dataloader'):
        total = registry[name].pop('total_batch_size', None)
        if total and 'batch_size' not in kwargs:
            world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if total % world:
                raise ValueError(f'Global batch {total} not divisible by world size {world}')
            kwargs['batch_size'] = total // world
    return create(name, registry, **kwargs)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def _tensor_mapping(value):
    return (isinstance(value, Mapping) and bool(value)
            and all(isinstance(key, str) and torch.is_tensor(v) for key, v in value.items()))


def checkpoint_state(path, branch):
    """Only use an explicitly supported state layout. No fallback from teacher EMA."""
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    # The trusted, user-specified native training files can include Python RNG metadata.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError('Checkpoint must be a state or named checkpoint mapping')
    choices = {
        'ema': [('ema', 'module'), ('ema',), ('model_ema', 'module'), ('model_ema',)],
        'model': [('model',)], 'state_dict': [('state_dict',)], 'raw': [()],
    }
    if branch == 'pretrain':
        branches = choices['ema'] + choices['model'] + choices['state_dict'] + [()]
    elif branch in choices:
        branches = choices[branch]
    else:
        raise ValueError(f'Unknown checkpoint state selection: {branch}')
    for parts in branches:
        value = payload
        for part in parts:
            value = value.get(part) if isinstance(value, Mapping) else None
        if not _tensor_mapping(value):
            continue
        has_ddp = all(key.startswith('module.') for key in value)
        state = {key[7:] if has_ddp else key: tensor for key, tensor in value.items()}
        return state, {'state_path': '.'.join(parts) or 'raw', 'ddp_prefix_removed': has_ddp}
    raise ValueError(f'Missing tensor state for {branch!r}; available checkpoint keys: {list(payload)[:25]}')


_HEAD = re.compile(r'decoder\.(?:denoising_class_embed\.weight|(?:enc_score_head|dec_score_head\.\d+)\.(?:weight|bias))$')


def build_student(cfg, initialize=True):
    native = deepcopy(cfg['native'])
    native['DINOv3STAs']['weights_path'] = None
    detector = component(native, native['model'])
    if not initialize:
        return detector, {}
    prefix = 'THESIS_KD_' + cfg['arch'].upper()
    backbone = cfg['resources'][prefix + '_BACKBONE_WEIGHTS']
    pretrain = cfg['resources'][prefix + '_DETECTOR_CHECKPOINT']
    state, backbone_layout = checkpoint_state(backbone, 'pretrain')
    detector.backbone.dinov3._model.load_state_dict(state, strict=True)
    del state
    state, layout = checkpoint_state(pretrain, 'pretrain')
    target = detector.state_dict()
    skipped = []
    for key in target.keys() & state.keys():
        if target[key].shape != state[key].shape and _HEAD.fullmatch(key):
            # Only the number of classes may differ, not decoder width.
            if state[key].shape[1:] != target[key].shape[1:]:
                raise ValueError(f'Wrong decoder width in {key}')
            state[key] = target[key]
            skipped.append(key)
    detector.load_state_dict(state, strict=True)
    return detector, {'backbone': backbone, 'backbone_layout': backbone_layout,
                      'detector': pretrain, 'detector_layout': layout,
                      'skipped_class_heads': sorted(skipped)}
