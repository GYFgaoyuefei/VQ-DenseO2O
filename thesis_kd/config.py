"""Resolve the isolated protocol without importing torch, engine or training resources."""
from copy import deepcopy
import os
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/tomato/thesis_kd'
SCALES = ('P3', 'P4', 'P5')
MODEL_KEYS = ('model', 'DEIM', 'DINOv3STAs', 'HybridEncoder', 'DEIMTransformer',
              'eval_spatial_size', 'use_focal_loss', 'PostProcessor')


def read_yaml(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected YAML mapping in {path}')
    return value


def merge(target, layer, sources=None, origin='', prefix=''):
    for key, value in layer.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            merge(target[key], value, sources, origin, name + '.')
        else:
            target[key] = deepcopy(value)
            if sources is not None:
                sources[name] = origin
    return target


def expanded_yaml(path, trail=(), sources=None):
    path = Path(path).resolve()
    if path in trail:
        raise ValueError(f'Cyclic YAML include: {path}')
    doc = read_yaml(path)
    out = {}
    for parent in doc.pop('__include__', []):
        merge(out, expanded_yaml(path.parent / parent, trail + (path,), sources))
    origin = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return merge(out, doc, sources, origin)


def _resources(paths):
    defaults = read_yaml(BASE / 'local/paths.example.yml')
    local = read_yaml(paths) if paths else {}
    unknown = local.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown resource fields: {sorted(unknown)}')
    values, origins = {}, {}
    for key, default in defaults.items():
        value = os.environ.get(key) or local.get(key) or default
        if value is not None and not isinstance(value, str):
            raise TypeError(f'{key} requires a string')
        if value and key not in ('THESIS_KD_TEACHER_VALIDATION', 'THESIS_KD_TEACHER_STATE',
                                 'THESIS_KD_OUTPUT_ROOT'):
            value = os.path.abspath(ROOT / Path(value).expanduser())
        values[key] = value
        origins['resources.' + key] = ('environment' if os.environ.get(key)
            else str(paths) if key in local else 'local/paths.example.yml')
    return values, origins


def resolve_config(experiment, paths=None):
    resources, origins = _resources(paths)
    return _compose(experiment, resources, origins)


def _compose(experiment, resources, path_sources):
    if not re.fullmatch(r'[a-z0-9_]+', experiment):
        raise ValueError(f'Invalid experiment name {experiment!r}')
    matches = [BASE / part / f'{experiment}.yml' for part in
               ('experiments', 'ablations', 'sensitivity')]
    experiment_file = next((p for p in matches if p.is_file()), None)
    if experiment_file is None:
        raise ValueError(f'Unknown experiment {experiment!r}')
    override = read_yaml(experiment_file)
    arch = override['arch']
    if arch not in ('m', 's'):
        raise ValueError(f'Unsupported architecture {arch}')
    cfg = {'experiment': experiment, 'native': {}}
    sources = {}
    native = cfg['native']

    def apply(path, target, prefix=''):
        return merge(target, read_yaml(BASE / path), sources, path, prefix)

    apply('data/tomato_rgb.yml', native, 'native.')
    model_file = f'models/student_{arch}.yml'
    spec = read_yaml(BASE / model_file)
    official_sources = {}
    official = expanded_yaml(ROOT / spec.pop('architecture_source'), sources=official_sources)
    for key in MODEL_KEYS:
        if key in official:
            merge(native, {key: official[key]})
            for name, source in official_sources.items():
                if name == key or name.startswith(key + '.'):
                    sources['native.' + name] = source
    merge(native, spec, sources, model_file, 'native.')
    apply('recipes/train_120e.yml', cfg)
    cfg['kd'] = {}
    apply('distillation/feature_mse.yml', cfg['kd'], 'kd.')
    merge(cfg, override, sources, experiment_file.relative_to(BASE).as_posix())
    cfg['resources'] = deepcopy(resources)
    sources.update(path_sources)
    data_root = cfg['resources']['TOMATO_DATA_ROOT']
    if data_root:
        for split in ('train', 'val'):
            dataset = native[split + '_dataloader']['dataset']
            for field in ('img_folder', 'ann_file'):
                dataset[field] = dataset[field].replace('${TOMATO_DATA_ROOT}', data_root)
    cfg['sources'] = sources
    kd = cfg['kd']
    if (not kd['active_scales'] or len(set(kd['active_scales'])) != len(kd['active_scales'])
            or set(kd['active_scales']) - set(SCALES) or kd['lambda_kd'] < 0
            or kd['reduction'] != 'active_scales_mean' or kd['teacher_precision'] != 'fp32'):
        raise ValueError('Invalid feature KD protocol')
    return cfg


def kd_enabled(cfg):
    return bool(cfg['kd']['enabled'] and cfg['kd']['lambda_kd'] > 0)


def pair_diff(cfg):
    """Compare all resolved non-KD training settings with the same-size baseline."""
    counterpart = cfg['arch'] + ('_kd' if cfg['experiment'].endswith('_base') else '_base')
    other = _compose(counterpart, cfg['resources'], {})
    excluded = {'experiment', 'resources', 'sources', 'kd'}
    differences = [key for key in (cfg.keys() | other.keys()) - excluded
                   if cfg.get(key) != other.get(key)]
    return {'compared': [cfg['experiment'], other['experiment']],
            'non_kd_differences': sorted(differences),
            'kd_differences': {k: [cfg['kd'][k], other['kd'][k]] for k in cfg['kd']
                               if cfg['kd'][k] != other['kd'][k]}}


def check_resources(cfg, *, resume=False):
    values = cfg['resources']
    checked = []
    missing = []
    def check(label, path, directory=False):
        if path and (Path(path).is_dir() if directory else Path(path).is_file()):
            checked.append(str(path))
        else:
            missing.append(f'{label}={path or "<unset>"}')
    if values['TOMATO_DATA_ROOT']:
        for split in ('train', 'val'):
            ds = cfg['native'][split + '_dataloader']['dataset']
            check(split + ' images', ds['img_folder'], directory=True)
            check(split + ' annotations', ds['ann_file'])
    else:
        missing.append('TOMATO_DATA_ROOT=<unset>')
    if not resume:
        for kind in ('BACKBONE_WEIGHTS', 'DETECTOR_CHECKPOINT'):
            name = f"THESIS_KD_{cfg['arch'].upper()}_{kind}"
            check(name, values[name])
    if kd_enabled(cfg):
        for name in ('THESIS_KD_TEACHER_CONFIG', 'THESIS_KD_TEACHER_CHECKPOINT'):
            check(name, values[name])
        if not (values['THESIS_KD_TEACHER_VALIDATION'] or '').strip():
            missing.append('THESIS_KD_TEACHER_VALIDATION=<unset>')
    if missing:
        raise FileNotFoundError('Missing required experiment resources: ' + ', '.join(missing))
    return checked
