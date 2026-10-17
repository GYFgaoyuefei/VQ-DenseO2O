"""Resolve independent Paper v2 switches into registered native components."""
import copy
import os
import re


def expand_environment(value):
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if not os.environ.get(name):
                raise ValueError(f'Required environment variable is unset: {name}')
            return os.environ[name]
        return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace, value)
    return value


def resolve_innovation(config, expand_env=True):
    cfg = copy.deepcopy(config)
    if expand_env:
        # Override only the unresolved default, never a saved run's resolved
        # sidecar. This accepts verified versioned filenames without renaming.
        tomato = cfg.get('tomato', {})
        if (cfg.get('paper_v2') and os.environ.get('TOMATO_VISIBLE_MASK_TRAIN') and
                '${TOMATO_DATA_ROOT}' in tomato.get('visible_mask_train', '')):
            tomato['visible_mask_train'] = os.environ['TOMATO_VISIBLE_MASK_TRAIN']
        cfg = expand_environment(cfg)
    if 'innovation' not in cfg:
        return cfg
    flags = cfg['innovation']
    allowed = {'instance_mask_sync', 'gr_vq', 'li_copyblend', 'dc_sta'}
    unknown = set(flags) - allowed
    if unknown:
        raise ValueError(f'Unknown or historical innovation switches: {sorted(unknown)}')
    sync = flags.get('instance_mask_sync', False)
    if not isinstance(sync, bool):
        raise ValueError('instance_mask_sync must be boolean')
    for name in ('gr_vq', 'li_copyblend', 'dc_sta'):
        opts = flags.get(name, {})
        if not isinstance(opts, dict) or not isinstance(opts.get('enabled', False), bool):
            raise ValueError(f'{name}.enabled must be boolean')
    gr = flags.get('gr_vq', {})
    li = flags.get('li_copyblend', {})
    dc = flags.get('dc_sta', {})
    if (gr.get('enabled', False) or li.get('enabled', False)) and not sync:
        raise ValueError('GR-VQ and LI-CopyBlend require instance_mask_sync=true')
    beta = gr.get('beta', 0.10)
    if not isinstance(beta, (int, float)) or not 0 <= beta <= 1:
        raise ValueError('gr_vq.beta must be finite and in [0,1]')

    # Always resolve from native definitions. Re-resolving a saved config is idempotent.
    cfg['criterion'] = 'DEIMCriterion'
    if gr.get('enabled', False):
        cfg['criterion'] = 'GRVQDEIMCriterion'
        cfg['GRVQDEIMCriterion'] = {**copy.deepcopy(cfg['DEIMCriterion']), 'gr_vq': gr}
    dataset = cfg['train_dataloader']['dataset']
    dataset['type'] = 'VisibleMaskCocoDetection' if sync else 'CocoDetection'
    dataset['return_masks'] = False
    if sync:
        dataset.update(visible_mask_sidecar=cfg['tomato']['visible_mask_train'],
                       visible_mask_required=True, split='train')
        if cfg['tomato'].get('illumination_map'):
            dataset['illumination_map'] = cfg['tomato']['illumination_map']
    else:
        for key in ('visible_mask_sidecar', 'visible_mask_required', 'split', 'illumination_map'):
            dataset.pop(key, None)
    for op in dataset['transforms']['ops']:
        if op['type'] in ('SanitizeBoundingBoxes', 'SanitizeVisibleBoundingBoxes'):
            op['type'] = 'SanitizeVisibleBoundingBoxes' if sync else 'SanitizeBoundingBoxes'
    cfg['train_dataloader']['collate_fn']['li_copyblend'] = copy.deepcopy(li)
    cfg['DINOv3STAs']['dc_sta'] = copy.deepcopy(dc)
    return cfg
