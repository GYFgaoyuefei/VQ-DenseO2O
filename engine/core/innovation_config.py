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
    allowed = {'instance_mask_sync', 'vq_dense_o2o', 'li_copyblend', 'dc_sta'}
    unknown = set(flags) - allowed
    if unknown:
        raise ValueError(f'Unknown or historical innovation switches: {sorted(unknown)}')
    sync = flags.get('instance_mask_sync', False)
    if not isinstance(sync, bool):
        raise ValueError('instance_mask_sync must be boolean')
    for name in ('vq_dense_o2o', 'li_copyblend', 'dc_sta'):
        opts = flags.get(name, {})
        if not isinstance(opts, dict) or not isinstance(opts.get('enabled', False), bool):
            raise ValueError(f'{name}.enabled must be boolean')
    vq = flags.get('vq_dense_o2o', {})
    li = flags.get('li_copyblend', {})
    dc = flags.get('dc_sta', {})
    if any(options.get('enabled', False) for options in (vq, li, dc)) and not sync:
        raise ValueError('Formal VQ-DenseO2O, LI-CopyBlend and DC-STA require instance_mask_sync=true')
    beta = vq.get('beta', 0.10)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or beta != 0.10:
        raise ValueError('Formal vq_dense_o2o.beta is fixed at 0.10')
    if vq.get('invalid_mask_policy', 'iou_fallback') != 'iou_fallback':
        raise ValueError('VQ invalid masks require the original IoU fallback')
    if vq.get('copyblend', {}).get('visibility_aware', False):
        raise ValueError('VQ cannot enable a second CopyBlend implementation')
    quality = vq.get('quality', {})
    if (quality.get('combination', 'geometric_mean') != 'geometric_mean' or
            not quality.get('use_coverage', True) or not quality.get('use_purity', True)):
        raise ValueError('VQ requires the original geometric mean of coverage and purity')
    apply_to = vq.get('apply_to', {})
    if (not apply_to.get('dense_o2o_positive_only', True) or
            apply_to.get('branches', ['main', 'aux', 'pre', 'encoder']) != ['main', 'aux', 'pre', 'encoder']):
        raise ValueError('VQ applies only to ordinary main/aux/pre/encoder positives')
    if any(vq.get(key, False) for key in ('modify_hungarian_matcher', 'modify_bbox_target', 'modify_class_target')):
        raise ValueError('VQ cannot change matching, localization or category identity')
    if 'alpha' in dc:
        raise ValueError('Residual-tanh alpha is not supported by the restored Sigmoid DC-STA')

    # Always resolve from native definitions. Re-resolving a saved config is idempotent.
    cfg['criterion'] = 'DEIMCriterion'
    cfg.pop('GRVQDEIMCriterion', None)
    cfg.pop('VQDEIMCriterion', None)
    if vq.get('enabled', False):
        cfg['criterion'] = 'VQDEIMCriterion'
        cfg['VQDEIMCriterion'] = {**copy.deepcopy(cfg['DEIMCriterion']), 'vq_dense_o2o': vq}
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
