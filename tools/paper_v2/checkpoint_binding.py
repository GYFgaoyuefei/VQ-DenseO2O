"""Read existing best.pth runs for evaluation without inventing training metadata.

Only inference configuration and checkpoint contents are required. Training
recipe/selection claims remain unverified unless their saved evidence exists.
This keeps strict tensor loading; evaluation never gates on training versions.
"""
import copy
import hashlib
import json
from pathlib import Path


METADATA_FILES = ('config_resolved.yml', 'run_info.json', 'selection.json', 'completed.json')


def inference_digest(cfg):
    """Hash effective inference choices, never training tags or completion files."""
    keys = {'model', 'postprocessor', 'num_classes', 'eval_spatial_size',
            'remap_mscoco_category', 'use_focal_loss', '_evaluation_dc_options'}

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value in cfg and value not in keys:
            keys.add(value)
            visit(cfg[value])

    visit(cfg.get('model'))
    visit(cfg.get('postprocessor'))
    inference = {key: cfg[key] for key in sorted(keys) if key in cfg}
    dataset = cfg.get('val_dataloader', {}).get('dataset', {})
    inference['evaluation_dataset'] = {key: dataset[key] for key in
                                      ('type', 'category_ids', 'return_masks', 'transforms') if key in dataset}
    return hashlib.sha256(json.dumps(inference, sort_keys=True, default=str,
                          ensure_ascii=False).encode('utf-8')).hexdigest()


def _json_if_present(path):
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return result


def selected_ema(state):
    """Reject absent/non-FP32 EMA; never silently switch to raw model weights."""
    import torch
    if not isinstance(state, dict):
        raise ValueError('best.pth must be a detector training checkpoint dictionary')
    wrapper = state.get('ema')
    ema = wrapper.get('module') if isinstance(wrapper, dict) else None
    if not isinstance(ema, dict) or not ema:
        raise ValueError('best.pth has no EMA module; this FP32 EMA evaluation cannot substitute model weights')
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in ema.values() if isinstance(value, torch.Tensor)):
        raise ValueError('best.pth EMA must retain FP32 tensors')
    return ema


def load_evaluation_state(model, ema, run_policy='strict', cfg=None):
    from tools.paper_v2.evaluation_state import load_evaluation_state as load
    return load(model, ema, run_policy, cfg)


def bind_existing_checkpoint(run_dir, model, seed):
    """Return cfg, best.pth, actual epoch binding, and honest source provenance.

    A saved resolved/native config takes priority. Without one, the caller's
    model label selects the repository architecture, which must subsequently
    pass load_state_dict(strict=True). That cannot prove historical training
    augmentation/loss choices, and the provenance explicitly says so.
    """
    import torch
    import yaml
    from tools.paper_v2.run import ALL_MODELS, load_resolved, config_path
    from engine.core.yaml_utils import load_config
    from tools.paper_v2.evaluation_state import prepare_inference_config

    if model not in ALL_MODELS or type(seed) is not int or seed not in (0, 42, 3407):
        raise ValueError('Unknown evaluation model or seed')
    directory = Path(run_dir).resolve()
    checkpoint = directory / 'best.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    before = checkpoint.stat()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selected_ema(state)
    epoch = state.get('last_epoch')
    if type(epoch) is not int or not 0 <= epoch < 120:
        raise ValueError('best.pth last_epoch must be zero-based within 0..119')
    del state
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('best.pth changed while reading its evaluation binding')

    saved_path = next((directory / name for name in
                       ('config_resolved.yml', 'config.yml', 'config.yaml')
                       if (directory / name).is_file()), None)
    if saved_path is None:
        cfg = load_resolved(model, expand_env=False)
        config_source = str(config_path(model).resolve())
        config_origin = 'repository_model_fallback'
    else:
        # Read resolved files as-is: resolving historical training innovations
        # here would rewrite their saved choices or reject irrelevant old names.
        cfg = yaml.safe_load(saved_path.read_text(encoding='utf-8'))
        if not isinstance(cfg, dict):
            raise ValueError(f'Expected a configuration mapping: {saved_path}')
        if cfg.get('__include__') and saved_path.name != 'config_resolved.yml':
            cfg = load_config(str(saved_path))
        config_source, config_origin = str(saved_path), 'saved_run_config'
    cfg = copy.deepcopy(cfg)
    cfg.pop('__include__', None)
    prepare_inference_config(cfg)
    # Strict tensor loading cannot detect shape-preserving evaluation choices.
    # Keep the saved architecture/normalization/postprocessor, enforcing only
    # the shared evaluation contract rather than today's training recipe tag.
    required = ('model', 'DINOv3STAs', 'PostProcessor', 'val_dataloader')
    if any(key not in cfg for key in required):
        raise ValueError(f'Incomplete inference config: {config_source}; expected {required}')
    if (cfg.get('num_classes') != 3 or cfg.get('eval_spatial_size') != [640, 640]
            or cfg['PostProcessor'].get('num_top_queries') != 300):
        raise ValueError('Evaluation requires the saved three-class, 640-input, top-300 detector')
    flags = cfg.get('innovation', {})
    dc = cfg.get('_evaluation_dc_options', cfg['DINOv3STAs'].get('dc_sta', flags.get('dc_sta', {}))) or {}
    if not isinstance(dc, dict) or bool(dc.get('enabled', False)) != ('dc_sta' in model):
        raise ValueError('Saved DC-STA architecture does not match the requested model directory')
    cfg['DINOv3STAs']['weights_path'] = None

    info = _json_if_present(directory / 'run_info.json')
    selection = _json_if_present(directory / 'selection.json')
    completion = _json_if_present(directory / 'completed.json')
    for owner, declared_seed in (('saved config', cfg.get('seed') if saved_path else None),
                                 ('run_info.json', (info or {}).get('seed'))):
        if declared_seed is not None and (type(declared_seed) is not int or declared_seed != seed):
            raise ValueError(f'{owner} seed differs from requested seed={seed}')
    if info and info.get('experiment', model) != model:
        raise ValueError('run_info.json experiment differs from the requested model directory')
    if selection and 'epoch' in selection and (
            type(selection['epoch']) is not int or selection['epoch'] != epoch):
        raise ValueError('best.pth epoch differs from selection.json')
    expected_selection = dict(model='ema', split='mixed_validation', precision='FP32',
                              autocast=False, tf32=False)
    verified = bool(selection is not None and selection.get('epoch') == epoch
                    and all(selection.get(key) == value and type(selection.get(key)) is type(value)
                            for key, value in expected_selection.items()))
    # Preserve declared facts separately from the current evaluation suite.
    provenance = {
        'mode': 'existing-checkpoint', 'config_origin': config_origin,
        'config_source': config_source,
        'inference_config_sha256': inference_digest(cfg),
        'dc_formula': ('residual_tanh' if any(key in dc for key in ('alpha', 'input_mean', 'input_std'))
                       else 'sigmoid') if dc.get('enabled') else None,
        'missing_metadata': [name for name in METADATA_FILES if not (directory / name).is_file()],
        'declared_algorithm_suite': cfg.get('paper_v2', {}).get('algorithm_suite'),
        'declared_innovation': copy.deepcopy(flags),
        'selection_verification': 'VERIFIED' if verified else 'UNVERIFIED',
        'selection_record': selection,
        'completion_marker_present': completion is not None,
        'training_recipe_verification': 'UNVERIFIED',
        'epoch_source': 'best.pth:last_epoch',
        'training_budget': cfg.get('epoches', 120),
        'training_budget_source': 'saved_config' if saved_path and 'epoches' in cfg else 'expected_protocol',
        'note': 'Strict EMA state loading verifies inference compatibility, not historical loss, '
                'augmentation, completion or checkpoint-selection procedure. Test never selects a checkpoint.',
    }
    if saved_path is None:
        formula = f"; DC formula assumed {provenance['dc_formula']}" if provenance['dc_formula'] else ''
        print(f'  CONFIG FALLBACK: {model}/seed_{seed} has no saved config; '
              f'using the repository model definition{formula}. Historical training settings are UNVERIFIED.',
              flush=True)
    cfg['seed'] = seed
    return cfg, checkpoint, {'epoch': epoch}, provenance
