"""Unified Paper v2 CLI for native training, explicit evaluation and reporting."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = ('baseline_native', 'baseline_maskaware', 'vq_dense_o2o', 'li_copyblend', 'dc_sta')
COMBINATIONS = {'vq_dense_o2o_li_copyblend': ('vq_dense_o2o', 'li_copyblend'),
                'vq_dense_o2o_dc_sta': ('vq_dense_o2o', 'dc_sta'),
                'li_copyblend_dc_sta': ('li_copyblend', 'dc_sta'),
                'vq_dense_o2o_li_copyblend_dc_sta': ('vq_dense_o2o', 'li_copyblend', 'dc_sta')}
ALL_MODELS = (*MODELS, *COMBINATIONS)
ALGORITHM_SUITE = 'beta_select_algorithms_v1'
CONFIG_DIR = ROOT / 'configs/tomato/paper_v2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def config_path(model):
    if model not in ALL_MODELS:
        raise ValueError(f'Unknown formal model: {model}; historical GR-VQ and residual DC-STA are retired')
    return CONFIG_DIR / (f'combinations/{model}.yml' if model in COMBINATIONS else f'{model}.yml')


def load_resolved(model, expand_env=True):
    from engine.core.yaml_utils import load_config
    from engine.core.innovation_config import resolve_innovation
    cfg = resolve_innovation(load_config(str(config_path(model))), expand_env=expand_env)
    cfg.pop('__include__', None)
    return cfg


def create_config(cfg):
    # Use native YAMLConfig properties without a second disk roundtrip.
    from engine.core import YAMLConfig, BaseConfig
    obj = YAMLConfig.__new__(YAMLConfig)
    BaseConfig.__init__(obj)
    obj.yaml_cfg = copy.deepcopy(cfg)
    for key in list(obj.__dict__):
        if not key.startswith('_') and key in cfg:
            setattr(obj, key, cfg[key])
    return obj


def write_yaml(path, cfg):
    import yaml
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')


def generate_combinations():
    directory = CONFIG_DIR / 'combinations'
    directory.mkdir(parents=True, exist_ok=True)
    for name, modules in COMBINATIONS.items():
        write_yaml(directory / f'{name}.yml', {'__include__': ['../common.yml'],
                   'innovation': {'instance_mask_sync': True,
                                  **{module: {'enabled': True} for module in modules}}})
    print('Generated C1-C4 from common.yml; no training launched.')


def verify_recipe(cfg):
    if cfg.get('paper_v2', {}).get('algorithm_suite') != ALGORITHM_SUITE:
        raise ValueError('This run is not from the new beta_select algorithm suite; old runs cannot be reused')
    assert cfg['num_classes'] == 3 and cfg['epoches'] == 120
    assert cfg['use_amp'] and cfg['use_ema']
    assert cfg['DEIMCriterion']['gamma'] == 1.5
    assert cfg['train_dataloader']['total_batch_size'] == 32
    assert cfg['val_dataloader']['total_batch_size'] == 64
    assert cfg['DINOv3STAs']['interaction_indexes'] == [5, 8, 11]
    assert cfg['lrsheduler'] == 'flatcosine'
    assert cfg['flat_epoch'] == 60 and cfg['no_aug_epoch'] == 14


def check_assets(cfg, evaluation=None, require_pretrained=True, train_only=False):
    needed = []
    loaders = ('train_dataloader',) if train_only else (('val_dataloader',) if evaluation else ('train_dataloader', 'val_dataloader'))
    for loader in loaders:
        ds = cfg[loader]['dataset']
        needed.extend((ds['img_folder'], ds['ann_file']))
        if ds.get('visible_mask_sidecar'):
            needed.append(ds['visible_mask_sidecar'])
        if ds.get('illumination_map'):
            needed.append(ds['illumination_map'])
        annotation_path = Path(ds['ann_file'])
        if annotation_path.exists():
            coco = json.loads(annotation_path.read_text(encoding='utf-8'))
            if set(ds['category_ids']) != {c['id'] for c in coco['categories']}:
                raise ValueError(f'Unexpected maturity category IDs in {annotation_path}')
            missing_images = [item['file_name'] for item in coco['images']
                              if not (Path(ds['img_folder']) / item['file_name']).is_file()]
            if missing_images:
                raise FileNotFoundError(f'{len(missing_images)} missing images; first: {missing_images[:3]}')
    if not evaluation and require_pretrained:
        needed += [cfg['DINOv3STAs']['weights_path'], cfg['tomato']['detector_checkpoint']]
    missing = [str(path) for path in needed if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('Required paths are missing:\n' + '\n'.join(missing))


def preflight(args):
    if git('branch', '--show-current') != 'paper_v3.0':
        raise ValueError('Run Paper v2 on branch paper_v3.0')
    for model in ALL_MODELS if args.model == 'all' else (args.model,):
        cfg = load_resolved(model, expand_env=not args.config_only)
        verify_recipe(cfg)
        if not args.config_only:
            check_assets(cfg)
            obj = create_config(cfg)
            # Constructor checks sidecar identity once, alongside native registry wiring.
            _ = obj.train_dataloader.dataset, obj.val_dataloader.dataset
            _ = obj.model, obj.criterion, obj.postprocessor
            print(f'{model}: PASS paths, sidecar, model, criterion, data constructors')
            del obj
        else:
            print(f'{model}: PASS configuration only; data/weights/model NOT RUN')


def output_path(args):
    return Path(args.output_dir).resolve() if args.output_dir else ROOT / f'outputs/paper_v2_beta_select_algorithms/{args.model}/seed_{args.seed}'


def guard_output(directory, resume=None):
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f'Output is nonempty; use --resume to continue: {directory}')
    if resume:
        if not Path(resume).is_file():
            raise FileNotFoundError(resume)
        if Path(resume).resolve().parent != directory.resolve():
            raise ValueError('--resume must be a checkpoint in this run output directory')
        if (directory / 'completed.json').exists():
            raise ValueError('Run already completed; refusing to restart it')


def train_command(cfg_path, args, directory, evaluation=False):
    cmd = [sys.executable]
    if args.nproc_per_node > 1:
        if not args.device.startswith('cuda'):
            raise ValueError('Distributed Paper v2 training requires CUDA')
        if 32 % args.nproc_per_node:
            raise ValueError('32 total batch size must be divisible by nproc-per-node')
        cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(args.nproc_per_node)]
    cmd += [str(ROOT / 'train.py'), '-c', str(cfg_path), '--seed', str(args.seed),
            '--output-dir', str(directory), '--use-amp']
    # DDP chooses LOCAL_RANK; never pin all processes to cuda:0.
    if args.nproc_per_node == 1:
        cmd += ['--device', args.device]
    if evaluation:
        cmd += ['--test-only', '--resume', str(args.checkpoint)]
    elif args.resume:
        cmd += ['--resume', str(args.resume)]
    else:
        cmd += ['--tuning', os.environ['TOMATO_DETECTOR_CHECKPOINT']]
    return cmd


def stream_process(cmd, path, append=False):
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    with path.open('a' if append else 'w', encoding='utf-8') as log:
        with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as proc:
            for line in proc.stdout:
                print(line, end='')
                log.write(line)
                log.flush()
            code = proc.wait()
    if code:
        raise RuntimeError(f'Native train.py exited {code}; see {path}')


def train(args):
    if git('branch', '--show-current') != 'paper_v3.0':
        raise ValueError('Expected branch paper_v3.0')
    directory = output_path(args)
    guard_output(directory, args.resume)
    if args.resume:
        saved = directory / 'config_resolved.yml'
        if not saved.exists():
            raise ValueError('Resume requires this run config_resolved.yml')
        import yaml
        previous = yaml.safe_load(saved.read_text(encoding='utf-8'))
        expected = load_resolved(args.model, expand_env=False)
        if previous['innovation'] != expected['innovation'] or previous.get('seed') != args.seed:
            raise ValueError('Resume model switches or seed differ from saved run')
        cfg = previous
    else:
        cfg = load_resolved(args.model)
    verify_recipe(cfg)
    check_assets(cfg, require_pretrained=not bool(args.resume))
    directory.mkdir(parents=True, exist_ok=True)
    cfg.update(output_dir=str(directory), seed=args.seed)
    write_yaml(directory / 'config_resolved.yml', cfg)
    cmd = train_command(directory / 'config_resolved.yml', args, directory)
    info_path = directory / 'run_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
    info.update(experiment=args.model, seed=args.seed, git_commit=git('rev-parse', 'HEAD'),
                source_dirty=bool(git('status', '--porcelain')), branch='paper_v3.0', algorithm_suite=ALGORITHM_SUITE,
                pretrained={'dinov3': cfg['DINOv3STAs']['weights_path'], 'detector': cfg['tomato']['detector_checkpoint']},
                command=cmd, started_utc=datetime.now(timezone.utc).isoformat(),
                selection='EMA FP32 Mixed Validation AP; autocast/TF32 disabled; exact ties: earliest epoch')
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    stream_process(cmd, directory / 'train.log', append=bool(args.resume))
    (directory / 'completed.json').write_text(json.dumps({'finished_utc': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')


def smoke(args):
    import torch
    from engine.misc.dist_utils import setup_seed
    setup_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = load_resolved(args.model, expand_env=not args.synthetic)
    size = 128 if args.synthetic else 640
    cfg['eval_spatial_size'] = [size, size]
    cfg['DINOv3STAs']['weights_path'] = None if args.synthetic else cfg['DINOv3STAs']['weights_path']
    cfg.update(device=args.device, output_dir=str(output_path(args)), use_amp=False)
    obj = create_config(cfg)
    model, criterion = obj.model.to(args.device).train(), obj.criterion.to(args.device)
    if args.synthetic:
        rgb = torch.rand(2, 3, size, size, device=args.device)
        mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
        std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        images = (rgb - mean) / std
        targets = []
        for _ in range(2):
            masks = torch.zeros(1, size, size, dtype=torch.bool, device=args.device)
            masks[:, 40:88, 40:88] = True
            target = {'boxes': torch.tensor([[.5, .5, .375, .375]], device=args.device),
                      'labels': torch.tensor([1], device=args.device)}
            if cfg['innovation']['instance_mask_sync']:
                target.update(masks=masks, visible_mask_valid=torch.tensor([True], device=args.device))
            targets.append(target)
    else:
        check_assets(cfg)
        # Load detector tuning via the real framework helper; no training loop is run.
        from engine.solver._solver import BaseSolver
        loader = BaseSolver(obj)
        loader.model = model
        loader.load_tuning_state(cfg['tomato']['detector_checkpoint'])
        cfg['train_dataloader'].update(total_batch_size=2, num_workers=0)
        cfg['train_dataloader']['collate_fn'].update(base_size_repeat=None)
        obj.yaml_cfg = cfg
        images, targets = next(iter(obj.train_dataloader))
        images = images.to(args.device)
        targets = [{k: v.to(args.device) for k, v in target.items()} for target in targets]
    optimizer = obj.optimizer
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images, targets)
    losses = criterion(outputs, targets, epoch=0)
    loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite smoke loss')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Missing or nonfinite model gradients')
    optimizer.step()
    model.eval()
    with torch.no_grad():
        predicted = model(images[:1])
    report = {'model': args.model, 'status': 'PASS', 'synthetic': args.synthetic,
              'input_size': size, 'device': args.device, 'loss': float(loss.detach()),
              'loss_terms': sorted(losses), 'inference_shape': list(predicted['pred_boxes'].shape),
              'pretrained_loaded': not args.synthetic, 'formal_training': 'NOT RUN'}
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')


def smoke_train(args):
    """Launch a bounded native Train run, retaining the formal batch and recipe."""
    if not 1 <= args.smoke_steps <= 10 or not 0 <= args.smoke_epoch < 120:
        raise ValueError('smoke-train requires 1-10 steps and a recipe epoch from 0 to 119')
    if args.synthetic or args.resume or args.checkpoint:
        raise ValueError('smoke-train uses real Train data and shared pretrained initialization')
    if not args.device.startswith('cuda'):
        raise ValueError('Real training batch acceptance requires CUDA')
    if git('branch', '--show-current') != 'paper_v3.0':
        raise ValueError('Expected branch paper_v3.0')
    directory = (Path(args.output_dir).resolve() if args.output_dir else
                 ROOT / f'outputs/paper_v2_beta_select_algorithms/checks/real_train/{args.model}/seed_{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}')
    guard_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'train_smoke_preflight.json'
    try:
        cfg = load_resolved(args.model)
        verify_recipe(cfg)
        check_assets(cfg, train_only=True)
    except (ValueError, FileNotFoundError, OSError) as error:
        report_path.write_text(json.dumps({'status': 'NOT RUN', 'model': args.model, 'seed': args.seed,
            'reason': str(error), 'real_train_batches': 0, 'test': 'NOT RUN'}, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Real Train smoke NOT RUN: {report_path}')
        raise
    cfg.update(output_dir=str(directory), seed=args.seed)
    # Keep optimizer, AMP, augmentation, scheduler and total batch unchanged.
    write_yaml(directory / 'config_resolved.yml', cfg)
    command = train_command(directory / 'config_resolved.yml', args, directory)
    command += ['--train-smoke-steps', str(args.smoke_steps), '--train-smoke-epoch', str(args.smoke_epoch)]
    report_path.write_text(json.dumps({'status': 'PASS', 'model': args.model, 'seed': args.seed,
        'effective_batch_size': 32, 'steps': args.smoke_steps, 'recipe_epoch': args.smoke_epoch,
        'command': command, 'test': 'NOT RUN'}, indent=2, ensure_ascii=False), encoding='utf-8')
    stream_process(command, directory / 'train_smoke.log')
    print(f'Bounded Train acceptance reports: {directory}; no 120-epoch fit or Test evaluation')


def evaluate(args):
    directory = output_path(args)
    cfg, args.checkpoint, _ = benchmark_binding(args)
    split = 'test' if args.command == 'evaluate-test' else 'validation'
    if split == 'test':
        root = Path(os.environ['TOMATO_DATA_ROOT'])
        cfg['val_dataloader']['dataset'].update(img_folder=str(root / 'images/test'),
                                               ann_file=str(root / 'annotations/instances_test.json'))
    check_assets(cfg, evaluation=True)
    # Resume already includes backbone weights; do not require the pretraining source to remain mounted.
    cfg['DINOv3STAs']['weights_path'] = None
    destination = directory / f'{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}'
    destination.mkdir(parents=True, exist_ok=False)
    cfg['output_dir'] = str(destination)
    write_yaml(destination / 'config_resolved.yml', cfg)
    cmd = train_command(destination / 'config_resolved.yml', args, destination, evaluation=True)
    stream_process(cmd, destination / 'evaluation.log')
    print(f'{split} metrics: {destination / "metrics.json"}')


def efficiency_path(args):
    return Path(args.efficiency_dir).resolve() if args.efficiency_dir else ROOT / 'outputs/paper_v2_beta_select_algorithms/efficiency'


def benchmark_binding(args):
    """Bind a formal measurement to one completed run and its selected EMA."""
    import yaml
    directory = output_path(args)
    cfg = yaml.safe_load((directory / 'config_resolved.yml').read_text(encoding='utf-8'))
    info = json.loads((directory / 'run_info.json').read_text(encoding='utf-8'))
    if info.get('experiment') != args.model or info.get('seed') != args.seed or cfg.get('seed') != args.seed:
        raise ValueError('Efficiency model/seed does not match saved run metadata')
    if not (directory / 'completed.json').is_file():
        raise ValueError('Formal efficiency reporting requires the completed training run (Epoch=120)')
    verify_recipe(cfg)
    expected = load_resolved(args.model, expand_env=False)['innovation']
    actual = cfg['innovation']
    if (actual['instance_mask_sync'] != expected['instance_mask_sync'] or
        any(actual[key]['enabled'] != expected[key]['enabled'] for key in ('vq_dense_o2o', 'li_copyblend', 'dc_sta'))):
        raise ValueError('Efficiency model name does not match saved algorithm switches')
    if cfg.get('eval_spatial_size') != [640, 640] or cfg['PostProcessor']['num_top_queries'] != 300:
        raise ValueError('Efficiency requires native 640 spatial size and tensor postprocessor top-k=300')
    checkpoint = Path(args.checkpoint or directory / 'best.pth').resolve()
    if checkpoint != (directory / 'best.pth').resolve():
        raise ValueError('Formal efficiency uses this run best.pth selected by EMA Validation AP')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selection = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
    if selection.get('model') != 'ema' or selection.get('split') != 'mixed_validation':
        raise ValueError('Efficiency requires EMA Mixed Validation selection')
    if (selection.get('precision') != 'FP32' or selection.get('autocast') is not False
            or selection.get('tf32') is not False):
        raise ValueError('Efficiency requires explicit FP32 Validation selection with autocast/TF32 disabled')
    return cfg, checkpoint, selection


def benchmark(args):
    import torch
    from tools.paper_v2.efficiency_protocol import protocol
    from tools.paper_v2.efficiency_measure import measure
    from tools.paper_v2.efficiency_summary import check_campaign
    if args.nproc_per_node != 1:
        raise ValueError('Efficiency runs in one process on one GPU; do not use torchrun')
    torch.set_num_threads(args.threads)
    directory = efficiency_path(args)
    check_campaign(directory)
    path = directory / 'raw' / args.model / f'seed_{args.seed}' / 'efficiency.json'
    if path.exists():
        raise FileExistsError(f'Efficiency result already exists: {path}; use a new --efficiency-dir for a new campaign')
    cfg, checkpoint, selection = benchmark_binding(args)
    checkpoint_stat = checkpoint.stat()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selected = state.get('last_epoch')
    if ('ema' not in state or type(selected) is not int or not 0 <= selected < 120
            or selected != selection.get('epoch')):
        raise ValueError('best.pth is not the recorded selected EMA epoch')
    # Construct only inference components, with no datasets, criterion, or pretrained resource reads.
    cfg['DINOv3STAs']['weights_path'] = None
    obj = create_config(cfg)
    model = obj.model
    model.load_state_dict(state['ema']['module'], strict=True)
    del state
    result = measure(model, obj.postprocessor, args.device)
    stat = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns):
        raise ValueError('Selected checkpoint changed during efficiency measurement; refusing to save misleading provenance')
    report = {'schema': 'paper_v2_efficiency_record_v1', 'status': 'MEASURED', 'algorithm_suite': ALGORITHM_SUITE,
              'protocol': protocol(), 'model': args.model, 'seed': args.seed,
              'epochs': cfg['epoches'], 'selected_epoch_zero_based': selected,
              'checkpoint': {'path': str(checkpoint), 'size_bytes': stat.st_size,
                             'mtime_ns': stat.st_mtime_ns, 'model_state': 'ema'},
              'git_commit': git('rev-parse', 'HEAD'), 'source_dirty': bool(git('status', '--porcelain')),
              'measured_utc': datetime.now(timezone.utc).isoformat(), **result}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f'Five fixed-protocol repeats saved: {path}')
    summarize_efficiency(args)


def summarize_efficiency(args):
    from tools.paper_v2.efficiency_summary import summarize as aggregate_efficiency
    result = aggregate_efficiency(efficiency_path(args), ap_csv=args.ap_csv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def ap_results(args):
    from tools.paper_v2 import ap_workflow
    if args.command == 'evaluate-ap':
        return ap_workflow.evaluate_and_summarize(args)
    return ap_workflow.summarize(args)


def summarize(args):
    directory = Path(args.output_dir) if args.output_dir else ROOT / 'outputs/paper_v2_beta_select_algorithms'
    rows = []
    for path in sorted(directory.glob('*/seed_*/run_info.json')):
        info = json.loads(path.read_text(encoding='utf-8'))
        if info.get('algorithm_suite') != ALGORITHM_SUITE:
            raise ValueError(f'Historical run cannot enter the new experiment summary: {path}')
        rows.append({'model': info['experiment'], 'seed': info['seed'],
                     'best_validation_ap': info.get('best_validation_ap', ''),
                     'best_validation_epoch': info.get('best_validation_epoch', ''),
                     'completed': (path.parent / 'completed.json').exists(), 'path': str(path.parent)})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'summary.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'seed', 'best_validation_ap', 'best_validation_epoch', 'completed', 'path'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} actual runs summarized in {path}; no missing results imputed.')
    summarize_efficiency(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'smoke', 'smoke-train', 'train', 'validate', 'evaluate-test', 'benchmark', 'summarize', 'summarize-efficiency', 'evaluate-ap', 'summarize-ap', 'audit-configs', 'generate-combinations', 'commands'])
    p.add_argument('--model', choices=[*MODELS, *COMBINATIONS, 'all'], default='all')
    p.add_argument('--seed', type=int, choices=[0, 42, 3407], default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-dir')
    p.add_argument('--resume')
    p.add_argument('--checkpoint')
    p.add_argument('--nproc-per-node', type=int, default=1)
    p.add_argument('--config-only', action='store_true')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--report')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--smoke-steps', type=int, default=3, help='smoke-train only: 1-10 actual Train batches')
    p.add_argument('--smoke-epoch', type=int, default=0, help='smoke-train only: recipe epoch; 10 probes the active CopyBlend stage')
    p.add_argument('--efficiency-dir', help='Efficiency campaign directory; defaults to outputs/paper_v2_beta_select_algorithms/efficiency')
    p.add_argument('--ap-csv', help='Explicit AP-by-seed CSV in percent units, with Model and Seed columns')
    p.add_argument('--ap-split', choices=['test', 'validation'], default='test', help='AP report split; never changes checkpoint selection')
    p.add_argument('--ap-dir', help='One-split AP campaign directory; default outputs/paper_v2_beta_select_algorithms/ap/<split>')
    p.add_argument('--runs-dir', help='AP only: parent of <model>/seed_<seed> completed run directories')
    p.add_argument('--ap-models', nargs='+', choices=[*MODELS, *COMBINATIONS], help='AP only: default five independent models; may include C1-C4')
    p.add_argument('--ap-seeds', nargs='+', type=int, choices=[0, 42, 3407], default=[0, 42, 3407], help='AP evaluation seeds; aggregates still require all three')
    p.add_argument('--data-root', help='AP only: evaluation dataset root; otherwise TOMATO_DATA_ROOT or saved Validation paths')
    p.add_argument('--illumination-map', help='AP only: verified JSON mapping of image IDs or filenames to acquisition light labels')
    return p


def main():
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.nproc_per_node < 1:
        raise ValueError('nproc-per-node must be positive')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if args.command not in ('preflight', 'summarize', 'summarize-efficiency', 'evaluate-ap', 'summarize-ap', 'audit-configs', 'generate-combinations', 'commands') and args.model == 'all':
        raise ValueError('Select one --model')
    if args.command == 'generate-combinations':
        generate_combinations()
    elif args.command == 'audit-configs':
        from tools.paper_v2.config_audit import audit
        audit()
    elif args.command == 'commands':
        for seed in (42, 0, 3407):
            for model in ALL_MODELS:
                print(f'bash scripts/paper_v2/autodl/train_one.sh {model} {seed} cuda:0')
    else:
        {'preflight': preflight, 'smoke': smoke, 'smoke-train': smoke_train, 'train': train, 'validate': evaluate,
         'evaluate-test': evaluate, 'benchmark': benchmark, 'summarize': summarize,
         'evaluate-ap': ap_results, 'summarize-ap': ap_results,
         'summarize-efficiency': summarize_efficiency}[args.command](args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(f'Paper v2: {error}')
