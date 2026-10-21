"""Explicit final Test + fixed efficiency campaign for all nine models/three seeds.

Run this script on the training server. Importing it, --help, --dry-run and
--mode summarize-only never open Test data or load a checkpoint tensor.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.paper_v2.efficiency_protocol import ALGORITHM_SUITE, MODELS, SEEDS
from tools.paper_v2.ap_workflow import REQUIRED_RUN_FILES


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def ap_command(args):
    command = [sys.executable, str(ROOT / 'tools/paper_v2/run.py'), 'evaluate-ap',
               '--device', args.device, '--ap-split', 'test',
               '--runs-dir', str(args.runs_dir), '--data-root', str(args.data_root),
               '--ap-dir', str(args.ap_dir), '--ap-models', *MODELS,
               '--ap-seeds', *(str(seed) for seed in SEEDS), '--run-policy', args.run_policy]
    if args.illumination_map:
        command.extend(['--illumination-map', str(args.illumination_map)])
    return command


def efficiency_command(args, model, seed):
    return [sys.executable, str(ROOT / 'tools/paper_v2/run.py'), 'benchmark',
            '--model', model, '--seed', str(seed), '--device', args.device,
            '--threads', str(args.threads),
            '--output-dir', str(args.runs_dir / model / f'seed_{seed}'),
            '--efficiency-dir', str(args.efficiency_dir), '--run-policy', args.run_policy]


def check_test_lights(args):
    """Fail on missing/conflicting light labels before any of the 27 inferences."""
    from faster_coco_eval import COCO
    from tools.paper_v2.ap_metrics import _light_groups
    annotation = args.data_root / 'annotations/instances_test.json'
    mapping = (json.loads(args.illumination_map.read_text(encoding='utf-8'))
               if args.illumination_map else None)
    groups = _light_groups(COCO(str(annotation)), mapping)
    counts = {name: len(ids) for name, ids in groups.items()}
    missing = [name for name, count in counts.items() if not count]
    if missing:
        raise ValueError(f'Test has no images for required illumination groups: {missing}')
    print('Test illumination image counts: ' + json.dumps(counts), flush=True)
    return counts


def mark_ap_pending(args):
    """Unverified caches stay excluded if preflight or the subprocess is interrupted."""
    from tools.paper_v2.ap_summary import validate_campaign
    validate_campaign(args.ap_dir, 'test')
    write_json(args.ap_dir / 'evaluation_attempts.json', {
        'algorithm_suite': ALGORITHM_SUITE, 'split': 'test', 'models': list(MODELS),
        'seeds': list(SEEDS), 'status': 'RUNNING',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'runs': [{'model': model, 'seed': seed, 'status': 'RUNNING',
                  'reason': 'Awaiting current final-Test run/data verification'}
                 for model in MODELS for seed in SEEDS],
    })


def reusable_efficiency(args, model, seed):
    """Validate cached evidence against the current selected checkpoint; no inference."""
    from tools.paper_v2 import run
    from tools.paper_v2.efficiency_summary import validate_record
    path = args.efficiency_dir / 'raw' / model / f'seed_{seed}/efficiency.json'
    if not path.exists():
        return False
    record = validate_record(json.loads(path.read_text(encoding='utf-8')))
    binding_args = run.parser().parse_args(efficiency_command(args, model, seed)[2:])
    cfg, checkpoint, selection = run.benchmark_binding(binding_args)
    stat = checkpoint.stat()
    identity = dict(path=str(checkpoint), size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, model_state='ema')
    if (record['model'] != model or record['seed'] != seed
            or record['selected_epoch_zero_based'] != selection.get('epoch')
            or any(record['checkpoint'].get(key) != value for key, value in identity.items())):
        raise ValueError(f'Stale efficiency evidence: {path}; choose a new --efficiency-dir')
    if record.get('run_provenance') and cfg.get('_evaluation_provenance'):
        if (record['run_provenance'].get('inference_config_sha256') !=
                cfg['_evaluation_provenance'].get('inference_config_sha256')):
            raise ValueError(f'Efficiency configuration/provenance changed: {path}; choose a new --efficiency-dir')
    return True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--runs-dir', type=Path, default=ROOT / 'outputs/paper_v2_beta_select_algorithms')
    p.add_argument('--data-root', type=Path, default=Path(os.environ.get('TOMATO_DATA_ROOT', '/root/autodl-tmp/dataset/tomato_coco')))
    p.add_argument('--report-dir', type=Path, help='Default: <runs-dir>/final_test')
    p.add_argument('--ap-dir', type=Path, help='Default: <runs-dir>/ap/test')
    p.add_argument('--efficiency-dir', type=Path, help='Default: <runs-dir>/efficiency')
    p.add_argument('--illumination-map', type=Path, help='Verified image_id/file_name -> acquisition light JSON')
    p.add_argument('--threads', type=int, default=2, help='Shared CPU thread setting for every efficiency run')
    p.add_argument('--run-policy', choices=('existing-checkpoint', 'strict'), default='existing-checkpoint',
                   help='Default: evaluate existing best.pth without training version/metadata restrictions')
    p.add_argument('--mode', choices=('all', 'ap-only', 'efficiency-only', 'summarize-only'), default='all')
    p.add_argument('--dry-run', action='store_true', help='List commands and missing required files; no data/model access')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.runs_dir = args.runs_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    for name, suffix in (('report_dir', 'final_test'), ('ap_dir', 'ap/test'), ('efficiency_dir', 'efficiency')):
        setattr(args, name, (getattr(args, name) or args.runs_dir / suffix).expanduser().resolve())
    if args.illumination_map:
        args.illumination_map = args.illumination_map.expanduser().resolve()
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    if len({args.report_dir, args.ap_dir, args.efficiency_dir}) != 3:
        raise ValueError('Report, AP and efficiency directories must be distinct')
    ap_requested = args.mode in ('all', 'ap-only')
    efficiency_requested = args.mode in ('all', 'efficiency-only')
    if args.dry_run:
        required = REQUIRED_RUN_FILES if args.run_policy == 'strict' else ('best.pth',)
        missing = {f'{model}/seed_{seed}': [name for name in required
                    if not (args.runs_dir / model / f'seed_{seed}' / name).is_file()]
                   for model in MODELS for seed in SEEDS}
        print(json.dumps({'models': MODELS, 'seeds': SEEDS, 'expected_runs': 27,
                          'run_policy': args.run_policy, 'required_run_files': required,
                          'missing_run_files': {key: names for key, names in missing.items() if names},
                          'ap_command': ap_command(args) if ap_requested else None,
                          'efficiency_commands': [efficiency_command(args, m, s) for m in MODELS for s in SEEDS]
                                                 if efficiency_requested else [],
                          'report_dir': str(args.report_dir)}, indent=2, ensure_ascii=False))
        return 0

    from tools.paper_v2.final_test_report import export_report
    from tools.paper_v2.run import stream_process

    attempts_path = args.report_dir / 'final_test_attempts.json'
    previous = json.loads(attempts_path.read_text(encoding='utf-8')) if attempts_path.exists() else {}
    excluded = {tuple(key) for key in previous.get('excluded_efficiency', [])}
    if args.mode == 'summarize-only':
        result = export_report(args.ap_dir, args.efficiency_dir, args.report_dir, excluded_efficiency=excluded)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['status'] == 'COMPLETE' else 1

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run this campaign with plain Python/bash, not torchrun')
    if efficiency_requested:
        # Also covers interruption during the earlier AP stage in mode=all.
        excluded.update((model, seed) for model in MODELS for seed in SEEDS)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    logs = args.report_dir / 'logs'
    logs.mkdir(exist_ok=True)
    attempts = {'mode': args.mode, 'run_policy': args.run_policy, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'runs_dir': str(args.runs_dir), 'status': 'RUNNING', 'efficiency': [], 'errors': []}

    def save():
        attempts['excluded_efficiency'] = sorted(excluded)
        write_json(attempts_path, attempts)

    save()
    interrupted = False
    try:
        if ap_requested:
            try:
                mark_ap_pending(args)
                attempts['illumination_image_counts'] = check_test_lights(args)
                stream_process(ap_command(args), logs / 'test_ap.log', append=True)
                attempts['ap'] = 'FINISHED'
            except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, AssertionError) as error:
                attempts['ap'] = 'FAIL'
                attempts['errors'].append(f'AP: {error}')
            save()
        if efficiency_requested:
            for model in MODELS:
                for seed in SEEDS:
                    key = (model, seed)
                    row = {'model': model, 'seed': seed, 'status': 'RUNNING'}
                    attempts['efficiency'].append(row)
                    excluded.add(key)
                    save()
                    try:
                        if reusable_efficiency(args, model, seed):
                            row['status'] = 'REUSED'
                        else:
                            stream_process(efficiency_command(args, model, seed),
                                           logs / f'efficiency_{model}_seed_{seed}.log', append=True)
                            if not reusable_efficiency(args, model, seed):
                                raise RuntimeError('Benchmark did not produce a verified efficiency record')
                            row['status'] = 'MEASURED'
                        excluded.discard(key)
                    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, AssertionError) as error:
                        row.update(status='FAIL', reason=f'{type(error).__name__}: {error}')
                        attempts['errors'].append(f'{model}/seed_{seed} efficiency: {error}')
                    save()
    except KeyboardInterrupt:
        interrupted = True
        attempts['errors'].append('Interrupted; unfinished measurements remain excluded')
    except Exception as error:
        attempts['errors'].append(f'Campaign aborted: {type(error).__name__}: {error}')
    finally:
        attempts['finished_utc'] = datetime.now(timezone.utc).isoformat()
        attempts['status'] = 'INTERRUPTED' if interrupted else ('FAIL' if attempts['errors'] else 'FINISHED')
        save()
        result = export_report(args.ap_dir, args.efficiency_dir, args.report_dir,
                               excluded_efficiency=excluded, errors=attempts['errors'])
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if interrupted:
        return 130
    if args.mode == 'ap-only':
        return 0 if result['ap_status'] == 'COMPLETE' and not attempts['errors'] else 1
    if args.mode == 'efficiency-only':
        return 0 if result['efficiency_status'] == 'COMPLETE' and not attempts['errors'] else 1
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    sys.exit(main())
