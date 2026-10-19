"""Serial final-AP evaluation and offline three-seed aggregation."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .efficiency_protocol import ALGORITHM_SUITE, MODELS, SEEDS

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_RUN_FILES = ('config_resolved.yml', 'run_info.json', 'selection.json', 'completed.json', 'best.pth')


def selected_models(args):
    explicit = getattr(args, 'ap_models', None)
    single = getattr(args, 'model', 'all')
    if explicit and single != 'all':
        raise ValueError('Use either --model or --ap-models for AP evaluation')
    models = list(explicit or (MODELS[:5] if single == 'all' else (single,)))
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError('AP models must be unique registered Paper v2 models')
    return models


def output_directory(args):
    return (Path(args.ap_dir).expanduser().resolve() if args.ap_dir else
            ROOT / 'outputs/paper_v2_beta_select_algorithms/ap' / args.ap_split)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def summarize(args):
    from .ap_summary import summarize as summarize_records
    if args.output_dir:
        raise ValueError('Use --ap-dir for AP reports, not --output-dir')
    result = summarize_records(output_directory(args), split=args.ap_split, models=selected_models(args))
    print(json.dumps({key: result[key] for key in
                      ('status', 'split', 'measured_model_seed_count', 'expected_model_seed_count', 'output_dir')
                      if key in result}, indent=2, ensure_ascii=False))
    return result


def evaluate_and_summarize(args):
    from .ap_evaluate import evaluate_run, reusable_record
    from .ap_summary import load_exclusions

    if args.output_dir or args.resume or args.checkpoint or args.synthetic:
        raise ValueError('AP evaluation uses each completed run best.pth; use --runs-dir and --ap-dir')
    if args.nproc_per_node != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('AP campaign uses one process; do not launch it with torchrun')
    models = selected_models(args)
    seeds = list(args.ap_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        raise ValueError('AP seeds must be unique values from 0, 42, 3407')
    runs = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else ROOT / 'outputs/paper_v2_beta_select_algorithms'
    output = output_directory(args)
    mapping = None
    if args.illumination_map:
        mapping = json.loads(Path(args.illumination_map).read_text(encoding='utf-8'))
        if not isinstance(mapping, dict):
            raise ValueError('--illumination-map must contain a JSON object mapping image IDs or filenames to light labels')
    exclusions = load_exclusions(output, args.ap_split)
    attempts = {'algorithm_suite': ALGORITHM_SUITE, 'split': args.ap_split, 'models': models, 'seeds': seeds,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING', 'runs': []}
    attempt_path = output / 'evaluation_attempts.json'

    def save_attempts():
        attempts['excluded_runs'] = list(exclusions.values())
        _write_json(attempt_path, attempts)

    for model in models:
        for seed in seeds:
            run_dir = runs / model / f'seed_{seed}'
            row = {'model': model, 'seed': seed, 'status': 'RUNNING'}
            attempts['runs'].append(row)
            exclusions[(model, seed)] = row.copy()
            save_attempts()
            print(f'AP {args.ap_split}: {model} seed={seed}', flush=True)
            try:
                kwargs = dict(run_dir=run_dir, model=model, seed=seed, output_dir=output,
                              split=args.ap_split, data_root=args.data_root, illumination_map=mapping)
                existing = output / 'raw' / model / f'seed_{seed}' / 'metrics.json'
                if existing.exists():
                    record = reusable_record(**kwargs)
                    row.update(status=record['status'], reused=True)
                else:
                    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
                    if missing:
                        row.update(status='NOT RUN', reason='Missing completed run files: ' + ', '.join(missing))
                    else:
                        record = evaluate_run(**kwargs, device=args.device)
                        row.update(status=record['status'], reused=False)
            except (ValueError, OSError, RuntimeError, KeyError, TypeError, AssertionError) as error:
                row.update(status='FAIL', reason=f'{type(error).__name__}: {error}')
            if row['status'] in ('COMPLETE', 'INCOMPLETE'):
                exclusions.pop((model, seed), None)
            else:
                exclusions[(model, seed)] = row.copy()
            save_attempts()
            print(f"  {row['status']}: {row.get('reason', 'cached result' if row.get('reused') else 'evaluation recorded')}", flush=True)
    failures = [row for row in attempts['runs'] if row['status'] == 'FAIL']
    attempts['status'] = ('FAIL' if failures else 'COMPLETE' if
                          all(row['status'] == 'COMPLETE' for row in attempts['runs']) else 'INCOMPLETE')
    attempts['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save_attempts()
    # A one-model/one-seed repair must not shrink the shared tables and hide
    # other models whose raw measurements already belong to this campaign.
    summary_args = copy.copy(args)
    summary_args.model = 'all'
    summary_args.ap_models = [model for model in MODELS if model in models or
                             (output / 'raw' / model).is_dir() or
                             any(key[0] == model for key in exclusions)]
    result = summarize(summary_args)
    if failures:
        raise RuntimeError(f'{len(failures)} AP runs failed; partial CSV reports were written to {output}. '
                           f'See {attempt_path}. Failed or unverified records are excluded.')
    return result
