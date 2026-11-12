"""Report complete three-seed comparisons, keeping Test, Validation and training costs separate."""
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile

from tools.paper_v2.efficiency_protocol import AP_COLUMNS, COCO_NAMES, PAPER_COLUMNS, SEEDS

from .safeguards import inspect_tree, reject_link

CORE = ('m_base', 'm_kd', 's_base', 's_kd')
EXTRA = ('Seed', 'TeacherID', 'ProtocolID', 'RunTag', 'Split', 'Status', 'FLOPs coverage')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write_csv(path, cols, rows):
    reject_link(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False,
                                     encoding='utf-8-sig', newline='') as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, cols, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _teacher_id(identity):
    if not identity:
        return None
    return identity['id']


def _summary(model, group, split):
    result = dict(Model=model, Split=split, Status='NOT RUN')
    if len(group) != 3 or {row['Seed'] for row in group} != set(SEEDS):
        result['Status'] = 'INCOMPLETE' if group else 'NOT RUN'
        return result
    result['Status'] = 'COMPLETE' if all(row['Status'] == 'COMPLETE' for row in group) else 'INCOMPLETE'
    for key in ('Epoch', 'Latency(ms)', *AP_COLUMNS):
        values = [row.get(key) for row in group]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            result[key] = f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}'
            if key == 'Latency(ms)':
                result['FPS'] = 1000 / statistics.mean(values)
    for key in ('Hardware', 'Params(M)', 'GFLOPs', 'FLOPs coverage'):
        if all(row.get(key) == group[0].get(key) for row in group):
            result[key] = group[0].get(key)
    return result


def _bind_efficiency(row, bench, signatures):
    from .benchmark import PROTOCOL
    if (bench.get('formal') is not True or bench['protocol'] != PROTOCOL
            or (bench['warmup'], bench['iterations'], bench['repeats']) != (100, 500, 5)):
        raise ValueError('Unexpected formal benchmark protocol')
    flops = bench['flops']
    signatures.add((bench['protocol'], bench['hardware'], bench['torch_version'], bench['cuda_version'],
                    bench['cudnn_version'], flops['tool'], flops['tool_version'], flops['counting_convention']))
    coverage = flops['coverage_status'] if flops['gflops'] is not None else flops['status']
    row.update({'Latency(ms)': bench['latency_ms'], 'Hardware': bench['hardware'],
                'Params(M)': bench['params_m'], 'GFLOPs': flops['gflops'],
                'FPS': 1000 / bench['latency_ms'], 'FLOPs coverage': coverage})


def summarize_runs(tag_dir, *, teacher_result=None, split='test'):
    if split not in ('val', 'test'):
        raise ValueError('Summary split must be val or test')
    tag_dir = Path(tag_dir)
    if not tag_dir.is_dir():
        raise FileNotFoundError(tag_dir)
    inspect_tree(tag_dir)
    rows, costs, paired = [], [], []
    teachers, signatures, data_fingerprints = set(), set(), set()
    geometry = {}
    for path in sorted(tag_dir.glob('*/seed_*/run_info.json')):
        run = path.parent
        info = _read(path)
        model, seed = info['experiment'], info['seed']
        if info.get('smoke'):
            continue
        if ((model, f'seed_{seed}') != (run.parent.name, run.name)
                or info['run_tag'] != tag_dir.name or info['protocol_id'] != tag_dir.parent.name):
            raise ValueError(f'Inconsistent run identity {path}')
        tid = _teacher_id(info['teacher'])
        if tid:
            teachers.add(tid)
        row = dict(Model=model, Seed=seed, TeacherID=tid, ProtocolID=info['protocol_id'],
                   RunTag=tag_dir.name, Split=split, Status='NOT RUN')
        selection_path = run / 'selection.json'
        selection = _read(selection_path) if selection_path.is_file() else None
        if selection:
            if any(selection[key] != info[key] for key in ('protocol_id', 'experiment', 'seed', 'run_tag', 'teacher')):
                raise ValueError(f'Inconsistent selection {run}')
            row['Epoch'] = selection['epoch']
        metrics_path = run / f'evaluation_{split}' / 'metrics.json'
        if metrics_path.is_file():
            record = _read(metrics_path)
            if (not selection or record['selection'] != selection or record['split'] != split
                    or split == 'test' and not record['frozen']):
                raise ValueError(f'Unbound {split} report {metrics_path}')
            row.update({k: record['metrics'][k] for k in AP_COLUMNS})
            row['Status'] = record['status']
            data_fingerprints.add((record['annotation_sha256'], record.get('illumination_map_sha256')))
        elif split == 'val' and selection:
            with (run / 'validation_history.csv').open(encoding='utf-8') as stream:
                history = next(r for r in csv.DictReader(stream) if int(r['epoch']) == selection['epoch'])
            if float(history['AP']) != selection['mixed_ap']:
                raise ValueError(f'Validation AP does not match selection {run}')
            for name, column in zip(COCO_NAMES, AP_COLUMNS):
                value = float(history[name])
                row[column] = value * 100 if value >= 0 else None
            row['Status'] = 'MIXED ONLY'
        if (run / 'benchmark.json').is_file():
            bench = _read(run / 'benchmark.json')
            if not selection or bench.get('selection') != selection or bench['formal'] is not True:
                raise ValueError(f'Unbound formal benchmark {run}')
            _bind_efficiency(row, bench, signatures)
            measured = (row['Params(M)'], row['GFLOPs'])
            if geometry.setdefault(info['arch'], measured) != measured:
                raise ValueError(f'Deploy Params/FLOPs differ for {info["arch"]}')
        diag = run / 'kd_diagnostics.json'
        if diag.is_file():
            epochs = _read(diag)
            if epochs:
                costs.append(dict(Model=model, Seed=seed, Epochs=len(epochs),
                                  OptimizerUpdates=epochs[-1]['optimizer_updates'],
                                  TrainSeconds=sum(e['train_seconds'] for e in epochs),
                                  PeakBytes=max(e['peak_memory_bytes'] for e in epochs)))
        rows.append(row)
    if len(teachers) > 1 or len(signatures) > 1 or len(data_fingerprints) > 1:
        raise ValueError('Cannot compare mixed Teacher, benchmark, or evaluation dataset identities')
    summary = [_summary(model, [r for r in rows if r['Model'] == model], split) for model in CORE]
    lookup = {(r['Model'], r['Seed']): r for r in rows}
    for arch in ('m', 's'):
        diffs = []
        for seed in SEEDS:
            base, kd = [lookup.get((arch + suffix, seed), {}) for suffix in ('_base', '_kd')]
            a, b = [r.get('Mixed AP (%)') for r in (base, kd)]
            if a is not None and b is not None:
                delta = b - a
                diffs.append(delta)
                paired.append(dict(Arch=arch, Seed=seed, DeltaAP_pp=delta))
        if len(diffs) == len(SEEDS):
            paired.append(dict(Arch=arch, Seed='all', DeltaAP_pp=statistics.mean(diffs),
                               SampleSD_pp=statistics.stdev(diffs)))
    if teacher_result:
        t0 = _read(Path(teacher_result))
        tid = t0['checkpoint_sha256']
        if (len(tid) != 64 or any(ch not in '0123456789abcdef' for ch in tid)
                or (teachers and t0['teacher_id'] not in teachers) or t0['split'] != split
                or not t0['teacher_id'].startswith(tid + '/')
                or not t0.get('validation_basis') or 'seed_mean' in t0):
            raise ValueError('T0 must reference the exact shared Validation-selected checkpoint and split')
        if data_fingerprints and (t0['annotation_sha256'], t0.get('illumination_map_sha256')) not in data_fingerprints:
            raise ValueError('T0 dataset identity differs from students')
        teacher_row = {'Model': 'T0', 'TeacherID': tid, 'Epoch': t0['epoch'], 'Split': split,
                       **{k: t0['metrics'][k] for k in AP_COLUMNS}, 'Status': 'FIXED CHECKPOINT'}
        bench_path = tag_dir / 'T0' / 'benchmark_teacher.json'
        if bench_path.is_file():
            bench = _read(bench_path)
            if bench['teacher']['id'] != t0['teacher_id']:
                raise ValueError('T0 efficiency and accuracy use different checkpoint states')
            _bind_efficiency(teacher_row, bench, signatures)
            if len(signatures) > 1:
                raise ValueError('T0 and students used different benchmark protocols')
        summary.insert(0, teacher_row)
    else:
        summary.insert(0, {'Model': 'T0', 'Split': split, 'Status': 'NOT RUN'})
    dest = tag_dir / 'summary' / split
    dest.mkdir(parents=True, exist_ok=True)
    _write_csv(dest / 'per_seed.csv', (*PAPER_COLUMNS, *EXTRA), rows)
    _write_csv(dest / 'paper.csv', (*PAPER_COLUMNS, 'Split', 'Status', 'TeacherID', 'FLOPs coverage'), summary)
    _write_csv(dest / 'paired_ap.csv', ('Arch', 'Seed', 'DeltaAP_pp', 'SampleSD_pp'), paired)
    _write_csv(dest / 'training_cost.csv', ('Model', 'Seed', 'Epochs', 'OptimizerUpdates',
                                             'TrainSeconds', 'PeakBytes'), costs)
    return {'directory': str(dest), 'rows': len(rows), 'split': split, 'teacher': bool(teacher_result)}
