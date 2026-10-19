"""Small effective-configuration audit for the complete three-factor matrix."""
import csv
import itertools
import json
from pathlib import Path

from . import run

FACTORS = ('vq_dense_o2o', 'li_copyblend', 'dc_sta')
PAIRS = (
    ('baseline_native', 'baseline_maskaware', 'mask'),
    ('baseline_maskaware', 'vq_dense_o2o', 'vq_dense_o2o'),
    ('baseline_maskaware', 'li_copyblend', 'li_copyblend'),
    ('baseline_maskaware', 'dc_sta', 'dc_sta'),
    ('vq_dense_o2o', 'vq_dense_o2o_li_copyblend', 'li_copyblend'),
    ('vq_dense_o2o', 'vq_dense_o2o_dc_sta', 'dc_sta'),
    ('li_copyblend', 'li_copyblend_dc_sta', 'dc_sta'),
    ('vq_dense_o2o_li_copyblend', 'vq_dense_o2o_li_copyblend_dc_sta', 'dc_sta'),
)


def flatten(value, prefix=''):
    if isinstance(value, dict) and value:
        return {path: leaf for key, item in value.items()
                for path, leaf in flatten(item, f'{prefix}.{key}' if prefix else key).items()}
    if isinstance(value, list) and value:
        return {path: leaf for index, item in enumerate(value)
                for path, leaf in flatten(item, f'{prefix}[{index}]').items()}
    return {prefix: value}


def _allowed(path, factor, before, after):
    if path == f'innovation.{factor}.enabled':
        return True
    if factor == 'vq_dense_o2o':
        return path == 'criterion' or path.startswith('VQDEIMCriterion.')
    if factor == 'li_copyblend':
        return path == 'train_dataloader.collate_fn.li_copyblend.enabled'
    if factor == 'dc_sta':
        return path == 'DINOv3STAs.dc_sta.enabled'
    if factor == 'mask':
        return (path == 'innovation.instance_mask_sync' or path in {
            'train_dataloader.dataset.type', 'train_dataloader.dataset.visible_mask_sidecar',
            'train_dataloader.dataset.visible_mask_required', 'train_dataloader.dataset.split',
            'train_dataloader.dataset.illumination_map'} or
            (path.startswith('train_dataloader.dataset.transforms.ops[') and path.endswith('].type')
             and (before, after) == ('SanitizeBoundingBoxes', 'SanitizeVisibleBoundingBoxes')))
    return False


def inspect_configs(configs):
    if set(configs) != set(run.ALL_MODELS):
        raise ValueError('Audit requires all nine formal model configurations')
    rows = []
    for code, model in zip(('B0', 'B1', 'V1', 'L1', 'D1', 'C1', 'C2', 'C3', 'C4'), run.ALL_MODELS):
        cfg = configs[model]
        run.verify_recipe(cfg)
        flags = cfg['innovation']
        if model == 'baseline_native' and (flags['instance_mask_sync'] or
                                           any(flags[name]['enabled'] for name in FACTORS)):
            raise ValueError('B0 must use the native data path with all innovation switches off')
        if flags['vq_dense_o2o']['beta'] != 0.10:
            raise ValueError('VQ beta must remain 0.10 in the common ablation protocol')
        if flags['vq_dense_o2o']['enabled']:
            expected = {**cfg['DEIMCriterion'], 'vq_dense_o2o': flags['vq_dense_o2o']}
            if cfg['criterion'] != 'VQDEIMCriterion' or cfg.get('VQDEIMCriterion') != expected:
                raise ValueError('VQ criterion must inherit every native criterion/matcher setting unchanged')
        elif cfg['criterion'] != 'DEIMCriterion':
            raise ValueError('Models without VQ must use the native DEIMCriterion')
        rows.append({'Code': code, 'Model': model, 'Mask': int(flags['instance_mask_sync']),
                     **{name: int(flags[name]['enabled']) for name in FACTORS},
                     'Beta': flags['vq_dense_o2o']['beta'], 'Epoch': cfg['epoches'],
                     'Batch': cfg['train_dataloader']['total_batch_size'],
                     'Output root': cfg['output_dir']})
    combinations = [tuple(row[name] for name in FACTORS) for row in rows[1:]]
    if sorted(combinations) != list(itertools.product((0, 1), repeat=3)) or any(row['Mask'] != 1 for row in rows[1:]):
        raise ValueError('B1-C4 must form the full mask-aware 2^3 factorial matrix')
    comparisons, unexpected = [], []
    for left, right, factor in PAIRS:
        before, after = flatten(configs[left]), flatten(configs[right])
        changes = []
        for key in sorted(set(before) | set(after)):
            if key in before and key in after and before[key] == after[key]:
                continue
            management = key in ('output_dir', 'experiment_name', 'seed')
            kind = '管理字段' if management else '允许差异' if _allowed(key, factor, before.get(key), after.get(key)) else '意外差异'
            item = {'field': key, 'before': before.get(key, '<absent>'), 'after': after.get(key, '<absent>'), 'kind': kind}
            changes.append(item)
            if kind == '意外差异':
                unexpected.append({'pair': [left, right], **item})
        comparisons.append({'left': left, 'right': right, 'factor': factor, 'changes': changes})
    return {'status': 'FAIL' if unexpected else 'PASS', 'rows': rows,
            'comparisons': comparisons, 'unexpected': unexpected}


def audit(matrix_path=None, report_path=None):
    result = inspect_configs({model: run.load_resolved(model, expand_env=False) for model in run.ALL_MODELS})
    matrix_path = Path(matrix_path or run.ROOT / 'outputs/paper_v2/config_ablation_matrix.csv')
    report_path = Path(report_path or run.ROOT / 'docs/paper_v2/ablation_config_audit.md')
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    lines = ['# 新算法配置消融审计', '', f"状态：**{result['status']}**。开发分支：paper_v3.0。",
             '', 'B1—C4组成完整2³矩阵；全部启用同一mask管线，VQ beta固定0.10。',
             '以下比较实际解析后的配置，输出目录/实验名/seed单列为管理字段。没有读取数据或checkpoint。', '']
    for comparison in result['comparisons']:
        lines += [f"## {comparison['left']} → {comparison['right']}", '',
                  f"唯一允许新增因素：`{comparison['factor']}`。", '',
                  '|实际字段|原值|新值|分类|', '|---|---|---|---|']
        for change in comparison['changes']:
            before, after = (json.dumps(change[key], ensure_ascii=False).replace('|', '\\|') for key in ('before', 'after'))
            lines.append(f"|{change['field']}|{before}|{after}|{change['kind']}|")
        lines.append('')
    lines += [f"意外差异：{len(result['unexpected'])}项。", '',
              '共同协议：120e；640²；3类；train batch32/val64；AdamW lr5e-4、DINO lr1.25e-5；',
              'FlatCosine warmup2000/flat60/no_aug14；EMA FP32 Mixed Validation AP选模，精确并列保留较早epoch。',
              'beta_select Baseline与当前共同配方数值一致；保留paper管理/同步修复，详见03_beta_select迁移说明.md。']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Configuration audit {result['status']}: {matrix_path}; {report_path}")
    if result['unexpected']:
        raise ValueError('Unexpected effective configuration differences; see audit report')
    return result
