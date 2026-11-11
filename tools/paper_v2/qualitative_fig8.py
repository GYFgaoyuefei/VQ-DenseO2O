"""Fig. 8 from saved COCO detections only; no checkpoint, inference or training.

Rank diagnostics are score-ordered greedy one-to-one matches, NOT COCO AP.
Synthetic fixtures belong in tests, never in a publication output directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.paper_v2.qualitative_fig8_data import (
    MODELS, COLORS, DIAGNOSTIC_DESCRIPTION, load_inputs, diagnose, case_evidence, candidate_rows,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to export')
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_roi(value, width, height):
    if (not isinstance(value, list) or len(value) != 4 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError('ROI must be four finite x1,y1,x2,y2 coordinates')
    x1, y1, x2, y2 = map(float, value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'ROI outside original {width}x{height} image or has nonpositive size: {value}')
    return [x1, y1, x2, y2]


def in_roi(box, roi, fully=False):
    x, y, w, h = box
    a, b, c, d = roi
    if fully:
        return a <= x and b <= y and x + w <= c and y + h <= d
    return x < c and y < d and x + w > a and y + h > b


def read_image(bundle, image_id, images_root):
    from PIL import Image
    meta = bundle['images'][image_id]
    path = Path(meta.get('_path', Path(images_root).expanduser() / meta['file_name'])).resolve()
    with Image.open(path) as source:
        # COCO coordinates refer to stored pixels; do not apply EXIF rotation.
        if source.size != (meta['width'], meta['height']):
            raise ValueError(f'COCO and stored image dimensions differ: {path}')
        image = source.convert('RGB').copy()
    return image, path


def diagnostics(bundle, image_id, threshold, roi=None):
    gt = bundle['gt'][image_id]
    if roi is not None:
        gt = [g for g in gt if in_roi(g['bbox'], roi, fully=True)]
    result = {}
    for alias in MODELS:
        predictions = bundle['predictions'][alias][image_id]
        if roi is not None:
            predictions = [p for p in predictions if in_roi(p['bbox'], roi, fully=True)]
        result[alias] = diagnose(gt, predictions, score_thr=threshold, iou_thr=.5)
    return result


def draw_boxes(ax, records, categories, roi, gt=False):
    from matplotlib.patches import Rectangle
    drawn = []
    for record in records:
        if not in_roi(record['bbox'], roi):
            continue
        x, y, w, h = record['bbox']
        cat = categories[record['category_id']]
        color = cat['color']
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                               linewidth=.8, linestyle=(0, (3, 2)) if gt else '-', clip_on=True))
        label = cat['name'] if gt else f"{cat['name']} {record['score']:.2f}"
        if gt and (record.get('iscrowd', 0) or record.get('ignore', 0)):
            label += ' (ignored)'
        ax.text(max(x, roi[0]), max(y, roi[1]), label, fontsize=5.2, color=color,
                ha='left', va='top', clip_on=True,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.15))
        drawn.append(record['id'] if gt else record['_index'])
    return drawn


def fit_labels(fig):
    """Keep complete class/score labels within the shared image viewport."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        boundary = ax.get_window_extent(renderer)
        for label in ax.texts:
            box = label.get_window_extent(renderer)
            if box.width > boundary.width - 2:
                label.set_fontsize(label.get_fontsize() * (boundary.width - 2) / box.width)
                box = label.get_window_extent(renderer)
            dx = max(0., boundary.x0 + 1 - box.x0) - max(0., box.x1 - boundary.x1 + 1)
            dy = max(0., boundary.y0 + 1 - box.y0) - max(0., box.y1 - boundary.y1 + 1)
            point = ax.transData.transform(label.get_position())
            label.set_position(ax.transData.inverted().transform(point + [dx, dy]))


def comparison_figure(bundle, ids, images_root, rois, titles, threshold, *, width=7.2):
    """Equal image/ROI/aspect/threshold per row, with vector box and text artists."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.image_inline': True, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    margin, gap, row_gap, title_h, header_h, footer = .10, .045, .12, .25, .23, .42
    panel_w = (width - 2 * margin - 5 * gap) / 6
    heights = [panel_w * (rois[i][3] - rois[i][1]) / (rois[i][2] - rois[i][0]) for i in ids]
    total_h = margin + header_h + footer + sum(h + title_h for h in heights) + row_gap * (len(ids) - 1)
    fig = plt.figure(figsize=(width, total_h), facecolor='white')
    y = total_h - margin - header_h
    for column, name in enumerate(('GT / Input', *MODELS)):
        fig.text((margin + column * (panel_w + gap) + panel_w / 2) / width,
                 (total_h - margin - .06) / total_h, name, fontsize=7, ha='center', va='top', weight='bold')
    panels, image_sources = [], []
    for row_index, (image_id, panel_h, title) in enumerate(zip(ids, heights, titles)):
        image, image_path = read_image(bundle, image_id, images_root)
        roi = rois[image_id]
        fig.text(margin / width, (y - .04) / total_h, title, fontsize=6.8, va='top')
        y -= title_h + panel_h
        image_sources.append({'image_id': image_id, 'path': str(image_path), 'sha256': sha256(image_path),
                              'original_size': list(image.size), 'roi': roi})
        for column, alias in enumerate(('GT / Input', *MODELS)):
            ax = fig.add_axes([(margin + column * (panel_w + gap)) / width, y / total_h,
                               panel_w / width, panel_h / total_h])
            ax.imshow(image, origin='upper', extent=(0, image.width, image.height, 0), interpolation='none')
            ax.set_xlim(roi[0], roi[2])
            ax.set_ylim(roi[3], roi[1])
            ax.set_aspect('equal')
            ax.set_axis_off()
            records = (bundle['gt'][image_id] if column == 0 else
                       [p for p in bundle['predictions'][alias][image_id] if p['score'] >= threshold])
            drawn = draw_boxes(ax, records, bundle['categories'], roi, gt=column == 0)
            panels.append({'row': row_index, 'column': column, 'image_id': image_id, 'model': alias,
                           'roi': roi, 'score_threshold': threshold, 'drawn_ids': drawn,
                           'drawn_count': len(drawn), 'image_sha256': image_sources[-1]['sha256'],
                           'pixels_per_inch_in_figure': (roi[2] - roi[0]) / panel_w})
        y -= row_gap
    handles = [Line2D([0], [0], color=color, lw=1.4, label=name) for name, color in COLORS.items()]
    handles += [Line2D([0], [0], color='#333333', lw=.8, linestyle='--', label='GT'),
                Line2D([0], [0], color='#333333', lw=.8, label='Prediction')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025), ncol=5,
               frameon=False, fontsize=6.2, handlelength=1.7, columnspacing=1.1)
    fit_labels(fig)
    return fig, panels, image_sources


def close_figure(fig):
    from matplotlib import pyplot as plt
    plt.close(fig)


def common_audit(bundle, args):
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode,
            'seed': args.seed, 'score_threshold': args.score_thr, 'match_iou_threshold': .5,
            'model_order': MODELS, 'categories': bundle['categories'], 'sources': bundle['sources'],
            'image_count': len(bundle['images']), 'invalid_predictions': bundle['invalid'],
            'diagnostic_protocol': DIAGNOSTIC_DESCRIPTION,
            'box_policy': 'Invalid coordinates/sizes/scores excluded and audited; original prediction JSON '
                'unchanged. No abs, endpoint swap or geometry repair. Boxes crossing ROI are display-clipped only.',
            'selection_policy': 'Candidates require human review; not an exhaustive performance claim.',
            'dpi': args.dpi, 'font': 'DejaVu Sans', 'figure_width_inches': 7.2}


def rank(bundle, args, destination):
    rows = candidate_rows(bundle, score_thr=args.score_thr)
    write_csv(destination / 'fig8_candidate_ranking.csv', rows)
    a = sorted((r for r in rows if r['A_supported']), key=lambda r: -r['A_score'])
    b = sorted((r for r in rows if r['B_supported']), key=lambda r: -r['B_score'])
    selected = []
    # Alternate the two candidate pools so limitations are not displaced by A.
    for index in range(max(len(a), len(b))):
        for pool in (a, b):
            if index < len(pool) and pool[index]['image_id'] not in selected:
                selected.append(pool[index]['image_id'])
    selected += [r['image_id'] for r in rows if r['image_id'] not in selected]
    selected = selected[:args.contact_count]
    rois = {i: [0., 0., float(bundle['images'][i]['width']), float(bundle['images'][i]['height'])] for i in selected}
    titles = [f"Candidate {i} | {bundle['illumination'][i]} | human review required" for i in selected]
    fig, panels, sources = comparison_figure(bundle, selected, args.images, rois, titles, args.score_thr)
    try:
        fig.savefig(destination / 'fig8_candidate_contact_sheet.png', dpi=args.dpi)
    finally:
        close_figure(fig)
    audit = common_audit(bundle, args)
    audit.update(candidate_count=len(rows), contact_sheet_image_ids=selected, panels=panels,
                 image_sources=sources, A_candidate_count=len(a), B_candidate_count=len(b))
    write_json(destination / 'Fig8_rank_audit.json', audit)


def captions(args, ids, bundle, supported):
    aliases = '; '.join(f'{alias}: {model}' for alias, model in MODELS.items())
    en = ('Fig. 8. Qualitative comparison on two manually selected Test images. Columns: ground truth/input, '
          'B1, VQ, LI, DC, and Full. Green, orange, and red denote Immature, Half-ripe, and Ripe, respectively. '
          'Dashed boxes indicate GT; solid boxes indicate saved predictions with class labels and confidence scores. '
          f'All models use seed {args.seed} and the same score threshold ({args.score_thr:.2f}), original RGB image, '
          'ROI, and display scale within each row. ')
    zh = (f'图8. 两张人工选择的Test图片的定性比较。列依次为GT/Input、B1、VQ、LI、DC和Full。绿色、橙色、红色'
          f'分别表示Immature、Half-ripe和Ripe。虚线为GT，实线为带类别与置信度的保存预测。所有模型使用seed={args.seed}、'
          f'统一阈值{args.score_thr:.2f}；每行共享原始RGB图像、ROI和显示比例。')
    for index, letter in enumerate(('A', 'B')):
        meta = f"image_id={ids[index]}, illumination={bundle['illumination'][ids[index]]}"
        if supported[index] and index == 0:
            en += f'(A) Example of improved detection by VQ relative to B1 ({meta}). '
            zh += f'（A）VQ相对B1改善检测的示例（{meta}）。'
        elif supported[index]:
            en += f'(B) Example illustrating a limitation: Full degrades relative to B1 or VQ under sodium light ({meta}). '
            zh += f'（B）局限性示例：钠黄光下Full相对B1或VQ退化（{meta}）。'
        else:
            en += f'({letter}) Selected comparison ({meta}); the requested case interpretation is not supported by the ROI diagnostic. '
            zh += f'（{letter}）所选比较（{meta}）；ROI诊断未支持预设的改善/退化解释。'
    en += ('Interpretations are assisted by same-class one-to-one IoU>=0.50 matches for fully visible boxes in the ROI, '
           'not COCO AP. These selected examples do not establish overall superiority. Invalid predicted boxes are '
           'excluded only from visualization/diagnostics and logged; official AP and source predictions are unchanged.\n' + aliases + '\n')
    zh += ('解释由ROI内完整可见框的同类别、IoU≥0.50一对一匹配辅助，不等于COCO AP，所选示例不代表整体性能。'
           '非法预测框仅在可视化/诊断中排除并审计，正式AP及原始预测不变。\n' + aliases + '\n')
    return en, zh


def render(bundle, args, destination):
    ids = args.image_ids
    if not ids or len(ids) != 2 or ids[0] == ids[1] or any(i not in bundle['images'] for i in ids):
        raise ValueError('--image-ids must name two distinct real COCO image IDs')
    roi_map = json.loads(args.roi_json.read_text(encoding='utf-8')) if args.roi_json else {}
    if not isinstance(roi_map, dict) or any(k not in {str(i) for i in bundle['images']} for k in roi_map):
        raise ValueError('ROI JSON must map real COCO image IDs to original-coordinate ROIs')
    rois, cases, titles, support = {}, [], [], []
    for index, image_id in enumerate(ids):
        meta = bundle['images'][image_id]
        rois[image_id] = validate_roi(roi_map.get(str(image_id), [0, 0, meta['width'], meta['height']]),
                                     meta['width'], meta['height'])
        full_diagnostics = diagnostics(bundle, image_id, args.score_thr)
        visible_diagnostics = diagnostics(bundle, image_id, args.score_thr, rois[image_id])
        evidence = case_evidence(visible_diagnostics, bundle['illumination'][image_id])
        supported = evidence['A_supported' if index == 0 else 'B_supported']
        support.append(supported)
        label = ('Example of improved detection' if index == 0 else 'Example illustrating a limitation') if supported else 'Selected comparison'
        titles.append(f'({"AB"[index]}) {label}')
        cases.append({'image_id': image_id, 'file_name': meta['file_name'],
                      'illumination': bundle['illumination'][image_id], 'seed': args.seed,
                      'original_size': [meta['width'], meta['height']], 'roi': rois[image_id],
                      'score_threshold': args.score_thr, 'gt_count': len(bundle['gt'][image_id]),
                      'diagnostics_full_image': full_diagnostics, 'diagnostics_fully_inside_roi': visible_diagnostics,
                      'evidence': evidence, 'title': titles[-1], 'interpretation_supported': supported})
    fig, panels, sources = comparison_figure(bundle, ids, args.images, rois, titles, args.score_thr)
    try:
        for suffix in ('png', 'tif', 'svg'):
            # Raw TIFF avoids depending on an optional platform libtiff codec.
            kwargs = {'pil_kwargs': {'compression': 'raw'}} if suffix == 'tif' else {}
            fig.savefig(destination / f'Fig8_Qualitative_Comparison.{suffix}', dpi=args.dpi, **kwargs)
        size_inches = list(fig.get_size_inches())
    finally:
        close_figure(fig)
    full_rois = {i: [0., 0., float(bundle['images'][i]['width']), float(bundle['images'][i]['height'])] for i in ids}
    fig, _, _ = comparison_figure(bundle, ids, args.images, full_rois,
                                  [f'{"AB"[j]} | image_id={i} | full image for review' for j, i in enumerate(ids)], args.score_thr)
    try:
        fig.savefig(destination / 'Fig8_full_image_contact_sheet.png', dpi=args.dpi)
    finally:
        close_figure(fig)
    selected_rows = []
    for case in cases:
        image_id = case['image_id']
        for alias, model in MODELS.items():
            d = case['diagnostics_full_image'][alias]
            panel = next(p for p in panels if p['image_id'] == image_id and p['model'] == alias)
            invalid = [p for p in bundle['invalid'] if p['alias'] == alias and p['image_id'] == image_id]
            selected_rows.append({
                'image_id': image_id, 'file_name': case['file_name'], 'illumination': case['illumination'],
                'seed': args.seed, 'model': model, 'alias': alias,
                'prediction_json': bundle['prediction_paths'][alias],
                'original_width': case['original_size'][0], 'original_height': case['original_size'][1],
                'roi_xyxy': json.dumps(case['roi']), 'score_threshold': args.score_thr,
                'category_mapping': json.dumps(bundle['categories'], ensure_ascii=False),
                'gt_count': case['gt_count'], 'drawn_prediction_count': panel['drawn_count'],
                'invalid_prediction_count': len(invalid), 'TP': d['tp'], 'FP': d['fp'], 'FN': d['fn'],
                'class_errors': d['class_errors'], 'matched_mean_iou': d['mean_iou'],
                'gt_matching': json.dumps(d, ensure_ascii=False), 'interpretation_supported': case['interpretation_supported'],
            })
    write_csv(destination / 'Fig8_selected_cases.csv', selected_rows)
    en, zh = captions(args, ids, bundle, support)
    (destination / 'Fig8_caption_en.txt').write_text(en, encoding='utf-8')
    (destination / 'Fig8_caption_zh.txt').write_text(zh, encoding='utf-8')
    from PIL import Image
    with Image.open(destination / 'Fig8_Qualitative_Comparison.png') as image:
        pixel_size = list(image.size)
    audit = common_audit(bundle, args)
    if args.roi_json:
        audit['roi_source'] = {'path': str(args.roi_json.resolve()), 'sha256': sha256(args.roi_json)}
    audit.update(cases=cases, panels=panels, image_sources=sources, selected_image_ids=ids,
                 figure_size_inches=size_inches, output_pixel_size=pixel_size,
                 roi_label_evidence='Only boxes fully inside the shared ROI; original xywh used for IoU, no clipping repair.',
                 files=sorted([p.name for p in destination.iterdir()] + ['Fig8_render_audit.json']))
    write_json(destination / 'Fig8_render_audit.json', audit)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('rank', 'render'), required=True)
    for key in ('annotations', 'images', 'pred-root', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-ids', type=int, nargs=2, metavar=('A', 'B'))
    p.add_argument('--score-thr', type=float, default=.30)
    p.add_argument('--roi-json', type=Path)
    p.add_argument('--illumination-map', type=Path)
    p.add_argument('--dataset-manifest', type=Path, help='Explicit dataset provenance for prediction exports without metrics.json')
    p.add_argument('--expected-images', type=int, default=400, help='Use a different count only for a documented dataset or test fixture')
    p.add_argument('--contact-count', type=int, default=12)
    p.add_argument('--dpi', type=int, default=300)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not math.isfinite(args.score_thr) or not 0 <= args.score_thr <= 1:
        raise ValueError('--score-thr must be in [0,1]')
    if args.dpi < 300 or args.contact_count < 1 or args.expected_images < 1:
        raise ValueError('dpi>=300 and positive contact/image counts are required')
    if args.mode == 'rank' and (args.roi_json or args.image_ids):
        raise ValueError('Rank always scans full images; ROI and image IDs belong to render')
    bundle = load_inputs(args.annotations, args.images, args.pred_root, seed=args.seed,
                         illumination_map=args.illumination_map, expected_images=args.expected_images,
                         dataset_manifest=args.dataset_manifest)
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Stage a complete artifact set; never overwrite another paper figure.
    staging = Path(tempfile.mkdtemp(prefix='.fig8_', dir=args.out.parent)).resolve()
    if staging.parent != args.out.parent or not staging.name.startswith('.fig8_'):
        raise RuntimeError('Temporary output directory is outside the intended output parent')
    try:
        (rank if args.mode == 'rank' else render)(bundle, args, staging)
        collisions = [p.name for p in staging.iterdir() if (args.out / p.name).exists()]
        if collisions:
            raise FileExistsError(f'Existing Fig8 outputs are preserved; choose a new --out: {collisions}')
        args.out.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.rename(args.out / path.name)
    finally:
        shutil.rmtree(staging)
    print(f'{args.mode}: exported real-input artifacts to {args.out}; diagnostics are not COCO AP.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Fig8: {exc}')
