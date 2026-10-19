"""Deterministic beta_select Baseline comparisons; no datasets or weights are read.

The reference code is read from Git, never checked out or registered in the live
registry. Existing test_data_pipeline/test_data_sync cover B0/B1 equivalence and
invalid-mask retention; these tests add the independent historical comparison.
"""
import ast
import copy
from functools import lru_cache
import inspect
from pathlib import Path
import random
import subprocess

from PIL import Image
import pytest
import torch
import yaml

from engine.core.yaml_utils import load_config, merge_dict
from engine.backbone.dinov3_adapter import DINOv3STAs
from engine.data._misc import convert_to_tv_tensor
from engine.data.dataloader import BatchImageCollateFunction
from engine.data.transforms.mosaic import Mosaic
from engine.deim.deim import DEIM
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.deim_decoder import DEIMTransformer
from engine.deim.hybrid_encoder import HybridEncoder
from engine.deim.matcher import HungarianMatcher


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_COMMIT = 'f92edb021fd468e354f5767bb81ac745e0347cfe'  # Audited beta_select HEAD.


@lru_cache(None)
def reference_source(path):
    result = subprocess.run(['git', 'show', f'{REFERENCE_COMMIT}:{path}'], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        pytest.skip(f'Audited beta_select reference {REFERENCE_COMMIT} is unavailable; fetch its history')
    return result.stdout


@lru_cache(None)
def reference_module(path):
    """Suppress registry writes only; execute the unchanged algorithm bodies."""
    tree = ast.parse(reference_source(path), filename=f'beta_select:{path}')
    namespace = {'__name__': '_beta_baseline_reference',
                 '__package__': path.rsplit('/', 1)[0].replace('/', '.')}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            node.decorator_list = [decorator for decorator in node.decorator_list
                if not (isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == 'register')]
    # Historical mask helpers were renamed during the field-sync fix. The
    # reference collator must retain its own helpers, without replacing ours.
    for node in list(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == 'visible_geometry':
            old_geometry = reference_module('engine/data/visible_geometry.py')
            namespace.update({alias.asname or alias.name: old_geometry[alias.name]
                              for alias in node.names})
            tree.body.remove(node)
    exec(compile(ast.fix_missing_locations(tree), f'beta_select:{path}', 'exec'), namespace)
    return namespace


@lru_cache(None)
def reference_config():
    base = yaml.safe_load(reference_source('configs/tomato/common_validation.yml'))
    baseline = yaml.safe_load(reference_source('configs/tomato/deimv2_l_baseline.yml'))
    return merge_dict(base, baseline)


def construct(cls, options, shared=None):
    options = {**(shared or {}), **options}
    return cls(**{name: copy.deepcopy(options[name])
                  for name in inspect.signature(cls).parameters if name in options})


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_public_training_recipe_matches_effective_beta_baseline():
    reference = copy.deepcopy(reference_config())
    current = load_config(str(ROOT / 'configs/tomato/paper_v2/baseline_native.yml'))
    assert reference['innovation']['dc_sta']['enabled'] is False
    assert reference['innovation']['vq_dense_o2o']['enabled'] is False
    assert current['innovation']['instance_mask_sync'] is False
    assert all(not current['innovation'][name]['enabled']
               for name in ('vq_dense_o2o', 'li_copyblend', 'dc_sta'))
    # Only experiment management and explicitly disabled additions differ.
    for cfg in (reference, current):
        for key in ('__include__', 'innovation', 'output_dir', 'experiment_name',
                    'paper_v2', 'print_freq', 'checkpoint_freq'):
            cfg.pop(key, None)
    assert current == reference


def test_full_detector_with_disabled_modules_matches_beta_weights_and_outputs():
    cfg = copy.deepcopy(reference_config())
    cfg['DINOv3STAs']['weights_path'] = None
    # Keep the full DEIMv2-L width, depth and 300 queries; reduce the synthetic
    # canvas in BOTH models so this deterministic CPU test is bounded.
    cfg['eval_spatial_size'] = [128, 128]
    reference_classes = [reference_module(path)[name] for path, name in (
        ('engine/backbone/dinov3_adapter.py', 'DINOv3STAs'),
        ('engine/deim/hybrid_encoder.py', 'HybridEncoder'),
        ('engine/deim/deim_decoder.py', 'DEIMTransformer'),
        ('engine/deim/deim.py', 'DEIM'))]

    def model(classes):
        backbone, encoder, decoder, detector = classes
        return detector(construct(backbone, cfg['DINOv3STAs']),
                        construct(encoder, cfg['HybridEncoder'], cfg),
                        construct(decoder, cfg['DEIMTransformer'], cfg)).eval()

    torch.manual_seed(3407)
    reference = model(reference_classes)
    torch.manual_seed(3407)
    current = model((DINOv3STAs, HybridEncoder, DEIMTransformer, DEIM))
    assert current.backbone.dc_sta is None
    assert reference.state_dict().keys() == current.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
    samples = torch.linspace(-2, 2, 3 * 128 * 128).reshape(1, 3, 128, 128)
    with torch.inference_mode():
        expected, actual = reference(samples), current(samples)
    assert actual.keys() == expected.keys()
    assert actual['pred_logits'].shape == (1, 300, 3)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('epoch', [0, 88])
def test_native_matcher_mal_and_gradients_equal_beta_before_and_after_switch(epoch):
    cfg = reference_config()
    old_matcher = reference_module('engine/deim/matcher.py')['HungarianMatcher']
    old_criterion = reference_module('engine/deim/deim_criterion.py')['DEIMCriterion']
    options = copy.deepcopy(cfg['DEIMCriterion'])
    options['losses'] = ['mal', 'boxes']
    torch.manual_seed(42)
    logits = torch.randn(2, 9, 3)
    boxes = torch.rand(2, 9, 4) * .6 + .1
    targets = [{'boxes': torch.tensor([[.3, .4, .2, .3], [.7, .6, .15, .2]]),
                'labels': torch.tensor([0, 2])} for _ in range(2)]
    records = []
    for matcher_cls, criterion_cls in ((old_matcher, old_criterion),
                                       (HungarianMatcher, DEIMCriterion)):
        matcher = construct(matcher_cls, options['matcher'], cfg)
        criterion = construct(criterion_cls, {**options, 'matcher': matcher}, cfg)
        outputs = {'pred_logits': logits.clone().requires_grad_(),
                   'pred_boxes': boxes.clone().requires_grad_()}
        differentiable_inputs = tuple(outputs.values())
        outputs['aux_outputs'] = [dict(pred_logits=outputs['pred_logits'] * .9,
                                       pred_boxes=outputs['pred_boxes'] * .95)]
        outputs['pre_outputs'] = dict(pred_logits=outputs['pred_logits'] * .8,
                                      pred_boxes=outputs['pred_boxes'] * .9)
        outputs['enc_aux_outputs'] = [dict(pred_logits=outputs['pred_logits'] * .7,
                                           pred_boxes=outputs['pred_boxes'] * .85)]
        outputs['enc_meta'] = {'class_agnostic': False}
        indices = matcher(outputs, targets, epoch=epoch)['indices']
        losses = criterion(outputs, targets, epoch=epoch)
        gradients = torch.autograd.grad(sum(losses.values()), differentiable_inputs)
        records.append((indices, losses, gradients))
    for left, right in zip(records[0][0], records[1][0]):
        for a, b in zip(left, right):
            assert torch.equal(a, b)
    assert records[0][1].keys() == records[1][1].keys()
    for name, expected in records[0][1].items():
        torch.testing.assert_close(records[1][1][name], expected, rtol=0, atol=0)
    for actual, expected in zip(records[1][2], records[0][2]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('mixup,copyblend', [(1., 0.), (0., 1.), (1., 1.)])
@pytest.mark.parametrize('seed', [0, 42, 3407])
def test_native_dense_augmentation_rgb_boxes_labels_match_beta(mixup, copyblend, seed):
    old_collate = reference_module('engine/data/dataloader.py')['BatchImageCollateFunction']
    options = dict(base_size=64, base_size_repeat=3, stop_epoch=106,
                   mixup_prob=mixup, mixup_epochs=[7, 60],
                   copyblend_prob=copyblend, copyblend_epochs=[7, 106],
                   with_expand=True, num_objects=3, area_threshold=100)
    samples = []
    for image_id in range(2):
        target = dict(boxes=torch.tensor([[.25, .3, .1, .1], [.7, .6, .1, .12]]),
                      labels=torch.tensor([image_id, 2]), area=torch.tensor([12., 15.]),
                      iscrowd=torch.zeros(2, dtype=torch.int64),
                      image_id=torch.tensor([image_id]), orig_size=torch.tensor([64, 64]))
        samples.append((torch.arange(3*64*64).remainder(251).reshape(3, 64, 64).float()/251 + image_id, target))
    results = []
    for collate_cls in (old_collate, BatchImageCollateFunction):
        collate = collate_cls(**options)
        collate.set_epoch(10)
        torch.manual_seed(seed)
        random.seed(seed)
        images, targets = collate(copy.deepcopy(samples))
        results.append((images, targets, random.getstate(), torch.random.get_rng_state()))
    expected, actual = results
    assert torch.equal(actual[0], expected[0])
    assert actual[2] == expected[2] and torch.equal(actual[3], expected[3])
    for old, new in zip(expected[1], actual[1]):
        for key in ('boxes', 'labels', 'area', 'mixup'):
            assert (key in old) == (key in new)
            if key in old:
                assert torch.equal(old[key], new[key]), key
        # Historical iscrowd was stale after concatenation; the new common
        # metadata synchronizer intentionally repairs this for B0 and B1 alike.
        assert len(new['iscrowd']) == len(new['labels'])


def test_native_mosaic_pixels_boxes_labels_match_beta():
    old_mosaic = reference_module('engine/data/transforms/mosaic.py')['Mosaic']
    images = [Image.new('RGB', (16, 16), (index * 20, 80, 120)) for index in range(4)]
    targets = [dict(boxes=convert_to_tv_tensor(torch.tensor([[2., 2., 8., 7.]]),
                       'boxes', spatial_size=(16, 16)), labels=torch.tensor([index % 3]),
                    area=torch.tensor([30.]), iscrowd=torch.tensor([0]),
                    image_id=torch.tensor([index]), orig_size=torch.tensor([16, 16]))
               for index in range(4)]
    expected = old_mosaic(output_size=16).create_mosaic_from_dataset(
        images, copy.deepcopy(targets), 16, 16)
    actual = Mosaic(output_size=16).create_mosaic_from_dataset(
        images, copy.deepcopy(targets), 16, 16)
    assert actual[0].tobytes() == expected[0].tobytes()
    for key in ('boxes', 'labels', 'area', 'iscrowd'):
        assert torch.equal(actual[1][key], expected[1][key])


def test_native_visualization_never_deletes_existing_files(tmp_path):
    existing = tmp_path / 'previous_visualization.txt'
    existing.write_text('preserve user output', encoding='utf-8')
    with pytest.raises(FileExistsError, match='not deleted'):
        BatchImageCollateFunction(mixup_prob=.5, data_vis=True, vis_save=str(tmp_path))
    assert existing.read_text(encoding='utf-8') == 'preserve user output'
