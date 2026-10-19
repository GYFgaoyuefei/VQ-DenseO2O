"""Actual registered 640px Paper v2 pipelines on temporary synthetic COCO data."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch

from engine.core import YAMLConfig
from engine.data.visible_geometry import assert_visible_alignment, instance_fields
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher


class SharedRecipeDataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        root = Path('outputs/paper_v2_checks')
        root.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=root, prefix='pipeline_')
        cls.path = Path(cls.tmp.name).resolve()
        (cls.path/'images/train').mkdir(parents=True)
        (cls.path/'annotations').mkdir()
        images, annotations, side_annotations = [], [], {}
        lights = ('natural', 'artificial', 'low_light', 'sodium_light')
        for image_id in range(4):
            image = Image.new('RGB', (96, 80), (80, 80, 80))
            draw = ImageDraw.Draw(image)
            draw.polygon([(10, 10), (25, 10), (10, 25)], fill=(160, 50, 25))
            draw.rectangle((50, 35, 66, 53), fill=(45, 135, 40))
            image.save(cls.path/f'images/train/{image_id}.png')
            images.append(dict(id=image_id, file_name=f'{image_id}.png', width=96, height=80, illumination=lights[image_id]))
            for object_index, (bbox, cat, valid) in enumerate((([10, 10, 15, 15], 3, True), ([50, 35, 16, 18], 1, False))):
                aid = image_id * 2 + object_index + 1
                annotations.append(dict(id=aid, image_id=image_id, category_id=cat, bbox=bbox, area=bbox[2]*bbox[3], iscrowd=0))
                side_annotations[str(aid)] = dict(image_id=image_id, category_id=cat,
                    visible_mask_valid=valid, segmentation=[[10, 10, 25, 10, 10, 25]] if valid else [], visibility_group='easy')
        coco = dict(images=images, annotations=annotations, categories=[dict(id=i, name=str(i)) for i in (1, 2, 3)])
        ann = cls.path/'annotations/instances_train.json'
        ann.write_text(json.dumps(coco), encoding='utf-8')
        sidecar = dict(version='visible-mask-sidecar-v1', coco_annotation_sha256=hashlib.sha256(ann.read_bytes()).hexdigest(), annotations=side_annotations)
        (cls.path/'annotations/visible_masks_train.json').write_text(json.dumps(sidecar), encoding='utf-8')
        # These paths are never opened: this suite instantiates data, not pretrained networks.
        cls.env = patch.dict(os.environ, TOMATO_DATA_ROOT=str(cls.path), DINOV3_WEIGHTS=str(cls.path/'unused_dino.pth'),
                             TOMATO_DETECTOR_CHECKPOINT=str(cls.path/'unused_detector.pth'))
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.tmp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def make_loader(self, model):
        with redirect_stdout(io.StringIO()):
            cfg = YAMLConfig(f'configs/tomato/paper_v2/{model}.yml',
                train_dataloader={'total_batch_size': 2, 'num_workers': 0, 'shuffle': False})
            return cfg.train_dataloader

    def run_batch(self, loader, epoch, seed):
        loader.set_epoch(epoch)
        random.seed(seed)
        torch.manual_seed(seed)
        with redirect_stdout(io.StringIO()):
            images, targets = next(iter(loader))
        for target in targets:
            assert_visible_alignment(target, *images.shape[-2:])
            for field in instance_fields(target):
                self.assertEqual(len(target[field]), len(target['labels']), field)
            self.assertEqual(target['image_id'].numel(), 1)
            self.assertEqual(target['orig_size'].numel(), 2)
        return images, targets

    def test_all_five_registered_recipes_at_four_schedule_stages(self):
        for model in ('baseline_native', 'baseline_maskaware', 'vq_dense_o2o', 'li_copyblend', 'dc_sta'):
            loader = self.make_loader(model)
            for epoch in (0, 10, 80, 110):
                with self.subTest(model=model, epoch=epoch):
                    images, targets = self.run_batch(loader, epoch, 42)
                    self.assertEqual(images.shape[:2], (2, 3))
                    for target in targets:
                        self.assertEqual('masks' in target, model != 'baseline_native')
                        if model != 'baseline_native':
                            self.assertTrue(target['source_is_train'].all())
                            self.assertTrue(((target['source_illumination_id'] >= 0) & (target['source_illumination_id'] < 4)).all())

    def test_b0_b1_full_pipeline_fixed_seeds_rgb_bbox_class_equivalence(self):
        routes = set()
        criterion = DEIMCriterion(
            matcher=HungarianMatcher({'cost_class': 2., 'cost_bbox': 5., 'cost_giou': 2.}, use_focal_loss=True),
            weight_dict={'loss_mal': 1.}, losses=['mal'], num_classes=3, gamma=1.5)
        for seed in (0, 42, 3407):
            a_loader, b_loader = self.make_loader('baseline_native'), self.make_loader('baseline_maskaware')
            for epoch in (0, 10, 80, 110):
                with self.subTest(epoch=epoch, seed=seed):
                    a, ta = self.run_batch(a_loader, epoch, seed)
                    a_state = random.getstate()
                    b, tb = self.run_batch(b_loader, epoch, seed)
                    self.assertEqual(a_state, random.getstate())
                    self.assertTrue(torch.equal(a, b))
                    for first, second in zip(ta, tb):
                        for key in ('boxes', 'labels', 'area', 'iscrowd'):
                            self.assertTrue(torch.equal(first[key], second[key]), key)
                        if second['image_illumination_id'].item() == -1:
                            routes.add('mixed')
                        if (second['annotation_ids'] < 0).any():
                            routes.add('copyblend')
                        elif 'mixup' in second:
                            routes.add('mixup')
                    # Reuse the existing paired augmentation test to verify the
                    # actual native matcher + MAL target and gradient path too.
                    generator = torch.Generator().manual_seed(71)
                    queries = max(len(t['labels']) for t in ta) + 4
                    boxes = torch.rand(len(ta), queries, 4, generator=generator)
                    logits = torch.randn(len(ta), queries, 3, generator=generator)
                    logits_a, logits_b = logits.clone().requires_grad_(), logits.clone().requires_grad_()
                    loss_a = criterion({'pred_boxes': boxes, 'pred_logits': logits_a,
                                        'aux_outputs': [], 'enc_aux_outputs': [],
                                        'enc_meta': {'class_agnostic': False}}, ta)['loss_mal']
                    loss_b = criterion({'pred_boxes': boxes, 'pred_logits': logits_b,
                                        'aux_outputs': [], 'enc_aux_outputs': [],
                                        'enc_meta': {'class_agnostic': False}}, tb)['loss_mal']
                    self.assertTrue(torch.equal(loss_a, loss_b))
                    grad_a, = torch.autograd.grad(loss_a, logits_a)
                    grad_b, = torch.autograd.grad(loss_b, logits_b)
                    self.assertTrue(torch.equal(grad_a, grad_b))
                    if b_loader.dataset._transforms.transforms[0].mosaic_cache:
                        routes.add('mosaic')
        self.assertTrue({'mosaic', 'mixup', 'copyblend'}.issubset(routes), routes)

    def test_collate_constructor_does_not_delete_vis_when_disabled(self):
        from engine.data.dataloader import BatchImageCollateFunction
        vis = self.path/'existing_vis'
        vis.mkdir(exist_ok=True)
        keep = vis/'keep.txt'
        keep.write_text('user file', encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            BatchImageCollateFunction(mixup_prob=.5, copyblend_prob=.5, vis_save=str(vis), data_vis=False)
        self.assertEqual(keep.read_text(encoding='utf-8'), 'user file')


if __name__ == '__main__':
    unittest.main()
