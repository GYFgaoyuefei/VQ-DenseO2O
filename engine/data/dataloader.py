"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os
from collections import Counter, defaultdict, deque
from .visible_geometry import (concatenate_instances, translated_record, append_records,
                               resize_masks, assert_visible_alignment)
from .li_copyblend import LICopyBlend


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        copyblend_prob=0.0,
        copyblend_epochs=[0, 0],
        copyblend_type='blend',
        conflict_with_mixup=False,
        area_threshold=100,
        num_objects=3,
        with_expand=False,
        expand_ratios=[0.1, 0.25],
        random_num_objects=False,
        data_vis=False,
        vis_save='./vis_dataset/',
        li_copyblend=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.mixup_prob, self.mixup_epochs = mixup_prob, mixup_epochs

        self.copyblend_prob, self.copyblend_epochs, self.copyblend_type = copyblend_prob, copyblend_epochs, copyblend_type
        self.area_threshold, self.num_objects = area_threshold, num_objects
        self.data_vis, self.vis_save = data_vis, vis_save
        self.with_expand, self.expand_ratios, self.random_num_objects = with_expand, expand_ratios, random_num_objects
        self.conflict_with_mixup = conflict_with_mixup  # 是否冲突
        self.li_copyblend = LICopyBlend(**(li_copyblend or {}))

        if self.mixup_prob > 0 or self.copyblend_prob > 0:
            if self.data_vis and os.path.isdir(self.vis_save):
                raise FileExistsError('Existing visualization files are not deleted; choose a fresh vis_save')
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None

            if self.mixup_prob > 0:
                print("     ### Using MixUp with Prob@{} in {} epochs ### ".format(mixup_prob, mixup_epochs))
            if self.copyblend_prob > 0:
                print("     ### Using CopyBlend-{} with Prob@{} in {} epochs ### ".format(copyblend_type, copyblend_prob, copyblend_epochs))
                print(f'     ### CopyBlend -- area threshold@{area_threshold} and num of object@{num_objects} ###     ')
                if self.with_expand:
                    print(f'     ### CopyBlend -- expand@{expand_ratios} ###     ')
                if self.random_num_objects:
                    print(f'     ### CopyBlend -- random num of objects@{[1, self.num_objects]} ###     ')

        if stop_epoch is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))
        self.print_info_flag = True
        self.print_copyblend_flag = True
        # self.interpolation = interpolation

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        # Log when Mixup is permanently disabled
        if self.epoch == self.mixup_epochs[-1] and self.print_info_flag:
            print(f"     ### Attention --- Mixup is closed after epoch@ {self.epoch} ###")
            self.print_info_flag = False

        MixUp_flag, CopyBlend_flag = False, False
        copyblend_stats = Counter()
        beta = round(random.uniform(0.45, 0.55), 6)
        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)
            MixUp_flag = True

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = []

            for i in range(len(targets)):
                # Concatenation already creates every output instance tensor;
                # no copied target is consumed when building this dictionary.
                updated_target = concatenate_instances([targets[i], shifted_targets[i]])

                # Add mixup ratio to targets
                updated_target['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), 
                    dtype=torch.float32
                    )
                updated_targets.append(updated_target)
            targets = updated_targets

        elif (self.copyblend_epochs[0] <= self.epoch < self.copyblend_epochs[-1] and random.random() < self.copyblend_prob):
            if self.epoch == self.copyblend_epochs[-1] and self.print_copyblend_flag:
                print(f"     ### Attention --- CopyBlend closed after epoch@ {self.epoch} ###")
                self.print_copyblend_flag = False

            CopyBlend_flag = True
            if self.li_copyblend.enabled:
                return self.li_copyblend(images, targets, alpha=(1.0-beta) if self.copyblend_type == 'blend' else 1.0,
                    area_threshold=self.area_threshold, num_objects=self.num_objects,
                    random_num_objects=self.random_num_objects, with_expand=self.with_expand,
                    expand_ratios=self.expand_ratios)
            objects_pool = defaultdict(list)
            img_height, img_width = images[0].shape[-2:]

            # get all valid objects in batch
            for i in range(len(images)):
                source_boxes = targets[i]['boxes']
                source_labels = targets[i]['labels']
                source_areas = targets[i]['area']
                
                # filter valid objects
                valid_objects = [idx for idx in range(len(source_boxes)) if source_areas[idx] >= self.area_threshold]
                for idx in valid_objects:
                    objects_pool['boxes'].append(source_boxes[idx])
                    objects_pool['labels'].append(source_labels[idx])
                    objects_pool['areas'].append(source_areas[idx])
                    objects_pool['image_idx'].append(i)
                    objects_pool['instance_idx'].append(idx)
                    objects_pool['image_height'].append(img_height)
                    objects_pool['image_width'].append(img_width)
            
            # check if objects_pool is empty
            if len(objects_pool['boxes']) == 0:
                return self.li_copyblend.record_stats(images, targets, copyblend_stats)
            
            # convert list to tensor for convenient operation
            for key in ['boxes', 'labels', 'areas']:
                objects_pool[key] = torch.stack(objects_pool[key]) if objects_pool[key] else torch.tensor([])
                
            # apply CopyBlend
            batch_size = len(images)
            updated_images = images.clone()
            updated_targets = deepcopy(targets)

            for i in range(batch_size):
                # randomly decide the number of objects to blend
                if self.random_num_objects:
                    num_objects = random.randint(1, min(self.num_objects, len(objects_pool['boxes'])))
                else:
                    num_objects = min(self.num_objects, len(objects_pool['boxes']))
                
                # randomly select objects to blend
                selected_indices = random.sample(range(len(objects_pool['boxes'])), num_objects)
                
                blend_boxes = []
                blend_labels = []
                blend_areas = []
                blend_mixup_ratios = []
                blend_records = []

                for idx in selected_indices:
                    copyblend_stats['attempted_pastes'] += 1
                    # get source object information
                    box = objects_pool['boxes'][idx]
                    label = objects_pool['labels'][idx]
                    area = objects_pool['areas'][idx]
                    source_idx = objects_pool['image_idx'][idx]
                    source_instance = objects_pool['instance_idx'][idx]
                    source_height = objects_pool['image_height'][idx]
                    source_width = objects_pool['image_width'][idx]
                    
                    # calculate source object size and position
                    cx, cy, w, h = box
                    x1_src, y1_src = int((cx - w / 2) * source_width), int((cy - h / 2) * source_height)
                    x2_src, y2_src = int((cx + w / 2) * source_width), int((cy + h / 2) * source_height)

                    # check if source object is out of bound
                    x1_src, y1_src = max(x1_src, 0), max(y1_src, 0)
                    x2_src, y2_src = min(x2_src, img_width), min(y2_src, img_height)
                    new_w_px, new_h_px = x2_src - x1_src, y2_src - y1_src
                    # check if source object is valid
                    if new_w_px <= 0 or new_h_px <= 0:
                        continue

                    # randomly determine blend position
                    x1 = random.randint(0, img_width - new_w_px) if new_w_px < img_width else 0
                    y1 = random.randint(0, img_height - new_h_px) if new_h_px < img_height else 0
                    # after the above limit, [x2, y2] will not be out of bound, so no need to check
                    x2, y2 = x1 + new_w_px, y1 + new_h_px
                    
                    # calculate new normalized coordinates
                    new_cx, new_cy = (x1 + new_w_px / 2) / img_width, (y1 + new_h_px / 2) / img_height
                    new_w, new_h = new_w_px / img_width, new_h_px / img_height

                    # add to blend list - use original unexpanded box
                    blend_boxes.append(torch.tensor([new_cx, new_cy, new_w, new_h]))
                    blend_labels.append(label)
                    blend_areas.append(area)
                    # mixup ratio
                    blend_mixup_ratios.append(1.0 - beta)

                    # handle expanded area
                    if self.with_expand:
                        alpha = round(random.uniform(self.expand_ratios[0], self.expand_ratios[1]), 6)
                        expand_w, expand_h = int(new_w_px * alpha), int(new_h_px * alpha)
                        # check if out of bound: get the best offset in GT image
                        x1_expand, y1_expand = x1_src - max(x1_src - expand_w, 0), y1_src - max(y1_src - expand_h, 0)
                        x2_expand, y2_expand = min(x2_src + expand_w, img_width) - x2_src, min(y2_src + expand_h, img_height) - y2_src
                        # check if out of bound: whether the expanded area is out of bound in blend image
                        new_x1_expand, new_y1_expand = x1 - max(x1 - x1_expand, 0), y1 - max(y1 - y1_expand, 0)
                        new_x2_expand, new_y2_expand = min(x2 + x2_expand, img_width) - x2, min(y2 + y2_expand, img_height) - y2
                        # update
                        x1_src, y1_src, x2_src, y2_src = x1_src - new_x1_expand, y1_src - new_y1_expand, x2_src + new_x2_expand, y2_src + new_y2_expand
                        x1, y1, x2, y2 = x1 - new_x1_expand, y1 - new_y1_expand, x2 + new_x2_expand, y2 + new_y2_expand

                    # blend original area first
                    copy_patch_orig = images[source_idx, :, y1_src:y2_src, x1_src:x2_src]
                    if self.copyblend_type == 'blend':
                        blended_patch = updated_images[i, :, y1:y2, x1:x2] * beta + copy_patch_orig * (1 - beta)
                        updated_images[i, :, y1:y2, x1:x2] = blended_patch
                    else:
                        updated_images[i, :, y1:y2, x1:x2] = copy_patch_orig

                    # Propagate optional metadata without changing native RGB, box,
                    # class, sampling, expansion, or opacity behavior.
                    if self.copyblend_type != 'blend' and 'masks' in updated_targets[i]:
                        updated_targets[i]['masks'][:, y1:y2, x1:x2] = 0
                        for record in blend_records:
                            record['masks'][:, y1:y2, x1:x2] = 0
                    record = translated_record(targets[source_idx], source_instance,
                        (x1_src, y1_src, x2_src, y2_src), (x1, y1, x2, y2), (img_height, img_width))
                    if 'blend_opacity' in record:
                        record['blend_opacity'].fill_((1-beta) if self.copyblend_type == 'blend' else 1.0)
                    blend_records.append(record)
                    copyblend_stats['accepted_pastes'] += 1
                    
                # add blended objects to targets
                if len(blend_boxes) > 0:
                    blend_boxes = torch.stack(blend_boxes)
                    blend_labels = torch.stack(blend_labels)
                    blend_areas = torch.stack(blend_areas)
                    
                    # add mixup ratio
                    updated_targets[i]['mixup'] = torch.tensor(
                        [1.0] * len(updated_targets[i]['boxes']) + blend_mixup_ratios, 
                        dtype=torch.float32
                    )
                    # update targets
                    updated_targets[i]['boxes'] = torch.cat([updated_targets[i]['boxes'], blend_boxes])
                    updated_targets[i]['labels'] = torch.cat([updated_targets[i]['labels'], blend_labels])
                    updated_targets[i]['area'] = torch.cat([updated_targets[i]['area'], blend_areas])
                    append_records(updated_targets[i], blend_records)
                    if 'image_illumination_id' in updated_targets[i]:
                        updated_targets[i]['image_illumination_id'].fill_(-1)
                    if 'masks' in updated_targets[i]:
                        assert_visible_alignment(updated_targets[i], img_height, img_width)

            images, targets = updated_images, updated_targets

            if self.data_vis and CopyBlend_flag:
                for i in range(len(updated_targets)):
                    image_tensor = images[i]
                    if image_tensor.min() < 0:  # use normalization
                        image_tensor = image_tensor * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) \
                            + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    image_tensor_uint8 = (image_tensor * 255).type(torch.uint8)
                    image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                    pilImage = Image.fromarray(image_numpy)
                    draw = ImageDraw.Draw(pilImage)
                    print('mix_vis:', i, 'boxes.len=', len(updated_targets[i]['boxes']))
                    for box in updated_targets[i]['boxes']:
                        draw.rectangle([int(box[0]*640 - (box[2]*640)/2), int(box[1]*640 - (box[3]*640)/2), 
                                        int(box[0]*640 + (box[2]*640)/2), int(box[1]*640 + (box[3]*640)/2)], outline=(255,255,0))
                    pilImage.save(self.vis_save + str(i) + "_"+ str(len(updated_targets[i]['boxes'])) +'_out.jpg')

        return self.li_copyblend.record_stats(images, targets, copyblend_stats)

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # Mixup
        images, targets = self.apply_mixup(images, targets)

        alignment_checked = False
        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    resize_masks(tg, sz)
                # resize_masks validates every target after its final mutation.
                # No geometry changes before return, so do not scan them again.
                alignment_checked = True

        if not alignment_checked:
            for tg in targets:
                assert_visible_alignment(tg, *images.shape[-2:])

        return images, targets
