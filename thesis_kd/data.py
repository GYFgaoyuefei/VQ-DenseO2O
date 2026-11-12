"""Seeded RGB-only loaders for isolated student training and evaluation."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from .factories import component


def seed_worker(worker_id):
    # Torch already seeded this worker with base_seed + worker_id.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _loader(cfg, key, seed, distributed=True):
    from engine.data import DataLoader
    native = deepcopy(cfg['native'])
    settings = native[key]
    ds = settings['dataset']
    if ds['type'] != 'CocoDetection' or ds.get('return_masks') or ds.get('visible_mask_sidecar'):
        raise ValueError('Student protocol requires RGB bbox-only CocoDetection')
    if ds.get('category_ids') != [1, 2, 3] or native['PostProcessor']['category_ids'] != [1, 2, 3]:
        raise ValueError('Student labels must map [0,1,2] to COCO [1,2,3]')
    distributed = distributed and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world = torch.distributed.get_world_size() if distributed else 1
    total_batch = settings['total_batch_size']
    if total_batch < world or total_batch % world:
        raise ValueError(f'global batch {total_batch} is not divisible by world_size {world}')
    generator = torch.Generator().manual_seed(int(seed) + 8191 * rank + int(key == 'val_dataloader'))
    native['kd_dataset'] = ds
    native['kd_collator'] = settings['collate_fn']
    dataset = component(native, 'kd_dataset')
    expected = {1: 'immature', 2: 'half', 3: 'ripe'}
    if dataset.category2name != expected:
        raise ValueError(f'Unexpected COCO categories: {dataset.category2name}, expected {expected}')
    collator = component(native, 'kd_collator')
    train = key == 'train_dataloader'
    if not train and settings['drop_last']:
        raise ValueError('Evaluation must include the final incomplete batch')
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=train, seed=seed) if distributed else None
    loader = DataLoader(dataset, batch_size=total_batch // world, shuffle=train and sampler is None,
                        sampler=sampler, collate_fn=collator, drop_last=settings['drop_last'],
                        num_workers=settings['num_workers'], persistent_workers=False,
                        pin_memory=settings.get('pin_memory', False), generator=generator,
                        worker_init_fn=seed_worker)
    return loader


def build_train_loader(cfg, seed):
    return _loader(cfg, 'train_dataloader', seed)


def build_val_loader(cfg, seed, *, distributed=True):
    return _loader(cfg, 'val_dataloader', seed, distributed)


def build_frozen_test_loader(cfg, seed, *, confirm_frozen):
    if confirm_frozen is not True:
        raise ValueError('Test requires explicit --confirm-frozen and fixed selection')
    isolated = deepcopy(cfg)
    root = Path(isolated['resources']['TOMATO_DATA_ROOT'])
    isolated['native']['val_dataloader']['dataset'].update(
        img_folder=str(root / 'images' / 'test'),
        ann_file=str(root / 'annotations' / 'instances_test.json'))
    return build_val_loader(isolated, seed, distributed=False)
