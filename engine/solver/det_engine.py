"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..data.li_copyblend import STAT_NAMES, summarize_copyblend_stats
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    step_observer = kwargs.get('step_observer')

    cur_iters = epoch * len(data_loader)
    copyblend_totals = torch.zeros(len(STAT_NAMES) + 1, dtype=torch.float64, device=device)
    if hasattr(criterion, 'diagnostic_summary'):
        criterion.diagnostic_summary(reset=True)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step_observer is not None:
            step_observer.begin_step(i)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=torch.device(device).type, cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                if step_observer is not None:
                    raise RuntimeError('Train smoke produced nonfinite boxes; no checkpoint saved')
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0 or step_observer is not None:
                scaler.unscale_(optimizer)
            if step_observer is not None:
                step_observer.observe_gradients(model, optimizer, ema)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if step_observer is not None:
                step_observer.observe_gradients(model, optimizer, ema)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        if step_observer is not None:
            step_observer.end_step(samples, targets, loss_dict, optimizer)

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if targets and 'li_copyblend_stats' in targets[0]:
            copyblend_totals[:-1] += targets[0]['li_copyblend_stats'].detach().double()
            copyblend_totals[-1] += 1

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(copyblend_totals)
    values = copyblend_totals.cpu().tolist()
    if values[-1] > 0:
        diagnostics = summarize_copyblend_stats(values[:-1])
        result.update({f'copyblend_{key}': value for key, value in diagnostics.items()})
        result['copyblend_batches'] = values[-1]
        if diagnostics['li_enabled_image_count'] > 0:
            result.update({f'li_{key}': value for key, value in diagnostics.items()})
            result['li_batches'] = values[-1]
    if hasattr(criterion, 'diagnostic_summary'):
        diagnostics = criterion.diagnostic_summary(reset=True)
        # Epoch boundary only. Criterion exposes detached compact aggregates.
        for branch, values in diagnostics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    result[f'vq_dense_o2o_{branch}_{key}'] = float(value)
            else:
                result[f'vq_dense_o2o_{branch}'] = float(values)
    return result


@contextmanager
def fp32_evaluation(model, device, enabled):
    """Make Paper selection FP32 explicit, including callers inside AMP scopes."""
    if not enabled:
        yield
        return
    floating = [value for value in (*model.parameters(), *model.buffers()) if value.is_floating_point()]
    if any(value.dtype != torch.float32 for value in floating):
        raise ValueError('Paper v2 selection requires FP32 EMA parameters and buffers')
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, *, fp32=False):
    with fp32_evaluation(model, device, fp32):
        return _evaluate(model, criterion, postprocessor, data_loader, coco_evaluator, device)


def _evaluate(model, criterion, postprocessor, data_loader, coco_evaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
