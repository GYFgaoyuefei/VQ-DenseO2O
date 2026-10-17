"""Bounded real-Train acceptance using the native optimizer/AMP/EMA train loop."""
from itertools import islice
import json
from pathlib import Path
import time

import torch

from engine.misc import dist_utils
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from engine.solver.det_engine import train_one_epoch


class BoundedLoader:
    """Keep the real epoch length for scheduling; consume only requested batches."""
    def __init__(self, loader, steps):
        self.loader, self.steps = loader, steps

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return islice(iter(self.loader), self.steps)


class TrainStepObserver:
    def __init__(self, device):
        self.device = torch.device(device)
        self.rows = []
        self.current = None

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def begin_step(self, index):
        self._sync()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        self.current = {'step': index, 'status': 'RUNNING'}
        self.rows.append(self.current)
        self.started = time.perf_counter()

    def observe_gradients(self, model, optimizer, ema):
        from engine.backbone.dc_sta import audit_dc_training
        gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
        self.current['gradient_tensors'] = len(gradients)
        self.current['gradients_finite'] = finite
        norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gradients])) if finite else None
        self.current['gradient_l2_before_clip'] = float(norm) if norm is not None and torch.isfinite(norm) else None
        if not finite or self.current['gradient_l2_before_clip'] is None:
            raise RuntimeError('Train smoke found missing or nonfinite gradients')
        if self.current['gradient_l2_before_clip'] == 0:
            raise RuntimeError('Train smoke found an entirely zero training gradient')
        self.current['dc_sta'] = audit_dc_training(model, optimizer, ema, check_gradients=True)
        if self.current['dc_sta']['status'] == 'FAIL':
            raise RuntimeError('Train smoke found invalid DC-STA optimizer, EMA or gradient coverage')

    def end_step(self, samples, targets, losses, optimizer):
        self._sync()
        self.current['iteration_ms'] = (time.perf_counter() - self.started) * 1000.
        self.current['peak_gpu_memory_bytes'] = (
            int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == 'cuda' else None)
        self.current['input_shape'] = list(samples.shape)
        self.current['gt_instances_per_image'] = [len(target['labels']) for target in targets]
        self.current['losses'] = {key: float(value.detach()) for key, value in losses.items()}
        self.current['loss'] = sum(self.current['losses'].values())
        if not all(torch.isfinite(torch.tensor(value)) for value in self.current['losses'].values()):
            # The failure report must itself remain valid strict JSON.
            self.current['losses'] = {key: value if torch.isfinite(torch.tensor(value)) else None
                                      for key, value in self.current['losses'].items()}
            self.current['loss'] = None
            raise RuntimeError('Train smoke found nonfinite loss')
        self.current['next_learning_rates'] = [group['lr'] for group in optimizer.param_groups]
        self.current['status'] = 'PASS'


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run_bounded_train(solver, *, steps, epoch):
    """Use the actual batch32 recipe; never evaluate or save model checkpoints."""
    cfg = solver.cfg
    if type(steps) is not int or not 1 <= steps <= 10 or not 0 <= epoch < 120:
        raise ValueError('Train smoke requires 1-10 steps and a recipe epoch from 0 to 119')
    if not cfg.yaml_cfg.get('paper_v2') or cfg.resume:
        raise ValueError('Train smoke requires the shared Paper v2 initialization, without resume')
    if not cfg.tuning or cfg.tuning != cfg.yaml_cfg['tomato']['detector_checkpoint']:
        raise ValueError('Train smoke requires the shared detector pretrained checkpoint')
    if not cfg.use_amp or not cfg.use_ema or cfg.epoches != 120:
        raise ValueError('Train smoke retains the shared AMP/EMA/120-epoch scheduler recipe')
    if cfg.yaml_cfg['train_dataloader'].get('total_batch_size') != 32:
        raise ValueError('Train smoke must retain the formal effective batch size of 32')
    rank = dist_utils.get_rank()
    report = {'status': 'RUNNING', 'real_train_data': True, 'rank': rank,
              'seed': cfg.seed, 'requested_steps': steps, 'recipe_epoch': epoch,
              'world_size': dist_utils.get_world_size(), 'effective_batch_size': 32,
              'formal_training': 'NOT RUN', 'validation': 'NOT RUN', 'test': 'NOT RUN',
              'pretrained': {'dinov3': cfg.yaml_cfg['DINOv3STAs']['weights_path'], 'detector': cfg.tuning},
              'timing_scope': 'synchronized wall clock including H2D, forward, loss, backward, gradient diagnostics, optimizer, EMA and scheduler; data fetching excluded'}
    path = Path(cfg.output_dir) / f'train_smoke_rank_{rank}.json'
    observer = None
    try:
        # Native setup loads tuning before constructing EMA. It creates neither
        # the validation loader nor its evaluator; only Train is accessed below.
        solver._setup()
        if solver.device.type != 'cuda':
            raise ValueError('Formal real-batch acceptance requires CUDA; CPU unit tests are separate')
        report['hardware'] = torch.cuda.get_device_name(solver.device)
        report['torch_version'] = str(torch.__version__)
        report['amp'] = cfg.use_amp
        solver.optimizer = cfg.optimizer
        # This initializes group initial_lr, as in BaseSolver.train().
        _ = cfg.lr_scheduler
        # LinearWarmup construction also changes the first update's LR. Preserve
        # native construction order even though FlatCosine controls later steps.
        _ = cfg.lr_warmup_scheduler
        solver.train_dataloader = dist_utils.warp_loader(cfg.train_dataloader, shuffle=cfg.train_dataloader.shuffle)
        loader = solver.train_dataloader
        if len(loader) < steps:
            raise ValueError('Train dataset has fewer complete batches than requested smoke steps')
        if loader.batch_size * dist_utils.get_world_size() != 32:
            raise ValueError('Observed distributed training batch differs from 32')
        loader.set_epoch(epoch)
        if dist_utils.is_dist_available_and_initialized():
            loader.sampler.set_epoch(epoch)
        solver.ema.set_decay(solver.ema.decay)
        from engine.backbone.dc_sta import audit_dc_training
        report['dc_sta_before_training'] = audit_dc_training(solver.model, solver.optimizer, solver.ema)
        audit = report['dc_sta_before_training']
        if audit['status'] == 'FAIL' or (audit['enabled'] and not audit['identity_initialized']):
            raise RuntimeError('DC-STA must start from identity with complete optimizer/EMA coverage')
        report['rank_batch_size'] = loader.batch_size
        scheduler = FlatCosineLRScheduler(solver.optimizer, cfg.lr_gamma, len(loader), cfg.epoches,
                                         cfg.warmup_iter, cfg.flat_epoch, cfg.no_aug_epoch)
        observer = TrainStepObserver(solver.device)
        report['metrics'] = train_one_epoch(True, scheduler, solver.model, solver.criterion,
            BoundedLoader(loader, steps), solver.optimizer, solver.device, epoch,
            max_norm=cfg.clip_max_norm, print_freq=1, ema=solver.ema,
            scaler=solver.scaler, step_observer=observer)
        report['steps'] = observer.rows
        if len(observer.rows) != steps or any(row['status'] != 'PASS' for row in observer.rows):
            raise RuntimeError('Incomplete bounded Train smoke')
        report['status'] = 'PASS'
        report['completed_steps'] = len(observer.rows)
        _write_report(path, report)
        print(f'Train smoke PASS: {path}; no validation, Test, or checkpoint was run')
        return report
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      steps=observer.rows if observer else [])
        if observer and observer.current and observer.current['status'] == 'RUNNING':
            observer.current['status'] = 'FAIL'
        _write_report(path, report)
        if dist_utils.is_dist_available_and_initialized():
            # A failed rank must not wait at the generic atexit barrier while
            # peers are blocked in a different training collective.
            torch.distributed.destroy_process_group()
        raise
