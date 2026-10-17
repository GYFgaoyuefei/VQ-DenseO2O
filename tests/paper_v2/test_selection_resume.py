"""Tiny CPU state machines verify selection/resume without detector training."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from engine.optim.ema import ModelEMA
from engine.solver.det_solver import DetSolver
from engine.solver.paper_v2_tracking import is_better, read_selection


class TinyLoader:
    def __init__(self):
        self.collate_fn = SimpleNamespace(stop_epoch=2, ema_restart_decay=.9)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __len__(self):
        return 1


class TinySolver(DetSolver):
    def __init__(self, directory, epochs):
        self.cfg = SimpleNamespace(yaml_cfg={'paper_v2': {'selection': 'ema_mixed_validation_ap'}},
                                   lrsheduler=None, epoches=epochs, clip_max_norm=0,
                                   print_freq=1, checkpoint_freq=1)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.ema = ModelEMA(self.model, decay=.95, warmups=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.)
        self.lr_warmup_scheduler = None
        self.criterion = torch.nn.Identity()
        self.postprocessor = torch.nn.Identity()
        self.scaler = None
        self.train_dataloader = TinyLoader()
        self.val_dataloader = []
        self.evaluator = None
        self.writer = None
        self.device = torch.device('cpu')
        self.last_epoch = -1
        self.output_dir = Path(directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        pass


def simulate_fit(solver, aps, crash_epoch=None):
    visited = []
    observed_decays = []

    def synthetic_step(self_scheduler, scheduler, model, criterion, loader, optimizer, device, epoch, **kwargs):
        if crash_epoch == epoch:
            raise RuntimeError('synthetic interrupted process')
        visited.append(epoch)
        observed_decays.append(kwargs['ema'].decay)
        with torch.no_grad():
            model.weight.fill_(epoch + 1)
            kwargs['ema'].module.weight.fill_(epoch + 10)
        # Mark a real optimizer step in this tiny fixture without training data.
        optimizer.step()
        return {'loss': 0.}

    def synthetic_validation(*args, **kwargs):
        epoch = visited[-1]
        return {'coco_eval_bbox': [aps[epoch], .5, .4, .3, .2, .1]}, None

    with mock.patch('engine.solver.det_solver.stats', return_value=(1, {'synthetic': True})), \
            mock.patch('engine.solver.det_solver.train_one_epoch', side_effect=synthetic_step), \
            mock.patch('engine.solver.det_solver.evaluate', side_effect=synthetic_validation):
        solver.fit()
    return visited, observed_decays


class EMAStageTests(unittest.TestCase):
    def test_dynamic_decay_changes_actual_averaging(self):
        source = torch.nn.Linear(1, 1, bias=False)
        source.weight.data.zero_()
        ema = ModelEMA(source, decay=.9, warmups=0)
        source.weight.data.fill_(1.)
        ema.set_decay(.5)
        ema.update(source)
        self.assertEqual(ema.module.weight.item(), .5)

    def test_legacy_default_and_checkpoint_shape_unchanged(self):
        source = torch.nn.Linear(1, 1, bias=False)
        source.weight.data.zero_()
        ema = ModelEMA(source, decay=.9, warmups=0)
        source.weight.data.fill_(1.)
        ema.decay = .5
        ema.update(source)
        self.assertAlmostEqual(ema.module.weight.item(), .1)
        self.assertEqual(set(ema.state_dict()), {'module', 'updates'})

    def test_effective_decay_and_warmup_resume_exact(self):
        source = torch.nn.Linear(1, 1, bias=False)
        source.weight.data.zero_()
        original = ModelEMA(source, decay=.9, warmups=4, start=1)
        original.set_decay(.7)
        original.update(source)
        source.weight.data.fill_(1.)
        original.update(source)
        restored = ModelEMA(source, decay=.99, warmups=100, start=0)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        original.update(source)
        restored.update(source)
        self.assertEqual(restored.decay, .7)
        self.assertTrue(torch.equal(original.module.weight, restored.module.weight))
        self.assertEqual(original.before_start, restored.before_start)
        self.assertEqual(original.updates, restored.updates)


class SelectionResumeTests(unittest.TestCase):
    def test_exact_tie_retains_earliest_epoch(self):
        self.assertFalse(is_better(.4, {'ap': .4, 'epoch': 3}))
        self.assertFalse(is_better(float('nan'), {'ap': .4, 'epoch': 3}))
        self.assertTrue(is_better(.400001, {'ap': .4, 'epoch': 3}))

    def test_global_best_before_rollback_last_after_rollback(self):
        with TemporaryDirectory() as directory:
            solver = TinySolver(directory, 4)
            visited, decays = simulate_fit(solver, [.4, .5, .6, .55])
            self.assertEqual(visited, [0, 1, 2, 3])
            self.assertEqual(decays, [.95, .95, .9, .9])
            best = torch.load(Path(directory) / 'best.pth', weights_only=False)
            last = torch.load(Path(directory) / 'last.pth', weights_only=False)
            self.assertEqual(best['last_epoch'], 2)
            self.assertEqual(best['ema']['module']['weight'].item(), 12.)
            self.assertEqual(last['last_epoch'], 3)
            self.assertEqual(last['ema']['module']['weight'].item(), 11.)
            self.assertAlmostEqual(last['ema']['decay'], .8999)
            self.assertEqual(last['paper_v2_stage_state']['best_stat'], {'epoch': -1})
            self.assertEqual(read_selection(directory)['epoch'], 2)

    def test_interrupted_resume_keeps_stage_decisions_and_effective_decay(self):
        aps = [.4, .5, .45, .46, .44]
        with TemporaryDirectory() as root:
            full = TinySolver(Path(root) / 'full', 5)
            simulate_fit(full, aps)
            partial = TinySolver(Path(root) / 'resume', 5)
            with self.assertRaisesRegex(RuntimeError, 'synthetic interrupted'):
                simulate_fit(partial, aps, crash_epoch=3)
            checkpoint = torch.load(partial.output_dir / 'last.pth', weights_only=False)
            resumed = TinySolver(partial.output_dir, 5)
            resumed.load_state_dict(checkpoint)
            visited, decays = simulate_fit(resumed, aps)
            self.assertEqual(visited, [3, 4])
            self.assertAlmostEqual(decays[0], .8999)
            actual = torch.load(resumed.output_dir / 'last.pth', weights_only=False)
            expected = torch.load(full.output_dir / 'last.pth', weights_only=False)
            self.assertEqual(actual['paper_v2_stage_state'], expected['paper_v2_stage_state'])
            self.assertEqual(actual['ema']['decay'], expected['ema']['decay'])
            self.assertTrue(torch.equal(actual['ema']['module']['weight'], expected['ema']['module']['weight']))
            self.assertTrue(torch.equal(actual['model']['weight'], expected['model']['weight']))
            self.assertEqual(read_selection(resumed.output_dir), read_selection(full.output_dir))

    def test_stage_reload_preserves_current_policy_and_requested_decay(self):
        with TemporaryDirectory() as directory:
            solver = TinySolver(directory, 3)
            simulate_fit(solver, [.4, .5, .45])
            expected = copy.deepcopy(solver.paper_v2_stage_state)
            solver._restart_paper_stage_one(.85)
            self.assertEqual(solver.paper_v2_stage_state, expected)
            self.assertEqual(solver.ema.decay, .85)
            self.assertTrue(solver.ema.dynamic_decay)


if __name__ == '__main__':
    unittest.main()
