"""Measurement contracts using a fake CUDA device, never publication timings."""

from contextlib import contextmanager
import csv
import json
import math
import sys
from types import SimpleNamespace
import weakref

import pytest

from tools.paper_v2 import efficiency_measure as measure_module
from tools.paper_v2.efficiency_protocol import HARDWARE, SCOPES, protocol


class FakeDevice:
    def __init__(self, name):
        self.name = name
        self.type = name.split(':')[0]

    def __str__(self):
        return self.name


class FakeCuda:
    def __init__(self, events):
        self.events = events
        self.current = 7
        self.name = HARDWARE
        self.available = True
        self.live_outputs = weakref.WeakSet()
        self.live_at_reset = []

    def current_device(self):
        return self.current

    def set_device(self, device):
        self.current = device

    def is_available(self):
        return self.available

    def get_device_name(self, device):
        return self.name

    def get_device_properties(self, device):
        return SimpleNamespace(total_memory=96 * 1024 ** 3, uuid='fake-gpu')

    def synchronize(self, device):
        self.events.append('sync')

    def reset_peak_memory_stats(self, device):
        self.events.append('reset')
        self.live_at_reset.append(len(self.live_outputs))

    def max_memory_allocated(self, device):
        return 1024


class FakeTorch:
    def __init__(self):
        self.events = []
        self.cuda = FakeCuda(self.events)
        self.precision = 'high'
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False, version=lambda: 9000))
        self.__version__ = 'fake-torch'
        self.version = SimpleNamespace(cuda='fake-cuda')
        self.float32 = 'float32'
        self.distributed_initialized = False
        self.distributed = SimpleNamespace(is_available=lambda: True,
                                          is_initialized=lambda: self.distributed_initialized)
        self.nn = SimpleNamespace(DataParallel=type('DataParallel', (), {}),
                                  parallel=SimpleNamespace(DistributedDataParallel=type('DDP', (), {})))
        self.jit = SimpleNamespace(ScriptModule=type('ScriptModule', (), {}))
        self.inference_active = False
        self.autocast_enabled = True
        self.allocations = []

    device = FakeDevice

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, value):
        self.precision = value

    def get_num_threads(self):
        return 1

    @contextmanager
    def inference_mode(self):
        self.inference_active = True
        try:
            yield
        finally:
            self.inference_active = False

    @contextmanager
    def autocast(self, *, device_type, enabled):
        assert device_type == 'cuda' and enabled is False
        previous = self.autocast_enabled
        self.autocast_enabled = enabled
        try:
            yield
        finally:
            self.autocast_enabled = previous

    def zeros(self, shape, *, device, dtype):
        value = SimpleNamespace(shape=list(shape), device=device, dtype=dtype, fill=0)
        self.allocations.append(value)
        return value

    def tensor(self, values, *, device):
        value = SimpleNamespace(values=values, device=device)
        self.allocations.append(value)
        return value

    def assert_fixed_runtime(self):
        assert self.precision == 'highest'
        assert self.backends.cuda.matmul.allow_tf32 is False
        assert self.backends.cudnn.allow_tf32 is False
        assert self.backends.cudnn.benchmark is False
        assert self.inference_active and not self.autocast_enabled

    def assert_restored(self):
        assert self.precision == 'high'
        assert self.backends.cuda.matmul.allow_tf32 is True
        assert self.backends.cudnn.allow_tf32 is True
        assert self.backends.cudnn.benchmark is True
        assert self.cuda.current == 7
        assert not self.inference_active and self.autocast_enabled


class FakeOutput:
    pass


class FakeDetector:
    def __init__(self, torch):
        self.torch = torch
        self.deployed = False
        self.calls = 0
        self.fail_at = None

    def to(self, device):
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def deploy(self):
        self.deployed = True
        return self

    def parameters(self):
        # Include a frozen parameter, and remove train-time branches at deploy.
        return [SimpleNamespace(numel=lambda: 20, requires_grad=False),
                SimpleNamespace(numel=lambda: 10 if self.deployed else 30, requires_grad=True)]

    def __call__(self, x):
        self.torch.assert_fixed_runtime()
        assert self.deployed
        assert x.shape == [1, 3, 640, 640] and x.dtype == 'float32' and x.fill == 0
        assert x.device.type == 'cuda'
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated detector failure')
        self.torch.events.append('detector')
        result = FakeOutput()
        self.torch.cuda.live_outputs.add(result)
        return result


class FakeProcessor(FakeDetector):
    num_top_queries = 300

    def deploy(self):
        self.deploy_mode = True
        return self

    def __call__(self, outputs, sizes):
        self.torch.assert_fixed_runtime()
        assert self.deploy_mode
        assert isinstance(outputs, FakeOutput)
        assert sizes.values == [[640, 640]] and sizes.device.type == 'cuda'
        self.calls += 1
        self.torch.events.append('processor')
        return ('labels', 'boxes', 'scores')


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeTorch()
    monkeypatch.setitem(sys.modules, 'torch', fake)
    for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(key, raising=False)
    detector, processor = FakeDetector(fake), FakeProcessor(fake)
    clocks = iter(range(20000))
    monkeypatch.setattr(measure_module.time, 'perf_counter', lambda: next(clocks) * .001)

    def profiler(model, x):
        fake.assert_fixed_runtime()
        assert model is detector and model.deployed
        assert detector.calls == 6005 and processor.calls == 6000
        assert len(fake.cuda.live_outputs) == 0
        fake.events.append('profiler')
        return {'tool': 'calflops', 'tool_version': '0.3.2', 'status': 'PARTIAL',
                'coverage_status': 'PARTIAL', 'gflops': .5,
                'coverage_notes': ['CPU fake profiler fixture; no actual FLOPs measurement.'],
                'count_scope': 'deploy detector forward only; postprocessor, criterion and training augmentation excluded',
                'raw_flops': .5e9, 'raw_macs': .25e9, 'raw_params': 30.,
                'oracle': {'operation': '[4,8] @ [8,12]', 'expected_macs': 384,
                           'expected_flops': 768, 'raw_macs': 384., 'raw_flops': 768.},
                'operator_coverage': {
                    'status': 'PARTIAL', 'exhaustive': False,
                    'method': 'CPU fake profiler fixture, not actual operator evidence',
                    'unsupported_operator_enumeration': 'UNAVAILABLE in simulated profiler',
                    'interpretation': 'Unit-test evidence only; no publication measurement.',
                }}

    monkeypatch.setitem(sys.modules, 'tools.paper_v2.efficiency_flops',
                        SimpleNamespace(profile_deploy_flops=profiler))
    return fake, detector, processor


def test_each_iteration_has_its_own_sync_wall_clock_and_exact_fixed_count(monkeypatch):
    fake = FakeTorch()
    calls = []
    clock_calls = []
    durations = iter(value for i in range(500) for value in (i * 2, i * 2 + (i + 1) / 1000))

    def clock():
        fake.events.append('clock')
        clock_calls.append(1)
        return next(durations)

    def operation():
        fake.events.append('operation')
        calls.append(1)

    monkeypatch.setattr(measure_module.time, 'perf_counter', clock)
    result = measure_module._time_scope(operation, fake, FakeDevice('cuda:0'))
    assert len(calls) == 600 and len(clock_calls) == 1000
    assert fake.events[:102] == ['operation'] * 100 + ['sync', 'reset']
    assert fake.events[102:] == ['sync', 'clock', 'operation', 'sync', 'clock'] * 500
    assert result['warmup_iterations'] == 100 and result['timed_iterations'] == 500
    assert result['samples_ms'] == pytest.approx(list(range(1, 501)))
    assert result['latency_ms'] == pytest.approx(250.5)
    assert result['p50_ms'] == pytest.approx(250.5)
    assert result['p95_ms'] == pytest.approx(475)
    assert result['fps'] == pytest.approx(1000 / 250.5)
    assert result['peak_gpu_memory_bytes'] == 1024


def test_five_independent_repeats_all_scopes_and_profiler_after_timing(runtime):
    fake, detector, processor = runtime
    report = measure_module.measure(detector, processor, 'cuda:0')
    assert report['full_params_m'] == 50 / 1e6
    assert report['deploy_params_m'] == 30 / 1e6
    assert report['hardware'] == HARDWARE and report['precision'] == 'FP32'
    assert report['backend'] == 'PyTorch eager eval/deploy'
    assert report['runtime']['float32_matmul_precision'] == 'highest'
    assert report['runtime']['torch_version'] == 'fake-torch'
    assert report['runtime']['cuda_version'] == 'fake-cuda'
    assert report['runtime']['cudnn_version'] == 9000
    assert isinstance(report['runtime']['torchvision_version'], str)
    assert report['runtime']['cpu_threads'] == 1
    assert detector.calls == 6005 and processor.calls == 6000
    assert len(fake.allocations) == 2
    assert fake.events[-1] == 'profiler'
    assert fake.events.count('sync') == 5 * 3 * 1001 + 1
    assert fake.cuda.live_at_reset == [0, 0, 1] * 5
    assert len(report['repeats']) == 5
    for index, repeat in enumerate(report['repeats'], start=1):
        assert repeat['repeat'] == index
        assert tuple(repeat['scopes']) == SCOPES
        for result in repeat['scopes'].values():
            assert result['latency_ms'] == pytest.approx(1.)
            assert result['fps'] == pytest.approx(1000.)
            assert len(result['samples_ms']) == 500
        # All clocks measure 1 ms: pipeline must be its own 1 ms measurement,
        # never the sum of separately measured detector + processor latencies.
        assert repeat['scopes'][SCOPES[0]]['latency_ms'] == pytest.approx(1.)
    json.dumps(report, allow_nan=False)
    fake.assert_restored()


def test_measure_result_crosses_real_summary_validation_and_csv_boundary(runtime, monkeypatch, tmp_path):
    """Only dependencies are simulated; measure's result schema is not recreated."""
    from tools.paper_v2.efficiency_summary import summarize, validate_record

    fake, detector, processor = runtime
    fake.__version__ = '2.8.0'
    fake.version.cuda = '12.8'
    monkeypatch.setattr(measure_module.importlib.metadata, 'version', lambda name: '0.23.0')
    measured = measure_module.measure(detector, processor, 'cuda:0')
    record = {
        'schema': 'paper_v2_efficiency_record_v1', 'status': 'MEASURED',
        'protocol': protocol(), 'model': 'baseline_native', 'seed': 42,
        'epochs': 120, 'selected_epoch_zero_based': 76,
        'checkpoint': {'path': 'fake-unit-test/best.pth', 'size_bytes': 1,
                       'mtime_ns': 1, 'model_state': 'ema'},
        **measured,
    }
    assert validate_record(record) is record
    path = tmp_path / 'raw/baseline_native/seed_42/efficiency.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record, allow_nan=False), encoding='utf-8')
    report = summarize(tmp_path)
    assert report['status'] == 'INCOMPLETE' and report['measured_model_seed_count'] == 1
    with (tmp_path / 'efficiency_by_seed.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]['Model'] == 'baseline_native' and rows[0]['Seed'] == '42'
    assert float(rows[0]['Latency(ms)']) == pytest.approx(1.)
    assert float(rows[0]['FPS']) == pytest.approx(1000.)
    assert float(rows[0]['Full Params(M)']) == 50 / 1e6
    assert float(rows[0]['Deploy Params(M)']) == 30 / 1e6
    assert rows[0]['GFLOPs status'] == 'PARTIAL'
    for name in ('efficiency_summary.csv', 'efficiency_protocol.json', 'paper_efficiency_table.csv'):
        assert (tmp_path / name).is_file()
    fake.assert_restored()


@pytest.mark.parametrize('device,hardware,available', [
    ('cpu', HARDWARE, True),
    ('cuda:0', 'NVIDIA RTX 4090', True),
    ('cuda:0', HARDWARE, False),
])
def test_rejects_nonstandard_hardware_before_any_measurement(runtime, device, hardware, available):
    fake, detector, processor = runtime
    fake.cuda.name, fake.cuda.available = hardware, available
    with pytest.raises(RuntimeError, match='requires'):
        measure_module.measure(detector, processor, device)
    assert not fake.allocations and detector.calls == 0
    fake.assert_restored()


@pytest.mark.parametrize('key,value', [('WORLD_SIZE', '2'), ('RANK', '0'), ('LOCAL_RANK', '0')])
def test_rejects_distributed_launcher(runtime, monkeypatch, key, value):
    fake, detector, processor = runtime
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match='single-process'):
        measure_module.measure(detector, processor, 'cuda:0')
    assert not fake.allocations


def test_rejects_initialized_process_group(runtime):
    fake, detector, processor = runtime
    fake.distributed_initialized = True
    with pytest.raises(RuntimeError, match='distributed process'):
        measure_module.measure(detector, processor, 'cuda:0')
    assert detector.calls == 0


def test_rejects_compiled_detector(runtime):
    fake, detector, processor = runtime
    detector._orig_mod = object()
    with pytest.raises(RuntimeError, match='eager execution'):
        measure_module.measure(detector, processor, 'cuda:0')
    assert detector.calls == 0


def test_rejects_changed_postprocessor_topk(runtime):
    fake, detector, processor = runtime
    processor.num_top_queries = 100
    with pytest.raises(ValueError, match='top-k=300'):
        measure_module.measure(detector, processor, 'cuda:0')
    assert detector.calls == 0


@pytest.mark.parametrize('failure_at', [1, 101, 1201])
def test_global_flags_restored_on_warmup_timing_or_cached_output_failure(runtime, failure_at):
    fake, detector, processor = runtime
    detector.fail_at = failure_at
    with pytest.raises(RuntimeError, match='simulated detector failure'):
        measure_module.measure(detector, processor, 'cuda:0')
    assert 'profiler' not in fake.events
    fake.assert_restored()


def test_global_flags_restored_on_profiler_failure(runtime, monkeypatch):
    fake, detector, processor = runtime

    def fail_profiler(model, x):
        raise RuntimeError('simulated profiler failure')

    monkeypatch.setitem(sys.modules, 'tools.paper_v2.efficiency_flops',
                        SimpleNamespace(profile_deploy_flops=fail_profiler))
    with pytest.raises(RuntimeError, match='simulated profiler failure'):
        measure_module.measure(detector, processor, 'cuda:0')
    fake.assert_restored()


@pytest.mark.parametrize('duration', [0., -1., math.nan, math.inf])
def test_invalid_clock_samples_refuse_latency_or_fps(monkeypatch, duration):
    fake = FakeTorch()
    values = iter([0., duration] * 500)
    monkeypatch.setattr(measure_module.time, 'perf_counter', lambda: next(values))
    with pytest.raises(ValueError, match='Invalid per-iteration'):
        measure_module._time_scope(lambda: None, fake, FakeDevice('cuda:0'))
