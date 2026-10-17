"""Fixed CUDA FP32 deployment timing shared by every Paper v2 experiment.

This module never constructs a criterion, dataset, loader, or evaluator. Callers
load the selected detector before entering ``measure``. There are deliberately
no command-line or function overrides for the protocol's iteration counts.
"""

from contextlib import contextmanager
import importlib.metadata
import math
import os
import statistics
import time

from .efficiency_protocol import HARDWARE, SCOPES, protocol


def _require_single_process(torch, model):
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1 or
            any(os.environ.get(key) not in (None, '-1') for key in ('RANK', 'LOCAL_RANK'))):
        raise RuntimeError('Efficiency requires plain single-process Python, without torchrun')
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError('Efficiency must not run in an initialized distributed process')
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        raise RuntimeError('Efficiency requires the unwrapped eager detector')
    # torch.compile wraps its original module in OptimizedModule. Do not silently
    # profile a compiled backend while labelling it PyTorch eager.
    if hasattr(model, '_orig_mod') or isinstance(model, torch.jit.ScriptModule):
        raise RuntimeError('Efficiency requires eager execution, not a compiled/scripted detector')


@contextmanager
def _fixed_fp32_runtime(torch, device):
    previous = {
        'device': torch.cuda.current_device(),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    try:
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = previous['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = previous['cudnn_tf32']
        torch.backends.cudnn.benchmark = previous['cudnn_benchmark']
        torch.cuda.set_device(previous['device'])


def _time_scope(function, torch, device):
    """Measure a single scope, with a separate synchronized wall clock per call."""
    spec = protocol()
    for _ in range(spec['warmup_iterations']):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(spec['timed_iterations']):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.)
    if not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError('Invalid per-iteration wall-clock latency; refusing to report FPS')
    latency = statistics.mean(samples)
    return {
        'latency_ms': latency,
        'fps': 1000. / latency,
        'p50_ms': statistics.median(samples),
        'p95_ms': sorted(samples)[math.ceil(.95 * len(samples)) - 1],
        'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated(device)),
        'samples_ms': samples,
        'warmup_iterations': spec['warmup_iterations'],
        'timed_iterations': spec['timed_iterations'],
    }


def measure(model, processor, device_name):
    """Convert and measure one loaded detector/seed using the immutable protocol.

    ``model.deploy()`` mutates the caller's detector, as does
    ``processor.deploy()``. Global execution flags and the current CUDA device
    are restored even when deployment, timing, or profiling raises an exception.
    """
    import torch

    _require_single_process(torch, model)
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The fixed efficiency protocol requires an available CUDA device')
    hardware = torch.cuda.get_device_name(device)
    if hardware != HARDWARE:
        raise RuntimeError(f'Efficiency requires {HARDWARE}; found {hardware}')
    if getattr(processor, 'num_top_queries', None) != 300:
        raise ValueError('Efficiency requires the native tensor postprocessor with top-k=300')
    spec = protocol()
    with _fixed_fp32_runtime(torch, device):
        model = model.to(device).float().eval()
        full_params = sum(parameter.numel() for parameter in model.parameters())
        model = model.deploy()
        if model is None:
            raise TypeError('Detector.deploy() must return the deployed detector')
        model = model.to(device).float().eval()
        deploy_params = sum(parameter.numel() for parameter in model.parameters())
        processor = processor.to(device).float().eval().deploy()
        if processor is None or not getattr(processor, 'deploy_mode', False):
            raise TypeError('Postprocessor.deploy() must enable the native GPU tensor path')
        properties = torch.cuda.get_device_properties(device)
        try:
            torchvision_version = importlib.metadata.version('torchvision')
        except importlib.metadata.PackageNotFoundError:
            torchvision_version = 'UNAVAILABLE'
        runtime = {
            'torch_version': str(torch.__version__),
            'torchvision_version': torchvision_version,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'device': str(device),
            'gpu_memory_total_bytes': int(properties.total_memory),
            'gpu_uuid': str(getattr(properties, 'uuid', 'UNAVAILABLE')),
            'autocast': False,
            'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
            'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cpu_threads': torch.get_num_threads(),
        }
        # Allocations and any host-to-device setup are outside every timed scope.
        x = torch.zeros(spec['input_shape'], device=device, dtype=torch.float32)
        sizes = torch.tensor([[640, 640]], device=device)
        repeats = []
        for repeat in range(1, spec['repeats_per_seed'] + 1):
            scopes = {}
            scopes[SCOPES[0]] = _time_scope(lambda: processor(model(x), sizes), torch, device)
            scopes[SCOPES[1]] = _time_scope(lambda: model(x), torch, device)
            # Only this scope retains detector outputs. Release its cache before
            # the next repeat so it cannot inflate pipeline/forward memory.
            cached = model(x)
            try:
                scopes[SCOPES[2]] = _time_scope(lambda: processor(cached, sizes), torch, device)
            finally:
                del cached
            repeats.append({'repeat': repeat, 'scopes': scopes})
        torch.cuda.synchronize(device)
        # Import/run profilers only after *all* timing: profiler hooks and patches
        # cannot contaminate any latency or memory measurement.
        from .efficiency_flops import profile_deploy_flops
        flops = profile_deploy_flops(model, x)
    return {
        'full_params_m': full_params / 1e6,
        'deploy_params_m': deploy_params / 1e6,
        'hardware': hardware,
        'backend': spec['backend'],
        'precision': spec['precision'],
        'runtime': runtime,
        'repeats': repeats,
        'flops': flops,
    }
