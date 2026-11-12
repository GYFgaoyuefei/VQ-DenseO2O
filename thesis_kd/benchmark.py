"""One timer and profiler for native T0, M and S deploy copies."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid

import torch

from .evaluate import load_deployed_student, selected_weights, _postprocessor


PROTOCOL = 'kd_eager_fp32_640_sync_wall_v1'


def _verify_gpu(device):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    raw_uuid = getattr(torch.cuda.get_device_properties(device), 'uuid', None)
    try:
        identifier = (uuid.UUID(bytes=raw_uuid) if isinstance(raw_uuid, bytes)
                      else uuid.UUID(str(raw_uuid).lower().removeprefix('gpu-')))
        current = 'gpu-' + str(identifier)
    except (TypeError, ValueError):
        current = ''
    others = []
    for row in result.stdout.splitlines():
        if not row.strip():
            continue
        gpu_id, pid = [field.strip() for field in row.split(',')]
        # When the driver omits UUID, be conservative across visible devices.
        if (not current or current == gpu_id.lower()) and int(pid) != os.getpid():
            others.append(int(pid))
    if others:
        raise RuntimeError(f'Formal benchmark requires an idle GPU; other compute PIDs: {others}')


def benchmark_run(cfg, run_dir, device, *, formal=False, teacher=False):
    from tools.paper_v2.efficiency_measure import _fixed_fp32_runtime, _require_single_process
    from tools.paper_v2.efficiency_flops import profile_deploy_flops
    run_dir = Path(run_dir)
    name = 'benchmark_teacher' if teacher else 'benchmark'
    target = run_dir / (name + ('.json' if formal else '_smoke.json'))
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Latency protocol requires CUDA')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if teacher:
        from .teacher import load_detector
        model, identity, native = load_detector(cfg)
        processor = _postprocessor({'native': native}, device)
        binding = {'teacher': identity}
    else:
        ckpt, selection, info = selected_weights(cfg, run_dir)
        model = load_deployed_student(cfg, ckpt, device)
        processor = _postprocessor(cfg, device)
        binding = {'selection': selection}
    _require_single_process(torch, model)
    warmup, iterations, repeats = (100, 500, 5) if formal else (2, 5, 1)
    with _fixed_fp32_runtime(torch, device):
        model = model.to(device).float().eval().deploy()
        processor = processor.float().eval().deploy()
        if processor.num_top_queries != 300 or not processor.deploy_mode:
            raise ValueError('Expected native deploy tensor postprocessor, top-k 300')
        sample = torch.zeros(1, 3, 640, 640, device=device)
        sizes = torch.tensor([[640, 640]], device=device)
        if formal:
            _verify_gpu(device)
        timings = []
        for _ in range(repeats):
            for _ in range(warmup):
                processor(model(sample), sizes)
            elapsed = []
            for _ in range(iterations):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                processor(model(sample), sizes)
                torch.cuda.synchronize(device)
                elapsed.append((time.perf_counter() - start) * 1000)
            timings.append(elapsed)
            if formal:
                _verify_gpu(device)
        flops = profile_deploy_flops(model, sample) if formal else {'status': 'NOT RUN'}
    latency = statistics.mean(statistics.mean(series) for series in timings)
    record = dict(**binding, protocol=PROTOCOL, formal=formal, hardware=torch.cuda.get_device_name(device),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(), input_shape=[1, 3, 640, 640],
                  precision='FP32', tf32=False, autocast=False, device=str(device),
                  scope='detector_forward_plus_tensor_postprocessor', warmup=warmup,
                  iterations=iterations, repeats=repeats, timings_ms=timings,
                  params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                  flops=flops, latency_ms=latency, fps=1000 / latency)
    with target.open('x', encoding='utf8') as output:
        json.dump(record, output, ensure_ascii=False, allow_nan=False, indent=2)
    return record
