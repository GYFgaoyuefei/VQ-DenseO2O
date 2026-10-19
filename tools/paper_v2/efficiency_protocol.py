"""One fixed, versioned efficiency protocol for every Paper v2 model/seed."""
import copy

HARDWARE = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
SEEDS = (0, 42, 3407)
MODELS = ('baseline_native', 'baseline_maskaware', 'vq_dense_o2o', 'li_copyblend', 'dc_sta',
          'vq_dense_o2o_li_copyblend', 'vq_dense_o2o_dc_sta', 'li_copyblend_dc_sta',
          'vq_dense_o2o_li_copyblend_dc_sta')
ALGORITHM_SUITE = 'beta_select_algorithms_v1'
LABELS = dict(zip(MODELS, ('Baseline', 'Baseline (mask-aware)', 'VQ-DenseO2O', 'LI-CopyBlend',
                         'DC-STA', 'VQ-DenseO2O + LI-CopyBlend', 'VQ-DenseO2O + DC-STA',
                         'LI-CopyBlend + DC-STA', 'VQ-DenseO2O + LI-CopyBlend + DC-STA')))
SCOPES = ('detector_plus_postprocessor', 'detector_forward', 'postprocessor')
COCO_NAMES = ('AP', 'AP50', 'AP75', 'AP-S', 'AP-M', 'AP-L',
              'AR1', 'AR10', 'AR100', 'AR-S', 'AR-M', 'AR-L')
AP_COLUMNS = tuple(f'Mixed {name} (%)' for name in COCO_NAMES) + (
    'Natural AP (%)', 'Artificial AP (%)', 'Low-light AP (%)', 'Sodium-light AP (%)')
PAPER_COLUMNS = ('Model', 'Epoch', 'Params(M)', 'GFLOPs', 'Latency(ms)', 'FPS', 'Hardware', *AP_COLUMNS)

_PROTOCOL = {
    'id': 'paper_v2_efficiency_fp32_20261017',
    'reference': '2026-10-17 standard_ablation_test PyTorch FP32 deployment protocol',
    'hardware': HARDWARE,
    'backend': 'PyTorch eager eval/deploy',
    'precision': 'FP32',
    'autocast': False,
    'tf32_matmul': False,
    'tf32_cudnn': False,
    'float32_matmul_precision': 'highest',
    'cudnn_benchmark': False,
    'batch_size': 1,
    'input_shape': [1, 3, 640, 640],
    'input': 'GPU-resident synthetic all-zero FP32 tensor',
    'warmup_iterations': 100,
    'timed_iterations': 500,
    'repeats_per_seed': 5,
    'seeds': list(SEEDS),
    'timing': 'Python time.perf_counter; CUDA synchronize before start and before stop of every iteration',
    'primary_scope': SCOPES[0],
    'independently_measured_scopes': list(SCOPES),
    'postprocessor': 'native PostProcessor.deploy() GPU tensor outputs, fixed top-k=300, no NMS/TTA',
    'postprocessor_only_input': 'cached detector outputs computed outside timing',
    'p50': 'median of 500 per-iteration samples per repeat',
    'p95': 'nearest-rank ceil(0.95*N) of 500 samples per repeat',
    'seed_latency': 'arithmetic mean of five repeat mean latencies',
    'summary_latency': 'arithmetic mean and sample standard deviation (ddof=1) of three seed latencies',
    'fps': '1000 / corresponding mean latency in milliseconds; never mean of per-iteration/repeat/seed FPS',
    'paper_params': 'Deploy Params(M), all model parameters after deploy conversion',
    'full_params': 'all detector parameters before deploy conversion, including frozen parameters',
    'gflops_tool': 'calflops',
    'gflops_scope': 'deploy detector forward only; postprocessor excluded',
    'gflops_convention': 'native calflops count without correction; contraction oracle requires FMA=2',
    'gflops_coverage': 'PARTIAL unless full operator coverage is proven; calflops does not enumerate all unsupported ops',
    'peak_gpu_memory': 'torch.cuda.max_memory_allocated after warmup/reset per measured scope; includes resident model/input; excludes profiling',
    'excluded': ['image IO', 'preprocessing', 'dataloader', 'H2D/D2H transfers',
                 'loss/criterion', 'VQ-DenseO2O training computation', 'LI-CopyBlend training augmentation',
                 'COCO evaluator'],
}


def protocol():
    """Return an independent snapshot; callers cannot mutate the official constants."""
    return copy.deepcopy(_PROTOCOL)


def validate_protocol(value):
    if value != _PROTOCOL:
        differences = [key for key in sorted(set(value) | set(_PROTOCOL))
                       if value.get(key) != _PROTOCOL.get(key)]
        raise ValueError(f'Nonstandard efficiency protocol: {differences}')
