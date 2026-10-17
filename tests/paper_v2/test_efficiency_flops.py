"""CPU contracts for profiler scope, native convention and honest coverage."""

import sys
import types
import warnings
import json

import pytest
import torch
import torch.nn.functional as functional

from tools.paper_v2 import efficiency_flops as profiler


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3, 2))
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        return torch.tanh(torch.matmul(value, self.weight))


def fake_patch_functionals():
    # Inspected only: the test must not install these assignments.
    F.linear = None


def fake_patch_tensor_methods():
    torch.matmul = None


def install_fake(monkeypatch, calculate):
    module = types.ModuleType("calflops")
    module.calculate_flops = calculate
    module.__version__ = "test-1.3.0"
    ops = types.ModuleType("calflops.pytorch_ops")
    ops._patch_functionals = fake_patch_functionals
    ops._patch_tensor_methods = fake_patch_tensor_methods
    monkeypatch.setitem(sys.modules, "calflops", module)
    monkeypatch.setitem(sys.modules, "calflops.pytorch_ops", ops)
    monkeypatch.setattr(profiler.metadata, "version", lambda name: "test-1.3.0")


def test_native_counts_copy_detector_only_and_capture_coverage(monkeypatch):
    detector = TinyDetector().eval()
    sample = torch.zeros(1, 3)
    calls = []

    def calculate(**kwargs):
        model, args = kwargs["model"], kwargs["args"]
        calls.append(kwargs)
        assert not torch.is_grad_enabled()
        assert not model.training
        assert args[0].dtype == torch.float32
        model(*args)
        if isinstance(model, profiler._MatmulOracle):
            assert args[0].shape == (4, 8)
            assert model.weight.shape == (8, 12)
            return 768, 384, 96
        assert model is not detector
        assert args[0] is sample
        print("Warning: unsupported custom_kernel")
        print("not supported: special_op", file=sys.stderr)
        warnings.warn("uncounted fused operation", UserWarning)
        model.weight.zero_()
        model.train()
        return 1200, 600, 6

    install_fake(monkeypatch, calculate)
    result = profiler.profile_deploy_flops(detector, sample)
    assert len(calls) == 2
    assert all(call["forward_mode"] == "forward" for call in calls)
    assert all(call["include_backPropagation"] is False for call in calls)
    assert all(call["output_as_string"] is False for call in calls)
    assert detector.forward_calls == 0 and not detector.training
    assert torch.all(detector.weight == 1)
    assert result["tool"] == "calflops" and result["tool_version"] == "test-1.3.0"
    assert result["status"] == result["coverage_status"] == "PARTIAL"
    assert result["gflops"] == pytest.approx(1200 / 1e9)
    assert (result["raw_flops"], result["raw_macs"], result["raw_params"]) == (1200, 600, 6)
    assert "postprocessor" in result["count_scope"] and "excluded" in result["count_scope"]
    evidence = result["operator_coverage"]
    assert evidence["exhaustive"] is False
    assert evidence["patched_apis"] == ["F.linear", "torch.matmul"]
    assert "tanh" in evidence["potentially_uncovered_detector_apis"]
    assert "scaled_dot_product_attention" in evidence["candidate_api_names_absent_from_patch_source"]
    assert "custom_kernel" in result["profiler_stdout"]
    assert "special_op" in result["profiler_stderr"]
    assert any("uncounted fused operation" in line for line in result["profiler_warnings"])


def test_missing_calflops_leaves_null_counts_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "calflops", None)

    def missing_version(name):
        raise profiler.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(profiler.metadata, "version", missing_version)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert result["status"] == "UNAVAILABLE"
    assert result["tool_version"] is None
    assert result["gflops"] is result["raw_flops"] is result["raw_macs"] is None
    assert "NOT RUN" in " ".join(result["coverage_notes"])


def test_version_falls_back_to_module_metadata(monkeypatch):
    def calculate(**kwargs):
        return (768, 384, 96) if isinstance(kwargs["model"], profiler._MatmulOracle) else (100, 50, 6)

    install_fake(monkeypatch, calculate)

    def missing_version(name):
        raise profiler.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(profiler.metadata, "version", missing_version)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert result["tool_version"] == "test-1.3.0"
    assert result["status"] == "PARTIAL"


def test_convention_mismatch_never_profiles_detector(monkeypatch):
    calls = []

    def calculate(**kwargs):
        calls.append(kwargs["model"])
        return 384, 384, 96

    install_fake(monkeypatch, calculate)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert len(calls) == 1
    assert result["status"] == "CONVENTION_MISMATCH"
    assert result["gflops"] is result["raw_flops"] is result["raw_macs"] is None
    assert result["oracle"]["raw_flops"] == 384


@pytest.mark.parametrize("failure_scope", ["oracle", "detector"])
def test_failed_profiler_restores_torch_functional_and_tensor_patches(monkeypatch, failure_scope):
    original = (torch.add, functional.linear, torch.Tensor.matmul)

    def calculate(**kwargs):
        is_oracle = isinstance(kwargs["model"], profiler._MatmulOracle)
        if is_oracle and failure_scope == "detector":
            return 768, 384, 96
        torch.add = lambda *args, **kw: None
        functional.linear = lambda *args, **kw: None
        torch.Tensor.matmul = lambda *args, **kw: None
        raise RuntimeError("synthetic profiler failure")

    install_fake(monkeypatch, calculate)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert (torch.add, functional.linear, torch.Tensor.matmul) == original
    assert result["status"] == "FAILED"
    assert result["gflops"] is result["raw_flops"] is result["raw_macs"] is None
    assert "synthetic profiler failure" in result["error"]


@pytest.mark.parametrize("counts", [
    (float("nan"), 1, 1), (1, float("inf"), 1), (-1, 1, 1), (1, 1, -1),
    (0, 0, 6), (1, 1, 0),
])
def test_invalid_native_counts_are_never_published(monkeypatch, counts):
    def calculate(**kwargs):
        return (768, 384, 96) if isinstance(kwargs["model"], profiler._MatmulOracle) else counts

    install_fake(monkeypatch, calculate)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert result["status"] == "FAILED"
    assert result["gflops"] is result["raw_flops"] is result["raw_macs"] is result["raw_params"] is None


def test_coverage_source_failure_still_reports_partial_not_complete(monkeypatch):
    def calculate(**kwargs):
        return (768, 384, 96) if isinstance(kwargs["model"], profiler._MatmulOracle) else (100, 50, 6)

    install_fake(monkeypatch, calculate)
    monkeypatch.setitem(sys.modules, "calflops.pytorch_ops", None)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert result["status"] == "PARTIAL"
    assert result["coverage_status"] == "PARTIAL"
    assert result["operator_coverage"]["patch_source_status"] == "UNAVAILABLE"
    assert result["operator_coverage"]["exhaustive"] is False


def test_non_fp32_input_fails_before_profiler_executes(monkeypatch):
    def calculate(**kwargs):
        pytest.fail("profiler must not execute for a non-FP32 sample")

    install_fake(monkeypatch, calculate)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3, dtype=torch.float64))
    assert result["status"] == "FAILED"
    assert result["gflops"] is None
    assert "FP32" in result["error"]


@pytest.mark.parametrize("oracle_counts", [
    (float("nan"), 384, 96), (float("inf"), 384, 96),
    (768, float("nan"), 96), (768, float("inf"), 96),
    (-1, 384, 96), (768, -1, 96),
])
def test_invalid_oracle_counts_produce_json_safe_failure(monkeypatch, oracle_counts):
    install_fake(monkeypatch, lambda **kwargs: oracle_counts)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert result["status"] == "FAILED"
    assert result["gflops"] is result["raw_flops"] is result["raw_macs"] is result["raw_params"] is None
    assert "oracle" not in result
    assert "invalid oracle counts" in result["error"]
    json.dumps(result, allow_nan=False)


def test_calflops_import_runtime_failure_preserves_null_report(monkeypatch):
    real_import = profiler.importlib.import_module

    def import_module(name):
        if name == "calflops":
            raise RuntimeError("incompatible installed dependency")
        return real_import(name)

    monkeypatch.setattr(profiler.importlib, "import_module", import_module)
    result = profiler.profile_deploy_flops(TinyDetector(), torch.zeros(1, 3))
    assert result["status"] == "FAILED"
    assert result["gflops"] is None
    assert "incompatible installed dependency" in result["error"]


def test_paper_training_never_runs_optional_profiler_or_constructs_loader(monkeypatch):
    from engine.misc.profiler_utils import stats

    detector = TinyDetector()

    class Config:
        yaml_cfg = {'paper_v2': True}
        model = detector

        @property
        def train_dataloader(self):
            pytest.fail('Training parameter report must not construct a loader')

    class BrokenProfiler(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError('Installed calflops dependency is incompatible')

    monkeypatch.setitem(sys.modules, 'calflops', BrokenProfiler('calflops'))
    before = torch.get_rng_state().clone()
    params, report = stats(Config())
    assert params == 6 and detector.forward_calls == 0
    assert torch.equal(before, torch.get_rng_state())
    assert 'FLOPs NOT RUN during training' in ' '.join(report)
    assert 'unified Paper v2 efficiency benchmark' in ' '.join(report)
