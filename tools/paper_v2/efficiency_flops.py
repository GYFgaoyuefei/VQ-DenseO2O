"""Native calflops counts for a deployed detector, with explicit partial coverage.

Call this only after every latency/memory measurement.  Profiling receives a
private detector copy, never the tensor postprocessor, criterion, or augmenter.
No alternative profiler or analytic estimate is substituted when calflops fails.
"""

import ast
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib
from importlib import metadata
import inspect
import io
import math
import re
import textwrap
import warnings

import torch
import torch.nn.functional as functional


_COVERAGE_CANDIDATES = (
    "scaled_dot_product_attention", "_native_multi_head_attention",
    "grid_sample", "sigmoid", "tanh", "mean", "amax", "sqrt", "exp",
)
_SCOPE = "deploy detector forward only; postprocessor, criterion and training augmentation excluded"
_CAVEAT = (
    "calflops does not provide an exhaustive unsupported-operator ledger. "
    "Static API evidence and profiler diagnostics do not establish full executed-operator coverage; "
    "reported GFLOPs are PARTIAL native counts, not complete exact FLOPs."
)


def _tool_version(module=None):
    try:
        return metadata.version("calflops")
    except metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


@contextmanager
def _restore_torch_callables():
    """Restore profiler monkeypatches even if its forward raises before cleanup.

    calflops patches these three namespaces.  Snapshotting callable references
    is inexpensive and preserves pre-existing wrappers rather than guessing the
    installed calflops version's private cleanup signatures.
    """
    # Tensor methods such as matmul are inherited from TensorBase, so vars()
    # alone would miss them and leave failed-profiler patches installed.
    saved = [(namespace, name, inspect.getattr_static(namespace, name), name in vars(namespace))
             for namespace in (torch, functional, torch.Tensor)
             for name in dir(namespace) if callable(inspect.getattr_static(namespace, name, None))]
    failures = []
    try:
        yield failures
    finally:
        for namespace, name, value, direct in saved:
            if inspect.getattr_static(namespace, name, None) is not value:
                try:
                    if direct:
                        setattr(namespace, name, value)
                    else:
                        delattr(namespace, name)
                except (AttributeError, TypeError) as error:
                    failures.append(f"{namespace.__name__}.{name}: {error}")


def _attribute_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _coverage_evidence(model):
    evidence = {
        "status": "PARTIAL", "exhaustive": False,
        "method": "installed patch-function source plus detector module class source and captured diagnostics",
        "unsupported_operator_enumeration": "UNAVAILABLE; calflops does not enumerate all unsupported operators",
        "patched_apis": [], "candidate_api_names_absent_from_patch_source": [],
        "detector_source_api_references": [], "potentially_uncovered_detector_apis": [],
    }
    try:
        ops = importlib.import_module("calflops.pytorch_ops")
        sources = [textwrap.dedent(inspect.getsource(getattr(ops, name)))
                   for name in ("_patch_functionals", "_patch_tensor_methods")]
        patched = set()
        for source in sources:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        name = _attribute_name(target)
                        if name and name.startswith(("F.", "torch.", "Tensor.")):
                            patched.add(name)
        names = {name.rsplit(".", 1)[-1] for name in patched}
        evidence["patch_source_status"] = "AVAILABLE"
        evidence["patched_apis"] = sorted(patched)
        evidence["candidate_api_names_absent_from_patch_source"] = [
            name for name in _COVERAGE_CANDIDATES if name not in names]
    except (ImportError, AttributeError, OSError, TypeError, SyntaxError) as error:
        evidence["patch_source_status"] = "UNAVAILABLE"
        evidence["patch_source_error"] = f"{type(error).__name__}: {error}"

    found, unreadable = set(), []
    for cls in {type(module) for module in model.modules()}:
        try:
            source = inspect.getsource(cls)
            found.update(name for name in _COVERAGE_CANDIDATES
                         if re.search(r"\b" + re.escape(name) + r"\b", source))
        except (OSError, TypeError):
            unreadable.append(f"{cls.__module__}.{cls.__qualname__}")
    evidence["detector_source_api_references"] = sorted(found)
    evidence["unreadable_detector_class_sources"] = sorted(unreadable)
    evidence["potentially_uncovered_detector_apis"] = sorted(
        found.intersection(evidence["candidate_api_names_absent_from_patch_source"]))
    evidence["interpretation"] = (
        "Named APIs absent from patch assignments are coverage risks, not an exhaustive list "
        "of executed unsupported operators. Source references may be inactive; helpers and "
        "aliases may be missed. Presence in a patch function does not prove exact counting."
    )
    return evidence


class _MatmulOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8, 12), requires_grad=False)

    def forward(self, value):
        return torch.matmul(value, self.weight)


def profile_deploy_flops(model, sample):
    """Return JSON-safe, native forward counts (or nulls), always marked partial.

    The caller supplies the already-deployed FP32 detector and its resident
    synthetic input. CPU tensors are supported for contract tests; only the
    enclosing fixed CUDA protocol constitutes a formal efficiency measurement.
    """
    result = {
        "tool": "calflops", "tool_version": _tool_version(),
        "status": "UNAVAILABLE", "gflops": None, "raw_flops": None,
        "raw_macs": None, "raw_params": None, "count_scope": _SCOPE,
        "coverage_status": "PARTIAL", "coverage_notes": [_CAVEAT],
        "operator_coverage": {"status": "PARTIAL", "exhaustive": False,
                              "unsupported_operator_enumeration": "NOT_RUN"},
        "counting_convention": "native calflops counts unchanged; contraction oracle requires FMA=2",
        "profiler_stdout": "", "profiler_stderr": "", "profiler_warnings": [],
    }
    try:
        calflops = importlib.import_module("calflops")
    except ImportError as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["coverage_notes"].append("calflops unavailable; GFLOPs was NOT RUN and remains null.")
        return result
    except Exception as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        result["coverage_notes"].append("calflops import failed; GFLOPs was NOT RUN and remains null.")
        return result

    result["tool_version"] = _tool_version(calflops)
    stdout, stderr = io.StringIO(), io.StringIO()
    cleanup_errors = []
    with warnings.catch_warnings(record=True) as captured, redirect_stdout(stdout), redirect_stderr(stderr):
        warnings.simplefilter("always")
        try:
            if sample.dtype != torch.float32:
                raise ValueError("GFLOPs input must be FP32")
            result["operator_coverage"] = _coverage_evidence(model)
            options = dict(output_as_string=False, print_detailed=False, print_results=False,
                           forward_mode="forward", include_backPropagation=False)
            with torch.inference_mode():
                oracle_model = _MatmulOracle().to(device=sample.device, dtype=torch.float32).eval()
                oracle_input = torch.zeros(4, 8, device=sample.device, dtype=torch.float32)
                with _restore_torch_callables() as cleanup_errors:
                    oracle_f, oracle_m, _ = calflops.calculate_flops(
                        model=oracle_model, args=[oracle_input], **options)
                if cleanup_errors:
                    raise RuntimeError(f"Profiler monkeypatch cleanup failed: {cleanup_errors}")
                oracle_f, oracle_m = float(oracle_f), float(oracle_m)
                if not all(math.isfinite(value) and value >= 0 for value in (oracle_f, oracle_m)):
                    raise ValueError("calflops returned invalid oracle counts")
                result["oracle"] = {
                    "operation": "[4,8] @ [8,12]", "expected_macs": 384, "expected_flops": 768,
                    "raw_macs": oracle_m, "raw_flops": oracle_f,
                }
                if oracle_m != 384 or oracle_f != 768:
                    result["status"] = "CONVENTION_MISMATCH"
                    result["error"] = "Installed calflops did not demonstrate two FLOPs per contraction MAC"
                else:
                    detector_copy = copy.deepcopy(model).eval()
                    try:
                        with _restore_torch_callables() as cleanup_errors:
                            flops, macs, params = calflops.calculate_flops(
                                model=detector_copy, args=[sample], **options)
                        if cleanup_errors:
                            raise RuntimeError(f"Profiler monkeypatch cleanup failed: {cleanup_errors}")
                    finally:
                        del detector_copy
                    flops, macs, params = float(flops), float(macs), float(params)
                    if not all(math.isfinite(value) and value >= 0 for value in (flops, macs, params)):
                        raise ValueError("calflops returned negative or non-finite counts")
                    if flops == 0 or params == 0:
                        raise ValueError("calflops returned empty detector FLOPs or parameter counts")
                    result.update(status="PARTIAL", gflops=flops / 1e9,
                                  raw_flops=flops, raw_macs=macs, raw_params=params)
        except Exception as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    result["profiler_cleanup_errors"] = cleanup_errors
    result["profiler_stdout"] = stdout.getvalue()
    result["profiler_stderr"] = stderr.getvalue()
    result["profiler_warnings"] = [f"{warning.category.__name__}: {warning.message}" for warning in captured]
    diagnostics = [line for stream in (result["profiler_stdout"], result["profiler_stderr"])
                   for line in stream.splitlines()
                   if any(word in line.lower() for word in ("unsupported", "not supported", "uncounted", "warning"))]
    result["operator_coverage"]["profiler_diagnostics"] = diagnostics + result["profiler_warnings"]
    risks = result["operator_coverage"].get("potentially_uncovered_detector_apis", [])
    if risks:
        result["coverage_notes"].append("Detector source names absent from calflops patch assignments: "
                                        + ", ".join(risks) + "; execution and omissions are not exhaustively enumerated.")
    if diagnostics or result["profiler_warnings"]:
        result["coverage_notes"].append("See captured profiler diagnostics for unsupported-operation or warning messages.")
    return result
