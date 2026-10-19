"""Paper configuration/runner contracts without real data or checkpoints."""

import copy
import csv
import json

import pytest
import torch

from engine.core.innovation_config import resolve_innovation
from engine.core.yaml_utils import load_config
from engine.deim.postprocessor import PostProcessor
from engine.solver.paper_v2_tracking import is_better, read_selection, record_validation
from tools.paper_v2 import run


MATRIX = {
    "baseline_native": (False, False, False, False),
    "baseline_maskaware": (True, False, False, False),
    "vq_dense_o2o": (True, True, False, False),
    "li_copyblend": (True, False, True, False),
    "dc_sta": (True, False, False, True),
}


def switches(config):
    flags = config["innovation"]
    return (flags["instance_mask_sync"], *(flags[name]["enabled"] for name in ("vq_dense_o2o", "li_copyblend", "dc_sta")))


@pytest.mark.parametrize("name,expected", MATRIX.items())
def test_five_independent_configs_resolve_exact_switch_matrix(name, expected):
    cfg = run.load_resolved(name, expand_env=False)
    run.verify_recipe(cfg)
    assert switches(cfg) == expected
    sync, gr, li, dc = expected
    assert cfg["criterion"] == ("VQDEIMCriterion" if gr else "DEIMCriterion")
    assert cfg["train_dataloader"]["dataset"]["type"] == ("VisibleMaskCocoDetection" if sync else "CocoDetection")
    assert cfg["train_dataloader"]["collate_fn"]["li_copyblend"]["enabled"] == li
    assert cfg["DINOv3STAs"]["dc_sta"]["enabled"] == dc
    assert cfg["DINOv3STAs"]["conv_inplane"] == 32
    assert cfg["DINOv3STAs"]["name"] == "dinov3_vits16"
    sanitizers = [op["type"] for op in cfg["train_dataloader"]["dataset"]["transforms"]["ops"]
                  if op["type"].startswith("Sanitize")]
    assert sanitizers and set(sanitizers) == {"SanitizeVisibleBoundingBoxes" if sync else "SanitizeBoundingBoxes"}
    assert cfg["val_dataloader"]["dataset"]["type"] == "CocoDetection"
    assert resolve_innovation(cfg, expand_env=False) == cfg


def test_yaml_load_and_resolve_do_not_contaminate_other_experiments():
    first = run.load_resolved("baseline_native", expand_env=False)
    original = copy.deepcopy(first)
    for model in ("vq_dense_o2o", "li_copyblend", "dc_sta", "baseline_maskaware"):
        cfg = run.load_resolved(model, expand_env=False)
        cfg["train_dataloader"]["dataset"]["transforms"]["ops"][0]["probability"] = -1
    assert first == original
    assert run.load_resolved("baseline_native", expand_env=False) == original
    raw = load_config(str(run.config_path("baseline_native")))
    untouched = copy.deepcopy(raw)
    resolve_innovation(raw, expand_env=False)
    assert raw == untouched


@pytest.mark.parametrize("module", ["vq_dense_o2o", "li_copyblend", "dc_sta"])
def test_mask_consumers_require_independent_mask_switch(module):
    raw = load_config(str(run.config_path("baseline_native")))
    raw["innovation"][module]["enabled"] = True
    with pytest.raises(ValueError, match="require instance_mask_sync"):
        resolve_innovation(raw, expand_env=False)


def test_dc_sta_shares_mask_pipeline_but_keeps_native_criterion():
    raw = load_config(str(run.config_path("baseline_maskaware")))
    raw["innovation"]["dc_sta"]["enabled"] = True
    cfg = resolve_innovation(raw, expand_env=False)
    assert cfg["criterion"] == "DEIMCriterion"
    assert cfg["train_dataloader"]["dataset"]["type"] == "VisibleMaskCocoDetection"
    assert cfg["DINOv3STAs"]["dc_sta"]["enabled"]


def test_missing_environment_variables_fail_explicitly(monkeypatch):
    for key in ("TOMATO_DATA_ROOT", "DINOV3_WEIGHTS", "TOMATO_DETECTOR_CHECKPOINT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="environment variable is unset"):
        run.load_resolved("baseline_native")
    values = {"TOMATO_DATA_ROOT": "D:/synthetic/tomato", "DINOV3_WEIGHTS": "D:/synthetic/dino.pth",
              "TOMATO_DETECTOR_CHECKPOINT": "D:/synthetic/detector.pth"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    cfg = run.load_resolved("baseline_native")
    assert cfg["DINOv3STAs"]["weights_path"] == values["DINOV3_WEIGHTS"]
    assert cfg["train_dataloader"]["dataset"]["ann_file"].startswith(values["TOMATO_DATA_ROOT"])
    monkeypatch.setenv("TOMATO_VISIBLE_MASK_TRAIN", "D:/synthetic/verified_visible_masks_vqcr.json")
    masked = run.load_resolved("vq_dense_o2o")
    assert masked["train_dataloader"]["dataset"]["visible_mask_sidecar"] == "D:/synthetic/verified_visible_masks_vqcr.json"
    monkeypatch.setenv("TOMATO_VISIBLE_MASK_TRAIN", "D:/other_run/masks.json")
    # A resolved checkpoint/run configuration cannot drift with a later shell.
    assert resolve_innovation(masked) == masked


def test_combination_generator_produces_four_parseable_configs(tmp_path, monkeypatch):
    # Resolve through the real common recipe while keeping generated files in tmp.
    original_common = run.CONFIG_DIR / "common.yml"
    run.write_yaml(tmp_path / "common.yml", {"__include__": [str(original_common)]})
    monkeypatch.setattr(run, "CONFIG_DIR", tmp_path)
    run.generate_combinations()
    assert sorted(p.stem for p in (tmp_path / "combinations").glob("*.yml")) == sorted(run.COMBINATIONS)
    for name, modules in run.COMBINATIONS.items():
        config = run.load_resolved(name, expand_env=False)
        assert switches(config) == (True, *(key in modules for key in ("vq_dense_o2o", "li_copyblend", "dc_sta")))
        run.verify_recipe(config)


@pytest.mark.parametrize("name", list(MATRIX))
def test_real_model_and_criterion_construct_from_each_config_without_weights(name):
    cfg = run.load_resolved(name, expand_env=False)
    cfg["DINOv3STAs"]["weights_path"] = None
    cfg["eval_spatial_size"] = [128, 128]
    obj = run.create_config(cfg)
    assert obj.model.backbone.dinov3.embed_dim == 384
    assert (obj.model.backbone.dc_sta is not None) == MATRIX[name][3]
    assert type(obj.criterion).__name__ == cfg["criterion"]
    assert obj.postprocessor.category_ids.tolist() == [1, 2, 3]


@pytest.mark.parametrize("name,modules", list(run.COMBINATIONS.items()))
def test_combination_model_and_criterion_construct_without_weights(name, modules):
    raw = load_config(str(run.CONFIG_DIR / "common.yml"))
    raw["innovation"]["instance_mask_sync"] = True
    for key in modules:
        raw["innovation"][key]["enabled"] = True
    cfg = resolve_innovation(raw, expand_env=False)
    cfg["DINOv3STAs"]["weights_path"] = None
    cfg["eval_spatial_size"] = [128, 128]
    obj = run.create_config(cfg)
    assert (obj.model.backbone.dc_sta is not None) == ("dc_sta" in modules)
    assert type(obj.criterion).__name__ == ("VQDEIMCriterion" if "vq_dense_o2o" in modules else "DEIMCriterion")


def test_postprocessor_returns_verified_coco_ids_and_keeps_deploy_class_indices():
    outputs = {"pred_logits": torch.tensor([[[10., -10., -10.], [-10., 11., -10.], [-10., -10., 12.]]]),
               "pred_boxes": torch.full((1, 3, 4), .5)}
    processor = PostProcessor(num_classes=3, num_top_queries=3, category_ids=[7, 23, 91])
    ordinary = processor(outputs, torch.tensor([[80, 100]]))
    assert set(ordinary[0]["labels"].tolist()) == {7, 23, 91}
    assert ordinary[0]["boxes"].shape == (3, 4)
    deployed = processor.deploy()(outputs, torch.tensor([[80, 100]]))
    assert set(deployed[0].flatten().tolist()) == {0, 1, 2}
    with pytest.raises(ValueError):
        PostProcessor(num_classes=3, category_ids=[1, 1, 3])


def test_selection_uses_strict_improvement_and_retains_ties_after_reload(tmp_path):
    assert read_selection(tmp_path) == {"ap": -1., "epoch": -1}
    selection = {"ap": .6, "epoch": 9, "model": "ema"}
    assert is_better(.60001, selection)
    assert not is_better(.6, selection)
    assert not is_better(.59, selection)
    assert not is_better(float("nan"), selection)
    assert not is_better(float("inf"), selection)
    (tmp_path / "run_info.json").write_text(json.dumps({"experiment": "dc_sta", "seed": 42}), encoding="utf-8")
    record_validation(tmp_path, 9, [.6, .8, .7, .4, .5, .6], selection)
    resumed = read_selection(tmp_path)
    assert resumed == selection
    record_validation(tmp_path, 10, [.6, .8, .7, .4, .5, .6], resumed)
    assert read_selection(tmp_path)["epoch"] == 9
    with (tmp_path / "epoch_validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [9, 10]
    info = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    assert info["best_validation_ap"] == .6 and info["best_validation_epoch"] == 9


def test_output_guard_blocks_overwrite_and_foreign_or_completed_resume(tmp_path):
    destination = tmp_path / "run"
    run.guard_output(destination)
    destination.mkdir()
    run.guard_output(destination)
    checkpoint = destination / "last.pth"
    checkpoint.touch()
    with pytest.raises(FileExistsError):
        run.guard_output(destination)
    run.guard_output(destination, checkpoint)
    foreign = tmp_path / "foreign.pth"
    foreign.touch()
    with pytest.raises(ValueError, match="this run"):
        run.guard_output(destination, foreign)
    with pytest.raises(FileNotFoundError):
        run.guard_output(destination, destination / "missing.pth")
    (destination / "completed.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already completed"):
        run.guard_output(destination, checkpoint)


def test_native_train_command_single_gpu_ddp_resume_and_test(monkeypatch, tmp_path):
    monkeypatch.setenv("TOMATO_DETECTOR_CHECKPOINT", "D:/synthetic/detector.pth")
    args = run.parser().parse_args(["train", "--model", "vq_dense_o2o", "--seed", "42", "--device", "cuda:1"])
    command = run.train_command("resolved.yml", args, tmp_path)
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--tuning") + 1] == "D:/synthetic/detector.pth"
    args.nproc_per_node = 2
    command = run.train_command("resolved.yml", args, tmp_path)
    assert "torch.distributed.run" in command and "--device" not in command
    assert command[command.index("--nproc_per_node") + 1] == "2"
    args.resume = "last.pth"
    resumed = run.train_command("resolved.yml", args, tmp_path)
    assert "--tuning" not in resumed and resumed[resumed.index("--resume") + 1] == "last.pth"
    args.checkpoint = "best.pth"
    evaluated = run.train_command("resolved.yml", args, tmp_path, evaluation=True)
    assert "--test-only" in evaluated and "--tuning" not in evaluated
    assert evaluated[evaluated.index("--resume") + 1] == "best.pth"
    args.device = "cpu"
    with pytest.raises(ValueError, match="requires CUDA"):
        run.train_command("resolved.yml", args, tmp_path)
    args.device, args.nproc_per_node = "cuda:0", 3
    with pytest.raises(ValueError, match="divisible"):
        run.train_command("resolved.yml", args, tmp_path)
