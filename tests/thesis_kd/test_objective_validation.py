from copy import deepcopy
import hashlib
import json

import pytest
from faster_coco_eval import COCO

from thesis_kd import objective_inputs
from thesis_kd.config import resolve_config
from thesis_kd.evaluate import validation_class_ap


def test_study02_native_and_resource_pins(monkeypatch, tmp_path):
    cfg = resolve_config('m_cwd_l5_t1')
    identity = dict(id=objective_inputs.TEACHER_ID, config_digest=objective_inputs.TEACHER_CONFIG_SHA256)
    assert objective_inputs.native_fingerprint(cfg['native']) == objective_inputs.NATIVE_SHA256
    wrong = deepcopy(cfg)
    wrong['native']['clip_max_norm'] *= 2
    with pytest.raises(ValueError, match='recipe'):
        objective_inputs.verify_study02_inputs(wrong, identity)
    with pytest.raises(ValueError, match='teacher'):
        objective_inputs.verify_study02_inputs(cfg, dict(identity, id='other'))
    with pytest.raises(ValueError, match='annotation'):
        annotation = tmp_path / 'val.json'
        annotation.write_text('{}')
        objective_inputs.verify_annotation('val', annotation)

    known_files = {}
    for split, digest in objective_inputs.ANNOTATION_SHA256.items():
        known_files[cfg['native'][split + '_dataloader']['dataset']['ann_file']] = digest
    for key in ('BACKBONE_WEIGHTS', 'DETECTOR_CHECKPOINT'):
        cfg['resources']['THESIS_KD_M_' + key] = key
        known_files[key] = 'a' * 64
    monkeypatch.setattr(objective_inputs, 'sha256_file', known_files.__getitem__)
    record = objective_inputs.verify_study02_inputs(cfg, identity)
    assert record['annotation_sha256'] == objective_inputs.ANNOTATION_SHA256
    assert set(record['student_init_sha256']) == {'BACKBONE_WEIGHTS', 'DETECTOR_CHECKPOINT'}
    assert 'student_init_sha256' not in objective_inputs.verify_study02_inputs(cfg, identity, resume=True)


def test_study02_maturity_ap_missed_and_absent():
    coco = COCO()
    coco.dataset = {
        'images': [{'id': 1, 'height': 80, 'width': 80}],
        'categories': [{'id': i, 'name': name} for i, name in enumerate(('immature', 'half', 'ripe'), 1)],
        'annotations': [dict(id=i, image_id=1, category_id=i, bbox=[1, 1, 24, 24],
                             area=576, iscrowd=0) for i in (1, 2)],
    }
    coco.createIndex()
    detection = dict(image_id=1, category_id=1, bbox=[1, 1, 24, 24], score=0.9)
    record = validation_class_ap(coco, [detection])
    assert record['immature'] == pytest.approx(100)
    assert record['half'] == 0
    assert record['ripe'] is None
    assert validation_class_ap(coco, []) == dict(immature=0, half=0, ripe=None)


@pytest.fixture
def objective_run(tmp_path):
    cfg = resolve_config('m_mse_l5')
    run = tmp_path / 'study02' / cfg['experiment'] / 'seed_42'
    run.mkdir(parents=True)
    identity = dict(experiment=cfg['experiment'], protocol_id=cfg['protocol_id'], run_tag='study02',
                    seed=42, teacher=dict(id=objective_inputs.TEACHER_ID,
                                          config_digest=objective_inputs.TEACHER_CONFIG_SHA256))
    (run / 'run_info.json').write_text(json.dumps(dict(**identity, arch='m', smoke=False)))
    weights = b'synthetic checkpoint binding'
    (run / 'best_student.pth').write_bytes(weights)
    selection = dict(**identity, checkpoint='best_student.pth', epoch=1, mixed_ap=0.5,
                     sha256=hashlib.sha256(weights).hexdigest(),
                     source='FP32 EMA Mixed Validation AP', tie_break='earliest_epoch')
    (run / 'selection.json').write_text(json.dumps(selection))
    return cfg, run


def write_history(run, epochs):
    (run / 'validation_history.csv').write_text('epoch,AP\n' + ''.join(f'{i},0.5\n' for i in epochs))


@pytest.mark.parametrize('epochs', [list(range(1, 120)),
                                    [1, 1, *range(3, 121)],
                                    [2, 1, *range(3, 121)]],
                         ids=['in-progress', 'duplicate-missing', 'out-of-order'])
def test_validation_rejects_unfinished_history(objective_run, epochs):
    from thesis_kd.evaluate import evaluate_run
    cfg, run = objective_run
    write_history(run, epochs)
    with pytest.raises(ValueError, match='complete, ordered history'):
        evaluate_run(cfg, run, 'cpu', split='val')
    assert not (run / 'evaluation_val').exists()


def test_validation_accepts_complete_history_and_checks_identity(objective_run, monkeypatch):
    from thesis_kd import evaluate
    cfg, run = objective_run
    write_history(run, range(1, 121))
    reached = []
    monkeypatch.setattr(objective_inputs, 'verify_annotation', lambda *args: None)

    def load_fixture(cfg, path, device):
        reached.append(path.name)
        raise InterruptedError('fixture reached model load')

    monkeypatch.setattr(evaluate, 'load_deployed_student', load_fixture)
    with pytest.raises(InterruptedError, match='reached model'):
        evaluate.evaluate_run(cfg, run, 'cpu', split='val')
    assert reached == ['best_student.pth']
    path = run / 'run_info.json'
    info = json.loads(path.read_text())
    path.write_text(json.dumps(dict(info, smoke=True)))
    with pytest.raises(ValueError, match='run identity'):
        evaluate.evaluate_run(cfg, run, 'cpu', split='val')
    assert reached == ['best_student.pth'] and not (run / 'evaluation_val').exists()
