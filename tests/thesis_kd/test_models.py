"""Native model integration tests. Architecture-only checks deliberately use no pretrained files."""
import pytest

from thesis_kd.config import ROOT, SCALES, check_resources, pair_diff, resolve_config

EXPERIMENTS = ('m_base', 'm_kd', 's_base', 's_kd', 'm_kd_p3', 'm_kd_p4', 'm_kd_p5',
               'm_kd_lambda_025', 'm_kd_lambda_050', 'm_kd_lambda_200')
L_CONFIG = ROOT / 'configs/tomato/paper_v2/vq_dense_o2o.yml'


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_protocol(experiment):
    c = resolve_config(experiment)
    n = c['native']
    assert c['protocol_id'] == 'feature_kd_ms_v1' and c['stage_policy'] == 'continuous_no_rollback'
    assert pair_diff(c)['non_kd_differences'] == []
    assert n['DEIMCriterion']['matcher']['matcher_change_epoch'] == 88
    assert n['DEIMCriterion']['losses'] == ['mal', 'boxes', 'local']
    assert n['train_dataloader']['dataset']['category_ids'] == [1, 2, 3]
    assert n['PostProcessor']['category_ids'] == [1, 2, 3]
    assert n['train_dataloader']['collate_fn']['base_size_repeat'] == 6
    assert n['train_dataloader']['collate_fn']['copyblend_epochs'] == [7, 106]
    assert n['epoches'] == 120
    assert n['DINOv3STAs']['name'] == ('vit_tinyplus' if c['arch'] == 'm' else 'vit_tiny')
    if experiment.endswith(('p3', 'p4', 'p5')):
        assert c['kd']['active_scales'] == [experiment[-2:].upper()]


def test_resource_isolation(tmp_path, monkeypatch):
    c = resolve_config('m_base')
    for key in c['resources']:
        monkeypatch.delenv(key, raising=False)
    for split in ('train', 'val'):
        (tmp_path / 'images' / split).mkdir(parents=True)
        (tmp_path / 'annotations').mkdir(exist_ok=True)
        (tmp_path / 'annotations' / f'instances_{split}.json').touch()
    f = tmp_path / 'standin'
    f.touch()
    monkeypatch.setenv('TOMATO_DATA_ROOT', str(tmp_path))
    for kind in ('BACKBONE_WEIGHTS', 'DETECTOR_CHECKPOINT'):
        monkeypatch.setenv('THESIS_KD_M_' + kind, str(f))
    check_resources(resolve_config('m_base'))
    kd = resolve_config('m_kd')
    with pytest.raises(FileNotFoundError) as missing:
        check_resources(kd)
    assert 'THESIS_KD_TEACHER_CONFIG' in str(missing.value)
    assert 'THESIS_KD_TEACHER_CHECKPOINT' in str(missing.value)
    kd['kd']['lambda_kd'] = 0.
    check_resources(kd)
    monkeypatch.setenv('THESIS_KD_TEACHER_CONFIG', str(L_CONFIG))
    monkeypatch.setenv('THESIS_KD_TEACHER_CHECKPOINT', str(f))
    check_resources(resolve_config('m_kd'))


@pytest.fixture(scope='module')
def torch_native():
    torch = pytest.importorskip('torch')
    pytest.importorskip('engine')
    prev = torch.get_num_threads()
    torch.set_num_threads(min(3, prev))
    yield torch
    torch.set_num_threads(prev)


def _native(arch):
    from thesis_kd.factories import build_student, component
    from thesis_kd.teacher import teacher_config
    if arch == 't':
        cfg = teacher_config(L_CONFIG)
        return component(cfg, cfg['model'])
    return build_student(resolve_config(arch + '_base'), initialize=False)[0]


def _architecture(model):
    return (model.backbone.dinov3.embed_dim, tuple(model.encoder.out_channels),
            len(model.decoder.decoder.layers), model.decoder.num_classes,
            sum(p.numel() for p in model.parameters()))


@pytest.mark.parametrize('order', [('m', 't', 's', 'm'), ('s', 't', 'm', 's')])
def test_native_construction_order(order, torch_native):
    from engine.core import GLOBAL_CONFIG
    before = repr(GLOBAL_CONFIG)
    prior, signatures = [], []
    for arch in order:
        detector = _native(arch)
        signature = _architecture(detector)
        prior.append(detector)
        signatures.append(signature)
        assert signature[1] == ({'m': (256,)*3, 's': (192,)*3, 't': (224,)*3}[arch])
        assert signature[2:4] == (4, 3)
        assert [_architecture(d) for d in prior] == signatures
    assert signatures[0] == signatures[-1]
    assert repr(GLOBAL_CONFIG) == before


@pytest.mark.parametrize('arch', ['m', 's'])
def test_native_seed_and_shapes(arch, torch_native):
    torch = torch_native
    from thesis_kd.factories import build_student
    torch.manual_seed(42)
    base, _ = build_student(resolve_config(arch + '_base'), initialize=False)
    torch.manual_seed(42)
    kd, _ = build_student(resolve_config(arch + '_kd'), initialize=False)
    kd_state = kd.state_dict()
    assert base.state_dict().keys() == kd_state.keys()
    for key, tensor in base.state_dict().items():
        assert torch.equal(tensor, kd_state[key]), key
    del kd_state, kd
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    base.eval().to(dev)
    with torch.no_grad():
        for size in (640, 480):
            images = torch.zeros(1, 3, size, size, device=dev)
            # Training uses dynamic positions. Enable the same native path when
            # verifying only the encoder at a non-validation resolution in eval.
            if size != 640:
                base.encoder.eval_spatial_size = None
            features = base.encoder(base.backbone(images))
            assert [tuple(x.shape) for x in features] == [
                (1, base.encoder.out_channels[i], size // stride, size // stride)
                for i, stride in enumerate([8, 16, 32])]


def test_native_l_dynamic_teacher(torch_native):
    torch = torch_native
    from thesis_kd.teacher import Teacher
    from engine.solver.det_engine import fp32_evaluation
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = _native('t').float().eval().to(dev).requires_grad_(False)
    with torch.no_grad(), fp32_evaluation(detector, dev, enabled=True):
        x = torch.rand((1, 3, 640, 640), device=dev)
        expected = detector.encoder(detector.backbone(x))
        teacher = Teacher(detector)
        buffers = {n: b.clone() for n,b in teacher.named_buffers() if 'running_' in n}
        got = teacher(x)
        for level, fixed in zip(SCALES, expected):
            torch.testing.assert_close(got[level], fixed)
        assert [tuple(t.shape) for t in got.values()] == [(1, 224, h, h) for h in (80, 40, 20)]
        smaller = teacher(x[:, :, :480, :480])
        assert [tuple(t.shape) for t in smaller.values()] == [(1, 224, h, h) for h in (60, 30, 15)]
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not any(module.training for module in teacher.modules())
    assert all(torch.equal(dict(teacher.named_buffers())[n], b) for n,b in buffers.items())


def test_state_selection_and_strict_loading(tmp_path, torch_native):
    torch = torch_native
    from thesis_kd.factories import checkpoint_state
    state = torch.nn.Linear(2, 1).state_dict()
    path = tmp_path / 'example.pt'
    torch.save({'model': state}, path)
    with pytest.raises(ValueError, match='ema'):
        checkpoint_state(path, 'ema')
    torch.save({'ema': {'module': {'module.' + k: v for k,v in state.items()}}}, path)
    loaded, layout = checkpoint_state(path, 'ema')
    assert layout == {'state_path': 'ema.module', 'ddp_prefix_removed': True}
    torch.nn.Linear(2, 1).load_state_dict(loaded, strict=True)
    with pytest.raises(RuntimeError):
        torch.nn.Linear(3, 1).load_state_dict(loaded, strict=True)


def _compact_state(state, torch):
    # Preserve real native keys, shapes and dtypes in a tiny generated fixture.
    # Each tensor shares a one-element storage with other tensors of its dtype.
    scalars = {t.dtype: torch.zeros(1, dtype=t.dtype) for t in state.values()}
    return {k: scalars[t.dtype].reshape(()).expand(t.shape) for k, t in state.items()}


def test_synthetic_full_t0_load(tmp_path, torch_native):
    from thesis_kd.teacher import build_teacher
    model = _native('t')
    state = _compact_state(model.state_dict(), torch_native)
    del model
    path = tmp_path / 'generated_t0.pt'
    cfg = resolve_config('m_kd')
    cfg['resources'].update(THESIS_KD_TEACHER_CONFIG=str(L_CONFIG),
                            THESIS_KD_TEACHER_CHECKPOINT=str(path),
                            THESIS_KD_TEACHER_VALIDATION=None)
    torch_native.save({'ema': {'module': state}}, path)
    teacher, identity, native = build_teacher(cfg)
    assert not identity['validation_basis']
    assert identity['state_path'] == 'ema.module' and teacher.channels == dict.fromkeys(SCALES, 224)
    assert native['thesis_kd_feature_extraction']['encoder_eval_spatial_size'] is None
    del teacher
    state.pop(next(iter(state)))
    torch_native.save({'ema': {'module': state}}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        build_teacher(cfg)


def test_synthetic_pretraining_head_adaptation(tmp_path, torch_native):
    from thesis_kd.factories import build_student
    torch = torch_native
    model = _native('m')
    ckpt = _compact_state(model.state_dict(), torch)
    head = 'decoder.enc_score_head.weight'
    ckpt[head] = torch.zeros((80, ckpt[head].shape[1]))
    backbone = tmp_path / 'generated_backbone.pt'
    detector = tmp_path / 'generated_detector.pt'
    torch.save(_compact_state(model.backbone.dinov3._model.state_dict(), torch), backbone)
    torch.save({'model': ckpt}, detector)
    del model
    cfg = resolve_config('m_base')
    cfg['resources'].update(THESIS_KD_M_BACKBONE_WEIGHTS=str(backbone),
                            THESIS_KD_M_DETECTOR_CHECKPOINT=str(detector))
    model, report = build_student(cfg)
    assert report['skipped_class_heads'] == [head]
    del model
    ckpt.pop('backbone.dinov3._model.cls_token')
    torch.save({'model': ckpt}, detector)
    with pytest.raises(RuntimeError, match='Missing key'):
        build_student(cfg)


def test_designated_teacher_if_configured(torch_native):
    from thesis_kd.teacher import build_teacher
    cfg = resolve_config('m_kd')
    if not all(cfg['resources'][k] for k in (
            'THESIS_KD_TEACHER_CONFIG', 'THESIS_KD_TEACHER_CHECKPOINT')):
        pytest.skip('NOT RUN: designated T0 configuration/checkpoint unavailable')
    teacher, identity, _ = build_teacher(cfg)
    y = teacher(torch_native.zeros(1, 3, 640, 640))
    assert identity['checkpoint_sha256'] and [t.shape[-1] for t in y.values()] == [80, 40, 20]
