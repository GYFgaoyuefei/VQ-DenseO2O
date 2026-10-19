"""Synthetic CPU checks; no dataset, pretrained download or training required."""

import pytest
import torch
from torch import nn
import importlib.util
from pathlib import Path

from engine.backbone.dc_sta import DCSTA
from engine.backbone import dinov3_adapter


def inputs():
    torch.manual_seed(13)
    images = torch.randn(2, 3, 64, 96)
    semantic = [torch.randn(2, 384, h, w) for h, w in [(8, 12), (4, 6), (2, 3)]]
    detail = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in [(64, 8, 12), (128, 4, 6), (128, 2, 3)]]
    return images, semantic, detail


def test_original_near_identity_gain_at_every_scale():
    module = DCSTA([64, 128, 128])
    images, semantic, detail = inputs()
    untouched_semantic = [x.clone() for x in semantic]
    descriptor = module.descriptor(images)
    for level, (sem, sta) in enumerate(zip(semantic, detail)):
        channel, spatial = module.compute_gates(descriptor, sem, sta, level)
        torch.testing.assert_close(channel, torch.full_like(channel, .999 ** .5))
        torch.testing.assert_close(spatial, torch.full_like(spatial, .999 ** .5))
        assert not torch.equal(channel, torch.ones_like(channel))
        assert module.channel[level][0].in_features == 8
        assert module.channel[level][0].out_features == 16
        assert module.spatial[level].in_channels == 10
        assert module.spatial[level].kernel_size == (3, 3)
    result = module(images, semantic, detail)
    assert len(result) == 3
    for actual, original in zip(result, detail):
        torch.testing.assert_close(actual, original * .999, atol=5e-7, rtol=5e-7)
    for actual, original in zip(semantic, untouched_semantic):
        assert torch.equal(actual, original)


def test_last_layer_gradients_are_finite_and_nonzero_at_beta_initialization():
    module = DCSTA([64, 128, 128])
    images, semantic, detail = inputs()
    loss = sum(x.square().mean() for x in module(images, semantic, detail))
    loss.backward()
    for mlp, spatial in zip(module.channel, module.spatial):
        for parameter in (mlp[-1].weight, mlp[-1].bias, spatial.weight, spatial.bias):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0
        # Zero final weights intentionally block the first MLP layer on step 1.
        assert torch.count_nonzero(mlp[0].weight.grad) == 0
    for value in detail:
        assert torch.isfinite(value.grad).all()


def test_sigmoid_range_and_formula_after_parameters_change():
    module = DCSTA([64, 128, 128])
    images, semantic, detail = inputs()
    with torch.no_grad():
        for mlp, conv in zip(module.channel, module.spatial):
            mlp[-1].weight.normal_(0, 8)
            mlp[-1].bias.normal_(0, 8)
            conv.weight.normal_(0, 8)
            conv.bias.fill_(0.4)
    descriptor = module.descriptor(images)
    result = module(images, semantic, detail)
    for level, (sem, sta) in enumerate(zip(semantic, detail)):
        channel, spatial = module.compute_gates(descriptor, sem, sta, level)
        for gate in (channel, spatial):
            assert torch.isfinite(gate).all()
            assert gate.min() >= 0
            assert gate.max() <= 1
        assert torch.equal(result[level], sta * channel * spatial)
    sum(x.square().mean() for x in result).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_descriptor_inverse_normalization_clamp_and_rgb_order():
    module = DCSTA([64, 128, 128])
    rgb = torch.tensor([0.2, 0.4, 0.6]).view(1, 3, 1, 1).expand(1, 3, 80, 80)
    normalized = normalize(rgb)
    descriptor = module.descriptor(normalized)
    luminance = 0.299 * 0.2 + 0.587 * 0.4 + 0.114 * 0.6
    expected = torch.tensor([[luminance, 1e-4, torch.sigmoid(torch.tensor(12 * (0.2 - luminance))),
                              torch.sigmoid(torch.tensor(12 * (luminance - 0.85))), .2, .4, .6, 0.]])
    torch.testing.assert_close(descriptor, expected, atol=1e-6, rtol=1e-6)
    clamped = module.descriptor(torch.cat([torch.full_like(normalized, -100), torch.full_like(normalized, 100)]))
    torch.testing.assert_close(clamped[:, 4:7], torch.tensor([[0., 0., 0.], [1., 1., 1.]]))
    assert descriptor.dtype == torch.float32


def normalize(rgb):
    return ((rgb - rgb.new_tensor([.485, .456, .406])[None, :, None, None])
            / rgb.new_tensor([.229, .224, .225])[None, :, None, None])


def test_descriptor_finite_difference_uses_original_normalization():
    module = DCSTA([64, 128, 128])
    ramp = torch.linspace(0, 1, 80).view(1, 1, 1, 80).expand(2, 3, 80, 80).contiguous()
    descriptor = module.descriptor(normalize(ramp))
    torch.testing.assert_close(descriptor[:, 7], torch.full((2,), 0.5 / 79))
    torch.testing.assert_close(descriptor[:, 0], torch.full((2,), 0.5))
    assert torch.isfinite(descriptor).all()


def test_cpu_autocast_near_identity_and_finite_backward():
    module = DCSTA([64, 128, 128])
    images, semantic, detail = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = module(images, semantic, detail)
        loss = sum(x.square().mean() for x in result)
    for actual, original in zip(result, detail):
        # Original sigmoid(.999**.5) rounds to one in BF16; no residual rescaling.
        torch.testing.assert_close(actual, original, atol=.01, rtol=.01)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize("options", [{"alpha": .1}, {"init_detail_gain": float("nan")},
                                    {"init_detail_gain": 1.0}, {"init_detail_gain": 0},
                                    {"descriptor_dim": 9}, {"levels": ["c2"]},
                                    {"gate_init": "residual"},
                                    {"channel_gate": False, "spatial_gate": False}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        DCSTA([64, 128, 128], options)


def test_invalid_scale_count_is_rejected():
    with pytest.raises(ValueError):
        DCSTA([64, 128])
    module = DCSTA([64, 128, 128])
    images, semantic, detail = inputs()
    with pytest.raises(ValueError):
        module(images, semantic[:2], detail)


class FakeDino(nn.Module):
    """Small deterministic semantic stand-in; adapter and fusion remain real."""
    embed_dim = 384

    def __init__(self, name):
        super().__init__()
        self.proj = nn.Conv2d(3, self.embed_dim, 16, stride=16)

    def get_intermediate_layers(self, images, n, return_class_token):
        features = self.proj(images).flatten(2).transpose(1, 2)
        return [(features, features.mean(1)) for _ in n]


def make_adapter(monkeypatch, options=None, **kwargs):
    monkeypatch.setattr(dinov3_adapter, "DinoVisionTransformer", FakeDino)
    return dinov3_adapter.DINOv3STAs(name="dinov3_vits16", interaction_indexes=[5, 8, 11],
                                  conv_inplane=32, hidden_dim=224, dc_sta=options, **kwargs).eval()


def test_adapter_preserves_native_state_rng_and_original_sigmoid_fusion(monkeypatch):
    torch.manual_seed(41)
    native = make_adapter(monkeypatch)
    next_native_random = torch.rand(4)
    torch.manual_seed(41)
    disabled = make_adapter(monkeypatch, {"enabled": False})
    next_disabled_random = torch.rand(4)
    torch.manual_seed(41)
    enabled = make_adapter(monkeypatch, {"enabled": True})
    next_enabled_random = torch.rand(4)
    assert native.dc_sta is None and disabled.dc_sta is None
    assert native.state_dict().keys() == disabled.state_dict().keys()
    for key, value in native.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key])
        assert torch.equal(value, enabled.state_dict()[key])
    assert torch.equal(next_native_random, next_disabled_random)
    assert torch.equal(next_native_random, next_enabled_random)
    assert [conv.in_channels for conv in enabled.convs] == [448, 512, 512]
    assert all(isinstance(norm, nn.SyncBatchNorm) for norm in enabled.norms)
    images = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        original = native(images)
        off = disabled(images)
        on = enabled(images)
        reference_dc = beta_reference()([64, 128, 128], {})
        reference_dc.load_state_dict({key: value for key, value in enabled.dc_sta.state_dict().items()
                                      if key != "_gate_architecture"})
        layers = native.dinov3.get_intermediate_layers(images, n=[5, 8, 11], return_class_token=True)
        semantic = [torch.nn.functional.interpolate(features.transpose(1, 2).view(2, 384, 4, 6),
                     size=(64 // stride, 96 // stride), mode="bilinear", align_corners=False)
                    for (features, _), stride in zip(layers, (8, 16, 32))]
        expected = [norm(conv(torch.cat([sem, sta], 1)))
                    for norm, conv, sem, sta in zip(native.norms, native.convs, semantic,
                                                   reference_dc(images, semantic, native.sta(images)))]
    assert [x.shape for x in on] == [(2, 224, 8, 12), (2, 224, 4, 6), (2, 224, 2, 3)]
    assert all(torch.equal(a, b) for a, b in zip(original, off))
    assert any(not torch.equal(a, b) for a, b in zip(original, on))
    for actual, reference in zip(on, expected):
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)


def test_existing_pretrained_loading_is_preserved(monkeypatch, tmp_path):
    pretrained = FakeDino("dinov3_vits16")
    with torch.no_grad():
        pretrained.proj.weight.fill_(.25)
        pretrained.proj.bias.fill_(.5)
    checkpoint = tmp_path / "synthetic_dino.pth"
    torch.save(pretrained.state_dict(), checkpoint)
    adapter = make_adapter(monkeypatch, {"enabled": True}, weights_path=str(checkpoint))
    assert all(torch.equal(value, adapter.dinov3.state_dict()[key]) for key, value in pretrained.state_dict().items())


def test_adapter_requires_sta_and_three_scales(monkeypatch):
    with pytest.raises(ValueError, match="requires STA"):
        make_adapter(monkeypatch, {"enabled": True}, use_sta=False)
    adapter = make_adapter(monkeypatch, {"enabled": True})
    with pytest.raises(ValueError, match="divisible by 32"):
        adapter(torch.randn(1, 3, 63, 64))


@pytest.mark.parametrize("enabled", [False, True])
def test_real_dinov3_small_input_cpu_forward_backward(enabled):
    # Real ViT-S/16, blocks 5/8/11, native detail channels and fusion. No weights
    # are downloaded; this only checks executable architecture and gradients.
    adapter = dinov3_adapter.DINOv3STAs(
        name="dinov3_vits16", weights_path=None, interaction_indexes=[5, 8, 11],
        conv_inplane=32, hidden_dim=224, dc_sta={"enabled": enabled},
    ).eval()
    output = adapter(torch.randn(1, 3, 32, 32))
    assert [x.shape for x in output] == [(1, 224, 4, 4), (1, 224, 2, 2), (1, 224, 1, 1)]
    assert all(torch.isfinite(x).all() for x in output)
    sum(x.square().mean() for x in output).backward()
    if enabled:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.dc_sta.parameters())


def test_key_bias_mask_initialization_and_checkpoint_override():
    from engine.backbone.dinov3.layers.attention import LinearKMaskedBias
    projection = LinearKMaskedBias(4, 12)
    assert torch.equal(projection.bias_mask, torch.tensor([1.] * 4 + [0.] * 4 + [1.] * 4))
    assert torch.isfinite(projection(torch.randn(2, 4))).all()
    checkpoint = projection.state_dict()
    checkpoint['bias_mask'] = torch.arange(12).float() / 12
    projection.load_state_dict(checkpoint)
    assert torch.equal(projection.bias_mask, checkpoint['bias_mask'])


def beta_reference():
    spec = importlib.util.spec_from_file_location("beta_dc_reference",
                                                Path(__file__).parent / "references" / "beta_dc_sta.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DCSTA


@pytest.mark.parametrize("options", [{}, {"use_global_gate": False}, {"use_local_reliability": False},
                                    {"use_luminance_stats": False, "use_color_stats": False,
                                     "use_high_frequency_proxy": False}])
def test_fixed_input_weights_descriptor_outputs_and_gradients_match_beta_select(options):
    torch.manual_seed(52)
    reference = beta_reference()([64, 128, 128], options)
    torch.manual_seed(52)
    migrated = DCSTA([64, 128, 128], options)
    assert not hasattr(migrated, "alpha")
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter, dict(migrated.named_parameters())[name]), name
    # Exercise trained, nonconstant gates as well as their original initialization.
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-.15, .15)
    transferred = dict(reference.state_dict(), _gate_architecture=torch.tensor(2, dtype=torch.int64))
    migrated.load_state_dict(transferred)
    images, semantic, detail = inputs()
    ref_images = images.detach().clone().requires_grad_()
    new_images = images.detach().clone().requires_grad_()
    ref_sem = [value.detach().clone().requires_grad_() for value in semantic]
    new_sem = [value.detach().clone().requires_grad_() for value in semantic]
    ref_detail = [value.detach().clone().requires_grad_() for value in detail]
    new_detail = [value.detach().clone().requires_grad_() for value in detail]
    torch.testing.assert_close(migrated.descriptor(new_images), reference.descriptor(ref_images), atol=1e-7, rtol=1e-6)
    expected = reference(ref_images, ref_sem, ref_detail)
    actual = migrated(new_images, new_sem, new_detail)
    for ref_value, new_value in zip(expected, actual):
        torch.testing.assert_close(new_value, ref_value, atol=1e-7, rtol=1e-6)
    sum(value.square().mean() for value in expected).backward()
    sum(value.square().mean() for value in actual).backward()
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(dict(migrated.named_parameters())[name].grad, parameter.grad,
                                   atol=1e-7, rtol=1e-6)
    for ref_value, new_value in zip([ref_images, *ref_sem, *ref_detail], [new_images, *new_sem, *new_detail]):
        if ref_value.grad is None:
            assert new_value.grad is None
        else:
            torch.testing.assert_close(new_value.grad, ref_value.grad, atol=1e-7, rtol=1e-6)


def test_nonfinite_descriptor_is_rejected_like_beta_select():
    images = torch.full((1, 3, 8, 8), float("nan"))
    for module in [beta_reference()([4, 8, 8], {}), DCSTA([4, 8, 8])]:
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            module.descriptor(images)
