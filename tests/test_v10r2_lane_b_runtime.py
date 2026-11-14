from __future__ import annotations

import pytest

from successor.experiments.v10r2_lane_b_runtime import (
    AccelerationHold,
    apply_qwen35_acceleration,
    validate_lane_b_optimizer,
)


class _FakeDelta:
    def __init__(self):
        self.chunk_gated_delta_rule = object()
        self.recurrent_gated_delta_rule = object()
        self.causal_conv1d_fn = None
        self.causal_conv1d_update = object()


class _Other:
    pass


class _FakeModel:
    def __init__(self):
        self.delta_a = _FakeDelta()
        self.delta_b = _FakeDelta()
        self.other = _Other()

    def named_modules(self):
        return [
            ('', self),
            ('model.layers.0.linear_attn', self.delta_a),
            ('model.layers.1.mlp', self.other),
            ('model.layers.2.linear_attn', self.delta_b),
        ]


def test_fla_backend_patches_delta_rule_but_leaves_conv_on_reference_path() -> None:
    model = _FakeModel()
    chunk = object()
    recurrent = object()
    conv_update = object()

    report = apply_qwen35_acceleration(
        model,
        backend='fla_triton',
        fla_chunk=chunk,
        fla_recurrent=recurrent,
        torch_conv_update=conv_update,
    )

    assert report['backend'] == 'fla_triton'
    assert report['patched_delta_modules'] == 2
    assert report['module_names'] == [
        'model.layers.0.linear_attn',
        'model.layers.2.linear_attn',
    ]
    assert report['causal_conv_backend'] == 'torch_reference'
    for module in (model.delta_a, model.delta_b):
        assert module.chunk_gated_delta_rule is chunk
        assert module.recurrent_gated_delta_rule is recurrent
        assert module.causal_conv1d_fn is None
        assert module.causal_conv1d_update is conv_update


def test_torch_reference_backend_does_not_require_fla() -> None:
    model = _FakeModel()
    chunk = object()
    recurrent = object()
    conv_update = object()

    report = apply_qwen35_acceleration(
        model,
        backend='torch_reference',
        torch_chunk=chunk,
        torch_recurrent=recurrent,
        torch_conv_update=conv_update,
    )

    assert report['patched_delta_modules'] == 2
    for module in (model.delta_a, model.delta_b):
        assert module.chunk_gated_delta_rule is chunk
        assert module.recurrent_gated_delta_rule is recurrent
        assert module.causal_conv1d_fn is None
        assert module.causal_conv1d_update is conv_update


def test_optimizer_policy_rejects_paged_and_unknown() -> None:
    assert validate_lane_b_optimizer('adamw_bnb_8bit') == 'adamw_bnb_8bit'
    assert validate_lane_b_optimizer('adamw_torch_8bit') == 'adamw_torch_8bit'
    with pytest.raises(AccelerationHold, match='paged'):
        validate_lane_b_optimizer('paged_adamw_8bit')
    with pytest.raises(AccelerationHold, match='unsupported'):
        validate_lane_b_optimizer('adamw_torch')


def test_fla_full_backend_adapts_qwen_conv_layout_and_return_shape() -> None:
    model = _FakeModel()
    chunk = object()
    recurrent = object()
    seen = {}

    class FakeTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
        def transpose(self, a, b):
            seen.setdefault('transposes', []).append((self.name, a, b))
            shape = list(self.shape)
            shape[a], shape[b] = shape[b], shape[a]
            return FakeTensor(self.name + '_t', tuple(shape))
        def contiguous(self):
            seen.setdefault('contiguous', []).append(self.name)
            return self

    def fla_conv(*, x, weight, bias=None, activation=None, backend=None, **kwargs):
        seen['conv'] = {
            'shape': x.shape,
            'weight': weight,
            'bias': bias,
            'activation': activation,
            'backend': backend,
            'kwargs': kwargs,
        }
        return FakeTensor('y', x.shape), None

    def fla_update(x, cache, residual=None, weight=None, bias=None, activation=None):
        seen['update'] = {
            'shape': x.shape,
            'cache': cache,
            'residual': residual,
            'weight': weight,
            'bias': bias,
            'activation': activation,
        }
        return FakeTensor('uy', x.shape), cache

    report = apply_qwen35_acceleration(
        model,
        backend='fla_triton_full',
        fla_chunk=chunk,
        fla_recurrent=recurrent,
        fla_conv=fla_conv,
        fla_conv_update=fla_update,
    )

    assert report['causal_conv_backend'] == 'fla_triton'
    module = model.delta_a
    x = FakeTensor('x', (2, 12, 17))
    y = module.causal_conv1d_fn(
        x=x,
        weight='w',
        bias='b',
        activation='silu',
        seq_idx=None,
    )
    assert seen['conv']['shape'] == (2, 17, 12)
    assert seen['conv']['backend'] == 'triton'
    assert y.shape == (2, 12, 17)

    ux = FakeTensor('ux', (2, 12, 1))
    uy = module.causal_conv1d_update(
        ux,
        'cache',
        'w',
        'b',
        'silu',
    )
    assert seen['update']['shape'] == (2, 1, 12)
    assert seen['update']['cache'] == 'cache'
    assert uy.shape == (2, 12, 1)


def test_fla_full_backend_rejects_seq_idx_until_equivalence_is_proved() -> None:
    model = _FakeModel()

    def fla_conv(**kwargs):
        raise AssertionError('must not be called with seq_idx')

    def fla_update(*args, **kwargs):
        raise AssertionError('unused')

    apply_qwen35_acceleration(
        model,
        backend='fla_triton_full',
        fla_chunk=object(),
        fla_recurrent=object(),
        fla_conv=fla_conv,
        fla_conv_update=fla_update,
    )
    with pytest.raises(AccelerationHold, match='seq_idx'):
        model.delta_a.causal_conv1d_fn(
            x=object(),
            weight=object(),
            seq_idx=object(),
        )
