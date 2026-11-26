from __future__ import annotations

import json
from pathlib import Path

import pytest

from successor.experiments.train_v10r2_lane_b import (
    CLAIM_CEILING,
    DevTrainingHold,
    effect_label,
    load_and_validate_dev_spec,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _lane_b_spec() -> Path:
    return (
        _repo_root()
        / 'successor'
        / 'experiments'
        / 'V10R2_LANE_B_DEV_SPEC_FLA_BNB8_STEP1_20261003_V1.json'
    )


def test_lane_b_bound_spec_validates() -> None:
    spec = load_and_validate_dev_spec(_lane_b_spec())
    assert spec['qualification']['claim_ceiling'] == CLAIM_CEILING
    assert spec['trainer']['optimizer'] == 'adamw_bnb_8bit'
    assert spec['development_window'] == {'start_row': 4096, 'row_count': 8}
    assert spec['trainer']['pad_to_multiple_of'] == 512


def test_effect_labels_are_nonpaged_lane_b_backends() -> None:
    assert effect_label('adamw_bnb_8bit', 1) == 'ONE_LOCAL_BNB_ADAMW_8BIT_OPTIMIZER_STEP'
    assert effect_label('adamw_torch_8bit', 2) == 'TWO_LOCAL_TORCHAO_ADAMW_8BIT_OPTIMIZER_STEPS'


def test_lane_b_spec_rejects_paged_optimizer(tmp_path: Path) -> None:
    value = json.loads(_lane_b_spec().read_text(encoding='utf-8'))
    value['trainer']['optimizer'] = 'paged_adamw_8bit'
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(DevTrainingHold, match='paged'):
        load_and_validate_dev_spec(path)


def test_lane_b_spec_rejects_window_mismatch(tmp_path: Path) -> None:
    value = json.loads(_lane_b_spec().read_text(encoding='utf-8'))
    value['development_window']['row_count'] = 16
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(DevTrainingHold, match='staged optimizer window'):
        load_and_validate_dev_spec(path)

def test_nested_resume_spec_rejects_cumulative_below_previous(tmp_path: Path) -> None:
    value = json.loads(_lane_b_spec().read_text(encoding='utf-8'))
    value['trainer']['max_optimizer_steps'] = 8
    value['development_window'] = {'start_row': 4160, 'row_count': 64}
    value['resume_adapter'] = {
        'path': str(tmp_path / 'prior' / 'adapter'),
        'adapter_model_sha256': '1' * 64,
        'source_receipt_path': 'source-receipt.json',
        'source_receipt_sha256': '2' * 64,
        'source_spec_path': 'source-spec.json',
        'source_spec_sha256': '3' * 64,
        'source_weight_digest_after': '4' * 64,
        'previous_optimizer_steps': 8,
        'source_cumulative_optimizer_steps': 4,
        'next_train_row': 4160,
    }
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(DevTrainingHold, match='source_cumulative_optimizer_steps'):
        load_and_validate_dev_spec(path)


def test_verify_resume_adapter_accepts_bound_lane_a_source_ceiling_only(
    tmp_path: Path,
) -> None:
    import hashlib
    from successor.experiments.train_v10r2_lane_b import verify_resume_adapter

    adapter_dir = tmp_path / 'prior' / 'adapter'
    adapter_dir.mkdir(parents=True)
    model = adapter_dir / 'adapter_model.safetensors'
    model.write_bytes(b'lane-a-adapter')
    (adapter_dir / 'adapter_config.json').write_text(
        json.dumps({
            'peft_type': 'LORA',
            'task_type': 'CAUSAL_LM',
            'r': 4,
            'lora_alpha': 16,
            'lora_dropout': 0,
            'bias': 'none',
        }),
        encoding='utf-8',
    )

    source_spec_value = json.loads(_lane_b_spec().read_text(encoding='utf-8'))
    source_spec_value['qualification']['claim_ceiling'] = (
        'LOCAL_DEVELOPMENT_ADAPTER_ONLY_NOT_EXTERNALLY_QUALIFIED'
    )
    source_spec_value['trainer']['max_optimizer_steps'] = 8
    source_spec_value['development_window'] = {'start_row': 320, 'row_count': 64}
    source_spec_value['output']['namespace'] = str(tmp_path / 'source-output')
    source_spec = tmp_path / 'lane-a-source-spec.json'
    source_spec.write_text(json.dumps(source_spec_value), encoding='utf-8')
    source_spec_sha = hashlib.sha256(source_spec.read_bytes()).hexdigest()

    with pytest.raises(DevTrainingHold, match='claim ceiling'):
        load_and_validate_dev_spec(source_spec)

    receipt = {
        'spec_sha256': source_spec_sha,
        'weight_digest_after': '5' * 64,
        'weight_digest_changed': True,
        'cumulative_optimizer_steps': 48,
    }
    receipt_sha = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        ).encode('utf-8')
    ).hexdigest()
    receipt['receipt_sha256'] = receipt_sha
    receipt_path = tmp_path / 'source-receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')

    resume = {
        'path': str(adapter_dir),
        'adapter_model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'source_receipt_path': str(receipt_path),
        'source_receipt_sha256': receipt_sha,
        'source_spec_path': str(source_spec),
        'source_spec_sha256': source_spec_sha,
        'source_weight_digest_after': '5' * 64,
        'previous_optimizer_steps': 8,
        'source_cumulative_optimizer_steps': 48,
        'next_train_row': 384,
    }
    result = verify_resume_adapter(tmp_path, resume)
    assert result['source_spec_sha256'] == source_spec_sha
    assert result['source_cumulative_optimizer_steps'] == 48
    assert result['next_train_row'] == 384


def test_verify_resume_adapter_preserves_source_cumulative_steps(tmp_path: Path) -> None:
    import hashlib
    from successor.experiments.train_v10r2_lane_b import verify_resume_adapter

    adapter_dir = tmp_path / 'prior' / 'adapter'
    adapter_dir.mkdir(parents=True)
    model = adapter_dir / 'adapter_model.safetensors'
    model.write_bytes(b'lane-b-adapter')
    (adapter_dir / 'adapter_config.json').write_text(
        json.dumps({
            'peft_type': 'LORA',
            'task_type': 'CAUSAL_LM',
            'r': 4,
            'lora_alpha': 16,
            'lora_dropout': 0,
            'bias': 'none',
        }),
        encoding='utf-8',
    )

    source_spec_value = json.loads(_lane_b_spec().read_text(encoding='utf-8'))
    source_spec_value['trainer']['max_optimizer_steps'] = 8
    source_spec_value['development_window'] = {'start_row': 4096, 'row_count': 64}
    source_spec_value['output']['namespace'] = str(tmp_path / 'source-output')
    source_spec = tmp_path / 'source-spec.json'
    source_spec.write_text(json.dumps(source_spec_value), encoding='utf-8')
    source_spec_sha = hashlib.sha256(source_spec.read_bytes()).hexdigest()

    receipt = {
        'spec_sha256': source_spec_sha,
        'weight_digest_after': '5' * 64,
        'weight_digest_changed': True,
        'cumulative_optimizer_steps': 12,
    }
    receipt_sha = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    ).hexdigest()
    receipt['receipt_sha256'] = receipt_sha
    receipt_path = tmp_path / 'source-receipt.json'
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')

    resume = {
        'path': str(adapter_dir),
        'adapter_model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'source_receipt_path': str(receipt_path),
        'source_receipt_sha256': receipt_sha,
        'source_spec_path': str(source_spec),
        'source_spec_sha256': source_spec_sha,
        'source_weight_digest_after': '5' * 64,
        'previous_optimizer_steps': 8,
        'source_cumulative_optimizer_steps': 12,
        'next_train_row': 4160,
    }
    result = verify_resume_adapter(tmp_path, resume)
    assert result['previous_optimizer_steps'] == 8
    assert result['source_cumulative_optimizer_steps'] == 12
    assert result['next_train_row'] == 4160

def test_optimizer_state_summary_proves_finite_floating_state() -> None:
    import torch
    from successor.experiments.train_v10r2_lane_b import _optimizer_state_summary

    class FakeOptimizer:
        state = {
            1: {
                'float_state': torch.tensor([1.0, 2.0], dtype=torch.float32),
                'quantized_state': torch.tensor([1, 2], dtype=torch.uint8),
            }
        }

    summary = _optimizer_state_summary(FakeOptimizer())
    assert summary['floating_state_tensor_count'] == 1
    assert summary['nonfinite_state_tensor_count'] == 0
    assert summary['all_floating_state_finite'] is True


def test_optimizer_state_summary_detects_nonfinite_floating_state() -> None:
    import torch
    from successor.experiments.train_v10r2_lane_b import _optimizer_state_summary

    class FakeOptimizer:
        state = {1: {'float_state': torch.tensor([1.0, float('nan')])}}

    summary = _optimizer_state_summary(FakeOptimizer())
    assert summary['nonfinite_state_tensor_count'] == 1
    assert summary['all_floating_state_finite'] is False


def test_weight_delta_summary_records_update_magnitude() -> None:
    import math
    import torch
    from successor.experiments.train_v10r2_lane_b import (
        _trainable_parameter_snapshot,
        _weight_delta_summary,
    )

    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    before = _trainable_parameter_snapshot(model, torch)
    with torch.no_grad():
        model.weight.add_(torch.tensor([[3.0, 4.0]]))

    summary = _weight_delta_summary(before, model, torch)
    assert summary['all_finite'] is True
    assert summary['changed_elements'] == 2
    assert summary['total_elements'] == 2
    assert summary['delta_l2'] == pytest.approx(5.0)
    assert summary['delta_max_abs'] == pytest.approx(4.0)
    assert summary['delta_mean_abs'] == pytest.approx(3.5)
    assert summary['relative_l2_to_before'] == pytest.approx(5.0 / math.sqrt(5.0))

def test_lane_b_torchao_spec_is_one_step_unpadded_and_isolated() -> None:
    path = (
        _repo_root()
        / 'successor'
        / 'experiments'
        / 'V10R2_LANE_B_DEV_SPEC_FLA_TORCHAO8_STEP1_20261003_V1.json'
    )
    spec = load_and_validate_dev_spec(path)
    assert spec['trainer']['optimizer'] == 'adamw_torch_8bit'
    assert spec['trainer']['max_optimizer_steps'] == 1
    assert spec['trainer']['pad_to_multiple_of'] is None
    assert spec['development_window'] == {'start_row': 4096, 'row_count': 8}
    assert spec['acceleration']['backend'] == 'fla_triton'
    assert spec['acceleration']['causal_conv_backend'] == 'torch_reference'
    assert spec['output']['namespace'].endswith('fla-torchao8-step1-v1')

def test_optimizer_state_summary_handles_torchao_optimstate8bit() -> None:
    import torch
    from torchao.optim.subclass_8bit import OptimState8bit
    from successor.experiments.train_v10r2_lane_b import _optimizer_state_summary

    state = OptimState8bit.zeros(
        (4096,), signed=True, block_size=256, device='cpu', dtype=torch.bfloat16
    )

    class FakeOptimizer:
        pass

    opt = FakeOptimizer()
    opt.state = {1: {'exp_avg': state}}
    summary = _optimizer_state_summary(opt)
    expected_bytes = sum(
        int(t.numel()) * int(t.element_size())
        for t in (state.codes, state.scale, state.qmap)
    )
    assert summary['state_object_count'] == 1
    assert summary['state_tensor_count'] == 3
    assert summary['state_tensor_bytes'] == expected_bytes
    assert summary['floating_state_tensor_count'] == 2
    assert summary['nonfinite_state_tensor_count'] == 0
    assert summary['all_floating_state_finite'] is True

def test_matched_bnb8_control_differs_only_in_optimizer_role_and_output() -> None:
    torchao_path = (
        _repo_root() / 'successor' / 'experiments' /
        'V10R2_LANE_B_DEV_SPEC_FLA_TORCHAO8_STEP1_20261003_V1.json'
    )
    bnb8_path = (
        _repo_root() / 'successor' / 'experiments' /
        'V10R2_LANE_B_DEV_SPEC_FLA_BNB8_STEP1_MATCHED_20261003_V1.json'
    )
    torchao = load_and_validate_dev_spec(torchao_path)
    bnb8 = load_and_validate_dev_spec(bnb8_path)
    assert torchao['trainer']['optimizer'] == 'adamw_torch_8bit'
    assert bnb8['trainer']['optimizer'] == 'adamw_bnb_8bit'
    assert torchao['trainer']['pad_to_multiple_of'] is None
    assert bnb8['trainer']['pad_to_multiple_of'] is None
    assert torchao['development_window'] == bnb8['development_window'] == {
        'start_row': 4096, 'row_count': 8
    }
    assert torchao['source_subject'] == bnb8['source_subject']
    assert torchao['acceleration'] == bnb8['acceleration']
    assert torchao['quantization'] == bnb8['quantization']
    assert torchao['model_load'] == bnb8['model_load']
    assert torchao['lora'] == bnb8['lora']
    ignored_trainer = {'optimizer'}
    assert {k:v for k,v in torchao['trainer'].items() if k not in ignored_trainer} == {
        k:v for k,v in bnb8['trainer'].items() if k not in ignored_trainer
    }
    assert torchao['output']['namespace'] != bnb8['output']['namespace']
    assert torchao['comparison']['role'] != bnb8['comparison']['role']