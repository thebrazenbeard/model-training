"""Source-only, fail-closed Qwen3.5-4B memory-fit experiment.

This file offers a *candidate* tiny NF4 LoRA optimizer-step kernel and a
separate DeepSpeed ZeRO-3 CPU-offload configuration hypothesis. It never
authorizes either physical execution. The existing R18 recipe is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from typing import Mapping

from trainingday.physical_launch_authority import require_physical_launch_authority


BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SCHEMA = "TRAININGDAY_LOWMEM_SOURCE_CANDIDATE_V1"
EXECUTION_TARGET = "LAPPY_UBUNTU_26_04_WSL2_LOCAL_RTX3050"


class LowMemoryExperimentError(ValueError):
    """Invalid hypothetical plan or missing real optimizer-step evidence."""


def validate_declaration(declaration: Mapping[str, object]) -> None:
    """Validate untrusted *declarations*, never treat them as host grants."""
    if type(declaration) is not dict:
        raise LowMemoryExperimentError("model declaration must be a mapping")
    if (any(type(declaration.get(key)) is not str for key in
            ("model_id", "revision", "loader_class", "quantization"))
            or declaration.get("model_id") != BASE_ID
            or declaration.get("revision") != BASE_REVISION
            or declaration.get("loader_class") != "Qwen3_5ForCausalLM"
            or declaration.get("quantization") != "NF4_DOUBLE_BF16"):
        raise LowMemoryExperimentError("official text-only base binding mismatch")
    files = declaration.get("files")
    if type(files) is not dict or not 1 <= len(files) <= 100:
        raise LowMemoryExperimentError("tokenizer/weight binding missing")
    for name, digest in files.items():
        if (type(name) is not str or not 1 <= len(name) <= 100
                or "/" in name or "\\" in name or name in {".", ".."}
                or type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise LowMemoryExperimentError("unsafe or unhashed snapshot artifact")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                 "model.safetensors.index.json"):
        value = files.get(name)
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise LowMemoryExperimentError(f"unbound or malformed model artifact: {name}")
    if not any(re.fullmatch(r"model\.safetensors-[0-9]{5}-of-[0-9]{5}\.safetensors", name)
               for name in files):
        raise LowMemoryExperimentError("weight shard binding missing")
    # This is syntax-only. The existing _required_base validates actual bytes.


def source_plan() -> dict:
    """No imports of torch/transformers, no GPU/file/host effects."""
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_HARD_HOLD",
        "execution_target": EXECUTION_TARGET,
        "execution_host_local": True,
        "cuda_runtime_verified": False,
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "authorizes_training": False,
        "optimizer_steps_observed": 0,
        "eval_access": False,
        "data": "DETERMINISTIC_SYNTHETIC_ONLY",
        "candidates": {
            "nf4_minimal_gpu": {
                "implementation": "UNEXECUTED_KERNEL",
                "batch": 1,
                "sequence_length": 64,
                "lora_r": 4,
                "target_modules": ["q_proj", "v_proj"],
                "frozen_base": True,
                "activation_checkpointing": True,
                "device_map": 0,
                "cpu_offload": False,
                "validity": "FITS_ONLY_IF_MEASURED_ON_AUTHORIZED_GPU",
            },
            "bf16_zero3_cpu_offload": {
                "implementation": "CONFIG_HYPOTHESIS_NOT_RUNNABLE",
                "optimizer_steps_observed": 0,
                "offload_param": "cpu",
                "offload_optimizer": "cpu",
                "requires": ["compatible DeepSpeed + Qwen3.5 training stack",
                             "substantial unallocated host RAM",
                             "independently verified host broker and GPU lease"],
                "validity": "UNQUALIFIED_HIGH_HOST_RAM_AND_PLATFORM_RISK",
            },
        },
    }


def proposed_zero3_config() -> dict:
    """A configuration hypothesis, not a run path or memory-fit guarantee.

    ZeRO-3 training requires DeepSpeed engine.backward/engine.step plus model
    and optimizer initialization under the engine. Inference device_map='auto'
    is not a supported substitution for this training path.
    """
    return {
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": False},
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "overlap_comm": False,
            "contiguous_gradients": True,
        },
    }


def evaluate_optimizer_witness(*, before: Mapping[str, bytes], after: Mapping[str, bytes],
                               loss: float, gradient_norm: float,
                               optimizer_step_returned: bool) -> dict:
    """Independent before/after bytes are required; no self-asserted pass.

    This pure check can be tested with fake tensors. It does not itself prove
    the bytes came from an authentic pinned Qwen model or authorized GPU.
    """
    if (type(optimizer_step_returned) is not bool or not optimizer_step_returned
            or type(loss) not in (int, float) or not math.isfinite(loss)
            or type(gradient_norm) not in (int, float)
            or not math.isfinite(gradient_norm) or gradient_norm <= 0):
        raise LowMemoryExperimentError("missing finite loss/gradient/optimizer-step evidence")
    if (not before or set(before) != set(after)
            or any("lora_B" not in key for key in before)
            or any(not isinstance(value, bytes) or len(value) == 0
                   or not isinstance(after[key], bytes)
                   or len(value) != len(after[key])
                   for key, value in before.items())):
        raise LowMemoryExperimentError("LoRA-B before/after witness incomplete or incompatible")
    changed = [key for key in sorted(before) if before[key] != after[key]]
    if not changed:
        raise LowMemoryExperimentError("optimizer returned but no adapter tensor changed")
    digest = lambda tensors: hashlib.sha256(b"".join(
        len(key).to_bytes(4, "big") + key.encode("utf-8")
        + len(tensors[key]).to_bytes(8, "big") + tensors[key]
        for key in sorted(tensors))).hexdigest()
    return {
        "status": "EVIDENCE_SHAPE_VALIDATED_NOT_AUTHENTICATED",
        "adapter_tensors_changed": len(changed),
        "adapter_tensors_checked": len(before),
        "before_sha256": digest(before),
        "after_sha256": digest(after),
        "loss": float(loss),
        "gradient_norm": float(gradient_norm),
        "physical_training_authorized": False,
        "proves_model_quality": False,
    }


def summarize_cuda_phase_peaks(*, load_allocated: int, load_reserved: int,
                               step_allocated: int, step_reserved: int) -> dict:
    """Retain both CUDA allocator phases, even if loading was the high-water mark.

    These process-local PyTorch allocator counters do not measure another
    process's VRAM, system RAM, transient driver allocations, or device-wide
    memory availability. They cannot independently establish memory fit.
    """
    values = (load_allocated, load_reserved, step_allocated, step_reserved)
    if (any(type(value) is not int or value < 0 for value in values)
            or load_allocated > load_reserved or step_allocated > step_reserved):
        raise LowMemoryExperimentError("invalid CUDA allocator phase peaks")
    return {
        "load_peak_cuda_allocated_bytes": load_allocated,
        "load_peak_cuda_reserved_bytes": load_reserved,
        "step_peak_cuda_allocated_bytes": step_allocated,
        "step_peak_cuda_reserved_bytes": step_reserved,
        # Preserve the old public keys, but make their scope explicit.
        "peak_cuda_allocated_bytes": max(load_allocated, step_allocated),
        "peak_cuda_reserved_bytes": max(load_reserved, step_reserved),
        "cuda_peak_scope": "PYTORCH_PROCESS_ALLOCATOR_ONLY",
        "physical_training_authorized": False,
    }


def _snapshot_lora_b(model) -> dict[str, bytes]:
    """Capture actual CPU bytes for adapter B tensors, never model inputs."""
    result = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            result[name] = param.detach().float().cpu().contiguous().numpy().tobytes()
    if not result:
        raise LowMemoryExperimentError("LoRA-B adapter parameters missing")
    return result


def one_step_kernel(base_binding_path):
    """Protected experimental code, deliberately unreachable until real broker.

    The first call is the production hard hold, *before* torch, model files,
    CUDA, or side effects. No environment variable, caller flag or source
    receipt can override it. A future independently authenticated host broker
    must replace the separate physical authority boundary, not this function.
    """
    require_physical_launch_authority(operation="low-memory synthetic fit experiment")

    # Unreachable under current source authority. Kept as a reviewable kernel;
    # do not infer a verified installed/runnable host implementation.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM
    from trainingday.train_neural import _required_base
    from trainingday.qwen_text_checkpoint import (
        index_binding_pin, prepare_text_loader, verify_text_load,
    )

    base, snapshot = _required_base(base_binding_path)
    validate_declaration(base)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise LowMemoryExperimentError("BF16 CUDA GPU unavailable")
    index_sha, shards = index_binding_pin(base)
    text_config, mapping = prepare_text_loader(
        snapshot, expected_index_sha256=index_sha, expected_shards=shards)
    torch.manual_seed(73504)
    # Begin the load-phase high-water mark before the quantized weight load.
    # A later step-only reset must not erase its potentially larger peak.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model, load_info = Qwen3_5ForCausalLM.from_pretrained(
        snapshot, quantization_config=quant, device_map=0, dtype=torch.bfloat16,
        local_files_only=True, config=text_config, key_mapping=mapping,
        output_loading_info=True)
    verify_text_load(model, load_info)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]))
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise LowMemoryExperimentError("unqualified CPU/disk device placement")
    placement = getattr(model, "hf_device_map", None) or {}
    if any(str(value).lower() in {"cpu", "disk"} for value in placement.values()):
        raise LowMemoryExperimentError("inference offload is not a training fallback")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise LowMemoryExperimentError("no trainable LoRA parameters")
    model.train()
    before = _snapshot_lora_b(model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    torch.cuda.synchronize()
    load_peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated())
    load_peak_cuda_reserved_bytes = int(torch.cuda.max_memory_reserved())
    # Reset only after persisting the load + adapter-preparation high-water mark.
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(10, 10_000, (1, 64), device="cuda")
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids)).loss
    if loss is None or not torch.isfinite(loss):
        raise LowMemoryExperimentError("nonfinite or absent synthetic loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    if not torch.isfinite(gradient_norm):
        raise LowMemoryExperimentError("nonfinite synthetic gradient norm")
    optimizer.step()
    torch.cuda.synchronize()
    after = _snapshot_lora_b(model)
    step_peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated())
    step_peak_cuda_reserved_bytes = int(torch.cuda.max_memory_reserved())
    witness = evaluate_optimizer_witness(
        before=before, after=after, loss=float(loss.item()),
        gradient_norm=float(gradient_norm.item()), optimizer_step_returned=True)
    witness.update(summarize_cuda_phase_peaks(
        load_allocated=load_peak_cuda_allocated_bytes,
        load_reserved=load_peak_cuda_reserved_bytes,
        step_allocated=step_peak_cuda_allocated_bytes,
        step_reserved=step_peak_cuda_reserved_bytes))
    witness.update(actual_cpu_offload_used=False, checkpoint_written=False,
                   eval_access=False)
    return witness


if __name__ == "__main__":
    # Pure source-only CLI: no physical operation switch exists.
    if len(sys.argv) != 1:
        print("TRAININGDAY HARD HOLD: execution arguments unsupported", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(source_plan(), sort_keys=True))
