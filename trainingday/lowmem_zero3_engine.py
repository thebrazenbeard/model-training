"""Protected, unqualified BF16 ZeRO-3 CPU-offload Qwen3.5 fit candidate.

The existing host authority HARD HOLD is deliberately the first operation of
the physical entrypoint. This source is not an installer or a bearer-token
verification mechanism. No source flag can grant a GPU/NAS/data lease.

Unlike inference ``device_map='auto'``, this prospective backend uses a real
DeepSpeed training engine. It has not been tested with official Qwen weights,
with WSL process custody, or on Lappy's constrained RAM/GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping

from trainingday.physical_launch_authority import require_physical_launch_authority


BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_SCHEMA = "VERA_TRAININGDAY_ZERO3_CPU_SOURCE_V1"
STEPS = 3
SEQUENCE_LENGTH = 64
MIN_AVAILABLE_RAM_BYTES = 16 * 2**30  # Conservative experiment policy, NOT a fit proof.
MIN_FREE_CUDA_BYTES = 3200 * 2**20
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class Zero3SourceError(ValueError):
    """Invalid source evidence or unqualified fit requirements."""


def zero3_config() -> dict:
    """Explicit single-rank ZeRO-3 BF16 training/CPU offload configuration.

    This is separate from R18's frozen NF4 benchmark recipe. The parameters
    are hypotheses until exact Qwen/PEFT/DeepSpeed versions and real hardware
    pass a forward/backward/save/reload qualification.
    """
    return {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-4, "weight_decay": 0.0}},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": False},
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": 2_000_000,
            "stage3_prefetch_bucket_size": 2_000_000,
            "stage3_param_persistence_threshold": 10_000,
            "stage3_gather_16bit_weights_on_model_save": False,
        },
    }


def source_plan() -> dict:
    """Safe to inspect without PyTorch, a model, network, or CUDA."""
    return {
        "schema": SOURCE_SCHEMA,
        "status": "SOURCE_ONLY_PHYSICAL_HARD_HOLD",
        "model_id": BASE_ID,
        "model_revision": BASE_REVISION,
        "authorizes_training": False,
        "optimizer_steps_observed": 0,
        "steps_requested": STEPS,
        "sequence_length": SEQUENCE_LENGTH,
        "data_access": "SYNTHETIC_ONLY",
        "mode": "BF16_LORA_ZERO3_CPU_PARAM_AND_OPTIMIZER_OFFLOAD_CANDIDATE",
        "candidate_is_qualified": False,
        "min_available_ram_bytes_policy": MIN_AVAILABLE_RAM_BYTES,
        "min_free_cuda_bytes_policy": MIN_FREE_CUDA_BYTES,
        "external_gpu_telemetry_available": False,
        "checkpoint_saved_or_reloaded": False,
    }


def summarize_step_evidence(records: list[Mapping], *, total_cuda_bytes: int,
                            physical_attestation=None) -> dict:
    """Check shape and continuity only; no caller may forge host attestation.

    Passing synthetic dictionaries here never proves real model execution.
    The host-owned broker must independently create and verify a physical
    receipt; this Python function intentionally cannot report that success.
    """
    if physical_attestation is not None:
        raise Zero3SourceError("Python-supplied physical attestation not accepted")
    if (type(records) is not list or len(records) != STEPS
            or type(total_cuda_bytes) is not int or total_cuda_bytes <= 0):
        raise Zero3SourceError("three bounded CUDA step records are required")
    previous_after = None
    for record in records:
        required = {"before_sha256", "after_sha256", "loss", "gradient_norm",
                    "cuda_peak_reserved_bytes", "cuda_peak_allocated_bytes",
                    "rss_peak_bytes", "elapsed_seconds"}
        if type(record) is not dict or set(record) != required:
            raise Zero3SourceError("incomplete or foreign step record")
        before, after = record["before_sha256"], record["after_sha256"]
        if (type(before) is not str or type(after) is not str
                or _HEX64.fullmatch(before) is None or _HEX64.fullmatch(after) is None
                or before == after or (previous_after is not None and before != previous_after)):
            raise Zero3SourceError("adapter state not changed continuously")
        previous_after = after
        for name in ("loss", "gradient_norm", "elapsed_seconds"):
            value = record[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise Zero3SourceError(f"missing finite {name}")
        for name in ("cuda_peak_reserved_bytes", "cuda_peak_allocated_bytes", "rss_peak_bytes"):
            if type(record[name]) is not int or record[name] <= 0:
                raise Zero3SourceError(f"invalid {name}")
        if (record["cuda_peak_allocated_bytes"] > record["cuda_peak_reserved_bytes"]
                or record["cuda_peak_reserved_bytes"] > total_cuda_bytes):
            raise Zero3SourceError("CUDA telemetry exceeds physical capacity")
    return {
        "status": "CONTINUOUS_SYNTHETIC_EVIDENCE_SHAPE_NOT_AUTHENTICATED",
        "step_count": STEPS,
        "adapter_initial_sha256": records[0]["before_sha256"],
        "adapter_final_sha256": records[-1]["after_sha256"],
        "max_cuda_reserved_bytes": max(x["cuda_peak_reserved_bytes"] for x in records),
        "max_rss_bytes": max(x["rss_peak_bytes"] for x in records),
        "physical_training_attested": False,
        "proves_memory_plateau": False,
        "proves_model_quality": False,
    }


def _gather_lora_b_bytes(model, deepspeed) -> dict[str, bytes]:
    """Gather actual ZeRO-3 adapter tensors (single rank, still collective)."""
    tensors = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                tensor = param.detach().float().cpu().contiguous()
                if tensor.numel() == 0:
                    raise Zero3SourceError("empty gathered adapter tensor")
                tensors[name] = tensor.numpy().tobytes()
    if not tensors:
        raise Zero3SourceError("no gathered trainable LoRA-B tensors")
    return tensors


def run_qualified_zero3_fit(base_binding_path):
    """Never runnable until independently installed host-owned authority.

    A future broker must bind *observed* caller identity, exact immutable run
    plan, protected rights/evaluator custody, exclusive GPU lease and process
    revocation before its controlled worker can cross this boundary.
    """
    require_physical_launch_authority(operation="Qwen3.5 BF16 ZeRO-3 synthetic fit")

    # Unreachable while the trusted host authority is not installed.
    import os
    import resource
    import time

    import deepspeed
    import psutil
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM
    from transformers.integrations import HfDeepSpeedConfig

    from trainingday.lowmem_experiment import evaluate_optimizer_witness
    from trainingday.qwen_text_checkpoint import (
        index_binding_pin, prepare_text_loader, verify_text_load,
    )
    from trainingday.train_neural import _required_base

    if (os.name != "posix" or os.environ.get("WORLD_SIZE") != "1"
            or os.environ.get("RANK") != "0" or os.environ.get("LOCAL_RANK") != "0"):
        raise Zero3SourceError("Linux single-rank broker-owned process environment required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise Zero3SourceError("BF16 CUDA unavailable")
    ram_before = int(psutil.virtual_memory().available)
    free_before, cuda_capacity = torch.cuda.mem_get_info()
    if ram_before < MIN_AVAILABLE_RAM_BYTES or free_before < MIN_FREE_CUDA_BYTES:
        raise Zero3SourceError("unqualified CPU RAM or unleased free GPU budget")

    base, snapshot = _required_base(base_binding_path)
    if (base.get("model_id") != BASE_ID or base.get("revision") != BASE_REVISION):
        raise Zero3SourceError("official pinned Qwen3.5-4B required")
    index_sha, shards = index_binding_pin(base)
    text_config, mapping = prepare_text_loader(
        snapshot, expected_index_sha256=index_sha, expected_shards=shards)
    config = zero3_config()
    # HfDeepSpeedConfig must exist before from_pretrained and remain alive;
    # otherwise HF's ZeRO-3-aware checkpoint loading is not activated.
    hf_zero3_config = HfDeepSpeedConfig(config)
    torch.manual_seed(73504)
    torch.cuda.manual_seed_all(73504)
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        snapshot, local_files_only=True, config=text_config, key_mapping=mapping,
        dtype=torch.bfloat16, output_loading_info=True)
    verify_text_load(model, loading_info)
    model.config.use_cache = False
    # Frozen input embeddings plus reentrant checkpointing can lose the
    # adapter gradient path. Use the nonreentrant implementation and retain
    # input gradients explicitly; exact Qwen/PEFT runtime still needs proof.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(p.requires_grad and "lora_" not in n
                            for n, p in model.named_parameters()):
        raise Zero3SourceError("trainable parameter escaped LoRA scope")
    engine, _optimizer, _loader, _scheduler = deepspeed.initialize(
        model=model, model_parameters=trainable, config=config)
    if not getattr(engine, "zero_optimization_stage", lambda: None)() == 3:
        raise Zero3SourceError("DeepSpeed did not activate ZeRO-3")

    engine.train()
    # No packing: Qwen3.5 linear-attention boundaries remain one sequence.
    vocab = int(engine.module.config.vocab_size)
    if vocab <= 12:
        raise Zero3SourceError("invalid pinned tokenizer vocabulary")
    ids = torch.randint(10, min(10_000, vocab),
                        (1, SEQUENCE_LENGTH), device=engine.device)
    mask = torch.ones_like(ids)
    model_load_peak = int(torch.cuda.max_memory_reserved())
    records = []
    for _ in range(STEPS):
        before = _gather_lora_b_bytes(engine.module, deepspeed)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        output = engine(input_ids=ids, labels=ids, attention_mask=mask)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise Zero3SourceError("nonfinite synthetic loss")
        engine.backward(loss)
        # ZeRO-3 owns/partitions gradients; param.grad alone is not reliable.
        grad_norm_squared = 0.0
        for p in trainable:
            grad = deepspeed.utils.safe_get_full_grad(p)
            if grad is not None:
                grad_norm_squared += float(torch.sum(grad.detach().float() ** 2).item())
        grad_norm = math.sqrt(grad_norm_squared)
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise Zero3SourceError("no nonzero gathered adapter gradients")
        engine.step()
        torch.cuda.synchronize()
        after = _gather_lora_b_bytes(engine.module, deepspeed)
        consistency = evaluate_optimizer_witness(
            before=before, after=after, loss=float(loss.item()),
            gradient_norm=grad_norm, optimizer_step_returned=True)
        record = {
            "before_sha256": consistency["before_sha256"],
            "after_sha256": consistency["after_sha256"],
            "loss": float(loss.item()),
            "gradient_norm": grad_norm,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "rss_peak_bytes": max(int(psutil.Process().memory_info().rss),
                                  int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
            "elapsed_seconds": time.monotonic() - started,
        }
        records.append(record)
    summary = summarize_step_evidence(
        records, total_cuda_bytes=int(cuda_capacity), physical_attestation=None)
    summary.update({
        "base_model": BASE_ID, "base_revision": BASE_REVISION,
        "model_loading_peak_reserved_bytes": model_load_peak,
        "host_ram_available_before_bytes": ram_before,
        "host_ram_available_after_bytes": int(psutil.virtual_memory().available),
        "cuda_free_before_bytes": int(free_before),
        "externally_observed_gpu_peak_available": False,
        "cpu_offload_requested": True,
        "cpu_offload_runtime_verified": False,
        "checkpoint_saved_or_reloaded": False,
        "protected_evaluation_access": False,
        "training_data": "SYNTHETIC_ONLY",
    })
    # Keep hf_zero3_config live through load and step; this reference is not a
    # qualification claim and is deliberately not serialized into the result.
    assert hf_zero3_config is not None
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("TRAININGDAY HARD HOLD: physical execution arguments unsupported",
              file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(source_plan(), sort_keys=True))
