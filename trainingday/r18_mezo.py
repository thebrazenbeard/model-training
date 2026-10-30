"""R18 experimental MeZO-style *LoRA-only* zeroth-order benchmark.

This is not full-parameter MeZO: the official Qwen3.5-4B base remains frozen
under NF4/double-quantization; only FP32 LoRA tensors are perturbed/updated.
Every update measures paired +/- epsilon losses on the SAME effective batch
under one seeded Gaussian direction, restores the original parameters, then
applies an actual weight update. No autograd/gradient surrogate is used.

Import/preflight is CPU-only. A CUDA run requires explicit operator authority;
this module does not pause services, reserve GPUs, or invoke evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import re
import time
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

from trainingday.admission import AdmissionError, _canonical_prompt, _hash_file, admit
from trainingday.labyrinth_intake import SCHEMA as LABYRINTH_SCHEMA, admit_labyrinth
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT, numeric_question
from trainingday.train_neural import (BASE_ID, R16_PRODUCER_HEAD,
                                     R16_V2_MANIFEST_SHA256, _required_base)


SCHEMA = "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1"
RECIPE = "R18_MEZO_STYLE_FP32_LORA_ZO_ON_FROZEN_NF4_QWEN35_4B_V1"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[a-z0-9_-]{1,64}\Z")
CHECKPOINT = re.compile(r"checkpoint-([1-9][0-9]*)\Z")
PROBE_SCHEMA = "VERA_TRAININGDAY_R18_FIT_PROBE_V1"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def selected_id_digest(rows: list[dict]) -> str:
    """The *existing R16 admission-order* hash, not a sorted-ID rewrite."""
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode("utf-8")).hexdigest()


def selected_rows(rows: list[dict], mode: str) -> list[dict]:
    by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[row["family"]].add(row["answer"])
    result = []
    for row in rows:
        answer = row["answer"]
        if mode == "corrupt":
            alternatives = sorted(by_family[row["family"]] - {answer})
            if not alternatives:
                raise AdmissionError("corrupt control requires different answers in every family")
            answer = alternatives[int(hashlib.sha256(row["id"].encode()).hexdigest(), 16)
                                  % len(alternatives)]
        result.append({"id": row["id"], "prompt": numeric_question(row["prompt"]),
                       "response": answer})
    return result


def _positive_int(value: object, name: str, ceiling: int) -> None:
    if type(value) is not int or not 1 <= value <= ceiling:
        raise AdmissionError(f"{name} outside positive bound 1–{ceiling}")


def _validate_prereg_plan(args, summary: dict, base: dict, train_content_sha: str) -> dict:
    """Never start a GPU without the standalone custodian's complete frozen plan."""
    try:
        from trainingday.r18_benchmark import validate_plan
    except ImportError as exc:
        raise AdmissionError("R18 benchmark plan validator must be integrated first") from exc
    if not args.plan.is_file() or args.plan.is_symlink():
        raise AdmissionError("immutable R18 preregistered plan missing")
    plan = _read_json(args.plan)
    plan_sha = validate_plan(plan)
    expected = {
        "base_model": BASE_ID, "base_revision": BASE_REVISION,
        "base_binding_sha256": _hash_file(args.base_binding),
        "tokenizer_binding_sha256": base["files"]["tokenizer_config.json"],
        "corpus_manifest_sha256": summary["manifest_sha256"],
        "selected_id_sha256": summary["selected_id_sha256"],
        "selected_count": args.cases,
        "train_content_sha256": train_content_sha,
        "source_admission_status": summary["status"],
        "seed": args.seed, "training_budget_seconds": args.budget_seconds,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise AdmissionError("R18 source/base/seed/time differs from immutable plan")
    shape = plan["primary_training_config"]
    if (shape["micro_batch_size"] != args.micro_batch_size
            or shape["effective_batch_size"] != args.examples_per_update
            or shape["actual_padded_sequence_length"] != args.max_seq_length):
        raise AdmissionError("R18 primary tensor shape differs from frozen plan")
    return {**plan, "_validated_plan_sha256": plan_sha}


def validate_source(args) -> tuple[dict, list[dict], Path]:
    """All source gates run before torch/transformers imports or GPU allocation."""
    if args.mode not in {"zero", "correct", "corrupt"}:
        raise AdmissionError("unsupported control")
    if type(args.steps) is not int or not 0 <= args.steps <= 4096:
        raise AdmissionError("steps outside bound 0–4096")
    if (args.mode == "zero") != (args.steps == 0):
        raise AdmissionError("zero arm requires zero updates; trained arms require steps")
    _positive_int(args.cases, "cases", 1024)
    _positive_int(args.micro_batch_size, "micro batch", 16)
    _positive_int(args.examples_per_update, "examples per update", 1024)
    if args.examples_per_update % args.micro_batch_size:
        raise AdmissionError("examples per update must divide into complete microbatches")
    _positive_int(args.max_seq_length, "sequence length", 2048)
    _positive_int(args.seed, "seed", 2**31 - 1)
    _positive_int(args.checkpoint_every, "checkpoint interval", 4096)
    if (not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 86400):
        raise AdmissionError("time budget must be 0–86400 seconds")
    if (not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1.0
            or not math.isfinite(args.epsilon) or not 1e-7 <= args.epsilon <= 0.1):
        raise AdmissionError("zeroth-order hyperparameter outside bounds")
    if (not SAFE_ID.fullmatch(args.run_id) or not args.output_root.is_dir()
            or args.output_root.is_symlink() or not args.work_root.is_dir()
            or args.work_root.is_symlink()):
        raise AdmissionError("unsafe run ID or output/work root")
    if (args.resume_checkpoint is None) != (args.resume_receipt_sha256 is None):
        raise AdmissionError("resume checkpoint and pinned receipt SHA must be supplied together")
    if args.resume_receipt_sha256 and not HEX64.fullmatch(args.resume_receipt_sha256):
        raise AdmissionError("resume receipt SHA malformed")
    if args.fit_sweep_only and args.resume_checkpoint:
        raise AdmissionError("fit sweep never resumes weights")
    if args.preflight_only and args.fit_sweep_only:
        raise AdmissionError("choose source preflight or GPU fit sweep")
    if not args.resume_checkpoint and (args.output_root / args.run_id).exists():
        raise AdmissionError("run output exists; never silently overwrite")
    if not args.resume_checkpoint and (args.output_root / args.run_id).is_symlink():
        raise AdmissionError("symlinked run output")
    if (args.expected_initial_digest is not None
            and not HEX64.fullmatch(args.expected_initial_digest)):
        raise AdmissionError("expected initial adapter digest malformed")
    if ((args.initial_adapter is None) != (args.initial_adapter_sha256 is None)
            or (args.initial_adapter is None) != (args.initial_adapter_config_sha256 is None)):
        raise AdmissionError("initial R16 zero adapter requires both external file SHA pins")
    if args.initial_adapter_sha256 and not HEX64.fullmatch(args.initial_adapter_sha256):
        raise AdmissionError("initial adapter SHA malformed")
    if (args.initial_adapter_config_sha256
            and not HEX64.fullmatch(args.initial_adapter_config_sha256)):
        raise AdmissionError("initial adapter config SHA malformed")
    if args.fit_probe_sha256 and not HEX64.fullmatch(args.fit_probe_sha256):
        raise AdmissionError("fit probe SHA malformed")
    if args.initial_adapter is not None:
        weights = args.initial_adapter / "adapter_model.safetensors"
        config = args.initial_adapter / "adapter_config.json"
        if (not args.initial_adapter.is_dir() or args.initial_adapter.is_symlink()
                or not weights.is_file() or weights.is_symlink()
                or not config.is_file() or config.is_symlink()
                or _hash_file(weights) != args.initial_adapter_sha256
                or _hash_file(config) != args.initial_adapter_config_sha256):
            raise AdmissionError("externally pinned R16 zero adapter missing or tampered")
    if args.mode != "zero" and not args.expected_initial_digest:
        raise AdmissionError("trained arm needs externally pinned zero-arm initial digest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") == LABYRINTH_SCHEMA:
        summary, rows = admit_labyrinth(args.manifest, args.work_root,
                                        sample_size=args.cases,
                                        producer_git_head=args.producer_git_head)
    else:
        summary, rows = admit(args.manifest, args.work_root, sample_size=args.cases)
    if selected_id_digest(rows) != summary["selected_id_sha256"]:
        raise AdmissionError("selected ID hash disagrees with full source admission")
    if len({row["id"] for row in rows}) != len(rows):
        raise AdmissionError("duplicate selected ID")
    cases = selected_rows(rows, args.mode)
    train_content_sha = canonical_hash([{
        "id": row["id"],
        "prompt_sha256": hashlib.sha256(_canonical_prompt(row["prompt"])).hexdigest(),
        "answer_sha256": hashlib.sha256(row["answer"].encode("utf-8")).hexdigest(),
    } for row in rows])
    base, model_path = _required_base(args.base_binding)
    if base["revision"] != BASE_REVISION:
        raise AdmissionError("R18 base must match the official R16 pinned revision")
    if (_hash_file(args.manifest) != R16_V2_MANIFEST_SHA256
            or args.producer_git_head != R16_PRODUCER_HEAD):
        raise AdmissionError("R18 requires exact PR140 R16 v2 corpus and producer")
    plan = _validate_prereg_plan(args, summary, base, train_content_sha)
    preflight = {
        "schema": SCHEMA, "recipe": RECIPE, "status": "SOURCE_ONLY_PREFLIGHT",
        "arm": "mezo", "control": args.mode, "run_id": args.run_id,
        "seed": args.seed, "base_model": BASE_ID, "base_revision": base["revision"],
        "base_binding_sha256": _hash_file(args.base_binding),
        "corpus": summary, "selected_id_sha256": summary["selected_id_sha256"],
        "train_content_sha256": train_content_sha,
        "corpus_manifest_sha256": summary["manifest_sha256"],
        "source_admission_status": summary["status"],
        "plan_sha256": plan["_validated_plan_sha256"],
        "eval_custody_sha256": plan["eval_custody_sha256"],
        "eval_index_sha256": plan["eval_index_sha256"],
        "training_budget_seconds": plan["training_budget_seconds"],
        # Match R16's exact prompt/response input byte contract, not a new
        # serialization carrying per-case IDs (those are attested separately).
        "training_input_sha256": hashlib.sha256(json.dumps(
            [{"prompt": case["prompt"], "response": case["response"]} for case in cases],
            sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "selected_cases_sha256": canonical_hash(cases),
        "initial_adapter_model_sha256": args.initial_adapter_sha256,
        "initial_adapter_config_sha256": args.initial_adapter_config_sha256,
        "question_contract": NUMERIC_QUESTION_CONTRACT,
        "selected_count": len(cases), "model_method": "MeZO-style LoRA-only ZO; frozen NF4 base",
        "steps_cap": args.steps, "micro_batch_size": args.micro_batch_size,
        "examples_per_update": args.examples_per_update,
        "max_seq_length": args.max_seq_length, "learning_rate": args.learning_rate,
        "epsilon": args.epsilon, "budget_seconds": args.budget_seconds,
        "selected_fit_config": {
            "micro_batch_size": args.micro_batch_size,
            "actual_padded_sequence_length": args.max_seq_length,
            "effective_batch_size": args.examples_per_update,
            "offload": False, "truncation_allowed": False,
        },
    }
    return preflight, cases, model_path


def direction_seed(seed: int, step: int) -> int:
    """Independent of run ID: paired control arms receive same directions."""
    digest = hashlib.sha256(f"R18-MEZO-ZO\0{seed}\0{step}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def finite_difference(plus: float, minus: float, epsilon: float) -> float:
    if not all(math.isfinite(x) for x in (plus, minus, epsilon)) or epsilon <= 0:
        raise RuntimeError("nonfinite zeroth-order loss or epsilon")
    estimate = (plus - minus) / (2.0 * epsilon)
    if not math.isfinite(estimate):
        raise RuntimeError("nonfinite zeroth-order gradient estimate")
    return estimate


def trainables(model, torch) -> list[tuple[str, object]]:
    tensors = [(name, p) for name, p in sorted(model.named_parameters()) if p.requires_grad]
    if not tensors or any(p.dtype != torch.float32 or p.device.type != "cuda"
                          for _, p in tensors):
        raise RuntimeError("MeZO LoRA-only requires nonempty FP32 CUDA trainable tensors")
    if any("lora_" not in name.lower() for name, _ in tensors):
        raise RuntimeError("non-LoRA parameter unexpectedly trainable")
    return tensors


def _noise_delta(tensors, torch, seed: int, scale: float) -> None:
    """Regenerate one direction in stable tensor order; never hold full noise."""
    generator = torch.Generator(device="cuda:0")
    generator.manual_seed(seed)
    with torch.no_grad():
        for _, p in tensors:
            z = torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype)
            p.add_(z, alpha=scale)


def _snapshot_trainables(tensors):
    """CPU-side exact recovery image; never keep extra 4GB GPU copies."""
    return {name: p.detach().to(device="cpu", copy=True).contiguous()
            for name, p in tensors}


def _restore_trainables(tensors, torch, saved) -> None:
    with torch.no_grad():
        for name, p in tensors:
            # Direct host-to-device copy avoids an extra CUDA adapter tensor.
            p.copy_(saved[name])


def paired_zo_step(tensors, torch, *, seed: int, epsilon: float,
                   learning_rate: float, loss_fn) -> tuple[float, float, float]:
    """Exact two-sided finite-difference update; restoration on loss failure.

    The loss_fn must evaluate the same effective sample set under each sign
    and return target-token-weighted cross entropy. It does NOT call backward.
    """
    original = _snapshot_trainables(tensors)
    try:
        _noise_delta(tensors, torch, seed, epsilon)
        plus = float(loss_fn())
        _noise_delta(tensors, torch, seed, -2.0 * epsilon)
        minus = float(loss_fn())
    finally:
        # Byte-exact FP32 restoration, including if a perturbation partially
        # failed (e.g. CUDA OOM). Replayed +/- noise alone can round-drift.
        _restore_trainables(tensors, torch, original)
    derivative = finite_difference(plus, minus, epsilon)
    if derivative == 0.0:
        raise RuntimeError("zero directional difference; no real update attested")
    _noise_delta(tensors, torch, seed, -learning_rate * derivative)
    return plus, minus, derivative


def tokenize_cases(tokenizer, cases: list[dict], max_length: int) -> tuple[list[dict], dict]:
    from successor.experiments.train_v10_qwen35_authorized import _generation_prefix

    eos = tokenizer.eos_token
    if not eos or tokenizer.pad_token_id is None:
        raise RuntimeError("pinned tokenizer missing EOS/padding")
    prepared = []
    lengths, targets = [], []
    for case in cases:
        prefix = _generation_prefix(tokenizer, case["prompt"])
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        tail_ids = tokenizer(case["response"] + eos,
                             add_special_tokens=False)["input_ids"]
        full_ids = prefix_ids + tail_ids
        target_count = len(tail_ids)
        if target_count < 1 or len(full_ids) > max_length:
            raise RuntimeError("completion missing or token budget exceeded; no truncation")
        prepared.append({"id": case["id"], "input_ids": full_ids,
                         "labels": [-100] * len(prefix_ids) + full_ids[len(prefix_ids):],
                         "target_tokens": target_count, "input_tokens": len(full_ids)})
        lengths.append(len(full_ids))
        targets.append(target_count)
    return prepared, {"max_input_tokens": max(lengths), "min_input_tokens": min(lengths),
                      "target_tokens_per_cycle": sum(targets),
                      "input_tokens_per_cycle": sum(lengths)}


def step_rows(prepared: list[dict], step: int, examples_per_update: int) -> list[dict]:
    return [prepared[(step * examples_per_update + offset) % len(prepared)]
            for offset in range(examples_per_update)]


def pack_batch(torch, tokenizer, rows: list[dict], length: int) -> dict:
    inputs, masks, labels = [], [], []
    for row in rows:
        pad = length - len(row["input_ids"])
        if pad < 0:
            raise RuntimeError("fit candidate below verified sequence length")
        inputs.append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
        masks.append([1] * len(row["input_ids"]) + [0] * pad)
        labels.append(row["labels"] + [-100] * pad)
    return {"input_ids": torch.tensor(inputs, dtype=torch.long, device="cuda:0"),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device="cuda:0"),
            "labels": torch.tensor(labels, dtype=torch.long, device="cuda:0")}


def paired_batch_loss(model, torch, tokenizer, selected: list[dict],
                      batch_size: int, max_length: int) -> float:
    """Per-target-token weighted loss over the whole effective update batch."""
    total_weighted_loss = 0.0
    total_targets = 0
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            rows = selected[offset:offset + batch_size]
            tensors = pack_batch(torch, tokenizer, rows, max_length)
            target_count = sum(r["target_tokens"] for r in rows)
            output = model(**tensors, use_cache=False)
            loss = float(output.loss.detach().float().item())
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite forward loss")
            total_weighted_loss += loss * target_count
            total_targets += target_count
            del output, tensors
    if total_targets <= 0:
        raise RuntimeError("no supervised target tokens")
    return total_weighted_loss / total_targets


def parameter_attestation(tensors, torch, initial=None):
    digest = hashlib.sha256()
    changed, delta_sq, total = 0, 0.0, 0
    capture = {} if initial is None else None
    for name, p in tensors:
        value = p.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        total += value.numel()
        if initial is None:
            capture[name] = value.clone()
        else:
            old = initial[name]
            if old.shape != value.shape:
                raise RuntimeError("initial LoRA tensor shape changed")
            changed += int(torch.count_nonzero(value != old).item())
            delta_sq += float(torch.sum((value - old).double().square()).item())
    return {"digest": digest.hexdigest(), "parameter_count": total,
            "changed_elements": changed, "delta_l2": math.sqrt(delta_sq)}, capture


def frozen_parameter_digest(model, torch) -> tuple[str, int]:
    """Hash every frozen tensor's *actual bytes* in bounded CPU chunks.

    This captures packed NF4 parameter storage as well as frozen embeddings,
    including tensor names, shapes and dtypes. It does not imply that all
    non-parameter quantizer metadata are independently hash-covered.
    """
    digest = hashlib.sha256()
    total_bytes = 0
    frozen = sorted((name, p) for name, p in model.named_parameters()
                    if not p.requires_grad)
    if not frozen:
        raise RuntimeError("frozen base parameters missing")
    for name, p in frozen:
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(p.shape)).encode("ascii"))
        digest.update(str(p.dtype).encode("ascii"))
        flat = p.detach().reshape(-1)
        stride = max(1, (8 * 2**20) // flat.element_size())
        for offset in range(0, flat.numel(), stride):
            cpu = flat[offset:offset + stride].to(device="cpu").contiguous()
            chunk = cpu.view(torch.uint8).numpy().tobytes()
            digest.update(chunk)
            total_bytes += len(chunk)
            del cpu, chunk
    return digest.hexdigest(), total_bytes


def _read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError("invalid checkpoint JSON") from exc
    if not isinstance(result, dict):
        raise AdmissionError("checkpoint JSON not an object")
    return result


def _run_contract(preflight: dict, args) -> dict:
    return {"schema": "VERA_TRAININGDAY_R18_MEZO_RUN_CONTRACT_V1",
            "preflight": preflight, "checkpoint_every": args.checkpoint_every,
            "expected_initial_digest": args.expected_initial_digest}


def prepare_run(args, preflight: dict) -> tuple[Path, dict, int, float, dict | None]:
    """Resume only externally pinned safetensors; no Python pickle deserialization."""
    output = args.output_root / args.run_id
    contract = _run_contract(preflight, args)
    contract_path = output / "run-contract.json"
    if not args.resume_checkpoint:
        if output.exists() or output.is_symlink():
            raise AdmissionError("run output already occupied")
        output.mkdir(exist_ok=False)
        with contract_path.open("x", encoding="utf-8") as stream:
            json.dump(contract, stream, sort_keys=True, indent=2)
        return output, contract, 0, 0.0, None
    checkpoint = args.resume_checkpoint
    if (not output.is_dir() or output.is_symlink() or not contract_path.is_file()
            or contract_path.is_symlink() or _read_json(contract_path) != contract
            or not checkpoint.is_dir() or checkpoint.is_symlink()
            or checkpoint.parent != output):
        raise AdmissionError("resume path/contract unsafe or changed")
    match = CHECKPOINT.fullmatch(checkpoint.name)
    if not match:
        raise AdmissionError("resume checkpoint name invalid")
    step = int(match.group(1))
    if not 0 < step < args.steps:
        raise AdmissionError("resume step outside training cap")
    state_path = checkpoint / "state.json"
    weights = checkpoint / "lora-trainables.safetensors"
    if (not state_path.is_file() or state_path.is_symlink() or not weights.is_file()
            or weights.is_symlink() or _hash_file(state_path) != args.resume_receipt_sha256):
        raise AdmissionError("resume state missing or externally pinned SHA mismatched")
    state = _read_json(state_path)
    if (state.get("schema") != "R18_MEZO_STATE_V1"
            or state.get("run_contract_sha256") != canonical_hash(contract)
            or state.get("completed_steps") != step
            or state.get("weights_sha256") != _hash_file(weights)
            or state.get("status") != "CHECKPOINT_BYTES_PINNED_NOT_RELOADED"):
        raise AdmissionError("resume state/weight custody mismatch")
    elapsed = state.get("optimizer_window_seconds")
    if type(elapsed) not in (int, float) or not 0 <= elapsed <= args.budget_seconds:
        raise AdmissionError("resume wall-time budget malformed")
    return output, contract, step, float(elapsed), state


def save_checkpoint(output: Path, tensors, torch, contract: dict,
                    step: int, elapsed: float, counters: dict) -> dict:
    from safetensors.torch import save_file

    directory = output / f"checkpoint-{step}"
    if directory.exists() or directory.is_symlink():
        raise RuntimeError("checkpoint path already exists; refuse overwrite")
    directory.mkdir(exist_ok=False)
    weights = directory / "lora-trainables.safetensors"
    save_file({name: p.detach().to(device="cpu", dtype=torch.float32).contiguous()
               for name, p in tensors}, str(weights))
    state = {"schema": "R18_MEZO_STATE_V1",
             "status": "CHECKPOINT_BYTES_PINNED_NOT_RELOADED",
             "run_contract_sha256": canonical_hash(contract),
             "completed_steps": step, "optimizer_window_seconds": elapsed,
             "weights_sha256": _hash_file(weights), "counters": counters}
    state_path = directory / "state.json"
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, sort_keys=True, indent=2)
    return {"checkpoint_path": str(directory),
            "checkpoint_state_sha256": _hash_file(state_path),
            "checkpoint_weights_sha256": state["weights_sha256"]}


def restore_checkpoint(tensors, torch, directory: Path, state: dict) -> None:
    from safetensors.torch import load_file

    values = load_file(str(directory / "lora-trainables.safetensors"), device="cpu")
    if set(values) != {name for name, _ in tensors}:
        raise AdmissionError("resume adapter tensor names do not match pinned base")
    with torch.no_grad():
        for name, p in tensors:
            value = values[name]
            if value.shape != p.shape or value.dtype != torch.float32:
                raise AdmissionError("resume FP32 LoRA tensor shape/dtype changed")
            p.copy_(value.to(device=p.device))


def _oom(exc: Exception, torch=None) -> bool:
    message = str(exc).lower()
    return (torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)) or (
        isinstance(exc, RuntimeError) and
        ("cuda out of memory" in message or "cublas_status_alloc_failed" in message
         or "cuda error: memory allocation" in message))


def _nvml_used_bytes() -> tuple[int, bool]:
    """Best-effort physical NVIDIA allocation; distinguish unavailable from 0."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        pynvml.nvmlShutdown()
        return used, True
    except (ImportError, Exception):
        return 0, False


def _fit_configurations():
    try:
        from trainingday.r18_benchmark import fit_candidates
    except ImportError as exc:
        raise AdmissionError("R18 fit protocol module not integrated") from exc
    return fit_candidates()


def _probe_child(args) -> dict:
    """One actual synthetic-input ZO step; caller MUST start a fresh process."""
    from trainingday.r18_benchmark import validate_plan

    config = _fit_configurations()[args.probe_index]
    started = time.perf_counter()
    plan = _read_json(args.plan)
    plan_sha = validate_plan(plan)
    if (plan_sha != args.expected_plan_sha256 or args.seed != plan["seed"]
            or args.probe_result.exists() or args.probe_result.is_symlink()
            or args.probe_result.parent.is_symlink()):
        raise AdmissionError("probe plan identity/output path differs from custody")
    base, model_path = _required_base(args.base_binding)
    if (base["model_id"] != BASE_ID or base["revision"] != BASE_REVISION
            or _hash_file(args.base_binding) != plan["base_binding_sha256"]):
        raise AdmissionError("probe official base differs from frozen plan")
    if args.initial_adapter is not None:
        weights = args.initial_adapter / "adapter_model.safetensors"
        config_file = args.initial_adapter / "adapter_config.json"
        if (not args.initial_adapter.is_dir() or args.initial_adapter.is_symlink()
                or not weights.is_file() or weights.is_symlink()
                or not config_file.is_file() or config_file.is_symlink()
                or _hash_file(weights) != args.initial_adapter_sha256
                or _hash_file(config_file) != args.initial_adapter_config_sha256):
            raise AdmissionError("probe initial adapter custody mismatch")
    torch = None
    result = {"schema": PROBE_SCHEMA, "plan_sha256": plan_sha,
              "arm": "mezo", "candidate_index": args.probe_index,
              "config": config, "fresh_process_id": f"pid:{os.getpid()}:nonce:{uuid.uuid4().hex}",
              "data_partition": "TRAIN_ONLY", "eval_access": False,
              "checkpoint_selection": False,
              "actual_padded_sequence_length": config["actual_padded_sequence_length"],
              "tensor_shape": [config["micro_batch_size"], config["actual_padded_sequence_length"]],
              "probe_data_kind": "SYNTHETIC_TEXT_TOKENS_NO_TRAINING_OR_EVAL_ROWS",
              "optimizer_step_completed": False, "outcome": "ERROR"}
    peak_nvml, nvml_available = _nvml_used_bytes()
    try:
        torch, psutil, tokenizer, model = _gpu_stack(args, model_path)
        tensors = trainables(model, torch)
        before, _ = parameter_attestation(tensors, torch)
        length = config["actual_padded_sequence_length"]
        batch = config["micro_batch_size"]
        text_ids = tokenizer("Synthetic VRAM capacity probe. No heldout data.",
                             add_special_tokens=False)["input_ids"]
        if not text_ids:
            raise RuntimeError("synthetic text failed pinned tokenizer encoding")
        repeated = (text_ids * ((length + len(text_ids) - 1) // len(text_ids)))[:length]
        # The physical allocation is [B,L] on EVERY microbatch. These are
        # repeated harmless text tokens with two labels, never heldout prompts.
        synthetic = [{"id": f"synthetic-{i}", "input_ids": repeated,
                      "labels": [-100] * (length - 2) + repeated[-2:],
                      "target_tokens": 2, "input_tokens": length}
                     for i in range(8)]
        torch.cuda.reset_peak_memory_stats(0)
        paired_zo_step(tensors, torch, seed=direction_seed(args.seed, 1),
                       epsilon=args.epsilon, learning_rate=args.learning_rate,
                       loss_fn=lambda: paired_batch_loss(model, torch, tokenizer,
                                                          synthetic, batch, length))
        after, _ = parameter_attestation(tensors, torch)
        if before["digest"] == after["digest"]:
            raise RuntimeError("synthetic fit probe did not change FP32 LoRA weights")
        torch.cuda.synchronize(0)
        result.update({"outcome": "FIT", "optimizer_step_completed": True,
                       "failure_artifact_sha256": None,
                       "paired_forward_calls": 16 // batch,
                       "padded_forward_tokens": 16 * length})
    except Exception as exc:
        outcome = "OOM" if _oom(exc, torch) else "ERROR"
        failure = args.probe_result.with_suffix(".failure.txt")
        with failure.open("x", encoding="utf-8") as stream:
            stream.write(traceback.format_exc())
        result.update(outcome=outcome, failure_artifact_sha256=_hash_file(failure),
                      error_type=type(exc).__name__, error_summary=str(exc)[:512])
    if torch is not None and torch.cuda.is_available():
        try:
            result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            result["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        except Exception:
            pass
    peak_after, nvml_after = _nvml_used_bytes()
    result["peak_nvml_used_bytes"] = max(peak_nvml, peak_after)
    result["nvml_available"] = nvml_available or nvml_after
    result.setdefault("peak_cuda_allocated_bytes", 0)
    result.setdefault("peak_cuda_reserved_bytes", 0)
    result["peak_cpu_rss_bytes"] = int(psutil.Process().memory_info().rss) if torch else 0
    result["elapsed_seconds"] = max(1e-9, time.perf_counter() - started)
    if result["elapsed_seconds"] > plan["probe_budget_seconds"]:
        result["outcome"] = "ERROR"
        result["optimizer_step_completed"] = False
        result["probe_budget_exhausted"] = True
    with args.probe_result.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    return result


def run_fit_sweep(args, preflight: dict) -> dict:
    """14 preregistered candidates, each in its OWN OS process.

    Stop B1 at OOM, then try B2 only at B1-fitting lengths, ascending until
    OOM. Each child must complete an actual symmetric ZO optimizer update.
    """
    configs = _fit_configurations()
    plan = _read_json(args.plan)
    out = args.output_root / args.run_id
    if out.exists() or out.is_symlink():
        raise AdmissionError("fit sweep output occupied")
    out.mkdir(exist_ok=False)
    records = []

    def launch(index: int) -> dict:
        candidate = configs[index]
        result_file = out / f"fit-probe-{index:02d}.json"
        command = [sys.executable, "-m", "trainingday.r18_mezo", "--probe-child",
                   "--manifest", str(args.manifest), "--base-binding", str(args.base_binding),
                   "--plan", str(args.plan), "--work-root", str(args.work_root),
                   "--output-root", str(args.output_root), "--run-id", args.run_id,
                   "--mode", "zero", "--seed", str(args.seed),
                   "--budget-seconds", str(args.budget_seconds),
                   "--learning-rate", str(args.learning_rate), "--epsilon", str(args.epsilon),
                   "--probe-index", str(index), "--probe-result", str(result_file),
                   "--expected-plan-sha256", preflight["plan_sha256"]]
        if args.initial_adapter:
            command += ["--initial-adapter", str(args.initial_adapter),
                        "--initial-adapter-sha256", args.initial_adapter_sha256,
                        "--initial-adapter-config-sha256", args.initial_adapter_config_sha256]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=plan["probe_budget_seconds"] + 2, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"candidate {index} timed out; no FIT/OOM claim") from exc
        if completed.returncode != 0 or not result_file.is_file():
            raise RuntimeError(f"candidate {index} subprocess failed without valid receipt")
        item = _read_json(result_file)
        if (item.get("plan_sha256") != preflight["plan_sha256"]
                or item.get("candidate_index") != index
                or item.get("config") != candidate):
            raise RuntimeError("fresh probe candidate/plan custody mismatch")
        if item.get("outcome") not in {"FIT", "OOM"}:
            raise RuntimeError("fit probe failed without attributable OOM")
        return item

    b1_fits = 0
    for index in range(7):
        record = launch(index)
        records.append(record)
        if record["outcome"] == "OOM":
            break
        b1_fits += 1
    for index in range(7, 7 + b1_fits):
        record = launch(index)
        records.append(record)
        if record["outcome"] == "OOM":
            break
    # The expected validated schedule is a prefix of each phase; never claim
    # full fit when the child failed or the first B1 length cannot fit.
    summary = {"schema": "VERA_TRAININGDAY_R18_FIT_SWEEP_V1",
               "arm": "mezo", "plan_sha256": preflight["plan_sha256"],
               "probes": records, "fit_probe_sha256": canonical_hash(records),
               "status": "FIT_PROBES_COLLECTED_NOT_TRAINED"}
    with (out / "r18-mezo-fit-probes.json").open("x", encoding="utf-8") as stream:
        json.dump(records, stream, sort_keys=True, indent=2)
    return {**preflight, **summary}


def _gpu_stack(args, model_path: Path):
    import psutil
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM, set_seed

    if psutil.virtual_memory().available < 9 * 2**30:
        raise RuntimeError("under 9 GiB available RAM")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native BF16 CUDA unavailable")
    if torch.cuda.get_device_properties(0).total_memory < 3900 * 2**20:
        raise RuntimeError("GPU under R16 3900 MiB physical floor")
    if torch.cuda.mem_get_info(0)[0] < 3200 * 2**20:
        raise RuntimeError("less than R16 3200 MiB torch-visible free VRAM")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("pinned tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, quantization_config=config, device_map=0,
        dtype=torch.bfloat16, local_files_only=True,
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    lora = LoraConfig(r=4, lora_alpha=16, lora_dropout=0, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    if args.initial_adapter is not None:
        # Read only the already source-pinned R16 zero-arm safetensors, not a
        # derivative adapter and never an unsafe pickle checkpoint.
        model = PeftModel.from_pretrained(model, str(args.initial_adapter),
                                         is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, lora)
    # PEFT often initializes adapters as BF16 with a BF16 parent; force FP32.
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name.lower():
                raise RuntimeError("non-LoRA trainable parameter")
            p.data = p.data.to(dtype=torch.float32)
    model.eval()  # fixed dropout and no accidental train-mode stochasticity
    return torch, psutil, tokenizer, model


def run(args) -> dict:
    external_wall_started = time.perf_counter()
    preflight, cases, model_path = validate_source(args)
    if args.preflight_only:
        if args.resume_checkpoint:
            preflight["resume_custody_status"] = "GPU_TOKENIZED_CONTRACT_NOT_YET_VALIDATED"
        return preflight
    if args.fit_sweep_only:
        return run_fit_sweep(args, preflight)
    if not args.fit_probes_file or not args.fit_probe_sha256:
        raise AdmissionError("training needs externally pinned full MeZO fit probe receipt")
    if not args.fit_probes_file.is_file() or args.fit_probes_file.is_symlink():
        raise AdmissionError("MeZO fit probe evidence file absent/symlinked")
    probe_records = json.loads(args.fit_probes_file.read_text(encoding="utf-8"))
    if (not isinstance(probe_records, list)
            or canonical_hash(probe_records) != args.fit_probe_sha256
            or not all(row.get("schema") == PROBE_SCHEMA
                       and row.get("plan_sha256") == preflight["plan_sha256"]
                       and row.get("arm") == "mezo" for row in probe_records)
            or not any(row.get("config") == preflight["selected_fit_config"]
                       and row.get("outcome") == "FIT" for row in probe_records)):
        raise AdmissionError("fit probe plan/padded primary fit not independently pinned")
    if args.resume_checkpoint:
        raise AdmissionError("resumed process cannot satisfy strict single-window R18 budget custody")
    # GPU-heavy load occurs strictly after full manifest, rights and pin gates.
    cold_started = time.perf_counter()
    torch, psutil, tokenizer, model = _gpu_stack(args, model_path)
    stage_load_seconds = time.perf_counter() - cold_started
    stage_setup_started = time.perf_counter()
    prepared, token_report = tokenize_cases(tokenizer, cases, args.max_seq_length)
    preflight["tokenizer_binding_sha256"] = _hash_file(model_path / "tokenizer_config.json")
    preflight["token_report"] = token_report
    preflight["tokenization_sha256"] = canonical_hash([
        {"id": item["id"], "input_ids": item["input_ids"], "labels": item["labels"]}
        for item in prepared])
    preflight["source_training_input_sha256"] = preflight["training_input_sha256"]
    # The R18 comparison contract is the *exact labels/token IDs*, not merely
    # equal Unicode text. This must match the QLoRA R18 runner byte-for-byte.
    preflight["training_input_sha256"] = preflight["tokenization_sha256"]
    tensors = trainables(model, torch)
    before, initial = parameter_attestation(tensors, torch)
    if args.expected_initial_digest and before["digest"] != args.expected_initial_digest:
        raise RuntimeError("fresh LoRA initial digest differs from zero control")
    output, contract, resumed_step, previously_elapsed, state = prepare_run(args, preflight)
    if state is not None:
        restore_checkpoint(tensors, torch, args.resume_checkpoint, state)
    frozen_before, frozen_byte_count = frozen_parameter_digest(model, torch)
    # One unperturbed diagnostic forward is outside the optimizer clock.
    # It is counted separately, never conflated with training presentations.
    diagnostic_rows = step_rows(prepared, resumed_step, args.examples_per_update)
    baseline_started = time.perf_counter()
    unperturbed_loss = paired_batch_loss(model, torch, tokenizer, diagnostic_rows,
                                        args.micro_batch_size, args.max_seq_length)
    unperturbed_loss_seconds = time.perf_counter() - baseline_started
    cold_start_seconds = time.perf_counter() - cold_started
    stage_setup_seconds = time.perf_counter() - stage_setup_started
    counters = {"example_presentations": resumed_step * args.examples_per_update,
                "input_tokens_presented": 0, "supervised_target_tokens": 0,
                "forward_sample_tokens": 0, "padded_forward_tokens": 0,
                "backward_sample_tokens": 0, "padded_backward_tokens": 0,
                "paired_forward_calls": 0, "zo_directions": resumed_step}
    if state:
        counters = dict(state["counters"])
    torch.cuda.reset_peak_memory_stats(0)
    train_start_ns = time.monotonic_ns()
    train_deadline_ns = train_start_ns + round(args.budget_seconds * 1e9)
    start = time.perf_counter()
    completed = resumed_step
    last_checkpoint = None
    sampled_peak_rss = psutil.Process().memory_info().rss
    peak_nvml, nvml_supported = _nvml_used_bytes()
    directional_log = []
    if args.mode != "zero":
        for step in range(resumed_step, args.steps):
            if time.monotonic_ns() >= train_deadline_ns:
                break
            selected = step_rows(prepared, step, args.examples_per_update)
            loss_fn = lambda: paired_batch_loss(model, torch, tokenizer, selected,
                                                args.micro_batch_size, args.max_seq_length)
            plus, minus, derivative = paired_zo_step(
                tensors, torch, seed=direction_seed(args.seed, step + 1),
                epsilon=args.epsilon, learning_rate=args.learning_rate, loss_fn=loss_fn,
            )
            torch.cuda.synchronize(0)
            completed = step + 1
            counters["example_presentations"] += len(selected)
            counters["input_tokens_presented"] += sum(r["input_tokens"] for r in selected)
            counters["supervised_target_tokens"] += sum(r["target_tokens"] for r in selected)
            counters["forward_sample_tokens"] += 2 * sum(r["input_tokens"] for r in selected)
            counters["padded_forward_tokens"] += 2 * len(selected) * args.max_seq_length
            counters["paired_forward_calls"] += 2 * len(selected) // args.micro_batch_size
            counters["zo_directions"] += 1
            directional_log.append({"step": completed, "loss_plus": plus,
                                    "loss_minus": minus, "directional_derivative": derivative})
            sampled_peak_rss = max(sampled_peak_rss, psutil.Process().memory_info().rss)
            nvml_current, nvml_current_supported = _nvml_used_bytes()
            peak_nvml = max(peak_nvml, nvml_current)
            nvml_supported = nvml_supported or nvml_current_supported
            if completed % args.checkpoint_every == 0 and completed < args.steps:
                elapsed = previously_elapsed + time.perf_counter() - start
                last_checkpoint = save_checkpoint(output, tensors, torch, contract,
                                                  completed, elapsed, counters)
    train_stop_ns = time.monotonic_ns()
    elapsed = (train_stop_ns - train_start_ns) / 1e9
    if train_stop_ns > train_deadline_ns:
        raise RuntimeError("R18 optimizer update overran strict monotonic budget; no equal-time claim")
    stage_save_started = time.perf_counter()
    after, _ = parameter_attestation(tensors, torch, initial)
    frozen_after, frozen_after_bytes = frozen_parameter_digest(model, torch)
    if frozen_before != frozen_after or frozen_byte_count != frozen_after_bytes:
        raise RuntimeError("frozen NF4 base tensor bytes changed during LoRA-only ZO")
    if (args.mode == "zero") != (before["digest"] == after["digest"]):
        raise RuntimeError("zeroth-order actual adapter delta attestation failed")
    if args.mode != "zero" and not completed > resumed_step:
        raise RuntimeError("no new zeroth-order weight update completed in budget")
    if args.mode != "zero" and (not last_checkpoint or completed != int(
            Path(last_checkpoint["checkpoint_path"]).name.split("-")[1])):
        last_checkpoint = save_checkpoint(output, tensors, torch, contract,
                                          completed, elapsed, counters)
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_config = adapter / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise RuntimeError("LoRA checkpoint was not saved")
    receipt = {**preflight,
               "status": "TRAINED_UNSCORED", "zero_update": args.mode == "zero",
               "fit_status": "fits", "optimizer_steps": completed,
               "new_optimizer_steps": completed - resumed_step,
               "unique_examples_seen": min(len(prepared), counters["example_presentations"]),
               "examples_seen": counters["example_presentations"],
               "tokens_seen": counters["input_tokens_presented"], **counters,
               "timed_training_seconds": elapsed, "optimizer_window_seconds": elapsed,
               "train_start_monotonic_ns": train_start_ns,
               "train_stop_monotonic_ns": train_stop_ns,
               "train_deadline_monotonic_ns": train_deadline_ns,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
               "peak_nvml_used_bytes": peak_nvml,
               "nvml_available": nvml_supported,
               "peak_cpu_rss_bytes": sampled_peak_rss,
               "max_prepared_example_tokens": token_report["max_input_tokens"],
               "cold_start_and_diagnostic_seconds": cold_start_seconds,
               "unperturbed_loss": unperturbed_loss,
               "unperturbed_loss_seconds": unperturbed_loss_seconds,
               "diagnostic_forward_calls": len(diagnostic_rows) // args.micro_batch_size,
               "diagnostic_forward_sample_tokens": sum(r["input_tokens"] for r in diagnostic_rows),
               "diagnostic_padded_forward_tokens": len(diagnostic_rows) * args.max_seq_length,
               "total_forward_sample_tokens_including_diagnostic": (
                   counters["forward_sample_tokens"] +
                   sum(r["input_tokens"] for r in diagnostic_rows)),
               "total_padded_forward_tokens_including_diagnostic": (
                   counters["padded_forward_tokens"] + len(diagnostic_rows) * args.max_seq_length),
               "frozen_parameter_sha256_before": frozen_before,
               "frozen_parameter_sha256_after": frozen_after,
               "frozen_parameter_bytes_hashed": frozen_byte_count,
               "trainable_parameter_count": before["parameter_count"],
               "trainable_digest_before": before["digest"],
               "trainable_digest_after": after["digest"],
               "changed_trainable_elements": after["changed_elements"],
               "actual_weight_delta_l2": after["delta_l2"],
               "resumed_from_step": resumed_step, "checkpoint": last_checkpoint,
               "adapter_model_sha256": _hash_file(adapter_weights),
               "adapter_config_sha256": _hash_file(adapter_config),
               "directional_updates": directional_log,
               "eval_accuracy": None, "eval_nll": None,
               "eval_examples": 0, "eval_target_tokens": 0,
               "eval_nll_sum": None, "eval_case_receipt_sha256": None,
               "eval_set_sha256": None,
               "eval_correct": None, "eval_access": False,
               "arm_freeze_sha256": None, "arm_frozen_at_utc_ns": None,
               "eval_unsealed_at_utc_ns": None,
               "fit_probe_sha256": args.fit_probe_sha256,
               "evidence_limit": ("saved LoRA only; no independent reload or heldout score; "
                                  "NF4 quantizer non-parameter metadata not hashed")}
    stage_save_seconds = time.perf_counter() - stage_save_started
    receipt["stage_seconds"] = {
        "load": stage_load_seconds, "setup": stage_setup_seconds,
        "train": elapsed, "save": stage_save_seconds, "eval": 0.0}
    receipt["external_process_wall_seconds"] = time.perf_counter() - external_wall_started
    with (output / "r18-mezo-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
    return receipt


def parse_int_list(value: str) -> list[int]:
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from exc
    if not result or any(n < 1 for n in result):
        raise argparse.ArgumentTypeError("candidate sizes must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-child", action="store_true",
                        help="Internal single-candidate fresh-process synthetic ZO update")
    parser.add_argument("--probe-index", type=int, default=0)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--producer-git-head", default=None)
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["zero", "correct", "corrupt"], required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--examples-per-update", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--expected-initial-digest", default=None)
    parser.add_argument("--fit-probe-sha256", default=None)
    parser.add_argument("--fit-probes-file", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fit-sweep-only", action="store_true")
    parser.add_argument("--sweep-batches", type=parse_int_list, default=[1])
    parser.add_argument("--sweep-lengths", type=parse_int_list,
                        default=[64, 128, 256, 384, 512, 768, 1024])
    parser.add_argument("--initial-adapter", type=Path,
                        help="Externally pinned R16 zero-update PEFT adapter directory")
    parser.add_argument("--initial-adapter-sha256", help="Exact SHA256 of its adapter_model.safetensors")
    parser.add_argument("--initial-adapter-config-sha256",
                        help="Exact SHA256 of R16 zero adapter_config.json")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-receipt-sha256")
    args = parser.parse_args()
    if args.probe_child:
        if (not args.probe_result or not 0 <= args.probe_index < 14
                or not args.expected_plan_sha256 or not HEX64.fullmatch(args.expected_plan_sha256)):
            parser.error("internal fit child requires exact candidate and plan custody")
        result = _probe_child(args)
    else:
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
