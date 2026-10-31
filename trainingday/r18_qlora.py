"""R18 measured QLoRA arm for the official, pinned Qwen3.5-4B base.

This is deliberately separate from the R16 runner.  Fit probes use synthetic
tokens in fresh processes, never held-out material or a training case.  The
final adapter is initialized in another fresh process, so a probe optimizer
step cannot contaminate the benchmark.  No smaller model or CPU/disk offload
fallback is permitted after a 4B out-of-memory result.

Source-only tests need neither torch nor a model snapshot.  GPU execution must
be explicitly invoked by the Lappy operator after a frozen R18 plan exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file
from trainingday.labyrinth_intake import admit_labyrinth
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT
from trainingday.train_neural import (
    R16_PRODUCER_HEAD, R16_V2_MANIFEST_SHA256, _control_rows,
    _parameter_digest, _required_base,
)


BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_HEAD = "9a6c3b72ef8ad425fa4df371ee10cf700972c8cd"
RECIPE = "VERA_R18_QLORA_NF4_ALL_LINEAR_FP32_ADAMW8BIT_V1"
PLAN_SCHEMA = "VERA_TRAININGDAY_R18_BENCHMARK_PLAN_V1"
RECEIPT_SCHEMA = "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1"
SEQUENCES = (64, 128, 256, 384, 512, 768, 1024)
EFFECTIVE_BATCH = 8
FIT_POLICY = "ASCENDING_LENGTH_BATCH1_THEN_BATCH2_UNTIL_FIRST_OOM"
EVAL_SOURCE = "R18_NEW_SEALED_NOT_R16_FRESH64"
PROBE_SCHEMA = "VERA_TRAININGDAY_R18_FIT_PROBE_V1"


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def selected_id_digest(rows: list[dict]) -> str:
    ids = [row["id"] for row in rows]
    if (not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise AdmissionError("R18 selected training IDs missing or duplicated")
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def train_content_digest(rows: list[dict]) -> str:
    """Bind uncorrupted admitted training material independently of arm labels."""
    return canonical_sha([{
        "id": row["id"],
        "prompt_sha256": hashlib.sha256(
            " ".join(unicodedata.normalize("NFKC", row["prompt"]).split())
            .casefold().encode("utf-8")).hexdigest(),
        "answer_sha256": hashlib.sha256(row["answer"].encode("utf-8")).hexdigest(),
    } for row in rows])


def candidate_schedule() -> list[dict]:
    def item(length: int, batch: int) -> dict:
        return {"actual_padded_sequence_length": length, "micro_batch_size": batch,
                "effective_batch_size": EFFECTIVE_BATCH, "offload": False,
                "truncation_allowed": False}
    return [item(length, batch) for batch in (1, 2) for length in SEQUENCES]


def _validate_fit_plan(plan: dict) -> None:
    if (plan.get("fit_policy") != FIT_POLICY
            or plan.get("fit_candidates") != candidate_schedule()
            or plan.get("eval_bank_source") != EVAL_SOURCE):
        raise AdmissionError("R18 fit policy, ascending candidates, or sealed bank changed")
    primary = plan.get("primary_training_config")
    if (not isinstance(primary, dict)
            or primary.get("actual_padded_sequence_length") not in SEQUENCES
            or primary.get("micro_batch_size") != 1
            or primary.get("effective_batch_size") != EFFECTIVE_BATCH
            or primary.get("offload") is not False
            or primary.get("truncation_allowed") is not False
            or set(primary) != {"actual_padded_sequence_length", "micro_batch_size",
                                "effective_batch_size", "offload", "truncation_allowed"}):
        raise AdmissionError("R18 primary matched training config missing/invalid")


def _valid_seconds(value) -> bool:
    return type(value) in (int, float) and 0 < value <= 86400


def next_step_guard_seconds(completed_step_seconds: list[float]) -> float:
    """Reserve enough time for an entire GA8 optimizer update, not a microbatch.

    An initial 120-second floor uses R16's ~66-second GA8 observation with a
    substantial cushion. Thereafter the guard is 1.5 times the *slowest*
    completed step plus two seconds, with a 30-second absolute floor. It is a
    conservative predictor, not a guarantee against an unexpected slow step:
    any actual deadline breach remains a failed, non-comparable run.
    """
    if type(completed_step_seconds) is not list:
        raise ValueError("R18 step guard requires a primitive completed-step list")
    if not completed_step_seconds:
        return 120.0
    if any(not _valid_seconds(x) for x in completed_step_seconds):
        raise ValueError("R18 step guard requires finite positive bounded completed-step timings")
    return max(30.0, 1.5 * max(completed_step_seconds) + 2.0)


def may_start_full_step(remaining_seconds: float,
                        completed_step_seconds: list[float]) -> bool:
    """A partial GA8 update is never an admitted use of the time budget."""
    if (type(remaining_seconds) not in (int, float)
            or not (-86400 <= remaining_seconds <= 86400)):
        raise ValueError("R18 remaining budget must be a finite bounded number")
    return remaining_seconds >= next_step_guard_seconds(completed_step_seconds)


def effective_batch_token_loss_scale(items: list[dict],
                                     effective_batch: list[dict]) -> float:
    """Weight a microbatch mean by its share of supervised completion tokens.

    The model's causal-LM loss is the mean over supervised target tokens in
    this microbatch. The paired MeZO arm uses the mean across *all* target
    tokens in the effective eight-example update. Scaling each microbatch
    loss by this fraction produces the same objective independent of GA size.
    """
    if not items or not effective_batch:
        raise ValueError("R18 target token counts require nonempty batches")
    counts = [row.get("target_tokens") for row in effective_batch]
    selected_counts = [row.get("target_tokens") for row in items]
    if any(type(n) is not int or n <= 0 for n in counts + selected_counts):
        raise ValueError("R18 target token counts must be positive integers")
    denominator = sum(counts)
    numerator = sum(selected_counts)
    if numerator > denominator:
        raise ValueError("R18 target token microbatch exceeds effective batch")
    return numerator / denominator


def _initial_adapter_custody(args) -> dict:
    directory = getattr(args, "initial_adapter", None)
    model_sha = getattr(args, "initial_adapter_model_sha256", None)
    config_sha = getattr(args, "initial_adapter_config_sha256", None)
    expected_digest = getattr(args, "expected_initial_digest", None)
    if expected_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise AdmissionError("R18 FP32 initial adapter digest malformed")
    if args.control != "zero" and expected_digest is None:
        raise AdmissionError("R18 trained arms require paired zero FP32 initial digest")
    if directory is None:
        if any(x is not None for x in (model_sha, config_sha)):
            raise AdmissionError("R18 initial adapter custody pins without path")
        return {"initialization_method": "FRESH_PINNED_SEED_VERIFY_FP32_DIGEST",
                "initial_adapter_model_sha256": None,
                "initial_adapter_config_sha256": None}
    directory = Path(directory)
    if (not directory.is_dir() or directory.is_symlink()
            or not isinstance(model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha)
            or not isinstance(config_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", config_sha)):
        raise AdmissionError("R18 physically shared adapter custody invalid")
    for name, pin in (("adapter_model.safetensors", model_sha),
                      ("adapter_config.json", config_sha)):
        path = directory / name
        if not path.is_file() or path.is_symlink() or _hash_file(path) != pin:
            raise AdmissionError("R18 initial adapter physical SHA custody mismatch")
    return {"initialization_method": "PHYSICAL_R16_ZERO_ADAPTER_FP32_DIGEST",
            "initial_adapter_model_sha256": model_sha,
            "initial_adapter_config_sha256": config_sha}


def preflight(args) -> tuple[dict, list[dict], Path, dict]:
    """Admit only the R16 v2 training selection; never open an eval shard."""
    if args.control not in {"frozen", "zero", "correct", "corrupt"}:
        raise AdmissionError("QLoRA arm needs frozen/zero/correct/corrupt control")
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", args.run_id):
        raise AdmissionError("invalid R18 run ID")
    if (not args.output_root.is_dir() or args.output_root.is_symlink()
            or (args.output_root / args.run_id).exists()
            or (args.output_root / args.run_id).is_symlink()):
        raise AdmissionError("R18 output namespace occupied/unsafe")
    if (_hash_file(args.manifest) != R16_V2_MANIFEST_SHA256
            or args.producer_git_head != R16_PRODUCER_HEAD):
        raise AdmissionError("R18 needs the exact R16 v2 train corpus/producer")
    if type(args.cases) is not int or not 8 <= args.cases <= 1024 or args.cases % 8:
        raise AdmissionError("R18 case count must be 8–1024, divisible by eight")
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise AdmissionError("R18 seed invalid")
    if not 0 < args.learning_rate <= 0.0002:
        raise AdmissionError("R18 learning rate outside bounded pilot")
    initial_custody = _initial_adapter_custody(args)
    base, base_path = _required_base(args.base_binding)
    if base["model_id"] != BASE_ID or base["revision"] != BASE_REVISION:
        raise AdmissionError("R18 requires official exact Qwen3.5-4B revision")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise AdmissionError("R18 frozen benchmark plan absent/invalid")
    _validate_fit_plan(plan)
    if (plan.get("base_model") != BASE_ID or plan.get("base_revision") != BASE_REVISION
            or plan.get("base_binding_sha256") != _hash_file(args.base_binding)
            or plan.get("tokenizer_binding_sha256") != base["files"]["tokenizer_config.json"]
            or plan.get("corpus_manifest_sha256") != R16_V2_MANIFEST_SHA256
            or plan.get("seed") != args.seed
            or plan.get("selected_count") != args.cases
            or not _valid_seconds(plan.get("training_budget_seconds"))
            or not _valid_seconds(plan.get("probe_budget_seconds"))
            or not isinstance(plan.get("tokenizer_binding_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", plan["tokenizer_binding_sha256"])
            or not isinstance(plan.get("eval_custody_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", plan["eval_custody_sha256"])):
        raise AdmissionError("R18 benchmark plan/base/budget mismatch")
    summary, selected = admit_labyrinth(args.manifest, args.work_root,
                                        sample_size=args.cases,
                                        producer_git_head=args.producer_git_head)
    ids_sha = selected_id_digest(selected)
    if ids_sha != plan.get("selected_id_sha256"):
        raise AdmissionError("R18 selected IDs differ from frozen plan")
    if train_content_digest(selected) != plan.get("train_content_sha256"):
        raise AdmissionError("R18 train content differs from frozen plan")
    rows = _control_rows(selected, "zero" if args.control == "frozen" else args.control)
    # IDs are bound to the position in the admitted selection.  Control labels
    # change train-input hash, never sample membership or prompt ordering.
    examples = [{"id": source["id"], **row}
                for source, row in zip(selected, rows, strict=True)]
    train_sha = canonical_sha(examples)
    report = {
        "schema": RECEIPT_SCHEMA, "arm": "qlora", "control": args.control,
        "run_id": args.run_id, "seed": args.seed, "recipe": RECIPE,
        "source_head": SOURCE_HEAD, "question_contract": NUMERIC_QUESTION_CONTRACT,
        **initial_custody,
        "base_model": BASE_ID, "base_revision": BASE_REVISION,
        "base_binding_sha256": _hash_file(args.base_binding),
        "tokenizer_binding_sha256": plan["tokenizer_binding_sha256"],
        "plan_sha256": canonical_sha(plan), "selected_id_sha256": ids_sha,
        "raw_training_input_sha256": train_sha,
        "training_input_sha256": None,
        "corpus_manifest_sha256": R16_V2_MANIFEST_SHA256,
        "selected_count": args.cases,
        "train_content_sha256": plan["train_content_sha256"],
        "source_admission_status": plan["source_admission_status"],
        "eval_set_sha256": plan["eval_custody_sha256"],
        "eval_custody_sha256": plan["eval_custody_sha256"],
        "eval_index_sha256": plan["eval_index_sha256"],
        "training_budget_seconds": plan["training_budget_seconds"],
        "probe_budget_seconds": plan["probe_budget_seconds"],
        "fit_policy": FIT_POLICY, "fit_candidates": plan["fit_candidates"],
        "primary_training_config": plan["primary_training_config"],
        "optimizer": "adamw_bnb_8bit_nonpaged", "gradient_accumulation": None,
        "lora_r": 4, "lora_alpha": 16, "lora_dropout": 0,
        "lora_target_modules": "all-linear", "quantization": "NF4_DOUBLE_BF16",
        "model_training_mode": True, "gradient_update_objective": "completion_only_causal_lm",
        "effective_batch_loss_weighting": "SUPERVISED_TARGET_TOKEN_MEAN",
        "activation_checkpointing": True, "cpu_offload": False,
        "disk_offload": False, "truncation_allowed": False,
        "eval_accuracy": None, "eval_nll": None,
        "eval_nll_sum": None, "eval_case_receipt_sha256": None,
        "eval_correct": None,
        "eval_examples": 0, "eval_target_tokens": 0,
        "status": "SOURCE_ONLY_PREFLIGHT", "fit_status": "NOT_RUN",
        "optimizer_steps": 0, "unique_examples_seen": 0,
        "example_presentations": 0, "examples_seen": 0,
        "input_tokens_presented": 0, "tokens_seen": 0,
        "forward_sample_tokens": 0, "backward_sample_tokens": 0,
        "padded_forward_tokens": 0, "padded_backward_tokens": 0,
        "supervised_target_tokens": 0, "optimizer_window_seconds": 0.0,
        "timed_training_seconds": 0.0,
        "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
        "peak_cpu_rss_bytes": 0, "peak_nvml_used_bytes": 0,
        "trainable_digest_before": None, "trainable_digest_after": None,
    }
    return report, examples, base_path, plan


def _tensorize(tokenizer, examples: list[dict]) -> list[dict]:
    from successor.experiments.train_v10_qwen35_authorized import _generation_prefix

    eos = tokenizer.eos_token
    if not eos:
        raise RuntimeError("pinned tokenizer missing EOS token")
    result = []
    for row in examples:
        prompt = _generation_prefix(tokenizer, row["prompt"])
        head = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        whole = tokenizer(prompt + row["response"] + eos,
                          add_special_tokens=False)["input_ids"]
        if whole[:len(head)] != head:
            raise RuntimeError("R18 prompt/completion token boundary is not stable")
        tail = whole[len(head):]
        if not head or not tail:
            raise RuntimeError("empty R18 prompt or supervised completion")
        result.append({"id": row["id"], "input_ids": whole,
                       "labels": [-100] * len(head) + tail,
                       "target_tokens": len(tail)})
    return result


def _pad_batch(examples: list[dict], seq: int, pad_id: int, torch):
    if any(len(x["input_ids"]) > seq for x in examples):
        raise RuntimeError("R18 no-truncation fit condition violated")
    ids = []
    labels = []
    masks = []
    for row in examples:
        count = seq - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * count)
        labels.append(row["labels"] + [-100] * count)
        masks.append([1] * len(row["input_ids"]) + [0] * count)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device="cuda"),
            "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device="cuda")}


def _load_model(base_path: Path, seed: int, lr: float, *, initial_adapter=None):
    import torch
    import transformers
    import psutil
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    from bitsandbytes.optim import AdamW8bit
    from successor.experiments.r12_adapter_precision import verify_trainable
    from successor.experiments.train_v10_qwen35_authorized import validate_lora_target_coverage

    if (psutil.virtual_memory().available < 9 * 2**30 or not torch.cuda.is_available()
            or not torch.cuda.is_bf16_supported()
            or torch.cuda.get_device_properties(0).total_memory < 3900 * 2**20):
        raise RuntimeError("R18 Lappy BF16/3900MiB VRAM/9GiB RAM gate not met")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 3200 * 2**20:
        raise RuntimeError("R18 less than 3200MiB free VRAM: do not disrupt other work")
    transformers.set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("pinned tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base_path, quantization_config=quant, device_map=0,
        dtype=torch.bfloat16, local_files_only=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    adapter = LoraConfig(r=4, lora_alpha=16, lora_dropout=0, bias="none",
                         task_type="CAUSAL_LM", target_modules="all-linear")
    if initial_adapter is None:
        model = get_peft_model(model, adapter)
    else:
        model = PeftModel.from_pretrained(
            model, Path(initial_adapter), is_trainable=True, local_files_only=True)
    precision = verify_trainable(model, torch, permit_cast=True)
    names = list(getattr(model, "targeted_module_names", []) or [])
    if not names:
        # A separately loaded PEFT adapter does not always expose the
        # targeted_module_names convenience attribute.
        names = [name for name, module in model.named_modules()
                 if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
    coverage = validate_lora_target_coverage(names)
    # Never accept an automatic CPU/disk offload map: bitsandbytes NF4 LoRA
    # updates here are explicitly single-GPU, with checkpointed activations.
    if any(param.device.type != "cuda" for param in model.parameters()):
        raise RuntimeError("R18 model/LoRA partially CPU-offloaded")
    placement = getattr(model, "hf_device_map", None) or getattr(model.base_model, "hf_device_map", {})
    if any(str(v).lower() in {"cpu", "disk"} for v in placement.values()):
        raise RuntimeError("R18 CPU/disk offload not an admitted fallback")
    model.train()
    optimizer = AdamW8bit((p for p in model.parameters() if p.requires_grad), lr=lr)
    return torch, tokenizer, model, optimizer, precision, coverage


def _cpu_peak() -> int:
    import psutil
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _cuda_peak(torch) -> tuple[int, int]:
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return ("out of memory" in message or "cublas_status_alloc_failed" in message
            or "cuda error: memory allocation" in message)


def probe_worker(args) -> dict:
    """A single synthetic, full-tensor-length optimizer step in a child process."""
    result = {"max_seq_length": args.probe_seq, "micro_batch_size": args.probe_batch,
              "actual_tensor_shape": [args.probe_batch, args.probe_seq],
              "probe_is_synthetic": True, "status": "ERROR"}
    torch = None
    started = time.perf_counter()
    try:
        base, path = _required_base(args.base_binding)
        if base["model_id"] != BASE_ID or base["revision"] != BASE_REVISION:
            raise AdmissionError("R18 exact official base pin invalid")
        if args.initial_adapter is not None:
            args.control = "zero"  # synthetic probe has no train control arm
            _initial_adapter_custody(args)
        torch, tokenizer, model, optimizer, _, _ = _load_model(
            path, args.seed, args.learning_rate, initial_adapter=args.initial_adapter)
        torch.cuda.reset_peak_memory_stats()
        pad = tokenizer.eos_token_id
        if pad is None:
            raise RuntimeError("EOS ID missing")
        synthetic = [{"id": f"probe-{i}", "input_ids": [pad] * args.probe_seq,
                      "labels": [-100] * (args.probe_seq - 2) + [pad, pad]}
                     for i in range(args.probe_batch)]
        batch = _pad_batch(synthetic, args.probe_seq, pad, torch)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite synthetic probe loss")
        loss.backward()
        optimizer.step()  # Exercise actual nonpaged 8-bit optimizer allocation.
        optimizer.zero_grad(set_to_none=True)
        allocated, reserved = _cuda_peak(torch)
        result.update(status="FIT", peak_cuda_allocated_bytes=allocated,
                      peak_cuda_reserved_bytes=reserved, peak_cpu_rss_bytes=_cpu_peak(),
                      failure_artifact_sha256=None)
    except Exception as exc:
        result.update(status="OOM" if _is_oom(exc) else "ERROR",
                      error_type=type(exc).__name__, error_summary=str(exc)[:350])
        failure = args.probe_result.with_suffix(".failure.txt")
        failure.write_text(traceback.format_exc(), encoding="utf-8")
        result["failure_artifact_sha256"] = _hash_file(failure)
        if torch is not None and torch.cuda.is_available():
            try:
                result["peak_cuda_allocated_bytes"], result["peak_cuda_reserved_bytes"] = _cuda_peak(torch)
            except Exception:
                pass
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result.setdefault("peak_cpu_rss_bytes", _cpu_peak())
    result.setdefault("peak_cuda_allocated_bytes", 0)
    result.setdefault("peak_cuda_reserved_bytes", 0)
    result["fresh_process_id"] = f"{os.getpid()}-{time.time_ns()}"
    # NVML is optional on the Windows Lappy environment. Zero here means
    # unavailable, never an invented nvidia-smi peak observation.
    result["peak_nvml_used_bytes"] = 0
    result["nvml_peak_status"] = "UNAVAILABLE_NOT_ESTIMATED"
    args.probe_result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def probe_sweep(args, output: Path, plan: dict) -> tuple[list[dict], dict | None, str]:
    """Two ascending B1/B2 phases, fresh worker for every hardware attempt."""
    probes = []
    best = None
    b1_highest = None
    for batch in (1, 2):
        for index, seq in enumerate(SEQUENCES):
            if batch == 2 and (b1_highest is None or seq > b1_highest):
                break
            candidate_index = index + (0 if batch == 1 else len(SEQUENCES))
            artifact = output / f"probe-{candidate_index:02d}.json"
            command = [sys.executable, "-m", "trainingday.r18_qlora", "_probe_child",
                       "--base-binding", str(args.base_binding), "--seed", str(args.seed),
                       "--learning-rate", str(args.learning_rate),
                       "--probe-seq", str(seq), "--probe-batch", str(batch),
                       "--probe-result", str(artifact)]
            if getattr(args, "initial_adapter", None) is not None:
                command += ["--initial-adapter", str(args.initial_adapter),
                            "--initial-adapter-model-sha256", args.initial_adapter_model_sha256,
                            "--initial-adapter-config-sha256", args.initial_adapter_config_sha256]
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=plan["probe_budget_seconds"], check=False)
            except subprocess.TimeoutExpired:
                probes.append({"config": candidate_schedule()[candidate_index],
                               "candidate_index": candidate_index, "status": "TIMEOUT"})
                return probes, best, "PROBE_BUDGET_EXHAUSTED"
            if not artifact.is_file():
                probes.append({"config": candidate_schedule()[candidate_index],
                               "candidate_index": candidate_index,
                               "status": "PROCESS_FAILED", "exit_code": completed.returncode})
                return probes, best, "PROBE_PROCESS_FAILED"
            probe = json.loads(artifact.read_text(encoding="utf-8"))
            probe.update(schema=PROBE_SCHEMA, plan_sha256=canonical_sha(plan), arm="qlora",
                         config=candidate_schedule()[candidate_index],
                         candidate_index=candidate_index,
                         actual_padded_sequence_length=seq,
                         outcome=probe.get("status"), eval_access=False,
                         data_partition="TRAIN_ONLY", checkpoint_selection=False,
                         optimizer_step_completed=probe.get("status") == "FIT")
            probe.setdefault("fresh_process_id", f"synthetic-{candidate_index}")
            probe.setdefault("peak_nvml_used_bytes", 0)
            probe.setdefault("failure_artifact_sha256", None)
            probe["elapsed_seconds"] = max(0.001, float(probe.get("elapsed_seconds", 0.001)))
            probes.append(probe)
            if completed.returncode != 0 or probe["status"] not in {"FIT", "OOM"}:
                return probes, best, "PROBE_PROCESS_FAILED"
            if probe["status"] == "OOM":
                if batch == 1 and best is None:
                    return probes, None, "BASE_4B_QLORA_OOM"
                break
            if batch == 1:
                b1_highest = seq
            chosen = {"max_seq_length": seq, "micro_batch_size": batch,
                      "effective_batch_size": EFFECTIVE_BATCH,
                      "gradient_accumulation": EFFECTIVE_BATCH // batch}
            if (best is None or seq * batch > best["max_seq_length"] * best["micro_batch_size"]):
                best = chosen
    return probes, best, "FIT" if best else "BASE_4B_QLORA_OOM"


def _training_loop(args, output: Path, examples: list[dict], chosen: dict,
                   receipt: dict) -> dict:
    import torch
    from successor.experiments.r12_adapter_precision import verify_trainable

    load_started = time.perf_counter()
    torch, tokenizer, model, optimizer, precision, coverage = _load_model(
        _required_base(args.base_binding)[1], args.seed, args.learning_rate,
        initial_adapter=getattr(args, "initial_adapter", None))
    receipt["cold_model_and_optimizer_load_seconds"] = round(
        time.perf_counter() - load_started, 3)
    setup_started = time.perf_counter()
    converted = _tensorize(tokenizer, examples)
    receipt["training_input_sha256"] = canonical_sha([
        {"id": x["id"], "input_ids": x["input_ids"], "labels": x["labels"]}
        for x in converted
    ])
    seq = chosen["max_seq_length"]
    if max(map(lambda x: len(x["input_ids"]), converted)) > seq:
        raise RuntimeError("selected max-fit length cannot hold full supervised examples")
    batch_size = chosen["micro_batch_size"]
    accumulation = chosen["gradient_accumulation"]
    if batch_size * accumulation != 8:
        raise RuntimeError("R18 effective batch size is not eight")
    before, count = _parameter_digest(model, torch)
    initial_fp32 = {name: param.detach().to(device="cpu", dtype=torch.float32).clone()
                    for name, param in model.named_parameters() if param.requires_grad}
    receipt.update(trainable_digest_before=before, trainable_parameter_count=count,
                   fp32_before=precision, target_coverage=coverage,
                   gradient_accumulation=accumulation, micro_batch_size=batch_size,
                   max_seq_length=seq,
                   max_untruncated_example_tokens=max(len(r["input_ids"]) for r in converted),
                   max_prepared_example_tokens=max(len(r["input_ids"]) for r in converted))
    if args.expected_initial_digest and before != args.expected_initial_digest:
        raise RuntimeError("R18 initial FP32 adapter digest differs from paired zero control")
    torch.cuda.reset_peak_memory_stats()
    setup_elapsed = time.perf_counter() - setup_started
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + round(receipt["training_budget_seconds"] * 1e9)
    unique = set()
    steps = presentations = input_tokens = target_tokens = tensor_tokens = 0
    completed_step_seconds = []
    stop_reason = "ZERO_UPDATE_CONTROL" if args.control in {"frozen", "zero"} else "UNEXPECTED_ABORT"
    try:
        if args.control not in {"frozen", "zero"}:
            for offset in range(0, len(converted), EFFECTIVE_BATCH):
                remaining_seconds = (deadline_ns - time.perf_counter_ns()) / 1e9
                if not may_start_full_step(remaining_seconds, completed_step_seconds):
                    stop_reason = "INSUFFICIENT_REMAINING_FOR_SAFE_FULL_STEP"
                    break
                step_started_ns = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)
                effective_rows = converted[offset:offset + EFFECTIVE_BATCH]
                if len(effective_rows) != EFFECTIVE_BATCH:
                    raise RuntimeError("R18 effective batch is incomplete")
                for index in range(offset, offset + EFFECTIVE_BATCH, batch_size):
                    items = converted[index:index + batch_size]
                    if len(items) != batch_size or any(r["id"] in unique for r in items):
                        raise RuntimeError("R18 would repeat or incompletely present a train case")
                    batch = _pad_batch(items, seq, tokenizer.pad_token_id, torch)
                    loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite R18 completion-only loss")
                    # Exactly match MeZO's completion-token-weighted objective.
                    # Unweighted 1/GA would overemphasize short completions.
                    scale = effective_batch_token_loss_scale(items, effective_rows)
                    (loss * scale).backward()
                    for item in items:
                        unique.add(item["id"])
                        presentations += 1
                        input_tokens += len(item["input_ids"])
                        target_tokens += item["target_tokens"]
                        tensor_tokens += seq
                optimizer.step()
                # CUDA work is asynchronous; timing must cover a *completed*
                # backward/optimizer update before predicting the next one.
                torch.cuda.synchronize()
                completed_step_seconds.append(
                    (time.perf_counter_ns() - step_started_ns) / 1e9)
                steps += 1
            else:
                stop_reason = "UNIQUE_TRAINING_POOL_EXHAUSTED"
        torch.cuda.synchronize()
    finally:
        stop_ns = time.perf_counter_ns()
        elapsed = (stop_ns - started_ns) / 1e9
        overshoot = max(0., elapsed - receipt["training_budget_seconds"])
        tolerance = max(30., receipt["training_budget_seconds"] * 0.05)
        receipt.update(optimizer_steps=steps, unique_examples_seen=len(unique),
                       example_presentations=presentations, examples_seen=presentations,
                       input_tokens_presented=input_tokens, tokens_seen=input_tokens,
                       forward_sample_tokens=input_tokens,
                       backward_sample_tokens=input_tokens,
                       padded_forward_tokens=tensor_tokens,
                       padded_backward_tokens=tensor_tokens,
                       supervised_target_tokens=target_tokens,
                       optimizer_window_seconds=round(elapsed, 3),
                       timed_training_seconds=round(elapsed, 3),
                       training_budget_overshoot_seconds=round(overshoot, 3),
                       training_budget_unused_seconds=round(
                           max(0., receipt["training_budget_seconds"] - elapsed), 3),
                       training_budget_tolerance_seconds=round(tolerance, 3),
                       training_budget_within_tolerance=(overshoot <= tolerance),
                       budget_stop_reason=stop_reason,
                       completed_step_seconds=[round(x, 3) for x in completed_step_seconds],
                       next_step_guard_seconds=round(
                           next_step_guard_seconds(completed_step_seconds), 3),
                       step_guard_policy="INITIAL_120S_THEN_1P5X_SLOWEST_PLUS_2S_MIN30S",
                       measured_training_window_excludes_load_and_fit=True,
                       train_start_monotonic_ns=started_ns,
                       train_stop_monotonic_ns=stop_ns,
                       train_deadline_monotonic_ns=deadline_ns)
    if stop_ns > deadline_ns:
        raise RuntimeError("R18 strict equal-time deadline exceeded by completed step")
    if args.control in {"correct", "corrupt"} and steps == 0:
        raise RuntimeError("R18 fail-closed: no complete optimizer step fits training budget")
    after, count_after = _parameter_digest(model, torch)
    delta_squared = 0.
    delta_max = 0.
    delta_elements = 0
    for name, param in model.named_parameters():
        if name not in initial_fp32:
            continue
        delta = param.detach().to(device="cpu", dtype=torch.float32) - initial_fp32[name]
        delta_squared += float(delta.double().square().sum().item())
        delta_max = max(delta_max, float(delta.abs().max().item()))
        delta_elements += int(torch.count_nonzero(delta).item())
    if (count != count_after or (before == after) != (args.control in {"frozen", "zero"})
            or (args.control in {"correct", "corrupt"} and steps == 0)):
        raise RuntimeError("R18 no actual FP32 LoRA delta or invalid zero control")
    after_precision = verify_trainable(model, torch, permit_cast=False)
    # CUDA's peak counters were reset after model/optimizer setup above.
    # These are the primary training process's counters, never the peaks
    # from the earlier isolated synthetic fit-frontier workers.
    allocated, reserved = _cuda_peak(torch)
    receipt.update(status="TRAINED_UNSCORED",
                   fit_status="FIT", optimizer_steps=steps,
                   trainable_digest_after=after, fp32_after=after_precision,
                   fp32_delta_l2=delta_squared ** 0.5,
                   fp32_delta_max_abs=delta_max,
                   fp32_changed_elements=delta_elements,
                   peak_cuda_allocated_bytes=allocated,
                   peak_cuda_reserved_bytes=reserved,
                   peak_cuda_metric_scope="PRIMARY_TRAIN_PROCESS_AFTER_SETUP_RESET",
                   # Windows peak_wset is process lifetime, not a resettable
                   # optimizer-window CPU RSS measurement.
                   peak_cpu_rss_bytes=_cpu_peak(),
                   peak_cpu_rss_metric_scope="PROCESS_LIFETIME_PEAK_OR_CURRENT_RSS",
                   peak_nvml_metric_scope="UNAVAILABLE_NOT_ESTIMATED",
                   adapter_delta_observed=(before != after),
                   runtime={"torch": torch.__version__})
    adapter_path = output / "adapter"
    save_started = time.perf_counter()
    model.save_pretrained(adapter_path, safe_serialization=True)
    for name, field in (("adapter_model.safetensors", "adapter_model_sha256"),
                        ("adapter_config.json", "adapter_config_sha256")):
        target = adapter_path / name
        if not target.is_file():
            raise RuntimeError("R18 adapter save incomplete")
        receipt[field] = _hash_file(target)
    receipt["stage_seconds"] = {
        "load": receipt["cold_model_and_optimizer_load_seconds"],
        "setup": round(setup_elapsed, 3),
        "train": receipt["optimizer_window_seconds"],
        "save": round(time.perf_counter() - save_started, 3), "eval": 0.,
    }
    return receipt


def run(args) -> dict:
    receipt, examples, _, plan = preflight(args)
    if args.preflight_only:
        return receipt
    wall_started = time.perf_counter()
    output = args.output_root / args.run_id
    output.mkdir(exist_ok=False)
    (output / "r18-plan-sha256.txt").write_text(receipt["plan_sha256"] + "\n")
    try:
        probes, chosen, fit_status = probe_sweep(args, output, plan)
        primary = plan["primary_training_config"]
        primary_seq = primary["actual_padded_sequence_length"]
        primary_fit = any(p.get("status") == "FIT"
                          and p.get("max_seq_length") == primary_seq
                          and p.get("micro_batch_size") == 1 for p in probes)
        receipt.update(fit_probes=probes, fit_probe_sha256=canonical_sha(probes),
                       fit_status=fit_status, secondary_max_fit=chosen,
                       selected_fit_config=primary,
                       primary_fit_observed=primary_fit,
                       fit_probe_peak_cuda_allocated_bytes=max(
                           (p.get("peak_cuda_allocated_bytes", 0) for p in probes), default=0),
                       fit_probe_peak_cuda_reserved_bytes=max(
                           (p.get("peak_cuda_reserved_bytes", 0) for p in probes), default=0),
                       fit_probe_peak_cpu_rss_bytes=max(
                           (p.get("peak_cpu_rss_bytes", 0) for p in probes), default=0),
                       fit_probe_memory_scope="ISOLATED_SYNTHETIC_FRESH_PROCESSES")
        if not primary_fit:
            receipt["status"] = "OOM_NO_TRAINING" if "OOM" in fit_status else "NOT_TRAINED_FIT_INCOMPLETE"
            receipt["fit_status"] = "PRIMARY_CONFIG_NOT_FIT"
        else:
            try:
                training_config = {"max_seq_length": primary_seq,
                                   "micro_batch_size": 1, "effective_batch_size": 8,
                                   "gradient_accumulation": 8}
                receipt = _training_loop(args, output, examples, training_config, receipt)
                receipt["secondary_fit_status"] = fit_status
            except Exception as exc:
                receipt.update(status="OOM_INCOMPLETE_TRAINING" if _is_oom(exc) else "TRAINING_FAILED",
                               fit_status="TRAINING_OOM" if _is_oom(exc) else "TRAINING_ERROR",
                               error_type=type(exc).__name__, error_summary=str(exc)[:350])
    except Exception as exc:
        receipt.update(status="NOT_TRAINED_FIT_INCOMPLETE", fit_status="PROBE_ERROR",
                       error_type=type(exc).__name__, error_summary=str(exc)[:350])
    receipt["total_benchmark_wall_seconds"] = round(time.perf_counter() - wall_started, 3)
    receipt["external_process_wall_seconds"] = receipt["total_benchmark_wall_seconds"]
    (output / "r18-qlora-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="R18 honest 4B QLoRA max-fit/benchmark")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "_probe_child"])
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--control", choices=["frozen", "zero", "correct", "corrupt"])
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--producer-git-head", default=R16_PRODUCER_HEAD)
    parser.add_argument("--expected-initial-digest")
    parser.add_argument("--initial-adapter", type=Path,
                        help="optional physically shared, SHA-pinned R16 zero adapter")
    parser.add_argument("--initial-adapter-model-sha256")
    parser.add_argument("--initial-adapter-config-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--probe-seq", type=int)
    parser.add_argument("--probe-batch", type=int)
    parser.add_argument("--probe-result", type=Path)
    args = parser.parse_args()
    if args.command == "_probe_child":
        if (args.probe_seq not in SEQUENCES or args.probe_batch not in (1, 2)
                or args.probe_result is None):
            parser.error("invalid internal probe settings")
        result = probe_worker(args)
    else:
        if any(x is None for x in (args.manifest, args.work_root, args.output_root,
                                  args.plan, args.run_id, args.control)):
            parser.error("R18 run requires manifest/work-root/output-root/plan/run-id/control")
        result = run(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
