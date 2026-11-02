from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Callable


class TrainingHold(RuntimeError):
    """Fail-closed refusal before any weight-changing training work."""


def canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingHold(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingHold(f"JSON artifact must be an object: {path}")
    return value


def assess_preflight(repo_root: Path | str) -> dict:
    from successor.evaluate_successor import assess_v10_experiment_state

    return assess_v10_experiment_state(Path(repo_root))


def gated_training_stack(
    preflight: dict,
    loader: Callable[[], object],
):
    if (
        preflight.get("status") != "READY_PRECONDITIONS"
        or preflight.get("training_allowed") is not True
        or preflight.get("reasons") not in ([], None)
    ):
        reasons = preflight.get("reasons")
        raise TrainingHold(
            "training preflight HOLD"
            + (f": {reasons}" if reasons else "")
        )
    return loader()


def load_training_stack() -> dict:
    """Heavy imports are intentionally isolated behind the READY gate."""
    import torch
    import transformers
    import trl
    import peft
    import bitsandbytes
    import datasets
    import accelerate
    import safetensors
    import huggingface_hub
    import tokenizers
    import jinja2
    import numpy
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        Qwen3_5ForCausalLM,
    )
    from trl import SFTConfig, SFTTrainer

    return {
        "torch": torch,
        "transformers": transformers,
        "trl": trl,
        "peft": peft,
        "bitsandbytes": bitsandbytes,
        "datasets": datasets,
        "accelerate": accelerate,
        "safetensors": safetensors,
        "huggingface_hub": huggingface_hub,
        "tokenizers": tokenizers,
        "jinja2": jinja2,
        "numpy": numpy,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Qwen3_5ForCausalLM": Qwen3_5ForCausalLM,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def load_verified_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> list[dict]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TrainingHold(f"could not read JSONL {path}: {exc}") from exc

    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_sha256:
        raise TrainingHold(
            f"corpus hash mismatch: {actual_sha} != {expected_sha256}"
        )

    rows: list[dict] = []
    try:
        for line_number, line in enumerate(
            raw.decode("utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TrainingHold(
                    f"row {line_number} must be a JSON object"
                )
            prompt = row.get("prompt")
            response = row.get("response")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(response, str)
                or not response.strip()
            ):
                raise TrainingHold(
                    f"row {line_number} missing nonempty prompt/response"
                )
            rows.append(row)
    except UnicodeDecodeError as exc:
        raise TrainingHold(f"JSONL is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TrainingHold(f"invalid JSONL row in {path}: {exc}") from exc

    if len(rows) != expected_rows:
        raise TrainingHold(
            f"corpus row count mismatch: {len(rows)} != {expected_rows}"
        )
    return rows


def _generation_prefix(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def _token_count(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


def prepare_sft_rows(
    tokenizer,
    rows: list[dict],
    *,
    max_length: int,
) -> tuple[list[dict], dict]:
    if max_length < 1:
        raise TrainingHold("max_length must be positive")

    prepared: list[dict] = []
    lengths: list[int] = []
    eos = tokenizer.eos_token or ""
    for index, row in enumerate(rows, start=1):
        prompt = _generation_prefix(tokenizer, row["prompt"])
        completion = row["response"] + eos
        length = _token_count(tokenizer, prompt + completion)
        lengths.append(length)
        prepared.append(
            {
                "prompt": prompt,
                "completion": completion,
            }
        )

    over_budget = sum(length > max_length for length in lengths)
    report = {
        "rows": len(lengths),
        "max_length": max_length,
        "max_tokens": max(lengths, default=0),
        "min_tokens": min(lengths, default=0),
        "over_budget": over_budget,
    }
    if over_budget:
        raise TrainingHold(
            "token budget exceeded: "
            f"max_length={max_length} over_budget={over_budget} "
            f"max_tokens={report['max_tokens']}"
        )
    return prepared, report


def validate_lora_target_coverage(
    targeted_module_names,
) -> dict:
    names = list(targeted_module_names)
    families = {
        "linear_attn": sum(".linear_attn." in name for name in names),
        "self_attn": sum(".self_attn." in name for name in names),
        "mlp": sum(".mlp." in name for name in names),
    }
    if not names or not all(families.values()):
        raise TrainingHold(
            "LoRA coverage incomplete: "
            + json.dumps(families, sort_keys=True)
        )
    if any(
        "vision" in name.casefold() or ".visual" in name.casefold()
        for name in names
    ):
        raise TrainingHold("vision module present in LoRA targets")
    return {
        "status": "PASS",
        "targeted_module_count": len(names),
        "families": families,
    }


def validate_output_namespace(
    output_dir: Path,
    authority: dict,
) -> None:
    expected = authority.get("output_namespace")
    if not isinstance(expected, str) or not expected.strip():
        raise TrainingHold("authority output namespace missing")
    if str(output_dir) != expected:
        raise TrainingHold(
            f"output namespace mismatch: {output_dir} != {expected}"
        )
    if output_dir.exists():
        raise TrainingHold(
            f"output namespace already exists: {output_dir}"
        )


def validate_runtime_observation(
    binding: dict,
    observed: dict,
) -> dict:
    reasons: list[str] = []
    target = binding.get("target")
    if not isinstance(target, dict):
        raise TrainingHold("runtime binding target missing")

    for field in (
        "python_version",
        "gpu",
        "driver",
        "cuda_runtime",
    ):
        if observed.get(field) != target.get(field):
            reasons.append(
                f"runtime target mismatch:{field}:"
                f"{observed.get(field)}!={target.get(field)}"
            )

    observed_vram = observed.get("vram_mib")
    target_vram = target.get("vram_mib")
    if not isinstance(observed_vram, int) or not isinstance(target_vram, int):
        reasons.append(
            f"runtime target mismatch:vram_mib:"
            f"{observed_vram}!={target_vram}"
        )
        vram_delta_mib = None
    else:
        vram_delta_mib = abs(observed_vram - target_vram)
        if vram_delta_mib > 1:
            reasons.append(
                f"runtime target mismatch:vram_mib:"
                f"{observed_vram}!={target_vram}"
            )

    expected_packages = binding.get("packages")
    observed_packages = observed.get("packages")
    if not isinstance(expected_packages, dict):
        raise TrainingHold("runtime binding packages missing")
    if not isinstance(observed_packages, dict):
        raise TrainingHold("runtime observation packages missing")
    for package, expected in sorted(expected_packages.items()):
        actual = observed_packages.get(package)
        if actual != expected:
            reasons.append(
                f"package mismatch:{package}:{actual}!={expected}"
            )

    expected_artifacts = binding.get("base_artifacts")
    observed_artifacts = observed.get("base_artifacts")
    if not isinstance(expected_artifacts, dict):
        raise TrainingHold("runtime binding base_artifacts missing")
    if not isinstance(observed_artifacts, dict):
        raise TrainingHold("runtime observation base_artifacts missing")
    for name, expected in sorted(expected_artifacts.items()):
        actual = observed_artifacts.get(name)
        if actual != expected:
            reasons.append(
                f"base artifact mismatch:{name}:{actual}!={expected}"
            )

    if reasons:
        raise TrainingHold("runtime observation mismatch: " + " | ".join(reasons))
    return {
        "schema": "V10_QWEN35_LIVE_RUNTIME_CHECK_V1",
        "status": "PASS",
        "binding_sha256": binding.get("binding_sha256"),
        "vram_delta_mib": vram_delta_mib,
    }


def default_execution_spec() -> dict:
    return {
        "schema": "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "trainer": {
            "method": "QLORA_SFT_ONLY",
            "seed": 20261001,
            "epochs": 1.0,
            "learning_rate": 2e-5,
            "lr_scheduler_type": "cosine",
            "warmup_optimizer_steps": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 512,
            "overflow_policy": "ERROR_NO_TRUNCATION",
            "optimizer": "adamw_torch",
            "completion_only_loss": True,
            "packing": False,
            "shuffle_dataset": True,
            "gradient_checkpointing": True,
            "bf16": True,
            "tf32": True,
            "gradient_checkpointing_use_reentrant": False,
            "logging_steps": 10,
            "save_strategy": "no",
            "eval_strategy": "no",
            "report_to": "none",
            "validation_role": (
                "POST_TRAIN_DIAGNOSTIC_ONLY_NO_RECIPE_OR_CHECKPOINT_SELECTION"
            ),
        },
        "quantization": {
            "load_in_4bit": True,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "model_load": {
            "device_map": {"": 0},
            "dtype": "bfloat16",
            "use_cache": False,
            "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
        },
        "lora": {
            "r": 4,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "artifact_policy": {
            "fresh_adapter_only": True,
            "parent_adapter": None,
            "save_strategy": "final_adapter_only",
            "validation_checkpoint_selection": False,
        },
    }


def validate_execution_spec(
    contract: dict,
    execution_spec: dict,
) -> dict:
    recipe = contract.get("training_recipe")
    if not isinstance(recipe, dict):
        raise TrainingHold("contract training_recipe missing")

    trainer = execution_spec.get("trainer")
    quant = execution_spec.get("quantization")
    lora = execution_spec.get("lora")
    if not isinstance(trainer, dict):
        raise TrainingHold("execution trainer spec missing")
    if not isinstance(quant, dict):
        raise TrainingHold("execution quantization spec missing")
    if not isinstance(lora, dict):
        raise TrainingHold("execution LoRA spec missing")

    trainer_fields = (
        "method",
        "seed",
        "epochs",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_optimizer_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "overflow_policy",
        "optimizer",
        "completion_only_loss",
        "packing",
        "shuffle_dataset",
        "gradient_checkpointing",
        "validation_role",
    )
    reasons: list[str] = []
    for field in trainer_fields:
        if trainer.get(field) != recipe.get(field):
            reasons.append(
                f"{field}:{trainer.get(field)}!={recipe.get(field)}"
            )

    for field in ("load_in_4bit", "type", "double_quant", "compute_dtype"):
        if quant.get(field) != recipe.get("quantization", {}).get(field):
            reasons.append(
                f"quantization.{field}:"
                f"{quant.get(field)}!="
                f"{recipe.get('quantization', {}).get(field)}"
            )

    for field in ("r", "alpha", "dropout", "target_modules"):
        if lora.get(field) != recipe.get("lora", {}).get(field):
            reasons.append(
                f"lora.{field}:{lora.get(field)}!="
                f"{recipe.get('lora', {}).get(field)}"
            )

    base = contract.get("base_model", {})
    policy = execution_spec.get("artifact_policy", {})
    if policy.get("fresh_adapter_only") is not True:
        reasons.append("fresh_adapter_only")
    if base.get("fresh_adapter_required") is not True:
        reasons.append("contract_fresh_adapter_required")
    if policy.get("parent_adapter") is not None:
        reasons.append("execution_parent_adapter_not_null")
    if base.get("parent_adapter") is not None:
        reasons.append("contract_parent_adapter_not_null")
    if policy.get("validation_checkpoint_selection") is not False:
        reasons.append("validation_checkpoint_selection_not_false")

    exact_trainer = {
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing_use_reentrant": False,
        "logging_steps": 10,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
    }
    for field, expected in exact_trainer.items():
        if trainer.get(field) != expected:
            reasons.append(
                f"{field}:{trainer.get(field)}!={expected}"
            )

    expected_model_load = {
        "device_map": {"": 0},
        "dtype": "bfloat16",
        "use_cache": False,
        "prepare_model_for_kbit_training_use_gradient_checkpointing": True,
    }
    model_load = execution_spec.get("model_load")
    if model_load != expected_model_load:
        reasons.append(
            "model_load:"
            + json.dumps(model_load, sort_keys=True)
            + "!="
            + json.dumps(expected_model_load, sort_keys=True)
        )

    if lora.get("bias") != "none":
        reasons.append(f"lora.bias:{lora.get('bias')}!=none")
    if lora.get("task_type") != "CAUSAL_LM":
        reasons.append(
            f"lora.task_type:{lora.get('task_type')}!=CAUSAL_LM"
        )

    if quant.get("compute_dtype") != model_load.get("dtype", None) if isinstance(model_load, dict) else True:
        reasons.append("model_load.dtype_must_equal_quantization.compute_dtype")

    if reasons:
        raise TrainingHold(
            "execution spec does not match frozen recipe: "
            + " | ".join(reasons)
        )
    return {
        "schema": "V10_QWEN35_EXECUTION_SPEC_CHECK_V1",
        "status": "PASS",
        "execution_spec_sha256": sha256_bytes(
            canonical_bytes(execution_spec)
        ),
    }


def write_run_start_receipt(
    output_dir: Path,
    *,
    authority: dict,
    contract_sha256: str,
    execution_spec_sha256: str,
    runtime_binding_sha256: str,
    sealed_commitment_sha256: str,
    train_sha256: str,
    validation_sha256: str,
    started_at: str,
) -> dict:
    validate_output_namespace(output_dir, authority)
    output_dir.mkdir(parents=True, exist_ok=False)

    receipt = {
        "schema": "V10_QWEN35_RUN_STARTED_V1",
        "status": "RUN_STARTED",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "started_at": started_at,
        "output_namespace": str(output_dir),
        "authority_receipt_sha256": authority.get("receipt_sha256"),
        "contract_sha256": contract_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "sealed_final_bank_commitment_sha256": sealed_commitment_sha256,
        "train_sha256": train_sha256,
        "validation_sha256": validation_sha256,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(receipt))
    path = output_dir / "RUN_STARTED.json"
    try:
        with path.open("xb") as handle:
            handle.write(
                (
                    json.dumps(receipt, indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
            )
    except FileExistsError as exc:
        raise TrainingHold(
            f"run start receipt already exists: {path}"
        ) from exc
    return receipt


def observe_lightweight_runtime(binding: dict) -> dict:
    """Observe package/base-file identity without importing the training stack."""
    target = binding.get("target")
    if not isinstance(target, dict):
        raise TrainingHold("runtime binding target missing")

    packages = {}
    for package in sorted(binding.get("packages", {})):
        distribution = package
        if package == "huggingface_hub":
            distribution = "huggingface-hub"
        try:
            packages[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TrainingHold(
                f"bound runtime package not installed:{package}"
            ) from exc

    base_path = Path(str(target.get("base_path", "")))
    artifacts = {}
    for name in sorted(binding.get("base_artifacts", {})):
        path = base_path / name
        if not path.is_file():
            raise TrainingHold(f"bound base artifact missing:{path}")
        artifacts[name] = sha256_file(path)

    # GPU fields require torch and are intentionally observed only after the
    # preflight READY gate. Keep this lightweight function CPU/import safe.
    return {
        "python_version": platform.python_version(),
        "packages": packages,
        "base_artifacts": artifacts,
    }


def _require_ready_preflight(preflight: dict) -> None:
    if (
        preflight.get("status") != "READY_PRECONDITIONS"
        or preflight.get("training_allowed") is not True
        or preflight.get("reasons") not in ([], None)
    ):
        raise TrainingHold(
            "training preflight HOLD: "
            + json.dumps(
                preflight.get("reasons", []),
                sort_keys=True,
            )
        )


def _observe_live_runtime(
    binding: dict,
    stack: dict,
) -> dict:
    import subprocess

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise TrainingHold("CUDA is not available on authorized runtime")

    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
        ).stdout.splitlines()[0].strip()
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        IndexError,
    ) as exc:
        raise TrainingHold(
            f"could not observe NVIDIA driver version: {exc}"
        ) from exc

    packages: dict[str, str] = {}
    for package in sorted(binding.get("packages", {})):
        distribution = (
            "huggingface-hub"
            if package == "huggingface_hub"
            else package
        )
        try:
            packages[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TrainingHold(
                f"bound runtime package not installed:{package}"
            ) from exc

    target = binding.get("target", {})
    base_path = Path(str(target.get("base_path", "")))
    artifacts: dict[str, str] = {}
    for name in sorted(binding.get("base_artifacts", {})):
        path = base_path / name
        if not path.is_file():
            raise TrainingHold(f"bound base artifact missing:{path}")
        artifacts[name] = sha256_file(path)

    props = torch.cuda.get_device_properties(0)
    return {
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_mib": int(props.total_memory // (1024 * 1024)),
        "driver": driver,
        "cuda_runtime": str(torch.version.cuda),
        "packages": packages,
        "base_artifacts": artifacts,
    }


def _validate_qwen_topology(model) -> dict:
    layer_types = list(getattr(model.config, "layer_types", []))
    if (
        len(layer_types) != 32
        or layer_types.count("linear_attention") != 24
        or layer_types.count("full_attention") != 8
    ):
        raise TrainingHold(
            "unexpected Qwen3.5 topology: "
            f"layers={len(layer_types)} types={layer_types}"
        )

    names = [name for name, _ in model.named_modules()]
    expected_counts = {
        "q_proj": 8,
        "k_proj": 8,
        "v_proj": 8,
        "o_proj": 8,
        "in_proj_qkv": 24,
        "in_proj_z": 24,
        "in_proj_b": 24,
        "in_proj_a": 24,
    }
    counts = {
        suffix: sum(name.endswith(suffix) for name in names)
        for suffix in expected_counts
    }
    for suffix, expected in expected_counts.items():
        if counts[suffix] != expected:
            raise TrainingHold(
                f"module topology mismatch {suffix}: "
                f"{counts[suffix]} != {expected}"
            )
    if any(
        "vision" in name.casefold() or ".visual" in name.casefold()
        for name in names
    ):
        raise TrainingHold("vision modules present in text-only model")
    return {
        "layers": 32,
        "linear_attention": 24,
        "full_attention": 8,
        "module_counts": counts,
    }


def _artifact_hash_manifest(path: Path) -> dict:
    files = {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not files:
        raise TrainingHold(f"adapter artifact directory is empty:{path}")
    return {
        "files": files,
        "manifest_sha256": sha256_bytes(canonical_bytes(files)),
    }


def _write_failure_receipt(
    output_dir: Path,
    *,
    run_start_receipt_sha256: str,
    error: Exception,
) -> None:
    if not output_dir.exists():
        return
    value = {
        "schema": "V10_QWEN35_RUN_FAILED_V1",
        "status": "FAILED_AFTER_RUN_START",
        "run_start_receipt_sha256": run_start_receipt_sha256,
        "error_type": type(error).__name__,
        "error": str(error),
        "retry_authorized": False,
    }
    value["receipt_sha256"] = sha256_bytes(canonical_bytes(value))
    path = output_dir / "RUN_FAILED.json"
    if not path.exists():
        path.write_bytes(
            (
                json.dumps(value, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
        )


def execute_authorized_training(
    repo_root: Path | str,
    *,
    train_jsonl: Path,
    validation_jsonl: Path,
    output_dir: Path,
    training_stack_loader: Callable[[], object] = load_training_stack,
) -> dict:
    # A locally supplied V10 preflight is not physical launch authority.
    # Enforce the host-owned boundary before reading artifacts, touching the
    # output namespace, importing CUDA-capable libraries, or training weights.
    from trainingday.physical_launch_authority import (
        PhysicalLaunchAuthorityUnavailable,
        require_physical_launch_authority,
    )

    try:
        require_physical_launch_authority(operation="V10 Qwen3.5-4B training")
    except PhysicalLaunchAuthorityUnavailable as exc:
        raise TrainingHold(str(exc)) from exc

    root = Path(repo_root)
    preflight = assess_preflight(root)
    _require_ready_preflight(preflight)

    experiment_dir = root / "successor" / "experiments"
    contract_path = (
        experiment_dir / "V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    )
    runtime_path = (
        experiment_dir
        / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    authority_path = (
        experiment_dir / "V10_QWEN35_TRAINING_AUTHORITY_V1.json"
    )
    sealed_path = (
        experiment_dir / "V10_SEALED_FINAL_BANK_COMMITMENT_V1.json"
    )

    contract = _read_json(contract_path)
    runtime_binding = _read_json(runtime_path)
    authority = _read_json(authority_path)
    sealed = _read_json(sealed_path)

    validate_output_namespace(output_dir, authority)
    execution_spec_path = (
        experiment_dir / "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1.json"
    )
    execution_spec = _read_json(execution_spec_path)
    spec_check = validate_execution_spec(contract, execution_spec)

    subject = contract.get("source_subject", {})
    train_rows = load_verified_jsonl(
        train_jsonl,
        expected_sha256=subject.get("train_sha256"),
        expected_rows=int(subject.get("train_rows")),
    )
    validation_rows = load_verified_jsonl(
        validation_jsonl,
        expected_sha256=subject.get("validation_sha256"),
        expected_rows=int(subject.get("validation_rows")),
    )

    stack = gated_training_stack(preflight, training_stack_loader)
    if not isinstance(stack, dict):
        raise TrainingHold("training stack loader returned invalid object")

    live_runtime = _observe_live_runtime(runtime_binding, stack)
    runtime_check = validate_runtime_observation(
        runtime_binding,
        live_runtime,
    )

    target = runtime_binding["target"]
    base_path = Path(target["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = execution_spec["trainer"]["max_length"]
    train_sft, train_budget = prepare_sft_rows(
        tokenizer,
        train_rows,
        max_length=max_length,
    )
    validation_sft, validation_budget = prepare_sft_rows(
        tokenizer,
        validation_rows,
        max_length=max_length,
    )

    torch = stack["torch"]
    quant = execution_spec["quantization"]
    bits_config = stack["BitsAndBytesConfig"](
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(
            torch,
            quant["compute_dtype"],
        ),
    )
    model_load = execution_spec["model_load"]
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        base_path,
        quantization_config=bits_config,
        device_map=model_load["device_map"],
        dtype=getattr(torch, model_load["dtype"]),
    )
    model.config.use_cache = model_load["use_cache"]
    topology = _validate_qwen_topology(model)
    model = stack["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=model_load[
            "prepare_model_for_kbit_training_use_gradient_checkpointing"
        ],
    )

    lora_spec = execution_spec["lora"]
    lora = stack["LoraConfig"](
        r=lora_spec["r"],
        lora_alpha=lora_spec["alpha"],
        lora_dropout=lora_spec["dropout"],
        bias=lora_spec["bias"],
        task_type=lora_spec["task_type"],
        target_modules=lora_spec["target_modules"],
    )

    started_at = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat().replace("+00:00", "Z")
    run_start = write_run_start_receipt(
        output_dir,
        authority=authority,
        contract_sha256=sha256_file(contract_path),
        execution_spec_sha256=spec_check[
            "execution_spec_sha256"
        ],
        runtime_binding_sha256=runtime_binding[
            "binding_sha256"
        ],
        sealed_commitment_sha256=sealed[
            "commitment_sha256"
        ],
        train_sha256=subject["train_sha256"],
        validation_sha256=subject["validation_sha256"],
        started_at=started_at,
    )

    try:
        trainer_spec = execution_spec["trainer"]
        sft_config = stack["SFTConfig"](
            output_dir=str(output_dir / "sft_work"),
            per_device_train_batch_size=trainer_spec[
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=trainer_spec[
                "gradient_accumulation_steps"
            ],
            num_train_epochs=trainer_spec["epochs"],
            learning_rate=trainer_spec["learning_rate"],
            lr_scheduler_type=trainer_spec["lr_scheduler_type"],
            warmup_steps=trainer_spec["warmup_optimizer_steps"],
            optim=trainer_spec["optimizer"],
            bf16=trainer_spec["bf16"],
            tf32=trainer_spec["tf32"],
            gradient_checkpointing=trainer_spec[
                "gradient_checkpointing"
            ],
            gradient_checkpointing_kwargs={
                "use_reentrant": trainer_spec[
                    "gradient_checkpointing_use_reentrant"
                ]
            },
            max_length=trainer_spec["max_length"],
            completion_only_loss=trainer_spec[
                "completion_only_loss"
            ],
            packing=trainer_spec["packing"],
            shuffle_dataset=trainer_spec["shuffle_dataset"],
            logging_steps=trainer_spec["logging_steps"],
            save_strategy=trainer_spec["save_strategy"],
            eval_strategy=trainer_spec["eval_strategy"],
            report_to=trainer_spec["report_to"],
            seed=trainer_spec["seed"],
            data_seed=trainer_spec["seed"],
        )
        trainer = stack["SFTTrainer"](
            model=model,
            args=sft_config,
            train_dataset=stack["Dataset"].from_list(train_sft),
            eval_dataset=stack["Dataset"].from_list(
                validation_sft
            ),
            processing_class=tokenizer,
            peft_config=lora,
        )
        train_result = trainer.train()
        trained_model = trainer.model
        coverage = validate_lora_target_coverage(
            getattr(trained_model, "targeted_module_names", [])
        )

        # Validation is post-train diagnostic only. It never selects a
        # checkpoint, recipe, or rerun.
        validation_metrics = trainer.evaluate()

        for _, parameter in trained_model.named_parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.bfloat16)

        adapter_dir = output_dir / "adapter"
        trained_model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
        artifact_manifest = _artifact_hash_manifest(adapter_dir)

        receipt = {
            "schema": "V10_QWEN35_TRAINING_COMPLETE_V1",
            "status": "TRAINING_COMPLETE_UNQUALIFIED",
            "experiment_id": contract.get("experiment_id"),
            "effect": "ONE_FRESH_QLORA_ADAPTER_TRAINED",
            "authority_receipt_sha256": authority.get(
                "receipt_sha256"
            ),
            "run_start_receipt_sha256": run_start[
                "receipt_sha256"
            ],
            "sealed_final_bank_commitment_sha256": sealed[
                "commitment_sha256"
            ],
            "runtime_binding_sha256": runtime_binding[
                "binding_sha256"
            ],
            "execution_spec_sha256": spec_check[
                "execution_spec_sha256"
            ],
            "train_sha256": subject["train_sha256"],
            "validation_sha256": subject[
                "validation_sha256"
            ],
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_token_budget": train_budget,
            "validation_token_budget": validation_budget,
            "training_loss": float(train_result.training_loss),
            "validation_metrics": validation_metrics,
            "base_topology": topology,
            "lora_coverage": coverage,
            "live_runtime": live_runtime,
            "live_runtime_check": runtime_check,
            "adapter_artifacts": artifact_manifest,
            "qualification_status": "NOT_EVALUATED",
            "deployment_status": "NOT_DEPLOYED",
            "merge_status": "NOT_MERGED",
            "claim_ceiling": (
                "AUTHORIZED_SINGLE_TRAINING_RUN_COMPLETE / "
                "MODEL_BENEFIT_UNPROVEN / FINAL_BANK_NOT_REVEALED_OR_SCORED / "
                "NOT_QUALIFIED / NOT_DEPLOYED"
            ),
        }
        receipt["receipt_sha256"] = sha256_bytes(
            canonical_bytes(receipt)
        )
        (output_dir / "TRAINING_COMPLETE.json").write_bytes(
            (
                json.dumps(receipt, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
        )
        return receipt
    except Exception as exc:
        _write_failure_receipt(
            output_dir,
            run_start_receipt_sha256=run_start[
                "receipt_sha256"
            ],
            error=exc,
        )
        raise


def plan_execution(
    repo_root: Path | str,
    *,
    training_stack_loader: Callable[[], object] = load_training_stack,
) -> dict:
    preflight = assess_preflight(repo_root)
    if (
        preflight.get("status") != "READY_PRECONDITIONS"
        or preflight.get("training_allowed") is not True
    ):
        return {
            "schema": "V10_QWEN35_AUTHORIZED_TRAINING_PLAN_V1",
            "status": "HOLD",
            "training_stack_loaded": False,
            "reasons": list(preflight.get("reasons", [])),
            "preflight": preflight,
            "effect": "READ_ONLY_PLAN_NO_WEIGHT_CHANGE",
        }

    # Source-level preconditions can be READY while the physical process is
    # forbidden. A plan must not load the model stack or advertise READY then.
    from trainingday.physical_launch_authority import (
        PhysicalLaunchAuthorityUnavailable,
        require_physical_launch_authority,
    )

    try:
        require_physical_launch_authority(operation="V10 Qwen3.5-4B training plan")
    except PhysicalLaunchAuthorityUnavailable as exc:
        return {
            "schema": "V10_QWEN35_AUTHORIZED_TRAINING_PLAN_V1",
            "status": "HOLD",
            "training_stack_loaded": False,
            "reasons": [str(exc)],
            "preflight": preflight,
            "effect": "READ_ONLY_PLAN_NO_WEIGHT_CHANGE",
        }

    stack = gated_training_stack(preflight, training_stack_loader)
    return {
        "schema": "V10_QWEN35_AUTHORIZED_TRAINING_PLAN_V1",
        "status": "READY",
        "training_stack_loaded": stack is not None,
        "reasons": [],
        "preflight": preflight,
        "effect": "READ_ONLY_PLAN_NO_WEIGHT_CHANGE",
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--validation-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.plan_only:
        result = plan_execution(args.repo_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "READY" else 2

    if (
        args.train_jsonl is None
        or args.validation_jsonl is None
        or args.output_dir is None
    ):
        raise TrainingHold(
            "--execute requires --train-jsonl, "
            "--validation-jsonl, and --output-dir"
        )
    result = execute_authorized_training(
        args.repo_root,
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except TrainingHold as exc:
        print(
            json.dumps(
                {
                    "schema": "V10_QWEN35_AUTHORIZED_TRAINING_PLAN_V1",
                    "status": "HOLD",
                    "training_allowed": False,
                    "reason": str(exc),
                    "effect": "NO_WEIGHT_CHANGE",
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2)
