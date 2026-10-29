from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

# Import reusable training primitives without depending on the V10
# final-bank authority gate. This is a development-training lane.
from successor.experiments.train_v10_qwen35_authorized import (
    _artifact_hash_manifest,
    _observe_live_runtime,
    _read_json,
    _validate_qwen_topology,
    canonical_bytes,
    load_training_stack,
    load_verified_jsonl,
    prepare_sft_rows,
    sha256_file,
    validate_lora_target_coverage,
    validate_runtime_observation,
)


class DevTrainingHold(RuntimeError):
    """Fail-closed refusal for V10R2 local development training."""


SPEC_SCHEMA = "V10R2_QWEN35_DEV_TRAINING_SPEC_V1"
AUTHORITY_SCOPE = "CHANGE_TRAINING_METHOD_AND_CONTINUE_LOCAL_MODEL_TRAINING"
ALLOWED_OPTIMIZERS = {
    "adamw_bnb_8bit",
    "adamw_torch_8bit",
}
CLAIM_CEILING = "LOCAL_DEVELOPMENT_ADAPTER_ONLY_NOT_EXTERNALLY_QUALIFIED"
RECIPE_SEMANTICS_CLASS = "CONTINUOUS_STATE_RECIPE_SEMANTICS"
STAGED_RESET_CLASS = "STAGED_RESET_RECIPE_SEMANTICS_CONTROL"


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def effect_label(optimizer_name: str, steps: int) -> str:
    if not isinstance(steps, int) or steps < 1:
        raise DevTrainingHold("effect label requires positive optimizer steps")
    backend = {
        "adamw_bnb_8bit": "BNB_ADAMW_8BIT",
        "adamw_torch_8bit": "TORCHAO_ADAMW_8BIT",
    }.get(optimizer_name)
    if backend is None:
        raise DevTrainingHold("effect label optimizer is unsupported")
    words = {
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        20: "TWENTY",
        32: "THIRTY_TWO",
    }
    count = words.get(steps)
    if count is None:
        raise DevTrainingHold("effect label step count is unsupported")
    suffix = "STEP" if steps == 1 else "STEPS"
    return f"{count}_LOCAL_{backend}_OPTIMIZER_{suffix}"


def load_and_validate_dev_spec(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise DevTrainingHold(f"development spec missing:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))

    if value.get("schema") != SPEC_SCHEMA:
        raise DevTrainingHold("development spec schema mismatch")
    if value.get("status") != "AUTHORIZED_LOCAL_DEVELOPMENT_TRAINING":
        raise DevTrainingHold("development training is not authorized")

    authority = value.get("authority")
    if not isinstance(authority, dict):
        raise DevTrainingHold("development authority missing")
    if authority.get("actor_id") != "PATRICK_USER_AUTHORITY":
        raise DevTrainingHold("development authority actor mismatch")
    if authority.get("source") != "EXPLICIT_CURRENT_USER_INSTRUCTION":
        raise DevTrainingHold("development authority source mismatch")
    if authority.get("scope") != AUTHORITY_SCOPE:
        raise DevTrainingHold("development authority scope mismatch")
    if authority.get("paid_compute_authorized") is not False:
        raise DevTrainingHold("paid compute is not authorized")
    if authority.get("merge_authorized") is not False:
        raise DevTrainingHold("merge authority must remain false")
    if authority.get("deploy_activate_authorized") is not False:
        raise DevTrainingHold("deploy/activate authority must remain false")

    subject = value.get("source_subject")
    if not isinstance(subject, dict):
        raise DevTrainingHold("source subject missing")
    train_sha = subject.get("train_sha256")
    if (
        not isinstance(train_sha, str)
        or len(train_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in train_sha)
    ):
        raise DevTrainingHold("train sha256 invalid")
    if subject.get("train_rows") != 50000:
        raise DevTrainingHold("bound train row count mismatch")

    trainer = value.get("trainer")
    if not isinstance(trainer, dict):
        raise DevTrainingHold("trainer spec missing")
    optimizer_name = trainer.get("optimizer")
    if not isinstance(optimizer_name, str):
        raise DevTrainingHold("optimizer name missing")
    if "paged" in optimizer_name.casefold():
        raise DevTrainingHold("paged optimizers are forbidden on this host")
    if optimizer_name not in ALLOWED_OPTIMIZERS:
        raise DevTrainingHold(
            "optimizer must be one of the approved non-paged 8-bit choices: "
            + ", ".join(sorted(ALLOWED_OPTIMIZERS))
        )
    sampling_strategy = trainer.get("train_sampling_strategy", "random")
    if sampling_strategy not in {"random", "sequential"}:
        raise DevTrainingHold("unsupported train sampling strategy")
    max_steps = trainer.get("max_optimizer_steps")
    experiment_class = value.get("experiment_class")
    is_recipe_semantics = experiment_class == RECIPE_SEMANTICS_CLASS
    if is_recipe_semantics:
        if max_steps != 20:
            raise DevTrainingHold(
                "recipe-semantics experiment requires exactly 20 optimizer steps"
            )
        if optimizer_name != "adamw_bnb_8bit":
            raise DevTrainingHold(
                "recipe-semantics experiment requires adamw_bnb_8bit"
            )
        if trainer.get("warmup_optimizer_steps") != 0:
            raise DevTrainingHold(
                "recipe-semantics experiment requires zero warmup"
            )
        if trainer.get("lr_scheduler_type") != "cosine":
            raise DevTrainingHold(
                "recipe-semantics experiment requires cosine scheduler"
            )
        if trainer.get("shuffle_dataset") is not False:
            raise DevTrainingHold(
                "recipe-semantics experiment requires shuffle_dataset=false"
            )
        if sampling_strategy != "sequential":
            raise DevTrainingHold(
                "recipe-semantics experiment requires sequential sampling"
            )
    elif experiment_class == "R13_FROZEN_DIVERSE_IDENTITY_AND_JUDGMENT_32_STEPS":
        if max_steps != 32:
            raise DevTrainingHold("R13 requires exactly 32 optimizer steps")
        if trainer.get("learning_rate") != 1e-5:
            raise DevTrainingHold("R13 requires learning rate exactly 1e-5")
        if optimizer_name != "adamw_bnb_8bit":
            raise DevTrainingHold("R13 requires nonpaged adamw_bnb_8bit")
        if trainer.get("shuffle_dataset") is not False or sampling_strategy != "sequential":
            raise DevTrainingHold("R13 requires deterministic sequential sampling")
        if value.get("adapter_precision_policy") != "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1":
            raise DevTrainingHold("R13 requires resumed FP32 LoRA weights")
        if value.get("resume_adapter") is None:
            raise DevTrainingHold("R13 requires verified R12 parent checkpoint")
        if trainer.get("warmup_optimizer_steps") != 0:
            raise DevTrainingHold("R13 requires zero warmup steps")
    elif not isinstance(max_steps, int) or max_steps < 1 or max_steps > 8:
        raise DevTrainingHold(
            "development optimizer steps must be an integer from 1 through 8"
        )
    if trainer.get("per_device_train_batch_size") != 1:
        raise DevTrainingHold("per-device train batch size must be 1")
    if trainer.get("gradient_accumulation_steps") != 8:
        raise DevTrainingHold("gradient accumulation must be 8")
    if trainer.get("max_length") != 512:
        raise DevTrainingHold("max length must be 512")
    if trainer.get("completion_only_loss") is not True:
        raise DevTrainingHold("completion-only loss must remain enabled")
    if trainer.get("gradient_checkpointing") is not True:
        raise DevTrainingHold("gradient checkpointing must remain enabled")
    if trainer.get("bf16") is not True:
        raise DevTrainingHold("bf16 must remain enabled")
    if trainer.get("packing") is not False:
        raise DevTrainingHold("packing change not admitted in optimizer revision")

    window = value.get("development_window")
    if window is not None:
        if not isinstance(window, dict):
            raise DevTrainingHold("development window invalid")
        start = window.get("start_row")
        count = window.get("row_count")
        if not isinstance(start, int) or start < 0:
            raise DevTrainingHold("development start row invalid")
        if not isinstance(count, int) or count < 8:
            raise DevTrainingHold("development row count must be at least 8")
        expected_rows = (
            max_steps * trainer["gradient_accumulation_steps"]
            * trainer["per_device_train_batch_size"]
        )
        if count != expected_rows:
            raise DevTrainingHold(
                f"development row count must equal staged optimizer window:"
                f"{count}!={expected_rows}"
            )
        if start + count > subject["train_rows"]:
            raise DevTrainingHold("development window escapes train corpus")
        if is_recipe_semantics and (start != 0 or count != 160):
            raise DevTrainingHold(
                "recipe-semantics experiment requires exact rows 0-159"
            )

    resume = value.get("resume_adapter")
    comparison = value.get("comparison")
    if comparison is not None and not isinstance(comparison, dict):
        raise DevTrainingHold("comparison binding must be an object")
    expected_initial_digest = (
        comparison.get("expected_initial_trainable_parameter_digest")
        if isinstance(comparison, dict)
        else None
    )
    is_fresh_recipe_arm = (
        experiment_class in {RECIPE_SEMANTICS_CLASS, STAGED_RESET_CLASS}
        and resume is None
    )
    if is_fresh_recipe_arm and not _is_sha256(expected_initial_digest):
        raise DevTrainingHold(
            "fresh recipe-semantics arm requires expected initial trainable digest"
        )
    if is_recipe_semantics and resume is not None:
        raise DevTrainingHold(
            "recipe-semantics experiment requires a fresh adapter"
        )
    if resume is not None:
        if not isinstance(resume, dict):
            raise DevTrainingHold("resume_adapter must be an object")
        if not isinstance(resume.get("path"), str) or not resume["path"].strip():
            raise DevTrainingHold("resume_adapter path missing")
        for field in (
            "adapter_model_sha256",
            "source_receipt_sha256",
            "source_weight_digest_after",
        ):
            if not _is_sha256(resume.get(field)):
                raise DevTrainingHold(f"resume_adapter {field} invalid")
        prior_steps = resume.get("previous_optimizer_steps")
        if not isinstance(prior_steps, int) or prior_steps < 1:
            raise DevTrainingHold(
                "resume_adapter previous_optimizer_steps invalid"
            )
        next_train_row = resume.get("next_train_row")
        if not isinstance(next_train_row, int) or next_train_row < 0:
            raise DevTrainingHold("resume_adapter next_train_row invalid")
        if window is None:
            raise DevTrainingHold(
                "resume_adapter requires explicit development_window"
            )
        if window["start_row"] != next_train_row:
            raise DevTrainingHold(
                "development_window start_row must equal resume_adapter "
                f"next_train_row:{window['start_row']}!={next_train_row}"
            )
        for field in ("source_receipt_path", "source_spec_path"):
            path_value = resume.get(field)
            if (
                not isinstance(path_value, str)
                or not path_value.strip()
            ):
                raise DevTrainingHold(
                    f"resume_adapter {field} invalid"
                )
        if not _is_sha256(resume.get("source_spec_sha256")):
            raise DevTrainingHold(
                "resume_adapter source_spec_sha256 invalid"
            )
        cumulative = resume.get("source_cumulative_optimizer_steps")
        if cumulative is not None and (
            not isinstance(cumulative, int) or cumulative < prior_steps
        ):
            raise DevTrainingHold(
                "resume_adapter source_cumulative_optimizer_steps invalid"
            )

    quant = value.get("quantization")
    if not isinstance(quant, dict):
        raise DevTrainingHold("quantization spec missing")
    expected_quant = {
        "load_in_4bit": True,
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    }
    if quant != expected_quant:
        raise DevTrainingHold("quantization recipe mismatch")

    lora = value.get("lora")
    if not isinstance(lora, dict):
        raise DevTrainingHold("LoRA spec missing")
    expected_lora = {
        "r": 4,
        "alpha": 16,
        "dropout": 0,
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if lora != expected_lora:
        raise DevTrainingHold("LoRA recipe mismatch")

    qualification = value.get("qualification")
    if not isinstance(qualification, dict):
        raise DevTrainingHold("qualification boundary missing")
    if qualification.get("external_final_bank_required_for_qualification") is not True:
        raise DevTrainingHold("external final bank must remain required for qualification")
    if qualification.get("development_training_may_precede_external_qualification") is not True:
        raise DevTrainingHold("development training permission missing")
    if qualification.get("claim_ceiling") != CLAIM_CEILING:
        raise DevTrainingHold("development claim ceiling mismatch")

    output = value.get("output")
    if not isinstance(output, dict):
        raise DevTrainingHold("output namespace missing")
    if not isinstance(output.get("namespace"), str) or not output["namespace"].strip():
        raise DevTrainingHold("output namespace invalid")
    if output.get("save_final_adapter") is not True:
        raise DevTrainingHold("development adapter must be saved")
    if output.get("overwrite_forbidden") is not True:
        raise DevTrainingHold("output overwrite must be forbidden")

    precision_policy = value.get("adapter_precision_policy", "LEGACY_BF16")
    if precision_policy not in {"LEGACY_BF16", "FP32_RESUMED_LORA_TRAIN_AND_SAVE_V1"}:
        raise DevTrainingHold("unrecognized adapter precision policy")
    if precision_policy != "LEGACY_BF16" and resume is None:
        raise DevTrainingHold("FP32 resumed LoRA policy requires an exact parent")
    return value


def summarize_optimizer(optimizer) -> dict:
    wrapper_chain: list[str] = []
    current = optimizer
    seen: set[int] = set()
    while (
        hasattr(current, "optimizer")
        and id(current) not in seen
        and getattr(current, "optimizer") is not current
    ):
        seen.add(id(current))
        wrapper_chain.append(type(current).__name__)
        current = getattr(current, "optimizer")

    args = getattr(current, "args", None)
    module = type(current).__module__
    class_name = type(current).__name__
    backend = (
        "bitsandbytes"
        if module.split(".")[0] == "bitsandbytes"
        else "torchao"
        if module.split(".")[0] == "torchao"
        else module.split(".")[0]
    )
    optim_bits = getattr(args, "optim_bits", None)
    if optim_bits is None and backend == "torchao" and "8bit" in class_name.casefold():
        optim_bits = 8

    return {
        "class": class_name,
        "module": module,
        "backend": backend,
        "is_paged": bool(getattr(current, "is_paged", False)),
        "optim_bits": optim_bits,
        "min_8bit_size": getattr(args, "min_8bit_size", None),
        "wrapper_chain": wrapper_chain,
    }


def _trainable_parameter_digest(model, torch) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        count += int(parameter.numel())
        tensor = (
            parameter.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    if count < 1:
        raise DevTrainingHold("no trainable parameters")
    return digest.hexdigest(), count


def _optimizer_state_summary(optimizer) -> dict:
    tensor_bytes = 0
    tensor_count = 0
    devices: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for item in state.values():
            if not hasattr(item, "numel") or not hasattr(item, "element_size"):
                continue
            tensor_count += 1
            tensor_bytes += int(item.numel()) * int(item.element_size())
            device = str(item.device)
            dtype = str(item.dtype)
            devices[device] = devices.get(device, 0) + 1
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
    return {
        "state_tensor_count": tensor_count,
        "state_tensor_bytes": tensor_bytes,
        "state_tensor_mib": round(tensor_bytes / (1024**2), 3),
        "state_tensor_devices": devices,
        "state_tensor_dtypes": dtypes,
    }


def _case_id(row: dict, index: int) -> str:
    for field in ("case_id", "source_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return f"bound-train-row-{index}"


def _targeted_lora_module_names(model) -> list[str]:
    names = getattr(model, "targeted_module_names", None)
    if names:
        return list(names)
    discovered: list[str] = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            discovered.append(name)
    return discovered


def _resolve_bound_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_resume_adapter(root: Path | str, resume: dict) -> dict:
    root = Path(root)
    adapter_dir = _resolve_bound_path(root, resume["path"])
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_model.is_file():
        raise DevTrainingHold(
            f"resume adapter model missing:{adapter_model}"
        )
    if not adapter_config_path.is_file():
        raise DevTrainingHold(
            f"resume adapter config missing:{adapter_config_path}"
        )
    actual_adapter_sha = sha256_file(adapter_model)
    if actual_adapter_sha != resume["adapter_model_sha256"]:
        raise DevTrainingHold(
            "resume adapter_model_sha256 mismatch:"
            f"{actual_adapter_sha}!={resume['adapter_model_sha256']}"
        )

    source_spec_path = _resolve_bound_path(
        root,
        resume["source_spec_path"],
    )
    if not source_spec_path.is_file():
        raise DevTrainingHold(
            f"resume source spec missing:{source_spec_path}"
        )
    actual_spec_sha = sha256_file(source_spec_path)
    if actual_spec_sha != resume["source_spec_sha256"]:
        raise DevTrainingHold(
            "resume source_spec_sha256 mismatch:"
            f"{actual_spec_sha}!={resume['source_spec_sha256']}"
        )
    source_spec = load_and_validate_dev_spec(source_spec_path)
    if (
        source_spec["trainer"]["max_optimizer_steps"]
        != resume["previous_optimizer_steps"]
    ):
        raise DevTrainingHold(
            "resume previous_optimizer_steps mismatch with source spec"
        )
    source_window = source_spec.get("development_window") or {
        "start_row": 0,
        "row_count": 8,
    }
    expected_next_row = (
        int(source_window["start_row"])
        + int(source_window["row_count"])
    )
    if expected_next_row != resume["next_train_row"]:
        raise DevTrainingHold(
            "resume next_train_row mismatch with source spec:"
            f"{resume['next_train_row']}!={expected_next_row}"
        )

    source_receipt_path = _resolve_bound_path(
        root,
        resume["source_receipt_path"],
    )
    if not source_receipt_path.is_file():
        raise DevTrainingHold(
            f"resume source receipt missing:{source_receipt_path}"
        )
    receipt = _read_json(source_receipt_path)
    claimed_receipt_sha = receipt.get("receipt_sha256")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256", None)
    actual_receipt_sha = _canonical_sha(unsigned_receipt)
    if claimed_receipt_sha != actual_receipt_sha:
        raise DevTrainingHold(
            "resume source receipt self-hash mismatch"
        )
    if actual_receipt_sha != resume["source_receipt_sha256"]:
        raise DevTrainingHold(
            "resume source_receipt_sha256 mismatch:"
            f"{actual_receipt_sha}!={resume['source_receipt_sha256']}"
        )
    if receipt.get("spec_sha256") != actual_spec_sha:
        raise DevTrainingHold(
            "resume source receipt/spec binding mismatch"
        )
    if (
        receipt.get("weight_digest_after")
        != resume["source_weight_digest_after"]
    ):
        raise DevTrainingHold(
            "resume source_weight_digest_after mismatch"
        )
    if receipt.get("weight_digest_changed") is not True:
        raise DevTrainingHold(
            "resume source receipt does not prove weight change"
        )
    source_cumulative = resume.get("source_cumulative_optimizer_steps")
    if not isinstance(source_cumulative, int):
        raise DevTrainingHold(
            "resume source_cumulative_optimizer_steps missing"
        )
    receipt_cumulative = receipt.get("cumulative_optimizer_steps")
    if receipt_cumulative != source_cumulative:
        raise DevTrainingHold(
            "resume source_cumulative_optimizer_steps mismatch:"
            f"{resume['source_cumulative_optimizer_steps']}!="
            f"{receipt_cumulative}"
        )

    adapter_config = _read_json(adapter_config_path)
    expected_adapter = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 4,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
    }
    for field, expected in expected_adapter.items():
        if adapter_config.get(field) != expected:
            raise DevTrainingHold(
                f"resume adapter config mismatch:{field}:"
                f"{adapter_config.get(field)}!={expected}"
            )

    return {
        "status": "RESUME_ADAPTER_VERIFIED",
        "adapter_dir": str(adapter_dir),
        "adapter_model_sha256": actual_adapter_sha,
        "source_spec_path": str(source_spec_path),
        "source_spec_sha256": actual_spec_sha,
        "source_receipt_path": str(source_receipt_path),
        "source_receipt_sha256": actual_receipt_sha,
        "source_weight_digest_after": receipt["weight_digest_after"],
        "previous_optimizer_steps": resume["previous_optimizer_steps"],
        "source_cumulative_optimizer_steps": resume[
            "source_cumulative_optimizer_steps"
        ],
        "next_train_row": resume["next_train_row"],
    }


def execute_dev_training(
    repo_root: Path | str,
    *,
    train_jsonl: Path,
    spec_path: Path,
    training_stack_loader=load_training_stack,
) -> dict:
    root = Path(repo_root)
    spec = load_and_validate_dev_spec(spec_path)
    # Fail before allocating the GPU if a requested step count cannot be
    # represented in the eventual completion receipt.
    receipt_effect = effect_label(
        spec["trainer"]["optimizer"],
        spec["trainer"]["max_optimizer_steps"],
    )

    output_dir = Path(spec["output"]["namespace"])
    if output_dir.exists():
        raise DevTrainingHold(f"output namespace already exists:{output_dir}")

    subject = spec["source_subject"]
    train_rows = load_verified_jsonl(
        train_jsonl,
        expected_sha256=subject["train_sha256"],
        expected_rows=subject["train_rows"],
    )
    window = spec.get("development_window") or {
        "start_row": 0,
        "row_count": 8,
    }
    start = int(window["start_row"])
    count = int(window["row_count"])
    selected_rows = train_rows[start : start + count]

    runtime_path = (
        root
        / "successor"
        / "experiments"
        / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    runtime_binding = _read_json(runtime_path)
    if runtime_binding.get("binding_sha256") != subject["runtime_binding_sha256"]:
        raise DevTrainingHold("runtime binding digest mismatch")

    stack = training_stack_loader()
    if not isinstance(stack, dict):
        raise DevTrainingHold("training stack loader returned invalid object")

    live_runtime = _observe_live_runtime(runtime_binding, stack)
    runtime_check = validate_runtime_observation(
        runtime_binding,
        live_runtime,
    )

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise DevTrainingHold("CUDA unavailable")

    base_path = Path(runtime_binding["target"]["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer_spec = spec["trainer"]
    prepared, token_budget = prepare_sft_rows(
        tokenizer,
        selected_rows,
        max_length=trainer_spec["max_length"],
    )

    quant = spec["quantization"]
    bits_config = stack["BitsAndBytesConfig"](
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["compute_dtype"]),
    )

    model_load = spec["model_load"]
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

    lora_spec = spec["lora"]
    lora = stack["LoraConfig"](
        r=lora_spec["r"],
        lora_alpha=lora_spec["alpha"],
        lora_dropout=lora_spec["dropout"],
        bias=lora_spec["bias"],
        task_type=lora_spec["task_type"],
        target_modules=lora_spec["target_modules"],
    )

    resume_check = None
    peft_config = lora
    if spec.get("resume_adapter") is not None:
        resume_check = verify_resume_adapter(
            root,
            spec["resume_adapter"],
        )
        model = stack["peft"].PeftModel.from_pretrained(
            model,
            resume_check["adapter_dir"],
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
        peft_config = None

    from successor.experiments.r12_adapter_precision import verify_trainable, POLICY
    fp32_adapter = spec.get("adapter_precision_policy") == POLICY
    precision_before = (
        verify_trainable(model, torch, permit_cast=True) if fp32_adapter else None
    )
    stack["transformers"].set_seed(trainer_spec["seed"])
    output_dir.mkdir(parents=True, exist_ok=False)
    work_dir = output_dir / "sft_work"

    sft_config = stack["SFTConfig"](
        output_dir=str(work_dir),
        per_device_train_batch_size=trainer_spec[
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=trainer_spec[
            "gradient_accumulation_steps"
        ],
        max_steps=trainer_spec["max_optimizer_steps"],
        num_train_epochs=1,
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
        train_sampling_strategy=trainer_spec.get(
            "train_sampling_strategy",
            "random",
        ),
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
        train_dataset=stack["Dataset"].from_list(prepared),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    coverage = validate_lora_target_coverage(
        _targeted_lora_module_names(trainer.model)
    )
    # TRL SFTTrainer 1.13 casts quantized trainable adapters back to BF16
    # during its constructor. Enforce the bound FP32 policy AFTER that
    # constructor, before the optimizer captures parameter references.
    precision_after_trainer = (
        verify_trainable(trainer.model, torch, permit_cast=True)
        if fp32_adapter else None
    )
    # PEFT can load an FP32 checkpoint through BF16 module parameters.
    # BF16 -> FP32 upcasting does not restore the lost low bits. On the
    # bounded R13 branch, restore the exact SHA-verified saved tensor bytes
    # after TRL's constructor and BEFORE optimizer parameter capture.
    exact_resume_rehydration = None
    if spec.get("experiment_class") == "R13_FROZEN_DIVERSE_IDENTITY_AND_JUDGMENT_32_STEPS":
        from successor.experiments.r13_resume_fp32_integrity import restore_exact_lora
        exact_resume_rehydration = restore_exact_lora(
            trainer.model,
            Path(resume_check["adapter_dir"]) / "adapter_model.safetensors",
            expected_sha=resume_check["adapter_model_sha256"],
            expected_tensors=precision_after_trainer["verified_trainable_tensors"],
            torch=torch,
        )

    optimizer = trainer.create_optimizer()
    if fp32_adapter:
        verify_trainable(trainer.model, torch, permit_cast=False)
    optimizer_info = summarize_optimizer(optimizer)
    requested_optimizer = trainer_spec["optimizer"]
    if optimizer_info["is_paged"] is not False:
        raise DevTrainingHold("paged optimizer instantiated on forbidden host")
    if optimizer_info["optim_bits"] != 8:
        raise DevTrainingHold(
            f"optimizer is not 8-bit:{optimizer_info['optim_bits']}"
        )
    if requested_optimizer == "adamw_bnb_8bit":
        if not (
            optimizer_info["backend"] == "bitsandbytes"
            and optimizer_info["class"] == "AdamW"
        ):
            raise DevTrainingHold(
                "requested adamw_bnb_8bit but instantiated "
                f"{optimizer_info['module']}.{optimizer_info['class']}"
            )
    elif requested_optimizer == "adamw_torch_8bit":
        if not (
            optimizer_info["backend"] == "torchao"
            and optimizer_info["class"] == "AdamW8bit"
        ):
            raise DevTrainingHold(
                "requested adamw_torch_8bit but instantiated "
                f"{optimizer_info['module']}.{optimizer_info['class']}"
            )
    else:
        raise DevTrainingHold("unreachable optimizer selection")

    weight_before, trainable_count = _trainable_parameter_digest(
        trainer.model,
        torch,
    )
    comparison = spec.get("comparison") or {}
    expected_initial_digest = comparison.get(
        "expected_initial_trainable_parameter_digest"
    )
    if (
        expected_initial_digest is not None
        and weight_before != expected_initial_digest
    ):
        raise DevTrainingHold(
            "initial trainable parameter digest mismatch:"
            f"{weight_before}!={expected_initial_digest}"
        )
    if (
        resume_check is not None
        and weight_before
        != resume_check["source_weight_digest_after"]
    ):
        raise DevTrainingHold(
            "reloaded resume adapter trainable digest mismatch:"
            f"{weight_before}!="
            f"{resume_check['source_weight_digest_after']}"
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_before = {
        "allocated_mib": round(
            torch.cuda.memory_allocated() / (1024**2), 3
        ),
        "reserved_mib": round(
            torch.cuda.memory_reserved() / (1024**2), 3
        ),
    }

    started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    weight_after, post_count = _trainable_parameter_digest(
        trainer.model,
        torch,
    )
    if post_count != trainable_count:
        raise DevTrainingHold("trainable parameter count changed")
    if weight_after == weight_before:
        raise DevTrainingHold("optimizer step did not change trainable weights")

    state_summary = _optimizer_state_summary(trainer.optimizer)
    optimizer_after = summarize_optimizer(trainer.optimizer)
    identity_fields = (
        "class",
        "module",
        "backend",
        "is_paged",
        "optim_bits",
        "min_8bit_size",
    )
    if any(
        optimizer_after.get(field) != optimizer_info.get(field)
        for field in identity_fields
    ):
        raise DevTrainingHold("optimizer identity changed during training")

    if fp32_adapter:
        precision_after = verify_trainable(trainer.model, torch, permit_cast=False)
    else:
        precision_after = None
        for _, parameter in trainer.model.named_parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(torch.bfloat16)

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
    )
    artifact_manifest = _artifact_hash_manifest(adapter_dir)

    gpu_after = {
        "allocated_mib": round(
            torch.cuda.memory_allocated() / (1024**2), 3
        ),
        "reserved_mib": round(
            torch.cuda.memory_reserved() / (1024**2), 3
        ),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024**2), 3
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / (1024**2), 3
        ),
    }

    receipt = {
        "schema": "V10R2_QWEN35_DEV_TRAINING_RECEIPT_V1",
        "status": "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE",
        "effect": receipt_effect,
        "experiment_class": spec.get("experiment_class"),
        "spec_sha256": sha256_file(spec_path),
        "source_parent_revision": subject["parent_revision"],
        "training_corpus_id": subject["training_corpus_id"],
        "train_sha256": subject["train_sha256"],
        "development_window": {
            "start_row": start,
            "row_count": count,
            "row_ids": [
                _case_id(row, start + offset)
                for offset, row in enumerate(selected_rows)
            ],
        },
        "resume_adapter": resume_check,
        "cumulative_optimizer_steps": (
            trainer_spec["max_optimizer_steps"]
            + (
                resume_check["source_cumulative_optimizer_steps"]
                if resume_check is not None
                else 0
            )
        ),
        "token_budget": token_budget,
        "optimizer": optimizer_info,
        "optimizer_state": state_summary,
        "training_order": {
            "shuffle_dataset": trainer_spec["shuffle_dataset"],
            "train_sampling_strategy": trainer_spec.get(
                "train_sampling_strategy",
                "random",
            ),
        },
        "trainable_parameter_count": trainable_count,
        "initial_trainable_parameter_digest": weight_before,
        "initialization_equivalence_gate": {
            "expected_digest": expected_initial_digest,
            "pass": (
                expected_initial_digest is None
                or weight_before == expected_initial_digest
            ),
        },
        "weight_digest_before": weight_before,
        "weight_digest_after": weight_after,
        "weight_digest_changed": True,
        "training_loss": float(train_result.training_loss),
        "train_metrics": dict(train_result.metrics),
        "elapsed_seconds_observed": round(elapsed, 3),
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "runtime_binding_sha256": runtime_binding[
            "binding_sha256"
        ],
        "live_runtime": live_runtime,
        "live_runtime_check": runtime_check,
        "base_topology": topology,
        "lora_coverage": coverage,
        "adapter_artifacts": artifact_manifest,
        "adapter_precision_policy": spec.get("adapter_precision_policy", "LEGACY_BF16"),
        "adapter_precision_initial": precision_before,
        "adapter_precision_after_trainer": precision_after_trainer,
        "exact_resume_rehydration": exact_resume_rehydration,
        "adapter_precision_final": precision_after,
        "qualification_status": "NOT_EXTERNALLY_EVALUATED",
        "deployment_status": "NOT_DEPLOYED",
        "merge_status": "NOT_MERGED",
        "paid_compute": False,
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    (output_dir / "DEV_TRAINING_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    del trainer
    del model
    torch.cuda.empty_cache()
    return receipt


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = execute_dev_training(
            args.repo_root,
            train_jsonl=args.train_jsonl,
            spec_path=args.spec,
        )
    except DevTrainingHold as exc:
        print(
            json.dumps(
                {
                    "schema": "V10R2_QWEN35_DEV_TRAINING_RECEIPT_V1",
                    "status": "HOLD",
                    "reason": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())