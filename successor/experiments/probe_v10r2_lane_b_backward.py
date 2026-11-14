from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from successor.experiments.v10r2_lane_b_runtime import apply_qwen35_acceleration

from successor.experiments.train_v10_qwen35_authorized import (
    _observe_live_runtime,
    _read_json,
    _validate_qwen_topology,
    assess_preflight,
    canonical_bytes,
    load_training_stack,
    load_verified_jsonl,
    prepare_sft_rows,
    sha256_file,
    validate_execution_spec,
    validate_lora_target_coverage,
    validate_runtime_observation,
)


class ProbeHold(RuntimeError):
    """Fail-closed refusal for the non-mutating backward probe."""


_ALLOWED_TRAINING_HOLD_REASONS = {
    "final_bank_cases_not_admitted",
    "fresh_evaluation_bank_frozen",
    "independent_bank_admission_verified",
    "patrick_exact_weight_change_authority",
    "semantic_contamination_screen_verified",
}


def validate_probe_preconditions(preflight: dict) -> dict:
    if not isinstance(preflight, dict):
        raise ProbeHold("preflight is not an object")

    runtime = preflight.get("training_runtime")
    if not isinstance(runtime, dict) or runtime.get("status") != "VERIFIED":
        raise ProbeHold("training runtime is not verified")

    execution = preflight.get("training_execution")
    if not isinstance(execution, dict) or execution.get("status") != "VERIFIED":
        raise ProbeHold("training execution is not verified")

    reasons = preflight.get("reasons")
    if reasons is None:
        reasons = []
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) for reason in reasons
    ):
        raise ProbeHold("preflight reasons are invalid")

    unexpected = sorted(set(reasons) - _ALLOWED_TRAINING_HOLD_REASONS)
    if unexpected:
        raise ProbeHold(
            "unexpected preflight blockers: " + " | ".join(unexpected)
        )

    return {
        "schema": "V10_QWEN35_BACKWARD_PROBE_PRECONDITION_V1",
        "status": "PROBE_ALLOWED_WITH_TRAINING_HOLD",
        "weight_change_authorized": False,
        "training_allowed": bool(preflight.get("training_allowed")),
        "training_preflight_status": preflight.get("status"),
        "training_hold_reasons": list(reasons),
        "runtime_binding_sha256": runtime.get("binding_sha256"),
        "execution_binding_sha256": execution.get("binding_sha256"),
        "effect": "READ_ONLY_GATE_FOR_EPHEMERAL_BACKWARD_PROBE",
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def apply_liger_qwen35_candidate(
    model,
    *,
    liger_path: Path | None = None,
    liger_wheel: Path | None = None,
    patch_function: Callable | None = None,
    version: str | None = None,
) -> dict:
    if liger_path is None:
        if any(value is not None for value in (liger_wheel, patch_function, version)):
            raise ProbeHold("liger candidate arguments require liger_path")
        return {"enabled": False}
    if not liger_path.is_dir():
        raise ProbeHold(f"liger source path missing:{liger_path}")
    if liger_wheel is None or not liger_wheel.is_file():
        raise ProbeHold("liger wheel missing")

    wheel_sha256 = sha256_file(liger_wheel)
    if patch_function is None:
        source = str(liger_path)
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from liger_kernel.transformers.monkey_patch import (
                apply_liger_kernel_to_qwen3_5,
            )
        except Exception as exc:
            raise ProbeHold(f"liger import failed:{exc}") from exc
        patch_function = apply_liger_kernel_to_qwen3_5
    if version is None:
        try:
            version = metadata.version("liger-kernel")
        except metadata.PackageNotFoundError as exc:
            raise ProbeHold("liger version metadata missing") from exc

    patch_kwargs = {
        "rope": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
        "swiglu": True,
        "model": model,
    }
    patch_function(**patch_kwargs)
    return {
        "enabled": True,
        "version": version,
        "source_path": str(liger_path),
        "wheel_path": str(liger_wheel),
        "wheel_sha256": wheel_sha256,
        **{key: value for key, value in patch_kwargs.items() if key != "model"},
    }


def _gradient_summary(model, torch, *, vector_out: Path | None = None) -> dict:
    digest = hashlib.sha256()
    layout_digest = hashlib.sha256()
    vectors = []
    tensor_count = 0
    nonfinite_tensor_count = 0
    zero_tensor_count = 0
    element_count = 0
    l2_sq = 0.0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise ProbeHold(f"missing trainable gradient:{name}")
        grad = parameter.grad.detach().float().cpu().contiguous()
        raw = grad.numpy().tobytes(order="C")
        shape = list(grad.shape)
        header = canonical_bytes({"name": name, "shape": shape, "dtype": "float32"})
        digest.update(header)
        digest.update(raw)
        layout_digest.update(header)
        tensor_count += 1
        element_count += grad.numel()
        finite = bool(torch.isfinite(grad).all().item())
        if not finite:
            nonfinite_tensor_count += 1
        if not bool(torch.count_nonzero(grad).item()):
            zero_tensor_count += 1
        l2_sq += float(grad.double().square().sum().item())
        vectors.append(grad.reshape(-1))

    if not vectors:
        raise ProbeHold("no trainable gradient vectors")
    if nonfinite_tensor_count:
        raise ProbeHold(f"nonfinite trainable gradients:{nonfinite_tensor_count}")

    vector = torch.cat(vectors)
    vector_file_sha256 = None
    if vector_out is not None:
        vector_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vector, vector_out)
        vector_file_sha256 = sha256_file(vector_out)

    return {
        "schema": "V10R2_TRAINABLE_GRADIENT_SUMMARY_V1",
        "tensor_count": tensor_count,
        "zero_tensor_count": zero_tensor_count,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "all_finite": nonfinite_tensor_count == 0,
        "element_count": element_count,
        "l2": math.sqrt(l2_sq),
        "gradient_sha256": digest.hexdigest(),
        "layout_sha256": layout_digest.hexdigest(),
        "vector_file_sha256": vector_file_sha256,
    }


def finalize_probe_receipt(
    *,
    precondition_check: dict,
    contract_sha256: str,
    execution_spec_sha256: str,
    train_sha256: str,
    row_index: int,
    row_case_id: str,
    trainable_parameter_count: int,
    targeted_module_count: int,
    loss: float,
    microbatch_losses: list[float],
    microbatches_completed: int,
    expected_gradient_accumulation_steps: int,
    gradients_present: bool,
    nonzero_gradient_parameter_count: int,
    optimizer_created: bool,
    weight_digest_before: str,
    weight_digest_after: str,
    cuda_memory: dict,
    output_artifacts_written: bool,
) -> dict:
    if precondition_check.get("status") != "PROBE_ALLOWED_WITH_TRAINING_HOLD":
        raise ProbeHold("probe precondition check is not allowed")
    for label, value in (
        ("contract_sha256", contract_sha256),
        ("execution_spec_sha256", execution_spec_sha256),
        ("train_sha256", train_sha256),
        ("weight_digest_before", weight_digest_before),
        ("weight_digest_after", weight_digest_after),
    ):
        if not _valid_sha256(value):
            raise ProbeHold(f"{label} is invalid")
    if not isinstance(row_index, int) or row_index < 0:
        raise ProbeHold("row_index is invalid")
    if not isinstance(row_case_id, str) or not row_case_id.strip():
        raise ProbeHold("row_case_id is invalid")
    if not isinstance(trainable_parameter_count, int) or trainable_parameter_count < 1:
        raise ProbeHold("no trainable parameters")
    if not isinstance(targeted_module_count, int) or targeted_module_count < 1:
        raise ProbeHold("no LoRA target modules")
    if not isinstance(loss, (float, int)) or not math.isfinite(float(loss)):
        raise ProbeHold("loss is not finite")
    if (
        not isinstance(expected_gradient_accumulation_steps, int)
        or expected_gradient_accumulation_steps < 1
    ):
        raise ProbeHold("expected gradient accumulation steps invalid")
    if (
        not isinstance(microbatches_completed, int)
        or microbatches_completed != expected_gradient_accumulation_steps
    ):
        raise ProbeHold("gradient accumulation window incomplete")
    if (
        not isinstance(microbatch_losses, list)
        or len(microbatch_losses) != microbatches_completed
        or any(
            not isinstance(value, (float, int))
            or not math.isfinite(float(value))
            for value in microbatch_losses
        )
    ):
        raise ProbeHold("microbatch losses invalid")
    if gradients_present is not True or nonzero_gradient_parameter_count < 1:
        raise ProbeHold("no trainable gradients")
    if optimizer_created is not False:
        raise ProbeHold("optimizer was created")
    if weight_digest_before != weight_digest_after:
        raise ProbeHold("trainable parameter digest changed")
    if output_artifacts_written is not False:
        raise ProbeHold("output artifacts were written")

    receipt = {
        "schema": "V10_QWEN35_BACKWARD_PROBE_RECEIPT_V1",
        "status": "BACKWARD_PROBE_PASS_NO_WEIGHT_CHANGE",
        "effect": (
            "EPHEMERAL_FORWARD_BACKWARD_ONLY_NO_OPTIMIZER_NO_WEIGHT_CHANGE"
        ),
        "contract_sha256": contract_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "train_sha256": train_sha256,
        "row_index": row_index,
        "row_case_id": row_case_id,
        "trainable_parameter_count": trainable_parameter_count,
        "targeted_module_count": targeted_module_count,
        "loss": float(loss),
        "microbatch_losses": [float(value) for value in microbatch_losses],
        "microbatches_completed": microbatches_completed,
        "expected_gradient_accumulation_steps": expected_gradient_accumulation_steps,
        "gradients_present": True,
        "nonzero_gradient_parameter_count": nonzero_gradient_parameter_count,
        "optimizer_created": False,
        "weight_digest_before": weight_digest_before,
        "weight_digest_after": weight_digest_after,
        "weight_digest_unchanged": True,
        "cuda_memory": dict(cuda_memory),
        "output_artifacts_written": False,
        "runtime_binding_sha256": precondition_check.get(
            "runtime_binding_sha256"
        ),
        "execution_binding_sha256": precondition_check.get(
            "execution_binding_sha256"
        ),
        "training_hold_reasons": list(
            precondition_check.get("training_hold_reasons", [])
        ),
        "qualification_status": "NOT_EVALUATED",
        "claim_ceiling": (
            "REAL_BOUND_CORPUS_FORWARD_BACKWARD_OBSERVED / "
            "NO_OPTIMIZER_STEP / NO_WEIGHT_CHANGE / NO_ADAPTER_SAVED / "
            "DOES_NOT_SATISFY_FINAL_BANK_OR_TRAINING_AUTHORITY"
        ),
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_bytes(receipt)
    ).hexdigest()
    return receipt


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
        raise ProbeHold("no trainable parameters after LoRA construction")
    return digest.hexdigest(), count


def _case_label(row: dict, index: int) -> str:
    for field in ("case_id", "source_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return f"bound-train-row-{index}"


def execute_backward_probe(
    repo_root: Path | str,
    *,
    train_jsonl: Path,
    row_index: int = 0,
    training_stack_loader: Callable[[], object] = load_training_stack,
    pad_to_multiple_of: int | None = None,
    acceleration_backend: str = "fla_triton",
    gradient_vector_out: Path | None = None,
    liger_path: Path | None = None,
    liger_wheel: Path | None = None,
) -> dict:
    if acceleration_backend not in {"fla_triton", "fla_triton_full"}:
        raise ProbeHold(
            f"unsupported probe acceleration backend:{acceleration_backend}"
        )
    root = Path(repo_root)
    preflight = assess_preflight(root)
    precondition_check = validate_probe_preconditions(preflight)

    experiment_dir = root / "successor" / "experiments"
    contract_path = (
        experiment_dir / "V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"
    )
    runtime_path = (
        experiment_dir / "V10R2_LANE_B_RUNTIME_BINDING_V2.json"
    )
    execution_spec_path = (
        experiment_dir / "V10_QWEN35_TRAINING_EXECUTION_SPEC_V1.json"
    )

    contract = _read_json(contract_path)
    runtime_binding = _read_json(runtime_path)
    precondition_check["runtime_binding_sha256"] = runtime_binding.get(
        "binding_sha256"
    )
    execution_spec = _read_json(execution_spec_path)
    spec_check = validate_execution_spec(contract, execution_spec)

    subject = contract.get("source_subject")
    if not isinstance(subject, dict):
        raise ProbeHold("contract source subject missing")
    train_rows = load_verified_jsonl(
        train_jsonl,
        expected_sha256=subject.get("train_sha256"),
        expected_rows=int(subject.get("train_rows")),
    )
    accumulation_steps = int(
        execution_spec["trainer"]["gradient_accumulation_steps"]
    )
    if (
        row_index < 0
        or row_index + accumulation_steps > len(train_rows)
    ):
        raise ProbeHold(
            "row window out of range:"
            f"{row_index}+{accumulation_steps}/{len(train_rows)}"
        )
    selected_rows = train_rows[
        row_index : row_index + accumulation_steps
    ]
    selected = selected_rows[0]

    stack = training_stack_loader()
    if not isinstance(stack, dict):
        raise ProbeHold("training stack loader returned invalid object")

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
    prepared, token_budget = prepare_sft_rows(
        tokenizer,
        selected_rows,
        max_length=max_length,
    )

    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise ProbeHold("CUDA is unavailable")

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
    acceleration = apply_qwen35_acceleration(
        model,
        backend=acceleration_backend,
    )
    liger = apply_liger_qwen35_candidate(
        model,
        liger_path=liger_path,
        liger_wheel=liger_wheel,
    )
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

    trainer_spec = execution_spec["trainer"]
    stack["transformers"].set_seed(trainer_spec["seed"])

    with tempfile.TemporaryDirectory(
        prefix="vera-v10-backward-probe-"
    ) as temp_dir:
        sft_config = stack["SFTConfig"](
            output_dir=temp_dir,
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
            pad_to_multiple_of=pad_to_multiple_of,
            completion_only_loss=trainer_spec[
                "completion_only_loss"
            ],
            packing=trainer_spec["packing"],
            shuffle_dataset=trainer_spec["shuffle_dataset"],
            logging_steps=trainer_spec["logging_steps"],
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=trainer_spec["seed"],
            data_seed=trainer_spec["seed"],
        )
        trainer = stack["SFTTrainer"](
            model=model,
            args=sft_config,
            train_dataset=stack["Dataset"].from_list(prepared),
            processing_class=tokenizer,
            peft_config=lora,
        )
        coverage = validate_lora_target_coverage(
            getattr(trainer.model, "targeted_module_names", [])
        )

        if trainer.optimizer is not None:
            raise ProbeHold("optimizer was created before backward probe")

        weight_before, trainable_count = _trainable_parameter_digest(
            trainer.model,
            torch,
        )

        dataloader = trainer.get_train_dataloader()
        iterator = iter(dataloader)

        trainer.model.train()
        trainer.model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()

        microbatch_losses: list[float] = []
        backward_started = time.perf_counter()
        for microbatch_index in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ProbeHold(
                    "gradient accumulation dataloader ended early:"
                    f"{microbatch_index}/{accumulation_steps}"
                ) from exc
            batch = trainer._prepare_inputs(batch)
            loss = trainer.compute_loss(trainer.model, batch)
            if not bool(torch.isfinite(loss.detach()).item()):
                raise ProbeHold(
                    f"loss is not finite at microbatch:{microbatch_index}"
                )
            microbatch_losses.append(
                float(loss.detach().float().cpu().item())
            )
            trainer.accelerator.backward(loss / accumulation_steps)
        torch.cuda.synchronize()
        backward_elapsed_seconds = time.perf_counter() - backward_started

        gradient_parameter_count = 0
        nonzero_gradient_parameter_count = 0
        for _, parameter in trainer.model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient_parameter_count += 1
            if bool(torch.count_nonzero(parameter.grad.detach()).item()):
                nonzero_gradient_parameter_count += 1

        gradient_summary = _gradient_summary(
            trainer.model,
            torch,
            vector_out=gradient_vector_out,
        )

        weight_after, post_trainable_count = _trainable_parameter_digest(
            trainer.model,
            torch,
        )
        if post_trainable_count != trainable_count:
            raise ProbeHold("trainable parameter count changed")

        optimizer_created = trainer.optimizer is not None
        memory = {
            "allocated_before_backward_mib": round(
                allocated_before / (1024**2), 3
            ),
            "reserved_before_backward_mib": round(
                reserved_before / (1024**2), 3
            ),
            "allocated_after_backward_mib": round(
                torch.cuda.memory_allocated() / (1024**2), 3
            ),
            "reserved_after_backward_mib": round(
                torch.cuda.memory_reserved() / (1024**2), 3
            ),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / (1024**2), 3
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved() / (1024**2), 3
            ),
        }

        trainer.model.zero_grad(set_to_none=True)

        receipt = finalize_probe_receipt(
            precondition_check=precondition_check,
            contract_sha256=sha256_file(contract_path),
            execution_spec_sha256=spec_check[
                "execution_spec_sha256"
            ],
            train_sha256=subject["train_sha256"],
            row_index=row_index,
            row_case_id=_case_label(selected, row_index),
            trainable_parameter_count=trainable_count,
            targeted_module_count=coverage["targeted_module_count"],
            loss=sum(microbatch_losses) / len(microbatch_losses),
            microbatch_losses=microbatch_losses,
            microbatches_completed=len(microbatch_losses),
            expected_gradient_accumulation_steps=accumulation_steps,
            gradients_present=gradient_parameter_count > 0,
            nonzero_gradient_parameter_count=(
                nonzero_gradient_parameter_count
            ),
            optimizer_created=optimizer_created,
            weight_digest_before=weight_before,
            weight_digest_after=weight_after,
            cuda_memory=memory,
            output_artifacts_written=False,
        )
        receipt["gradient_parameter_count"] = gradient_parameter_count
        receipt["gradient_summary"] = gradient_summary
        receipt["token_budget"] = token_budget
        receipt["microbatch_row_indices"] = list(
            range(row_index, row_index + accumulation_steps)
        )
        receipt["microbatch_row_ids"] = [
            _case_label(row, row_index + offset)
            for offset, row in enumerate(selected_rows)
        ]
        receipt["base_topology"] = topology
        receipt["acceleration"] = acceleration
        receipt["acceleration_backend_requested"] = acceleration_backend
        receipt["liger"] = liger
        receipt["backward_elapsed_seconds"] = backward_elapsed_seconds
        receipt["pad_to_multiple_of"] = pad_to_multiple_of
        receipt["lora_coverage"] = coverage
        receipt["live_runtime"] = live_runtime
        receipt["live_runtime_check"] = runtime_check
        receipt["scratch_output_directory"] = "TEMPORARY_REMOVED_ON_EXIT"
        receipt["receipt_sha256"] = hashlib.sha256(
            canonical_bytes(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_sha256"
                }
            )
        ).hexdigest()

    del trainer
    del model
    torch.cuda.empty_cache()
    return receipt


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--pad-to-multiple-of", type=int, default=None)
    parser.add_argument(
        "--acceleration-backend",
        choices=("fla_triton", "fla_triton_full"),
        default="fla_triton",
    )
    parser.add_argument("--gradient-vector-out", type=Path)
    parser.add_argument("--liger-path", type=Path)
    parser.add_argument("--liger-wheel", type=Path)
    args = parser.parse_args(argv)

    result = execute_backward_probe(
        args.repo_root,
        train_jsonl=args.train_jsonl,
        row_index=args.row_index,
        pad_to_multiple_of=args.pad_to_multiple_of,
        acceleration_backend=args.acceleration_backend,
        gradient_vector_out=args.gradient_vector_out,
        liger_path=args.liger_path,
        liger_wheel=args.liger_wheel,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except ProbeHold as exc:
        print(
            json.dumps(
                {
                    "schema": "V10_QWEN35_BACKWARD_PROBE_RECEIPT_V1",
                    "status": "HOLD",
                    "reason": str(exc),
                    "effect": "NO_OPTIMIZER_NO_WEIGHT_CHANGE",
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2)
