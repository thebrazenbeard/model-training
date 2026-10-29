"""VERA::TRAININGDAY bounded Qwen3.5 4B QLoRA pilot.

No model is loaded until every supplied corpus shard passes admission. This
runner starts a NEW adapter from a pinned official base: it deliberately does
not inherit the provenance-unresolved R13 research adapter. Each control arm
uses a separate immutable output namespace and the same selected rows/seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from trainingday.admission import AdmissionError, admit, _hash_file
from trainingday.labyrinth_intake import SCHEMA as LABYRINTH_SCHEMA, admit_labyrinth


BASE_ID = "Qwen/Qwen3.5-4B"
RECIPE = "TRAININGDAY_QWEN35_FRESH_QLORA_V1"


def _required_base(binding: Path) -> tuple[dict, Path]:
    if not binding.is_file() or binding.is_symlink():
        raise AdmissionError("base binding missing or symlinked")
    data = json.loads(binding.read_text(encoding="utf-8"))
    if (data.get("schema") != "TRAININGDAY_PINNED_BASE_V1" or data.get("model_id") != BASE_ID
            or not isinstance(data.get("revision"), str) or len(data["revision"]) != 40
            or any(c not in "0123456789abcdef" for c in data["revision"])
            or data.get("loader_class") != "Qwen3_5ForCausalLM"
            or data.get("quantization") != "NF4_DOUBLE_BF16"):
        raise AdmissionError("official base binding incomplete or unpinned")
    directory = Path(data.get("snapshot_path", ""))
    if not directory.is_absolute() or not directory.is_dir() or directory.is_symlink():
        raise AdmissionError("pinned local base snapshot absent")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise AdmissionError("base snapshot custody file hashes missing")
    for name in ("config.json", "tokenizer_config.json"):
        if name not in files:
            raise AdmissionError(f"base {name} hash missing")
    if not any(name.endswith(".safetensors") for name in files):
        raise AdmissionError("base weight tensors not pinned")
    for name, expected in files.items():
        if (not isinstance(name, str) or Path(name).name != name
                or name in {".", ".."} or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise AdmissionError("base file binding malformed")
        path = directory / name
        if not path.is_file() or path.is_symlink() or _hash_file(path) != expected:
            raise AdmissionError(f"base file custody mismatch:{name}")
    frozen_tensors = {name for name in files if name.endswith(".safetensors")}
    actual_tensors = {path.name for path in directory.glob("*.safetensors")}
    if actual_tensors != frozen_tensors:
        raise AdmissionError("unbound or missing safetensors in pinned base snapshot")
    return data, directory


def _check_cuda_free_memory(total_mib: float, free_mib: float) -> float:
    """Check post-CUDA-init free VRAM without demanding impossible free memory.

    CUDA context creation consumes device memory before the first model tensor.
    On a 4096 MiB GPU, keeping an unconditional 3500 MiB post-context floor
    can reject an otherwise idle GPU. Retain the 3900 MiB physical-capacity
    gate; require all but 800 MiB of VRAM on constrained devices, and the
    original 3500 MiB floor on larger GPUs. Model load/optimizer can still
    fail with a real OOM; this changes admission only, not training capacity.
    """
    if total_mib < 3900:
        raise RuntimeError("GPU capacity below the frozen 3900 MiB floor")
    threshold_mib = min(3500, total_mib - 800)
    if free_mib < threshold_mib:
        raise RuntimeError("GPU not sufficiently free; pause only authorized workloads")
    return threshold_mib


def _control_rows(rows: list[dict], mode: str) -> list[dict]:
    if mode not in {"zero", "correct", "corrupt"}:
        raise AdmissionError("unknown neural-control mode")
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row["answer"])
    cases = []
    for row in rows:
        copy = {"prompt": row["prompt"], "response": row["answer"]}
        if mode == "corrupt":
            alternatives = sorted(set(by_family[row["family"]]) - {row["answer"]})
            if not alternatives:
                raise AdmissionError("corrupted-label control needs two distinct answers per family")
            index = int(hashlib.sha256(row["id"].encode()).hexdigest(), 16) % len(alternatives)
            copy["response"] = alternatives[index]
        cases.append(copy)
    return cases


def _parameter_digest(model, torch) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        count += param.numel()
        value = param.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if not count:
        raise RuntimeError("no trainable LoRA tensors")
    return digest.hexdigest(), count


def validate_run(args) -> tuple[dict, list[dict], dict, Path]:
    if not (args.mode == "zero" and args.steps == 0
            or args.mode in {"correct", "corrupt"} and 1 <= args.steps <= 8):
        raise AdmissionError("control steps must be 0 or 1–8")
    if not isinstance(args.cases, int) or args.cases < args.steps * 8 or args.cases > 1024:
        raise AdmissionError("insufficient/more-than-bounded cases")
    if not (0 < args.learning_rate <= 0.0002):
        raise AdmissionError("learning rate outside bounded pilot")
    if not args.output_root.is_dir() or args.output_root.is_symlink():
        raise AdmissionError("isolated output root absent/symlinked")
    if (not args.run_id or len(args.run_id) > 64
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id)
            or (args.output_root / args.run_id).exists()):
        raise AdmissionError("unsafe or occupied run output")
    if args.mode != "zero" and not args.expected_initial_digest:
        raise AdmissionError("trained arms require a bound zero-update initial digest")
    if args.expected_initial_digest and (len(args.expected_initial_digest) != 64
                                        or any(c not in "0123456789abcdef" for c in args.expected_initial_digest)):
        raise AdmissionError("initial digest malformed")
    manifest_head = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest_head.get("schema") == LABYRINTH_SCHEMA:
        summary, selected = admit_labyrinth(
            args.manifest, args.work_root, sample_size=args.cases,
            producer_git_head=args.producer_git_head,
        )
    else:
        summary, selected = admit(args.manifest, args.work_root, sample_size=args.cases)
    cases = _control_rows(selected, args.mode)
    binding, model_path = _required_base(args.base_binding)
    return summary, cases, binding, model_path


def train(args) -> dict:
    summary, cases, base, model_path = validate_run(args)
    preflight = {"status": "SOURCE_ONLY_PREFLIGHT", "recipe": RECIPE,
                 "corpus": summary, "base_model": base["model_id"],
                 "base_revision": base["revision"], "mode": args.mode,
                 "steps": args.steps, "new_optimizer_steps": 0,
                 "training_input_sha256": hashlib.sha256(
                     json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")
                 ).hexdigest()}
    if args.preflight_only:
        return preflight

    # Heavy imports and any GPU allocation start *after* manifest/source gates.
    import psutil
    import torch
    from successor.experiments.r12_adapter_precision import verify_trainable
    from successor.experiments.train_v10_qwen35_authorized import (
        load_training_stack, prepare_sft_rows, validate_lora_target_coverage,
    )
    from successor.experiments.train_v10r2_dev import summarize_optimizer

    if psutil.virtual_memory().available < 9 * 2**30:
        raise RuntimeError("less than 9 GiB free system RAM; do not disrupt other work")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native BF16 CUDA training unavailable")
    gpu_total_mib = torch.cuda.get_device_properties(0).total_memory / (2**20)
    free_gpu, _ = torch.cuda.mem_get_info(0)
    _check_cuda_free_memory(gpu_total_mib, free_gpu / (2**20))

    stack = load_training_stack()
    stack["transformers"].set_seed(args.seed)
    tokenizer = stack["AutoTokenizer"].from_pretrained(model_path)
    if not tokenizer.chat_template:
        raise RuntimeError("pinned tokenizer lacks a chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared, tokens = prepare_sft_rows(tokenizer, cases, max_length=512)
    config = stack["BitsAndBytesConfig"](
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        model_path, quantization_config=config, device_map="auto", dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = stack["prepare_model_for_kbit_training"](
        model, use_gradient_checkpointing=True,
    )
    lora = stack["LoraConfig"](
        r=4, lora_alpha=16, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    output = args.output_root / args.run_id
    if output.exists():
        raise RuntimeError("run output occupied after source preflight")
    output.mkdir(exist_ok=False)
    try:
        cfg = stack["SFTConfig"](
            output_dir=str(output / "trainer-work"),
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            max_steps=max(args.steps, 1), num_train_epochs=1,
            learning_rate=args.learning_rate, lr_scheduler_type="constant",
            warmup_steps=0, optim="adamw_bnb_8bit", bf16=True, tf32=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=512, completion_only_loss=True, packing=False,
            shuffle_dataset=False, train_sampling_strategy="sequential",
            logging_steps=1, save_strategy="no", eval_strategy="no",
            report_to=[], seed=args.seed, data_seed=args.seed,
        )
        trainer = stack["SFTTrainer"](
            model=model, args=cfg,
            train_dataset=stack["Dataset"].from_list(prepared),
            processing_class=tokenizer, peft_config=lora,
        )
        # TRL's constructor may recast trainables to BF16, so the fix belongs
        # after it and before the optimizer captures trainable references.
        precision = verify_trainable(trainer.model, torch, permit_cast=True)
        names = list(getattr(trainer.model, "targeted_module_names", []) or [])
        coverage = validate_lora_target_coverage(names)
        before, count = _parameter_digest(trainer.model, torch)
        if args.expected_initial_digest and before != args.expected_initial_digest:
            raise RuntimeError("control-arm initial FP32 adapter digest mismatch")

        training_result = None
        optimizer_identity = None
        started = time.perf_counter()
        if args.mode != "zero":
            optimizer_identity = summarize_optimizer(trainer.create_optimizer())
            if (optimizer_identity["backend"] != "bitsandbytes"
                    or optimizer_identity["is_paged"] is not False
                    or optimizer_identity["optim_bits"] != 8):
                raise RuntimeError("unexpected optimizer backend/precision/paging")
            training_result = trainer.train()
        observed_steps = int(trainer.state.global_step)
        if observed_steps != args.steps:
            raise RuntimeError(f"optimizer step count mismatch:{observed_steps}!={args.steps}")
        after, count_after = _parameter_digest(trainer.model, torch)
        if count_after != count or ((before == after) != (args.mode == "zero")):
            raise RuntimeError("invalid LoRA tensor-change outcome")
        precision_after = verify_trainable(trainer.model, torch, permit_cast=False)
        adapter_dir = output / "adapter"
        trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
        saved_model = adapter_dir / "adapter_model.safetensors"
        saved_config = adapter_dir / "adapter_config.json"
        if not saved_model.is_file() or not saved_config.is_file():
            raise RuntimeError("adapter save incomplete")
        receipt = {
            **preflight,
            "status": "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED",
            "new_optimizer_steps": observed_steps,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "seed": args.seed, "trainable_parameter_count": count,
            "trainable_digest_before": before, "trainable_digest_after": after,
            "adapter_model_sha256": _hash_file(saved_model),
            "adapter_config_sha256": _hash_file(saved_config),
            "base_binding_sha256": _hash_file(args.base_binding),
            "token_budget": tokens, "target_coverage": coverage,
            "fp32_before": precision, "fp32_after": precision_after,
            "training_loss": float(training_result.training_loss) if training_result else None,
            "optimizer_identity": optimizer_identity,
            "runtime": {"torch": torch.__version__, "transformers": stack["transformers"].__version__},
        }
        (output / "training_receipt.json").write_text(
            json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return receipt
    except Exception:
        # Preserve failed-run artifacts for forensics; never reuse this run_id.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--producer-git-head", default=None,
                        help="Exact Labyrinth PR source head for its corpus schema")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["zero", "correct", "corrupt"], required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--expected-initial-digest", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, default=str))


if __name__ == "__main__":
    main()
