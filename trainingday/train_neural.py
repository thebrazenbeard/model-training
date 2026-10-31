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
import math
import time
from collections import defaultdict
from pathlib import Path

from trainingday.admission import AdmissionError, admit, _hash_file
from trainingday.labyrinth_intake import SCHEMA as LABYRINTH_SCHEMA, admit_labyrinth
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT, numeric_question
from trainingday.checkpoint_custody import canonical_sha, create_custody, validate_custody


BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RECIPE = "TRAININGDAY_QWEN35_FRESH_QLORA_R15_NUMERIC_MATCHED"
R16_RECIPE = "TRAININGDAY_QWEN35_QLORA_R16_CHECKPOINTABLE_MATCHED"
R16_V2_MANIFEST_SHA256 = "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836"
R16_PRODUCER_HEAD = "1d38e4f25b5663bf8a84ff5e139875d4152e3680"


def _r16(args) -> bool:
    return getattr(args, "r16_checkpointable", False) is True


def _r16_resume_requested(args) -> bool:
    return getattr(args, "resume_checkpoint", None) is not None


def _r16_contract(args, preflight: dict) -> dict:
    if not _r16(args):
        raise AdmissionError("R16 checkpoint contract requested for R15")
    return {
        "schema": "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1",
        "recipe": R16_RECIPE,
        "question_contract": NUMERIC_QUESTION_CONTRACT,
        "manifest_sha256": preflight["corpus"]["manifest_sha256"],
        "producer_git_head": args.producer_git_head,
        "base_model": preflight["base_model"],
        "base_revision": preflight["base_revision"],
        "base_binding_sha256": _hash_file(args.base_binding),
        "mode": args.mode, "steps": args.steps, "cases": args.cases,
        "seed": args.seed, "learning_rate": args.learning_rate,
        "run_id": args.run_id,
        "initial_digest": args.expected_initial_digest,
        "training_input_sha256": preflight["training_input_sha256"],
        "optimizer": "adamw_bnb_8bit", "gradient_accumulation": 8,
        "lora_r": 4, "lora_alpha": 16, "max_length": 512,
        "save_strategy": "steps", "save_steps": 8,
    }


def _r16_prepare_output(args, preflight: dict) -> tuple[Path, Path | None, int, str]:
    """Gate any checkpoint deserialization before importing GPU dependencies."""
    output = args.output_root / args.run_id
    contract = _r16_contract(args, preflight)
    contract_hash = canonical_sha(contract)
    contract_file = output / "r16-run-contract.json"
    if not _r16_resume_requested(args):
        if output.exists() or output.is_symlink():
            raise AdmissionError("R16 fresh output already exists")
        output.mkdir(exist_ok=False)
        contract_file.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
        return output, None, 0, contract_hash
    checkpoint = Path(args.resume_checkpoint)
    custody_file = getattr(args, "resume_custody", None)
    expected_custody_sha = getattr(args, "resume_custody_sha256", None)
    if (not output.is_dir() or output.is_symlink() or not contract_file.is_file()
            or contract_file.is_symlink() or not checkpoint.is_dir()
            or checkpoint.is_symlink() or checkpoint.parent != output / "trainer-work"
            or not custody_file or not expected_custody_sha
            or any(path.is_symlink() for path in output.parents)):
        raise AdmissionError("R16 resume output/checkpoint/custody missing or unconfined")
    if json.loads(contract_file.read_text(encoding="utf-8")) != contract:
        raise AdmissionError("R16 run contract mismatch")
    custody_path = Path(custody_file)
    if (not custody_path.is_file() or custody_path.is_symlink()
            or custody_path.is_relative_to(output)
            or _hash_file(custody_path) != expected_custody_sha):
        raise AdmissionError("R16 externally pinned checkpoint receipt absent or changed")
    custody = json.loads(custody_path.read_text(encoding="utf-8"))
    step = validate_custody(checkpoint, custody, contract_sha256=contract_hash)
    if step % 8 or step >= args.steps:
        raise AdmissionError("R16 resume step outside bounded schedule")
    return output, checkpoint, step, contract_hash


def _required_base(binding: Path) -> tuple[dict, Path]:
    if (not binding.is_file()
            or any(path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                   for path in (binding, *binding.parents))):
        raise AdmissionError("base binding missing or symlink/junction ancestor")
    data = json.loads(binding.read_text(encoding="utf-8"))
    if (data.get("schema") != "TRAININGDAY_PINNED_BASE_V1" or data.get("model_id") != BASE_ID
            or data.get("revision") != BASE_REVISION
            or data.get("loader_class") != "Qwen3_5ForCausalLM"
            or data.get("quantization") != "NF4_DOUBLE_BF16"):
        raise AdmissionError("official base revision or binding incomplete or unpinned")
    directory = Path(data.get("snapshot_path", ""))
    if (not directory.is_absolute() or not directory.is_dir()
            or directory.is_symlink()
            or any(parent.is_symlink() for parent in directory.parents)):
        raise AdmissionError("pinned local base snapshot absent")
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise AdmissionError("base snapshot custody file hashes missing")
    for name in ("config.json", "tokenizer_config.json"):
        if name not in files:
            raise AdmissionError(f"base {name} hash missing")
    if not any(name.endswith(".safetensors") for name in files):
        raise AdmissionError("base weight tensors not pinned")
    # A loader may read versioned tokenizer JSON and nested chat templates,
    # not only *.safetensors. Require an exact recursive set of regular files,
    # with canonical POSIX-relative names and a digest for every file.
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or name.startswith("/")
                or "\\" in name or ":" in name or "\x00" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise AdmissionError("base file binding malformed")
        path = directory.joinpath(*name.split("/"))
        if (not path.is_file() or path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents
                       if parent == directory or directory in parent.parents)
                or not path.resolve().is_relative_to(directory.resolve())
                or _hash_file(path) != expected):
            raise AdmissionError(f"base file custody mismatch:{name}")

    # The fast tokenizer may select a versioned filename from this JSON rather
    # than tokenizer.json. Require every named candidate to be byte-pinned so
    # a loader/version change cannot select an unbound local asset.
    try:
        tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError("base tokenizer configuration unreadable") from exc
    if not isinstance(tokenizer_config, dict):
        raise AdmissionError("base tokenizer configuration invalid")
    fast_files = tokenizer_config.get("fast_tokenizer_files", [])
    if fast_files is None:
        fast_files = []
    if (not isinstance(fast_files, list)
            or any(not isinstance(name, str) or name not in files
                   for name in fast_files)):
        raise AdmissionError("base fast tokenizer candidate unpinned")

    actual = set()

    def visit(folder: Path, prefix: tuple[str, ...]) -> None:
        try:
            entries = tuple(folder.iterdir())
        except OSError as exc:
            raise AdmissionError("base snapshot inventory unreadable") from exc
        for entry in entries:
            if entry.is_symlink():
                raise AdmissionError("symlink in pinned base snapshot")
            relative = prefix + (entry.name,)
            if entry.is_dir():
                visit(entry, relative)
            elif entry.is_file():
                actual.add("/".join(relative))
            else:
                raise AdmissionError("unsupported base snapshot entry")

    visit(directory, ())
    if actual != set(files):
        raise AdmissionError("unbound or missing files in pinned base snapshot")
    return data, directory


def _control_rows(rows: list[dict], mode: str) -> list[dict]:
    if mode not in {"zero", "correct", "corrupt"}:
        raise AdmissionError("unknown neural-control mode")
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row["answer"])
    cases = []
    for row in rows:
        copy = {"prompt": numeric_question(row["prompt"]), "response": row["answer"]}
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
        if not torch.isfinite(value).all().item():
            raise RuntimeError("nonfinite trainable LoRA parameter")
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if not count:
        raise RuntimeError("no trainable LoRA tensors")
    return digest.hexdigest(), count


def _check_base_unchanged(binding: Path, expected_sha256: str,
                          expected_data: dict, expected_directory: Path) -> None:
    """Detect persistent loader-side mutation, not same-principal atomic custody."""
    if _hash_file(binding) != expected_sha256:
        raise AdmissionError("base binding custody changed during training")
    observed, directory = _required_base(binding)
    if observed != expected_data or directory != expected_directory:
        raise AdmissionError("base snapshot custody changed during training")


def validate_run(args) -> tuple[dict, list[dict], dict, Path]:
    if (type(args.steps) is not int or type(args.cases) is not int
            or type(args.seed) is not int):
        raise AdmissionError("steps, cases, and seed must be exact integers")
    step_cap = 32 if _r16(args) else 8
    if not (args.mode == "zero" and args.steps == 0
            or args.mode in {"correct", "corrupt"} and 1 <= args.steps <= step_cap):
        raise AdmissionError(f"control steps must be 0 or 1–{step_cap}")
    if not isinstance(args.cases, int) or args.cases < args.steps * 8 or args.cases > 1024:
        raise AdmissionError("insufficient/more-than-bounded cases")
    if not (0 < args.learning_rate <= 0.0002):
        raise AdmissionError("learning rate outside bounded pilot")
    if not args.output_root.is_dir() or args.output_root.is_symlink():
        raise AdmissionError("isolated output root absent/symlinked")
    if (not args.run_id or len(args.run_id) > 64
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.run_id)
            or ((args.output_root / args.run_id).exists() and
                not (_r16(args) and _r16_resume_requested(args)))):
        raise AdmissionError("unsafe or occupied run output")
    if _r16(args):
        if (_hash_file(args.manifest) != R16_V2_MANIFEST_SHA256
                or args.producer_git_head != R16_PRODUCER_HEAD):
            raise AdmissionError("R16 needs exact v2 manifest and producer")
        if args.mode in {"correct", "corrupt"} and (args.steps < 8 or args.steps % 8):
            raise AdmissionError("R16 trained steps must be checkpoint boundaries 8–32")
        if _r16_resume_requested(args) and args.mode == "zero":
            raise AdmissionError("zero-update arm cannot resume")
    elif (_r16_resume_requested(args) or getattr(args, "resume_custody", None)
          or getattr(args, "resume_custody_sha256", None)):
        raise AdmissionError("R15 cannot resume optimizer state")
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
    # The zero and trained arms must use one frozen source selection. A weight
    # digest alone cannot bind the admitted corpus or intended control inputs.
    # This is a local equality commitment, not independent rights attestation.
    frozen_source = {
        "schema": "TRAININGDAY_R15_CONTROL_SOURCE_V1",
        "question_contract": NUMERIC_QUESTION_CONTRACT,
        "manifest_sha256": summary["manifest_sha256"],
        "selected_id_sha256": summary["selected_id_sha256"],
        "selected_count": summary["selected_count"],
        "producer_identity": (getattr(args, "producer_git_head", None)
                              or summary.get("source_commit")),
        "base_model": base["model_id"], "base_revision": base["revision"],
        "base_binding_sha256": _hash_file(args.base_binding),
        "seed": args.seed, "cases": args.cases,
        "selected_prompt_sha256": hashlib.sha256(json.dumps(
            [item["prompt"] for item in cases], ensure_ascii=False,
            separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }
    control_source_sha256 = canonical_sha(frozen_source)
    if not _r16(args) and args.mode != "zero":
        expected = getattr(args, "expected_control_source_sha256", None)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)
                or expected != control_source_sha256):
            raise AdmissionError("trained arm control source does not match zero arm")
    preflight = {"status": "SOURCE_ONLY_PREFLIGHT",
                 "recipe": R16_RECIPE if _r16(args) else RECIPE,
                 "question_contract": NUMERIC_QUESTION_CONTRACT,
                 "corpus": summary, "base_model": base["model_id"],
                 "base_revision": base["revision"], "mode": args.mode,
                 "steps": args.steps, "new_optimizer_steps": 0,
                 "control_source_sha256": control_source_sha256,
                 "training_input_sha256": hashlib.sha256(
                     json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")
                 ).hexdigest()}
    if args.preflight_only:
        if _r16(args) and _r16_resume_requested(args):
            # A read-only, fully checked checkpoint resume preflight.
            _r16_prepare_output(args, preflight)
        return preflight

    r16_output = None
    resume_checkpoint = None
    resumed_from_step = 0
    r16_contract_sha = None
    if _r16(args):
        r16_output, resume_checkpoint, resumed_from_step, r16_contract_sha = (
            _r16_prepare_output(args, preflight)
        )

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
    if torch.cuda.get_device_properties(0).total_memory < 3900 * 2**20:
        raise RuntimeError("GPU capacity below the frozen 3900 MiB floor")
    free_gpu, _ = torch.cuda.mem_get_info(0)
    # Windows WDDM reports about 3306 MiB torch-visible free VRAM while
    # nvidia-smi reports about 3962 MiB. An NF4 Qwen3.5-4B load bounded
    # to 3300 MiB completed all 426 tensors on CUDA (2977 MiB allocated).
    # This floor permits the zero arm; backward may still be OOM.
    if free_gpu < 3200 * 2**20:
        raise RuntimeError("less than 3200 MiB torch-visible free VRAM; do not disrupt other work")

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
    # Force the entire 4-bit base onto CUDA for training: auto placement
    # rejected a partial CPU/disk map even though capped CUDA-only load fit.
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        model_path, quantization_config=config, device_map=0, dtype=torch.bfloat16,
    )
    _check_base_unchanged(args.base_binding, frozen_source["base_binding_sha256"],
                          base, model_path)
    model.config.use_cache = False
    model = stack["prepare_model_for_kbit_training"](
        model, use_gradient_checkpointing=True,
    )
    lora = stack["LoraConfig"](
        r=4, lora_alpha=16, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    output = args.output_root / args.run_id
    if not _r16(args):
        if output.exists():
            raise RuntimeError("run output occupied after source preflight")
        output.mkdir(exist_ok=False)
    else:
        if output != r16_output:
            raise RuntimeError("R16 output contract path changed")
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
            logging_steps=1,
            save_strategy="steps" if _r16(args) else "no",
            save_steps=8 if _r16(args) else 500,
            eval_strategy="no",
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
            training_result = trainer.train(
                resume_from_checkpoint=str(resume_checkpoint)
                if _r16(args) and resume_checkpoint else None
            )
        if type(trainer.state.global_step) is not int:
            raise RuntimeError("optimizer step counter is not an exact integer")
        observed_steps = trainer.state.global_step
        if observed_steps != args.steps:
            raise RuntimeError(f"optimizer step count mismatch:{observed_steps}!={args.steps}")
        after, count_after = _parameter_digest(trainer.model, torch)
        if count_after != count or ((before == after) != (args.mode == "zero")):
            raise RuntimeError("invalid LoRA tensor-change outcome")
        precision_after = verify_trainable(trainer.model, torch, permit_cast=False)
        training_loss = float(training_result.training_loss) if training_result else None
        if training_loss is not None and not math.isfinite(training_loss):
            raise RuntimeError("nonfinite training loss")
        _check_base_unchanged(args.base_binding, frozen_source["base_binding_sha256"],
                              base, model_path)
        adapter_dir = output / "adapter"
        trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
        saved_model = adapter_dir / "adapter_model.safetensors"
        saved_config = adapter_dir / "adapter_config.json"
        if not saved_model.is_file() or not saved_config.is_file():
            raise RuntimeError("adapter save incomplete")
        receipt = {
            **preflight,
            "status": ("R16_ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
                       if _r16(args) else "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"),
            "new_optimizer_steps": observed_steps - resumed_from_step,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "seed": args.seed, "trainable_parameter_count": count,
            "trainable_digest_before": before, "trainable_digest_after": after,
            "adapter_model_sha256": _hash_file(saved_model),
            "adapter_config_sha256": _hash_file(saved_config),
            "base_binding_sha256": _hash_file(args.base_binding),
            "token_budget": tokens, "target_coverage": coverage,
            "fp32_before": precision, "fp32_after": precision_after,
            "training_loss": training_loss,
            "optimizer_identity": optimizer_identity,
            "runtime": {"torch": torch.__version__, "transformers": stack["transformers"].__version__},
        }
        if _r16(args):
            receipt["r16_run_contract_sha256"] = r16_contract_sha
            receipt["resumed_from_step"] = resumed_from_step
            receipt["optimizer_resume_equivalence"] = "UNVERIFIED_REQUIRES_UNINTERRUPTED_MATCH"
            if args.mode != "zero":
                checkpoint_path = output / "trainer-work" / f"checkpoint-{observed_steps}"
                custody = create_custody(checkpoint_path, contract_sha256=r16_contract_sha)
                custody_path = output / "latest-checkpoint-custody.json"
                custody_path.write_text(json.dumps(custody, sort_keys=True, indent=2) + "\n",
                                        encoding="utf-8")
                receipt["checkpoint_custody_file_sha256"] = _hash_file(custody_path)
                receipt["checkpoint_step"] = custody["step"]
                receipt["checkpoint_custody_state"] = custody["state"]
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
    parser.add_argument("--expected-control-source-sha256", default=None,
                        help="Frozen control_source_sha256 emitted by the zero-update arm")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--r16-checkpointable", action="store_true",
                        help="Bounded R16 experiment: 8–32 steps with optimizer/RNG checkpoint custody")
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Exact externally inventoried R16 Trainer checkpoint")
    parser.add_argument("--resume-custody", type=Path,
                        help="Externally pinned checksum receipt (not inside run output)")
    parser.add_argument("--resume-custody-sha256",
                        help="Expected SHA256 of the separate checkpoint custody receipt")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, default=str))


if __name__ == "__main__":
    main()
