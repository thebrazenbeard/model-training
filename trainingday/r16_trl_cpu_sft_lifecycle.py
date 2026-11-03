"""Tiny real TRL SFTTrainer+PEFT post-save/pause/resume source experiment.

Public synthetic tokens and randomly initialized GPT-2 only. This is not a
Qwen model, QLoRA, authorized TRAIN operation, or SFT generalization result.
The orchestrator runs three *separate* CPU-only processes to test a planned
16-step run paused after saved checkpoint-8 against an uninterrupted control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from trainingday.checkpoint_custody import (
    canonical_sha, create_custody, validate_custody, write_external_custody,
)
from trainingday.checkpoint_stop import build_checkpoint_stop_callback


SEED = 41
PAUSE_STEP = 8
TARGET_STEPS = 16


def _canonical_state_sha256(value, torch) -> str:
    """Hash actual optimizer/scheduler state without process-local objects.

    Parameter IDs and tensors come from the framework's state_dict; we never
    deserialize a guessed checkpoint or substitute synthetic state values.
    """
    def normalize(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device="cpu").contiguous()
            return {"type": "tensor", "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
        if isinstance(item, dict):
            return {str(key): normalize(part)
                    for key, part in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise AssertionError(f"unsupported framework state field: {type(item).__name__}")

    encoded = json.dumps(normalize(value), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_phase(root: Path, phase: str, *, external_sha256: str | None) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
        TrainerCallback, set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_seed(SEED)

    class StepWitness(TrainerCallback):
        def __init__(self):
            self.ends: list[int] = []
            self.optimizer_events = 0

        def on_step_end(self, args, state, control, **kwargs):
            self.ends.append(int(state.global_step))
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.optimizer_events += 1
            return control

    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"w{i}": i for i in range(4, 64)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", unk_token="[UNK]",
    )
    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_ctx=32, n_embd=32,
        n_layer=1, n_head=4, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    ))
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                      target_modules=["c_attn"], fan_in_fan_out=True,
                      task_type="CAUSAL_LM")
    # A real HF datasets.Dataset is mandatory: TRL SFTTrainer rejects plain
    # torch.utils.data.Dataset. Deliberately omit prebuilt labels: TRL must
    # derive actual -100 prompt masks from completion_mask itself.
    row = {"input_ids": [1, 4, 7, 10, 13, 16, 19, 2],
           "attention_mask": [1] * 8,
           "completion_mask": [0, 0, 0, 0, 1, 1, 1, 1]}
    expected_completion_labels = [-100, -100, -100, -100, 13, 16, 19, 2]
    dataset = Dataset.from_list([dict(row) for _ in range(32)])

    namespace = "paused" if phase in {"pause", "resume"} else "control"
    work = root / namespace / "trainer-work"
    work.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(work), use_cpu=True, max_steps=TARGET_STEPS,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=0.0001, optim="adamw_torch", lr_scheduler_type="constant",
        warmup_steps=0, seed=SEED, data_seed=SEED, dataloader_num_workers=0,
        save_strategy="steps", save_steps=PAUSE_STEP, save_only_model=False,
        logging_strategy="no", eval_strategy="no", report_to="none",
        disable_tqdm=True, remove_unused_columns=False,
        gradient_checkpointing=False, shuffle_dataset=False, packing=False,
        completion_only_loss=True, max_length=32,
    )
    contract = {"schema": "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1",
                "model": "tiny-random-GPT2-TRL-SFT-PEFT-only",
                "planned_steps": TARGET_STEPS, "seed": SEED}
    contract_sha = canonical_sha(contract)
    if phase == "pause":
        (work.parent / "r16-run-contract.json").write_text(
            json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
        )
    stopper = (build_checkpoint_stop_callback(
        TrainerCallback, trainer_work=work, contract_sha256=contract_sha,
        stop_step=PAUSE_STEP,
    ) if phase == "pause" else None)
    witness = StepWitness()
    trainer = SFTTrainer(
        model=base, args=config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora,
        callbacks=[witness] + ([stopper] if stopper is not None else []),
    )
    # Inspect a genuine batch produced by TRL's configured collator. This is
    # the semantic contract for completion-only supervision, not just a copy
    # of the input dataset's label declaration.
    batch = next(iter(trainer.get_train_dataloader()))
    actual_ids = batch["input_ids"][0].tolist()
    actual_labels = batch["labels"][0].tolist()
    if (actual_ids != row["input_ids"]
            or actual_labels != expected_completion_labels):
        raise AssertionError(
            f"TRL completion mask mismatch: ids={actual_ids}, labels={actual_labels}"
        )
    if phase == "resume":
        checkpoint = work / "checkpoint-8"
        external = root / "external-custody.json"
        if (external_sha256 is None or len(external_sha256) != 64
                or not external.is_file() or external.is_symlink()
                or hashlib.sha256(external.read_bytes()).hexdigest() != external_sha256):
            raise AssertionError("external custody receipt pin missing or changed")
        receipt = json.loads(external.read_text(encoding="utf-8"))
        validate_custody(checkpoint, receipt, contract_sha256=contract_sha)
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        trainer.train()

    observed = int(trainer.state.global_step)
    required = PAUSE_STEP if phase == "pause" else TARGET_STEPS
    if observed != required or (stopper is not None and not stopper.triggered):
        raise AssertionError(f"{phase}: observed {observed}, expected {required}")
    expected_ends = (list(range(1, 9)) if phase == "pause"
                     else list(range(9, 17)) if phase == "resume"
                     else list(range(1, 17)))
    if witness.ends != expected_ends:
        raise AssertionError(f"{phase}: Trainer stepped outside phase: {witness.ends}")
    if witness.optimizer_events != len(expected_ends):
        raise AssertionError(f"{phase}: optimizer event count not phase length")
    if phase == "pause":
        custody = create_custody(work / "checkpoint-8", contract_sha256=contract_sha)
        if custody != stopper.custody:
            raise AssertionError("checkpoint changed after TRL on_save")

    optimizer_state = trainer.optimizer.state_dict()
    if (not optimizer_state.get("state")
            or not any("exp_avg" in slot and "exp_avg_sq" in slot
                       for slot in optimizer_state["state"].values())):
        raise AssertionError("real AdamW optimizer moments unavailable")
    if trainer.lr_scheduler is None:
        raise AssertionError("actual Trainer LR scheduler unavailable")
    optimizer_sha = _canonical_state_sha256(optimizer_state, torch)
    scheduler_sha = _canonical_state_sha256(trainer.lr_scheduler.state_dict(), torch)

    adapter = trainer.model
    h = hashlib.sha256()
    count = 0
    for name, param in sorted(adapter.named_parameters()):
        if param.requires_grad:
            count += param.numel()
            h.update(name.encode("utf-8"))
            h.update(param.detach().to(device="cpu", dtype=torch.float32)
                     .contiguous().numpy().tobytes())
    if not count:
        raise AssertionError("TRL did not create trainable LoRA adapter")
    return {"phase": phase, "global_step": observed,
            "step_sequence": witness.ends,
            "optimizer_callback_events": witness.optimizer_events,
            "completion_mask_verified_on_actual_trl_batch": True,
            "optimizer_state_sha256": optimizer_sha,
            "lr_scheduler_state_sha256": scheduler_sha,
            "trainable_count": count, "trainable_sha256": h.hexdigest(),
            "tiny_random_model": True, "trl_sfttrainer_exercised": True,
            "qwen_model_exercised": False, "cuda_visible": False,
            "training_authority": False}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pause", "resume", "control"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--external-sha256")
    options = parser.parse_args()
    if options.phase:
        if options.root is None:
            raise SystemExit("phase needs root")
        result = _run_phase(options.root, options.phase,
                            external_sha256=options.external_sha256)
        print("R16_TRL_CPU_PHASE " + json.dumps(result, sort_keys=True))
        return
    with tempfile.TemporaryDirectory(prefix="trainingday-r16-trl-cpu-") as temp:
        root = Path(temp).resolve()
        phases = []
        sha = None
        for phase in ("pause", "resume", "control"):
            command = [sys.executable, "-m", "trainingday.r16_trl_cpu_sft_lifecycle",
                       "--phase", phase, "--root", str(root)]
            if phase == "resume":
                if sha is None:
                    raise AssertionError("pause receipt pin missing")
                command += ["--external-sha256", sha]
            child = subprocess.run(
                command, text=True, capture_output=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                     "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
            if child.returncode:
                raise AssertionError(f"{phase} process failure ({child.returncode}): "
                                     f"{child.stderr[-5000:]}\n{child.stdout[-2000:]}")
            receipts = [s.removeprefix("R16_TRL_CPU_PHASE ")
                        for s in child.stdout.splitlines()
                        if s.startswith("R16_TRL_CPU_PHASE ")]
            if len(receipts) != 1:
                raise AssertionError(f"{phase}: missing unique receipt")
            phases.append(json.loads(receipts[0]))
            if phase == "pause":
                sha = write_external_custody(
                    root / "paused/trainer-work/checkpoint-8",
                    root / "external-custody.json",
                )["receipt_sha256"]
        pause, resume, control = phases
        if resume["trainable_sha256"] != control["trainable_sha256"]:
            raise AssertionError("TRL fresh resume and uninterrupted LoRA weights differ")
        if resume["trainable_count"] != control["trainable_count"]:
            raise AssertionError("TRL LoRA parameter counts differ")
        if resume["optimizer_state_sha256"] != control["optimizer_state_sha256"]:
            raise AssertionError("TRL optimizer moments differ after fresh 8+8 resume")
        if resume["lr_scheduler_state_sha256"] != control["lr_scheduler_state_sha256"]:
            raise AssertionError("TRL scheduler state differs after fresh 8+8 resume")
        print(json.dumps({"status": "R16_TINY_TRL_SFT_CPU_LIFECYCLE_PASS",
                          "phases": phases, "post_process_custody_sha_verified": True,
                          "tiny_trl_resume_equivalence": True,
                          "completion_only_mask_verified": True,
                          "optimizer_state_resume_equivalence": True,
                          "scheduler_state_resume_equivalence": True,
                          "qwen_model_exercised": False, "gpu_operators": 0,
                          "training_authority": False}, sort_keys=True))


if __name__ == "__main__":
    _main()
