"""Public CPU-only *real Trainer* exercise of the R16 post-save callback.

This uses a randomly initialized tiny GPT-2 and a genuine PEFT LoRA adapter.
It never downloads model weights, touches a GPU, admits a TRAIN corpus, or
qualifies the separate Qwen3.5 SFT/QLoRA runtime.  The orchestrator launches
fresh Python processes for pause, resume and uninterrupted control phases.
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
PLANNED_STEPS = 16
PAUSE_STEP = 8


def _train_phase(root: Path, phase: str, *, custody_sha256: str | None) -> dict:
    # Never expose CUDA to this test, even on a runner with available devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments,
        set_seed,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_seed(SEED)

    class TinyCorpus(torch.utils.data.Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, index):
            # Fixed, local, synthetic token IDs; no model/tokenizer download.
            tokens = torch.tensor([1, 4, 7, 10, 13, 16, 19, 2], dtype=torch.long)
            return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens),
                    "labels": tokens.clone()}

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

    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_ctx=32, n_embd=32,
        n_layer=1, n_head=4, bos_token_id=1, eos_token_id=2,
    ))
    lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                      target_modules=["c_attn"], fan_in_fan_out=True,
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)

    work = root / ("paused" if phase in {"pause", "resume"} else "control") / "trainer-work"
    work.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(work), use_cpu=True,
        max_steps=PLANNED_STEPS, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=0.0001,
        optim="adamw_torch", lr_scheduler_type="constant", warmup_steps=0,
        seed=SEED, data_seed=SEED, dataloader_num_workers=0,
        save_strategy="steps", save_steps=PAUSE_STEP, save_only_model=False,
        save_safetensors=True, logging_strategy="no", eval_strategy="no",
        report_to="none", disable_tqdm=True, remove_unused_columns=False,
    )
    witness = StepWitness()
    contract = {"schema": "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1",
                "model": "random-tiny-GPT2-PEFT-only", "planned_steps": PLANNED_STEPS,
                "seed": SEED}
    contract_sha = canonical_sha(contract)
    if phase == "pause":
        (work.parent / "r16-run-contract.json").write_text(
            json.dumps(contract, sort_keys=True) + "\n", encoding="utf-8"
        )
    stopper = (build_checkpoint_stop_callback(
        TrainerCallback, trainer_work=work, contract_sha256=contract_sha,
        stop_step=PAUSE_STEP,
    ) if phase == "pause" else None)
    callbacks = [witness] + ([stopper] if stopper is not None else [])
    trainer = Trainer(model=model, args=args, train_dataset=TinyCorpus(),
                      callbacks=callbacks)
    if phase == "resume":
        checkpoint = work / "checkpoint-8"
        external = root / "external-custody.json"
        if (custody_sha256 is None or len(custody_sha256) != 64
                or not external.is_file() or external.is_symlink()
                or hashlib.sha256(external.read_bytes()).hexdigest() != custody_sha256):
            raise AssertionError("fresh-process external checkpoint receipt pin mismatch")
        receipt = json.loads(external.read_text(encoding="utf-8"))
        validate_custody(checkpoint, receipt, contract_sha256=contract_sha)
        trainer.train(resume_from_checkpoint=str(checkpoint))
    else:
        trainer.train()

    step = int(trainer.state.global_step)
    expected = PAUSE_STEP if phase == "pause" else PLANNED_STEPS
    if step != expected or (stopper is not None and not stopper.triggered):
        raise AssertionError(f"{phase}: Trainer global step {step}, expected {expected}")
    if phase == "pause":
        check = create_custody(work / "checkpoint-8", contract_sha256=contract_sha)
        if check != stopper.custody:
            raise AssertionError("checkpoint changed after on_save")
    if phase == "resume" and witness.ends != list(range(9, 17)):
        raise AssertionError(f"resume step sequence unexpected: {witness.ends}")
    if phase == "pause" and witness.ends != list(range(1, 9)):
        raise AssertionError(f"pause executed outside 1..8: {witness.ends}")
    if phase == "control" and witness.ends != list(range(1, 17)):
        raise AssertionError(f"uninterrupted steps unexpected: {witness.ends}")
    if witness.optimizer_events != len(witness.ends):
        raise AssertionError("optimizer event count does not match trainer step events")

    h = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        count += parameter.numel()
        h.update(name.encode("utf-8"))
        h.update(parameter.detach().to(dtype=torch.float32, device="cpu")
                 .contiguous().numpy().tobytes())
    if not count:
        raise AssertionError("no trainable LoRA parameters")
    return {"phase": phase, "global_step": step,
            "step_end_sequence": witness.ends,
            "optimizer_callback_events": witness.optimizer_events,
            "trainable_count": count, "trainable_sha256": h.hexdigest(),
            "checkpoint_source_only": True, "model": "tiny-random-GPT2-PEFT",
            "gpu_operators": 0, "qwen_optimizer_steps": 0}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pause", "resume", "control"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--custody-sha256")
    options = parser.parse_args()
    if options.phase:
        if options.root is None:
            raise SystemExit("phase root missing")
        result = _train_phase(options.root, options.phase,
                              custody_sha256=options.custody_sha256)
        print("R16_CPU_PHASE_RECEIPT " + json.dumps(result, sort_keys=True))
        return

    with tempfile.TemporaryDirectory(prefix="trainingday-r16-cpu-") as tmp:
        root = Path(tmp).resolve()
        phases = []
        custody_sha256 = None
        for phase in ("pause", "resume", "control"):
            command = [sys.executable, "-m", "trainingday.r16_cpu_trainer_lifecycle",
                       "--phase", phase, "--root", str(root)]
            if phase == "resume":
                command.extend(["--custody-sha256", custody_sha256])
            child = subprocess.run(command, text=True, capture_output=True,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                        "HF_HUB_OFFLINE": "1"})
            if child.returncode:
                raise AssertionError(
                    f"{phase} fresh process failed ({child.returncode}): "
                    f"{child.stderr[-5000:]}\n{child.stdout[-2000:]}"
                )
            marker = [line.removeprefix("R16_CPU_PHASE_RECEIPT ")
                      for line in child.stdout.splitlines()
                      if line.startswith("R16_CPU_PHASE_RECEIPT ")]
            if len(marker) != 1:
                raise AssertionError(f"{phase} returned no unique receipt: {child.stdout[-2000:]}")
            phases.append(json.loads(marker[0]))
            if phase == "pause":
                external = root / "external-custody.json"
                custody_sha256 = write_external_custody(
                    root / "paused/trainer-work/checkpoint-8", external
                )["receipt_sha256"]
        pause, resume, control = phases
        if resume["trainable_sha256"] != control["trainable_sha256"]:
            raise AssertionError("fresh-process resume and uninterrupted LoRA digests differ")
        if resume["trainable_count"] != control["trainable_count"]:
            raise AssertionError("trainable parameter counts differ")
        print(json.dumps({"status": "R16_TINY_CPU_TRAINER_LIFECYCLE_PASS",
                          "phases": phases, "checkpoint_verified_after_process_exit": True,
                          "exact_resume_equivalence_for_tiny_model": True,
                          "trl_sfttrainer_exercised": False,
                          "qwen_model_exercised": False,
                          "training_authority": False,
                          "gpu_operators": 0}, sort_keys=True))


if __name__ == "__main__":
    _main()
