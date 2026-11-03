"""Tiny real TRL SFTTrainer+PEFT post-save/pause/resume source experiment.

Public synthetic tokens and randomly initialized GPT-2 only. This is not a
Qwen model, QLoRA, authorized TRAIN operation, or SFT generalization result.
The orchestrator runs three *separate* CPU-only positive processes to test a
planned 16-step run paused after checkpoint-8 against uninterrupted control,
then a separately isolated negative resume with a corrupted real collator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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
SYNTHETIC_ROWS = 32


def _synthetic_example(index: int) -> dict:
    """A distinct, public token-only row with an observable sample identity."""
    if type(index) is not int or not 0 <= index < SYNTHETIC_ROWS:
        raise ValueError("synthetic sample index must be 0..31")
    # Vocabulary ID 4+index is unique and occurs at a fixed *prompt* offset.
    # No private data, external tokenizer, or model download is involved.
    tokens = [1, 4 + index, 7, 10,
              16 + index, 48 + (index % 8), 56 + (index // 8), 2]
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": [-100] * 4 + tokens[4:],
            "completion_mask": [0] * 4 + [1] * 4}


def _verify_sample_continuity(paused: list, resumed: list, control: list) -> list[int]:
    """Reject reorder, replay, omissions, uniform samples and forged counts.

    Source tests deliberately perturb this comparator. Runtime evidence still
    requires actual forwards, witnessed *after* TRL has built its real batches.
    """
    for sequence, expected in ((paused, PAUSE_STEP),
                               (resumed, TARGET_STEPS - PAUSE_STEP),
                               (control, TARGET_STEPS)):
        if (type(sequence) is not list or len(sequence) != expected
                or any(type(sample) is not int or not 0 <= sample < SYNTHETIC_ROWS
                       for sample in sequence)):
            raise AssertionError("invalid actual model-forward sample witness")
    if len(set(control)) != TARGET_STEPS:
        raise AssertionError("sample witness lacks distinct control examples")
    if paused + resumed != control:
        raise AssertionError("TRL actual model-forward sample sequence changed on resume")
    return control


def _effective_batch_sha256(sample: int, input_ids: list,
                            attention_mask: list, labels: list) -> str:
    """Bind every actually consumed token and loss label to source truth."""
    try:
        expected = _synthetic_example(sample)
    except ValueError as exc:
        raise AssertionError("R16_EFFECTIVE_BATCH_MISMATCH sample identity") from exc
    actual = {"input_ids": input_ids, "attention_mask": attention_mask,
              "labels": labels}
    if any(type(v) is not list or len(v) != 8
           or any(type(token) is not int for token in v) for v in actual.values()):
        raise AssertionError("R16_EFFECTIVE_BATCH_MISMATCH tensor contents invalid")
    if any(actual[key] != expected[key] for key in actual):
        raise AssertionError("R16_EFFECTIVE_BATCH_MISMATCH actual model-forward row/labels")
    return canonical_sha(actual)


def _verify_effective_batch_continuity(paused: list, resumed: list,
                                       control: list) -> list[str]:
    """Full consumed batch hashes, not only per-example ID, must match."""
    for sequence, expected in ((paused, PAUSE_STEP),
                               (resumed, TARGET_STEPS - PAUSE_STEP),
                               (control, TARGET_STEPS)):
        if (type(sequence) is not list or len(sequence) != expected
                or any(type(digest) is not str or len(digest) != 64
                       or any(ch not in "0123456789abcdef" for ch in digest)
                       for digest in sequence)):
            raise AssertionError("invalid effective batch sequence witness")
    if paused + resumed != control:
        raise AssertionError("TRL effective batch sequence changed on resume")
    return control


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

    class ModelForwardSampleWitness:
        """Observe the real GPT-2 forward, not dataloader prefetch or a mock."""

        def __init__(self):
            self.samples: list[int] = []
            self.batch_sha256: list[str] = []

        def observe(self, _model, _args, kwargs):
            tensors = {key: kwargs.get(key) for key in
                       ("input_ids", "attention_mask", "labels")}
            if any(not isinstance(t, torch.Tensor) or tuple(t.shape) != (1, 8)
                   for t in tensors.values()):
                raise AssertionError("R16_EFFECTIVE_BATCH_MISMATCH forward tensor shape")
            rows = {key: tensor.detach().to("cpu").tolist()[0]
                    for key, tensor in tensors.items()}
            sample = rows["input_ids"][1] - 4
            if not 0 <= sample < SYNTHETIC_ROWS:
                raise AssertionError("TRL model-forward sample identity missing")
            digest = _effective_batch_sha256(sample, **rows)
            self.samples.append(sample)
            self.batch_sha256.append(digest)

    class StepWitness(TrainerCallback):
        def __init__(self, sample_probe: ModelForwardSampleWitness):
            self.ends: list[int] = []
            self.optimizer_events = 0
            self.samples_at_step: list[int] = []
            self.batch_sha256_at_step: list[str] = []
            self.sample_probe = sample_probe

        def on_step_end(self, args, state, control, **kwargs):
            self.ends.append(int(state.global_step))
            # One batch and one forward per optimizer step by contract.
            # A prefetch-only or duplicate forward is not credited.
            if len(self.sample_probe.samples) != len(self.ends):
                raise AssertionError("sample forward count does not match optimizer steps")
            self.samples_at_step.append(self.sample_probe.samples[-1])
            self.batch_sha256_at_step.append(self.sample_probe.batch_sha256[-1])
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
    # torch.utils.data.Dataset. The completion-only mask is explicit.
    dataset = Dataset.from_list([_synthetic_example(i) for i in range(SYNTHETIC_ROWS)])

    namespace = ("paused" if phase in {"pause", "resume"} else
                 "negative" if phase == "negative" else "control")
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
    sample_probe = ModelForwardSampleWitness()
    witness = StepWitness(sample_probe)
    trainer = SFTTrainer(
        model=base, args=config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora,
        callbacks=[witness] + ([stopper] if stopper is not None else []),
    )
    if phase == "negative":
        # This is a real TRL collator plus a controlled, synthetic-only fault,
        # not a fake batch or a comparator-only perturbation. It runs solely in
        # a copied checkpoint directory, never the successful resume output.
        real_collator = trainer.data_collator

        class CorruptingResumeCollator:
            def __call__(self, examples):
                batch = real_collator(examples)
                labels = batch.get("labels")
                if (not isinstance(labels, torch.Tensor)
                        or tuple(labels.shape) != (1, 8)
                        or int(labels[0, 5]) < 0):
                    raise AssertionError("negative collator cannot find completion label")
                corrupted = labels.clone()
                corrupted[0, 5] = (int(corrupted[0, 5]) + 1) % 64
                batch["labels"] = corrupted
                return batch

        trainer.data_collator = CorruptingResumeCollator()
    # GPT-2's base object remains inside the PEFT wrapper built by TRL.
    # A forward pre-hook observes the actual input IDs consumed by that model.
    forward_handle = base.register_forward_pre_hook(sample_probe.observe, with_kwargs=True)
    try:
        if phase in {"resume", "negative"}:
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
    finally:
        forward_handle.remove()

    observed = int(trainer.state.global_step)
    required = PAUSE_STEP if phase == "pause" else TARGET_STEPS
    if observed != required or (stopper is not None and not stopper.triggered):
        raise AssertionError(f"{phase}: observed {observed}, expected {required}")
    expected_ends = (list(range(1, 9)) if phase == "pause"
                     else list(range(9, 17)) if phase in {"resume", "negative"}
                     else list(range(1, 17)))
    if witness.ends != expected_ends:
        raise AssertionError(f"{phase}: Trainer stepped outside phase: {witness.ends}")
    if witness.optimizer_events != len(expected_ends):
        raise AssertionError(f"{phase}: optimizer event count not phase length")
    if (witness.samples_at_step != sample_probe.samples
            or len(witness.samples_at_step) != len(expected_ends)):
        raise AssertionError("real TRL batch sequence not linked to optimizer steps")
    if (witness.batch_sha256_at_step != sample_probe.batch_sha256
            or len(witness.batch_sha256_at_step) != len(expected_ends)):
        raise AssertionError("real TRL full effective batch not linked to optimizer steps")
    if phase == "pause":
        custody = create_custody(work / "checkpoint-8", contract_sha256=contract_sha)
        if custody != stopper.custody:
            raise AssertionError("checkpoint changed after TRL on_save")

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
            "model_forward_sample_sequence": witness.samples_at_step,
            "model_forward_effective_batch_sha256": witness.batch_sha256_at_step,
            "sample_witness_from_real_model_forward": True,
            "optimizer_callback_events": witness.optimizer_events,
            "trainable_count": count, "trainable_sha256": h.hexdigest(),
            "tiny_random_model": True, "trl_sfttrainer_exercised": True,
            "qwen_model_exercised": False, "cuda_visible": False,
            "training_authority": False}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pause", "resume", "control", "negative"))
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
        sample_sequence = _verify_sample_continuity(
            pause["model_forward_sample_sequence"],
            resume["model_forward_sample_sequence"],
            control["model_forward_sample_sequence"],
        )
        effective_sequence = _verify_effective_batch_continuity(
            pause["model_forward_effective_batch_sha256"],
            resume["model_forward_effective_batch_sha256"],
            control["model_forward_effective_batch_sha256"],
        )
        # Hostile negative: clone only the already pinned step-8 checkpoint,
        # then run an actual resumed TRL trainer whose real collator corrupts
        # one completion target. Detection must come from the model-forward
        # witness, not from changing a comparison list after the fact.
        negative_checkpoint = root / "negative/trainer-work/checkpoint-8"
        negative_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / "paused/trainer-work/checkpoint-8", negative_checkpoint)
        negative = subprocess.run(
            [sys.executable, "-m", "trainingday.r16_trl_cpu_sft_lifecycle",
             "--phase", "negative", "--root", str(root),
             "--external-sha256", sha],
            text=True, capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        if negative.returncode == 0 or "R16_EFFECTIVE_BATCH_MISMATCH" not in negative.stderr:
            raise AssertionError("real resume-only TRL collator corruption was not detected: "
                                 f"exit={negative.returncode} {negative.stderr[-3000:]}")
        print(json.dumps({"status": "R16_TINY_TRL_SFT_CPU_LIFECYCLE_PASS",
                          "phases": phases, "post_process_custody_sha_verified": True,
                          "tiny_trl_resume_equivalence": True,
                          "real_forward_sample_sequence_verified": True,
                          "real_forward_effective_batch_sequence_verified": True,
                          "resume_only_real_collator_corruption_detected": True,
                          "full_effective_batch_sequence_sha256": canonical_sha({
                              "batches": effective_sequence}),
                          "full_sample_sequence_sha256": canonical_sha({"samples": sample_sequence}),
                          "qwen_model_exercised": False, "gpu_operators": 0,
                          "training_authority": False}, sort_keys=True))


if __name__ == "__main__":
    _main()
