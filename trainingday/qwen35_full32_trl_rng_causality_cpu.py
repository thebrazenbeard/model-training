"""Source-only stochastic RNG-causality witness for tiny random Qwen3.5 LoRA.

This is deliberately NOT a physical TRAININGDAY authorization path. It uses
generated arithmetic, official tokenizer bytes and a CPU-only width-64 model.
The checkpoint is deleted after this test, not durably stored across sessions.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from trainingday import qwen35_full32_trl_integrated_cpu as integrated
from trainingday import qwen35_full32_trl_optimizer_resume_cpu as resume


SCHEMA = "TRAININGDAY_QWEN35_TINY_FULL32_TRL_RNG_CAUSALITY_V1"
STOCHASTIC_LORA_DROPOUT = 0.2


def assert_causal_resume(*, initial: str, first: str, direct: dict,
                         restored: dict, rng_omitted: dict) -> None:
    """Require a non-noop second step, exact restoration, and RNG-dependent change."""
    for name, digest in (("initial", initial), ("first", first)):
        if not isinstance(digest, str) or len(digest) != 64:
            raise AssertionError(f"{name} adapter SHA missing")
    for result in (direct, restored, rng_omitted):
        if (not isinstance(result, dict) or
                not all(isinstance(result.get(k), str) and len(result[k]) == 64
                        for k in ("adapter_sha256", "optimizer_state_sha256", "logits_sha256"))):
            raise AssertionError("missing actual trained state digests")
    if initial == first:
        raise AssertionError("first optimizer step did not modify LoRA")
    if direct["adapter_sha256"] == first:
        raise AssertionError("second optimizer step did not modify LoRA")
    if any(direct[k] != restored[k] for k in
           ("adapter_sha256", "optimizer_state_sha256", "logits_sha256")):
        raise AssertionError("RNG-restored fresh child failed uninterrupted parity")
    if rng_omitted["adapter_sha256"] == direct["adapter_sha256"]:
        raise AssertionError("omitting RNG state did not change stochastic LoRA training")
    if (restored.get("after_resume_optimizer_steps") != [2] or
            rng_omitted.get("after_resume_optimizer_steps") != [2]):
        raise AssertionError("the child did not take a real second AdamW step")


def _child(path: Path, *, omit_rng: bool) -> dict:
    integrated._offline_cpu()
    import torch
    torch.set_num_threads(2)
    if omit_rng:
        # A deliberate NEGATIVE control: the original resume code still checks
        # checkpoint hashes, loads adapter and AdamW moments, and trains, but
        # we suppress *only* applying the saved Torch RNG bytes. This code is
        # separate from the production-shaped resume entrypoint.
        from unittest.mock import patch
        with patch.object(torch, "set_rng_state", return_value=None):
            result = resume._resume(path)
        result["control"] = "RNG_RESTORE_OMITTED_NEGATIVE"
    else:
        result = resume._resume(path)
        result["control"] = "RNG_RESTORED"
    return result


def _invoke(path: Path, mode: str) -> dict:
    p = subprocess.run([sys.executable, "-m",
                        "trainingday.qwen35_full32_trl_rng_causality_cpu",
                        mode, str(path)], capture_output=True, text=True,
                       timeout=180, check=False, env=os.environ.copy())
    if p.returncode:
        raise AssertionError(f"stochastic {mode} child failed: {p.stderr[-1200:]}")
    result = json.loads(p.stdout.splitlines()[-1])
    if result.get("pid") == os.getpid() or result.get("cuda_initialized"):
        raise AssertionError("stochastic child identity/CUDA boundary failed")
    return result


def run() -> dict:
    integrated._offline_cpu()
    import torch
    torch.set_num_threads(2)
    model, trainable, batches, frozen_sha = resume._context(
        torch, lora_dropout=STOCHASTIC_LORA_DROPOUT)
    if abs(model.peft_config["default"].lora_dropout - STOCHASTIC_LORA_DROPOUT) > 1e-12:
        raise AssertionError("stochastic dropout was not installed in PEFT adapter")
    opt = torch.optim.AdamW([p for _, p in trainable], lr=resume.LR)
    initial = resume._trainable_digest(model)
    with tempfile.TemporaryDirectory(prefix="trainingday-qwen35-rng-causal-") as tmp:
        path = Path(tmp)
        first_loss = resume._step(torch, model, opt, batches[0])
        first = resume._trainable_digest(model)
        resume._save(path, model, opt, batches[1], frozen_sha,
                     [n for n, _ in trainable], torch)
        second_loss = resume._step(torch, model, opt, batches[1])
        ids = torch.tensor([integrated._synthetic_rows(integrated._load_tokenizer(
            Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])))[0]["input_ids"]],
                           dtype=torch.long)
        direct = {
            "adapter_sha256": resume._trainable_digest(model),
            "optimizer_state_sha256": resume._optimizer_digest(opt),
            "logits_sha256": integrated._sha_tensor(integrated._probe_logits(model, ids, torch)),
        }
        del model, opt, trainable, batches, ids
        gc.collect()
        restored = _invoke(path, "--restore-rng")
        omitted = _invoke(path, "--omit-rng-control")
        if abs(restored["second_step_loss"] - second_loss) > 1e-7:
            raise AssertionError("RNG-restored second masked loss differs")
        assert_causal_resume(initial=initial, first=first, direct=direct,
                             restored=restored, rng_omitted=omitted)
    return {
        "schema": SCHEMA, "status": "STOCHASTIC_RNG_RESTORED_CAUSAL_PARITY_PASS",
        "tiny_model": "32_LAYER_24_GDN_8_ATTN_WIDTH64_RANDOM_152_LORA_B",
        "lora_dropout": STOCHASTIC_LORA_DROPOUT, "optimizer_steps": 2,
        "first_step_loss": first_loss, "second_step_loss": second_loss,
        "first_adapter_sha256": first, "direct_and_restored_adapter_sha256": direct["adapter_sha256"],
        "direct_and_restored_optimizer_sha256": direct["optimizer_state_sha256"],
        "direct_and_restored_logits_sha256": direct["logits_sha256"],
        "rng_omitted_adapter_sha256": omitted["adapter_sha256"],
        "rng_omitted_changed_training": True, "checkpoint_survives_test": False,
        "official_pretrained_weights_loaded": False, "training_authorized": False,
        "cuda_initialized": False,
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--restore-rng":
        print(json.dumps(_child(Path(sys.argv[2]), omit_rng=False), sort_keys=True))
    elif len(sys.argv) == 3 and sys.argv[1] == "--omit-rng-control":
        print(json.dumps(_child(Path(sys.argv[2]), omit_rng=True), sort_keys=True))
    elif len(sys.argv) == 1:
        print(json.dumps(run(), sort_keys=True))
    else:
        raise SystemExit("stochastic CPU test only: no physical training entrypoint")
