"""Standalone fresh-loader check for R16 Qwen3.5 LoRA adapter custody.

Run the CLI in a *new Python process* after training ends. This verifies that
adapter tensors saved to disk can be loaded and reproduce the FP32 trainable
parameter digest. It does not score heldout prompts, attest optimizer resume,
or prove that the fine-tuning improved reasoning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import R16_PRODUCER_HEAD, _r16_adapter
from trainingday.train_neural import _parameter_digest, _required_base


def _admit_reload(receipt_path: Path, binding_path: Path, output_path: Path,
                  *, mode: str, expected_receipt_sha256: str) -> tuple[dict, Path, Path]:
    """Complete all cheap, source-only checks before any CUDA import/allocation."""
    if (not isinstance(expected_receipt_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256) is None
            or not receipt_path.is_file() or receipt_path.is_symlink()
            or _hash_file(receipt_path) != expected_receipt_sha256):
        raise AdmissionError("R16 receipt must match an externally recorded SHA256")
    if (output_path.exists() or output_path.is_symlink()
            or not output_path.parent.is_dir() or output_path.parent.is_symlink()
            or any(part.is_symlink() for part in output_path.parents)):
        raise AdmissionError("fresh-loader verification output must be unoccupied and isolated")
    receipt, adapter_dir = _r16_adapter(
        receipt_path, binding_path,
        producer_head=R16_PRODUCER_HEAD, required_mode=mode,
    )
    base, model_path = _required_base(binding_path)
    if (base["model_id"] != receipt.get("base_model")
            or base["revision"] != receipt.get("base_revision")):
        raise AdmissionError("R16 adapter and pinned model revision differ")
    count = receipt.get("trainable_parameter_count")
    if type(count) is not int or count <= 0:
        raise AdmissionError("R16 trainable parameter count absent/invalid")
    return receipt, adapter_dir, model_path


def _gpu_reload(model_path: Path, adapter_dir: Path) -> tuple[str, int, str]:
    """The sole GPU path: fresh NF4 base load followed by PEFT adapter reload."""
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native BF16 CUDA unavailable for R16 fresh loader")
    free_gpu, _ = torch.cuda.mem_get_info(0)
    if free_gpu < 3200 * 2**20:
        raise RuntimeError("under 3200 MiB torch-visible GPU memory: loader denied")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, quantization_config=quant, device_map=0, dtype=torch.bfloat16,
    )
    reloaded = PeftModel.from_pretrained(
        base_model, adapter_dir, is_trainable=True,
    )
    digest, count = _parameter_digest(reloaded, torch)
    return digest, count, torch.__version__


def verify_r16_reload(receipt_path: Path, binding_path: Path, output_path: Path,
                      *, mode: str, expected_receipt_sha256: str) -> dict:
    """Verify R16 adapter bytes in an isolated new-loader invocation."""
    receipt, adapter_dir, model_path = _admit_reload(
        receipt_path, binding_path, output_path,
        mode=mode, expected_receipt_sha256=expected_receipt_sha256,
    )
    actual, count, runtime_torch = _gpu_reload(model_path, adapter_dir)
    if (actual != receipt["trainable_digest_after"]
            or count != receipt["trainable_parameter_count"]):
        raise RuntimeError("R16 freshly loaded adapter FP32 digest/count mismatch")
    report = {
        "status": "R16_DISK_RELOADED_FP32_TENSORS_MATCH",
        "source_training_receipt_sha256": expected_receipt_sha256,
        "run_contract_sha256": receipt["r16_run_contract_sha256"],
        "mode": mode,
        "optimizer_steps_in_training": receipt["new_optimizer_steps"],
        "base_revision": receipt["base_revision"],
        "base_binding_sha256": receipt["base_binding_sha256"],
        "adapter_model_sha256": receipt["adapter_model_sha256"],
        "fp32_trainable_tensor_sha256": actual,
        "trainable_parameter_count": count,
        "runtime_torch": runtime_torch,
        "optimizer_resume_equivalence": "NOT_ATTESTED_BY_ADAPTER_RELOAD",
        "behavioral_efficacy": "NOT_EVALUATED",
        "process_boundary": "RUN_STANDALONE_CLI_AFTER_TRAINING_FOR_INDEPENDENT_PROBE",
    }
    # Exclusive create prevents a prior verification receipt from being replaced.
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--base-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["zero", "correct", "corrupt"], required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_r16_reload(
        args.receipt, args.base_binding, args.output, mode=args.mode,
        expected_receipt_sha256=args.expected_receipt_sha256,
    ), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
