"""Fresh-process QLoRA adapter reload with exact tensor and custody checks.

This is intentionally separate from the training process: a save receipt is
not proof that the adapter can be reconstructed and used later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file
from trainingday.train_neural import _parameter_digest, _required_base


def validate_receipt(
    receipt_path: Path, binding_path: Path, *,
    expected_recipe: str | None = None,
    expected_question_contract: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_producer_git_head: str | None = None,
) -> tuple[dict, Path]:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise AdmissionError("saved training receipt missing or symlinked")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
            or receipt.get("mode") not in {"correct", "corrupt"}
            or not isinstance(receipt.get("new_optimizer_steps"), int)
            or receipt["new_optimizer_steps"] < 1):
        raise AdmissionError("no actual trained adapter indicated")
    if not binding_path.is_file() or binding_path.is_symlink():
        raise AdmissionError("base binding unavailable")
    if _hash_file(binding_path) != receipt.get("base_binding_sha256"):
        raise AdmissionError("base binding changed since training")
    # Optional protocol expectations are used by the R15 evaluator. Leave
    # generic R14 fresh-reload custody checks backward compatible.
    if expected_recipe is not None and receipt.get("recipe") != expected_recipe:
        raise AdmissionError("training recipe differs from evaluation protocol")
    if (expected_question_contract is not None
            and receipt.get("question_contract") != expected_question_contract):
        raise AdmissionError("training question contract differs from evaluator")
    corpus = receipt.get("corpus")
    if expected_manifest_sha256 is not None and (
            not isinstance(corpus, dict)
            or corpus.get("manifest_sha256") != expected_manifest_sha256):
        raise AdmissionError("training corpus manifest differs from evaluator")
    if expected_producer_git_head is not None and (
            not isinstance(corpus, dict)
            or corpus.get("producer_git_head") != expected_producer_git_head):
        raise AdmissionError("training corpus producer differs from evaluator")
    adapter_dir = receipt_path.parent / "adapter"
    if not adapter_dir.is_dir() or adapter_dir.is_symlink():
        raise AdmissionError("adapter directory absent")
    for filename, key in (("adapter_model.safetensors", "adapter_model_sha256"),
                          ("adapter_config.json", "adapter_config_sha256")):
        file_path = adapter_dir / filename
        if (not file_path.is_file() or file_path.is_symlink()
                or _hash_file(file_path) != receipt.get(key)):
            raise AdmissionError(f"adapter custody mismatch:{filename}")
    expected = receipt.get("trainable_digest_after")
    if (not isinstance(expected, str) or len(expected) != 64
            or any(ch not in "0123456789abcdef" for ch in expected)):
        raise AdmissionError("trained adapter tensor digest malformed")
    return receipt, adapter_dir


def verify(receipt_path: Path, binding_path: Path, output_path: Path) -> dict:
    receipt, adapter_dir = validate_receipt(receipt_path, binding_path)
    if (output_path.exists() or output_path.is_symlink()
            or not output_path.parent.is_dir() or output_path.parent.is_symlink()):
        raise AdmissionError("independent reload receipt target occupied or unsafe")
    binding, model_path = _required_base(binding_path)
    if binding["revision"] != receipt.get("base_revision"):
        raise AdmissionError("pinned model revision differs from training")

    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 unavailable for exact fresh reload")
    gpu_free, _ = torch.cuda.mem_get_info(0)
    if gpu_free < 3200 * 2**20:
        raise RuntimeError("less than 3200 MiB torch-visible free VRAM")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, quantization_config=quant, device_map=0, dtype=torch.bfloat16,
    )
    reloaded = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    actual, count = _parameter_digest(reloaded, torch)
    if actual != receipt["trainable_digest_after"]:
        raise RuntimeError("freshly reloaded adapter tensor digest mismatch")
    if count != receipt["trainable_parameter_count"]:
        raise RuntimeError("freshly reloaded trainable count mismatch")
    report = {
        "status": "INDEPENDENT_ADAPTER_RELOAD_VERIFIED",
        "source_training_receipt": str(receipt_path.resolve()),
        "training_mode": receipt["mode"],
        "original_optimizer_steps": receipt["new_optimizer_steps"],
        "actual_fp32_tensor_sha256": actual,
        "matched_training_fp32_sha256": receipt["trainable_digest_after"],
        "trainable_parameters": count,
        "adapter_sha256": receipt["adapter_model_sha256"],
        "base_binding_sha256": receipt["base_binding_sha256"],
        "runtime_torch": torch.__version__,
        "behavioral_efficacy": "NOT_MEASURED",
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--base-binding", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt, args.base_binding, args.output), indent=2))


if __name__ == "__main__":
    main()
