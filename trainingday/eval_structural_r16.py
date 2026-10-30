"""One-shot R16 development evaluation on *new* v2 operation structures.

The 64 cases are selected without opening a model, and the prior 16 scored
cases are used only to exclude their entire operation signatures. No prompt,
expected answer, or generated text is included in the public receipt.

This is a development diagnostic, not the sealed Vera final bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file
from trainingday.checkpoint_custody import canonical_sha, validate_custody
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT, numeric_question
from trainingday.reload_verify import validate_receipt
from trainingday.structural_eval import preregister_r15_fresh_structural64
from trainingday.train_neural import RECIPE, _required_base


BANK_ID_SHA256 = "a975c41c12c177d40a5578bb5936b356fcc868b1a45c1f91c92a26416e14181c"
OLD_ID_SHA256 = "212cc8d173c012743899f2ddf610468bd5fed3e318c35fcf90ce08635fb9165c"
OLD_EVAL_SHA256 = "f7a8078ac9f67ad7fda36368060f80cdcfaf9560dca48a153f1c89b9e1e7573"
CORPUS_SHA256 = "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836"
PROTOCOL = "R16_FRESH_STRUCTURAL64_SCORED16_EXCLUDED_DEV_V1"
R16_RECIPE = "TRAININGDAY_QWEN35_QLORA_R16_CHECKPOINTABLE_MATCHED"
R16_STATUS = "R16_ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
R16_PRODUCER_HEAD = "1d38e4f25b5663bf8a84ff5e139875d4152e3680"


def _old_ids(receipt_path: Path) -> list[str]:
    """Read the old base-scoring receipt inside the isolated evaluator only."""
    if (not receipt_path.is_file() or receipt_path.is_symlink()
            or _hash_file(receipt_path) != OLD_EVAL_SHA256):
        raise AdmissionError("missing/symlinked original base evaluation receipt")
    item = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (item.get("status") != "HELDOUT_DIAGNOSTIC_NOT_FINAL_QUALIFICATION"
            or item.get("corpus_manifest_sha256") != CORPUS_SHA256
            or item.get("case_ids_sha256") != OLD_ID_SHA256
            or item.get("cases") != 16 or item.get("mode") != "base"
            or not isinstance(item.get("base_revision"), str)):
        raise AdmissionError("old evaluator receipt provenance mismatch")
    results = item.get("results")
    if not isinstance(results, list) or len(results) != 16:
        raise AdmissionError("original evaluation rows missing")
    ids = [row.get("id") for row in results]
    if (any(not isinstance(i, str) for i in ids) or len(set(ids)) != 16
            or hashlib.sha256("\n".join(ids).encode()).hexdigest() != OLD_ID_SHA256):
        raise AdmissionError("old evaluated IDs not bound to receipt")
    return ids


def preflight(manifest: Path, old_base_eval: Path) -> dict:
    if _hash_file(manifest) != CORPUS_SHA256:
        raise AdmissionError("pinned v2 manifest changed")
    old = _old_ids(old_base_eval)
    selection = preregister_r15_fresh_structural64(
        manifest, previously_scored_16_ids=old,
    )
    if (selection.get("selected_count") != 64
            or selection.get("excluded_scored_signature_count") != 13
            or selection.get("selected_id_sha256") != BANK_ID_SHA256):
        raise AdmissionError("R16 structural bank selection changed")
    return selection


def _zero_adapter(receipt_path: Path, binding_path: Path, producer_head: str) -> tuple[dict, Path]:
    """Admit an unchanged zero-step adapter without mislabeling it as trained."""
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise AdmissionError("zero receipt missing/symlinked")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    corpus = receipt.get("corpus")
    if (receipt.get("status") != "ADAPTER_SAVED_NOT_INDEPENDENTLY_RELOADED"
            or receipt.get("mode") != "zero" or receipt.get("new_optimizer_steps") != 0
            or receipt.get("trainable_digest_before") != receipt.get("trainable_digest_after")
            or receipt.get("recipe") != RECIPE
            or receipt.get("question_contract") != NUMERIC_QUESTION_CONTRACT
            or not isinstance(corpus, dict)
            or corpus.get("manifest_sha256") != CORPUS_SHA256
            or corpus.get("producer_git_head") != producer_head
            or _hash_file(binding_path) != receipt.get("base_binding_sha256")):
        raise AdmissionError("zero-update control provenance invalid")
    adapter = receipt_path.parent / "adapter"
    if not adapter.is_dir() or adapter.is_symlink():
        raise AdmissionError("zero adapter missing")
    for name, key in (("adapter_model.safetensors", "adapter_model_sha256"),
                      ("adapter_config.json", "adapter_config_sha256")):
        path = adapter / name
        if not path.is_file() or path.is_symlink() or _hash_file(path) != receipt.get(key):
            raise AdmissionError("zero adapter file custody mismatch")
    return receipt, adapter


def _r16_adapter(receipt_path: Path, binding_path: Path, *,
                 producer_head: str, required_mode: str) -> tuple[dict, Path]:
    """Admit an R16 adapter using its actual run/checkpoint custody.

    Byte custody is NOT proof of optimizer resume equivalence, model behavior,
    or fresh-process tensor reload. Those remain separate qualification gates.
    """
    if required_mode not in {"zero", "correct", "corrupt"}:
        raise AdmissionError("unsupported R16 adapter arm")
    if (not receipt_path.is_file() or receipt_path.is_symlink()
            or receipt_path.parent.is_symlink()):
        raise AdmissionError("R16 training receipt missing/unsafe")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    contract_path = receipt_path.parent / "r16-run-contract.json"
    if not contract_path.is_file() or contract_path.is_symlink():
        raise AdmissionError("R16 run contract absent")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    digest = canonical_sha(contract)
    corpus = receipt.get("corpus")
    if (producer_head != R16_PRODUCER_HEAD
            or contract.get("schema") != "VERA_TRAININGDAY_R16_RUN_CONTRACT_V1"
            or contract.get("recipe") != R16_RECIPE
            or contract.get("question_contract") != NUMERIC_QUESTION_CONTRACT
            or contract.get("manifest_sha256") != CORPUS_SHA256
            or contract.get("producer_git_head") != producer_head
            or contract.get("mode") != required_mode
            or contract.get("base_model") != "Qwen/Qwen3.5-4B"
            or not isinstance(contract.get("base_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", contract["base_revision"]) is None
            or contract.get("base_binding_sha256") != _hash_file(binding_path)
            or contract.get("optimizer") != "adamw_bnb_8bit"
            or contract.get("gradient_accumulation") != 8
            or contract.get("lora_r") != 4
            or contract.get("lora_alpha") != 16
            or contract.get("max_length") != 512
            or contract.get("save_strategy") != "steps"
            or contract.get("save_steps") != 8
            or type(contract.get("cases")) is not int
            or not 1 <= contract["cases"] <= 1024
            or type(contract.get("seed")) is not int
            or type(contract.get("learning_rate")) not in (float, int)
            or not 0 < contract["learning_rate"] <= 0.0002
            or not isinstance(contract.get("run_id"), str)
            or contract["run_id"] != receipt_path.parent.name
            or not isinstance(contract.get("training_input_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", contract["training_input_sha256"]) is None
            or receipt.get("r16_run_contract_sha256") != digest
            or receipt.get("status") != R16_STATUS
            or receipt.get("recipe") != R16_RECIPE
            or receipt.get("question_contract") != NUMERIC_QUESTION_CONTRACT
            or receipt.get("mode") != required_mode
            or receipt.get("base_model") != contract.get("base_model")
            or receipt.get("base_revision") != contract.get("base_revision")
            or receipt.get("base_binding_sha256") != contract.get("base_binding_sha256")
            or receipt.get("training_input_sha256") != contract.get("training_input_sha256")
            or not isinstance(corpus, dict)
            or corpus.get("manifest_sha256") != CORPUS_SHA256
            or corpus.get("producer_git_head") != producer_head
            or receipt.get("seed") != contract.get("seed")):
        raise AdmissionError("R16 recipe/run/manifest/base contract mismatch")
    initial = receipt.get("trainable_digest_before")
    final = receipt.get("trainable_digest_after")
    if any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
           for v in (initial, final)):
        raise AdmissionError("R16 tensor digest/initialization invalid")
    adapter = receipt_path.parent / "adapter"
    if not adapter.is_dir() or adapter.is_symlink():
        raise AdmissionError("R16 adapter directory absent")
    for name, key in (("adapter_model.safetensors", "adapter_model_sha256"),
                      ("adapter_config.json", "adapter_config_sha256")):
        path = adapter / name
        if not path.is_file() or path.is_symlink() or _hash_file(path) != receipt.get(key):
            raise AdmissionError("R16 adapter bytes changed")

    if required_mode == "zero":
        if (contract.get("steps") != 0 or receipt.get("new_optimizer_steps") != 0
                or receipt.get("resumed_from_step") != 0 or initial != final
                or contract.get("initial_digest") not in (None, initial)
                or any(key in receipt for key in ("checkpoint_step", "checkpoint_custody_file_sha256"))):
            raise AdmissionError("R16 zero control was not a zero-update")
        return receipt, adapter

    step = contract.get("steps")
    resumed = receipt.get("resumed_from_step")
    if (contract.get("initial_digest") != initial
            or type(step) is not int or step < 8 or step > 32 or step % 8
            or contract["cases"] < step * 8
            or type(resumed) is not int or resumed < 0 or resumed >= step
            or resumed % 8 or receipt.get("new_optimizer_steps") != step - resumed
            or initial == final or receipt.get("checkpoint_step") != step
            or receipt.get("optimizer_resume_equivalence") !=
            "UNVERIFIED_REQUIRES_UNINTERRUPTED_MATCH"
            or not isinstance(receipt.get("optimizer_identity"), dict)):
        raise AdmissionError("R16 optimizer/step/continuation receipt inconsistent")
    checkpoint = receipt_path.parent / "trainer-work" / f"checkpoint-{step}"
    custody_path = receipt_path.parent / "latest-checkpoint-custody.json"
    if (not custody_path.is_file() or custody_path.is_symlink()
            or _hash_file(custody_path) != receipt.get("checkpoint_custody_file_sha256")):
        raise AdmissionError("R16 checkpoint custody receipt tampered")
    custody = json.loads(custody_path.read_text(encoding="utf-8"))
    if (validate_custody(checkpoint, custody, contract_sha256=digest) != step
            or receipt.get("checkpoint_custody_state") != custody.get("state")):
        raise AdmissionError("R16 checkpoint file/step mismatch")
    return receipt, adapter


def evaluate(args) -> dict:
    if args.mode not in {"frozen", "zero", "correct", "corrupt"}:
        raise AdmissionError("invalid R16 control arm")
    if (args.output.exists() or not args.output.parent.is_dir()
            or args.output.parent.is_symlink() or args.output.is_symlink()):
        raise AdmissionError("evaluation output must be a new isolated file")
    bank = preflight(args.manifest, args.old_base_eval)
    base, model_path = _required_base(args.base_binding)
    original = json.loads(args.old_base_eval.read_text(encoding="utf-8"))
    if original["base_revision"] != base["revision"]:
        raise AdmissionError("old evaluation and current base revision differ")

    adapter_dir = None
    if args.mode != "frozen":
        if args.adapter_receipt is None:
            raise AdmissionError("trained arm requires adapter receipt")
        contents = json.loads(args.adapter_receipt.read_text(encoding="utf-8"))
        if contents.get("recipe") == R16_RECIPE:
            receipt, adapter_dir = _r16_adapter(
                args.adapter_receipt, args.base_binding,
                producer_head=args.producer_git_head, required_mode=args.mode,
            )
        elif args.mode == "zero":
            receipt, adapter_dir = _zero_adapter(
                args.adapter_receipt, args.base_binding, args.producer_git_head,
            )
        else:
            receipt, adapter_dir = validate_receipt(
                args.adapter_receipt, args.base_binding,
                expected_recipe=RECIPE,
                expected_question_contract=NUMERIC_QUESTION_CONTRACT,
                expected_manifest_sha256=CORPUS_SHA256,
                expected_producer_git_head=args.producer_git_head,
            )
        if (receipt.get("mode") != args.mode
                or receipt.get("base_revision") != base["revision"]
                or receipt.get("base_model") != base["model_id"]):
            raise AdmissionError("wrong control arm or base provenance")
    elif args.adapter_receipt is not None:
        raise AdmissionError("frozen arm must not load an adapter")

    summary = {
        "status": "R16_SOURCE_PREFLIGHT_NO_MODEL_EXECUTION",
        "protocol": PROTOCOL,
        "bank_status": "NEW_STRUCTURE_DEVELOPMENT_NOT_SEALED_FINAL",
        "selected_count": 64,
        "bank_id_sha256": BANK_ID_SHA256,
        "excluded_previous_signature_count": 13,
        "prior16_id_sha256": OLD_ID_SHA256,
        "old_base_evaluation_sha256": OLD_EVAL_SHA256,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "base_model": base["model_id"],
        "base_revision": base["revision"],
        "question_contract": NUMERIC_QUESTION_CONTRACT,
        "mode": args.mode,
        "adapter_receipt_sha256": _hash_file(args.adapter_receipt) if args.adapter_receipt else None,
        "evaluator_source_sha256": _hash_file(Path(__file__)),
        "selector_source_sha256": _hash_file(Path(__file__).with_name("structural_eval.py")),
    }
    if args.preflight_only:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    # Heavy CUDA work happens only after every provenance/selection gate.
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA inference unavailable")
    if torch.cuda.mem_get_info(0)[0] < 3200 * 2**20:
        raise RuntimeError("not enough unoccupied GPU memory")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, quantization_config=quant, device_map=0, dtype=torch.bfloat16,
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    correct = 0
    nlls: list[float] = []
    # Only aggregate scores leave this function, not prediction/answer text.
    for row in bank["cases"]:
        question = numeric_question(row["prompt"])
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            output = model.generate(
                **tokens, do_sample=False, max_new_tokens=24,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][tokens["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()
        correct += int(bool(re.fullmatch(r"[+-]?\d+", generated))
                       and int(generated) == int(row["expected"]))
        if args.teacher_force_nll:
            completion = str(row["expected"]) + (tokenizer.eos_token or "")
            target_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            prefix_count = tokens["input_ids"].shape[1]
            whole = torch.cat((tokens["input_ids"],
                               torch.tensor([target_ids], device="cuda:0")), dim=1)
            with torch.inference_mode():
                logits = model(input_ids=whole).logits
            pred = logits[0, prefix_count - 1:prefix_count + len(target_ids) - 1]
            nlls.append(float(torch.nn.functional.cross_entropy(
                pred.float(), whole[0, prefix_count:], reduction="mean",
            ).item()))
    summary.update(status="R16_MODEL_SCORED_DEVELOPMENT_ONLY", exact_match=correct)
    if args.teacher_force_nll:
        summary["mean_answer_eos_nll_per_token"] = sum(nlls) / len(nlls)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-base-eval", required=True, type=Path)
    parser.add_argument("--base-binding", required=True, type=Path)
    parser.add_argument("--producer-git-head", required=True)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["frozen", "zero", "correct", "corrupt"], required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--teacher-force-nll", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
