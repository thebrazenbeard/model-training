"""Tiny, independently reserved Labyrinth arithmetic probe for QLoRA arms.

This is an in-distribution diagnostic (not the sealed final bank). It opens
eval plaintext only in this separate evaluation process, never in training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file
from trainingday.labyrinth_intake import admit_labyrinth
from trainingday.reload_verify import validate_receipt
from trainingday.train_neural import _required_base


def _integer_oracle(row: dict) -> int:
    if row.get("kind") != "verified_reasoning" or type(row.get("start")) is not int:
        raise AdmissionError("not a verified reasoning case")
    result = row["start"]
    operations = row.get("operations")
    if not isinstance(operations, list) or not 2 <= len(operations) <= 5:
        raise AdmissionError("heldout operation-chain shape invalid")
    for item in operations:
        if (not isinstance(item, list) or len(item) != 2
                or type(item[1]) is not int):
            raise AdmissionError("heldout operation malformed")
        op, number = item
        if op == "add":
            result += number
        elif op == "subtract":
            result -= number
        elif op == "multiply":
            result *= number
        else:
            raise AdmissionError("heldout operation unknown")
    if type(row.get("answer")) is not int or row["answer"] != result:
        raise AdmissionError("heldout arithmetic answer disagrees with oracle")
    return result


def select_heldout(manifest_path: Path, count: int) -> tuple[list[dict], str]:
    if not 1 <= count <= 16:
        raise AdmissionError("heldout pilot must have 1–16 examples")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    scored = []
    seen = set()
    for shard in manifest["shards"]:
        if shard["split"] != "eval":
            continue
        path = root / shard["path"]
        # Caller must have verified every shard byte hash before this is read.
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row.get("kind") != "verified_reasoning":
                    continue
                _integer_oracle(row)
                case_id = row.get("id")
                if not isinstance(case_id, str) or case_id in seen:
                    raise AdmissionError("heldout case ID missing or duplicate")
                seen.add(case_id)
                score = hashlib.sha256(("r14-heldout-v1:" + case_id).encode()).hexdigest()
                scored.append((score, case_id, row))
    if len(scored) < count:
        raise AdmissionError("insufficient heldout verified cases")
    selected = [row for _, _, row in sorted(scored)[:count]]
    case_digest = hashlib.sha256("\n".join(row["id"] for row in selected).encode()).hexdigest()
    return selected, case_digest


def evaluate(args) -> dict:
    if args.output.exists() or not args.output.parent.is_dir():
        raise AdmissionError("output must be a new file under an existing directory")
    corpus, _ = admit_labyrinth(
        args.manifest, args.work_root, sample_size=1,
        producer_git_head=args.producer_git_head,
    )
    heldout, digest = select_heldout(args.manifest, args.cases)
    binding, base_path = _required_base(args.base_binding)
    adapter_dir = None
    if args.adapter_receipt:
        receipt, adapter_dir = validate_receipt(args.adapter_receipt, args.base_binding)
        if receipt["base_revision"] != binding["revision"]:
            raise AdmissionError("trained adapter base revision mismatch")

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA inference unavailable")
    if torch.cuda.mem_get_info(0)[0] < 3200 * 2**20:
        raise RuntimeError("insufficient unoccupied CUDA memory")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForCausalLM.from_pretrained(
        base_path, quantization_config=quant, device_map=0, dtype=torch.bfloat16,
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    results = []
    for row in heldout:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=12,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        exact = bool(re.fullmatch(r"[+-]?\d+", text)) and int(text) == row["answer"]
        results.append({"id": row["id"], "correct": exact,
                        "expected": row["answer"], "generated": text})
    result = {"status": "HELDOUT_DIAGNOSTIC_NOT_FINAL_QUALIFICATION",
              "case_ids_sha256": digest, "cases": len(results),
              "exact_match": sum(x["correct"] for x in results),
              "mode": "adapter" if adapter_dir else "base",
              "adapter_receipt": str(args.adapter_receipt) if adapter_dir else None,
              "corpus_manifest_sha256": _hash_file(args.manifest),
              "base_revision": binding["revision"], "results": results,
              "provenance": corpus.get("status")}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--producer-git-head", required=True)
    parser.add_argument("--base-binding", required=True, type=Path)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
