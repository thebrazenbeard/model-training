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
from trainingday.train_neural import RECIPE, _required_base
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT, numeric_question


# This is an exact-corpus R15 development experiment. Do not accidentally
# call a v1/v3 corpus a held-out v2 structural evaluation.
R15_V2_MANIFEST_SHA256 = "90b55cf8742f9222c91f88e5727dcd0113322e4721e4fd3d1bdcdf65f40d0836"


def _require_r15_v2_manifest(manifest_path: Path) -> str:
    digest = _hash_file(manifest_path)
    if digest != R15_V2_MANIFEST_SHA256:
        raise AdmissionError("evaluator requires pinned v2 corpus manifest")
    return digest


def _eval_contract_metadata(protocol: str) -> dict:
    if protocol not in {"v2", "v3"}:
        raise AdmissionError("unknown heldout protocol")
    # Deliberately preserve the R14 ID-selection namespace so comparisons
    # address the exact same development cases. Those IDs are NOT fresh.
    return {
        "protocol": "R15_NUMERIC_ONLY_" + protocol.upper() + "_DEVELOPMENT",
        "question_contract": NUMERIC_QUESTION_CONTRACT,
        "selection_namespace": "r14-heldout-" + protocol + ":",
        "bank_status": "REUSED_R14_DEVELOPMENT_NOT_FINAL",
    }


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


def select_heldout(manifest_path: Path, count: int,
                   protocol: str = "v2") -> tuple[list[dict], str]:
    if not 1 <= count <= 16:
        raise AdmissionError("heldout pilot must have 1–16 examples")
    if protocol not in {"v2", "v3"}:
        raise AdmissionError("unknown heldout protocol")
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
    # V1's four cases were exhausted by a truncated-format pilot (base 0/4),
    # so never recycle them as the revised instruction-following benchmark.
    old_ids = {case_id for _, case_id, _ in sorted(scored)[:4]}
    fresh = sorted((hashlib.sha256(("r14-heldout-v2:" + case_id).encode()).hexdigest(),
                    case_id, row) for _, case_id, row in scored if case_id not in old_ids)
    if protocol == "v3":
        # V2's eight development cases have now been scored by all three
        # one-step arms. Reserve a disjoint V3 bank before the eight-step
        # arms are evaluated; never reuse tuned development examples.
        old_ids.update(case_id for _, case_id, _ in fresh[:8])
        fresh = sorted((hashlib.sha256(("r14-heldout-v3:" + case_id).encode()).hexdigest(),
                        case_id, row) for _, case_id, row in scored if case_id not in old_ids)
    selected = [row for _, _, row in fresh[:count]]
    case_digest = hashlib.sha256("\n".join(row["id"] for row in selected).encode()).hexdigest()
    return selected, case_digest


def evaluate(args) -> dict:
    if args.output.exists() or not args.output.parent.is_dir():
        raise AdmissionError("output must be a new file under an existing directory")
    manifest_sha256 = _require_r15_v2_manifest(args.manifest)
    corpus, _ = admit_labyrinth(
        args.manifest, args.work_root, sample_size=1,
        producer_git_head=args.producer_git_head,
    )
    heldout, digest = select_heldout(args.manifest, args.cases, args.protocol)
    binding, base_path = _required_base(args.base_binding)
    adapter_dir = None
    if args.adapter_receipt:
        receipt, adapter_dir = validate_receipt(
            args.adapter_receipt, args.base_binding,
            expected_recipe=RECIPE,
            expected_question_contract=NUMERIC_QUESTION_CONTRACT,
            expected_manifest_sha256=manifest_sha256,
            expected_producer_git_head=args.producer_git_head,
        )
        if receipt["base_revision"] != binding["revision"]:
            raise AdmissionError("trained adapter base revision mismatch")
        if receipt.get("base_model") != binding["model_id"]:
            raise AdmissionError("trained adapter base model mismatch")

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
        # Explicit answer format is fixed before seeing this V2 case bank.
        question = numeric_question(row["prompt"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=24,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        exact = bool(re.fullmatch(r"[+-]?\d+", text)) and int(text) == row["answer"]
        scored = {"id": row["id"], "correct": exact,
                  "expected": row["answer"], "generated": text}
        if args.teacher_force_nll:
            # Companion diagnostic: likelihood of the original numeric
            # completion, not a claim of successful free generation.
            completion = str(row["answer"]) + (tokenizer.eos_token or "")
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            prefix_size = inputs["input_ids"].shape[1]
            full_ids = torch.cat((inputs["input_ids"],
                                  torch.tensor([completion_ids], device="cuda:0")), dim=1)
            with torch.inference_mode():
                logits = model(input_ids=full_ids).logits
            predicted = logits[0, prefix_size - 1:prefix_size + len(completion_ids) - 1]
            scored["answer_nll_per_token"] = float(torch.nn.functional.cross_entropy(
                predicted.float(), full_ids[0, prefix_size:], reduction="mean"
            ).item())
        results.append(scored)
    result = {"status": "HELDOUT_DIAGNOSTIC_NOT_FINAL_QUALIFICATION",
              **_eval_contract_metadata(args.protocol),
              "case_ids_sha256": digest, "cases": len(results),
              "exact_match": sum(x["correct"] for x in results),
              "mode": "adapter" if adapter_dir else "base",
              "adapter_receipt": str(args.adapter_receipt) if adapter_dir else None,
              "corpus_manifest_sha256": manifest_sha256,
              "base_revision": binding["revision"], "results": results,
              "provenance": corpus.get("status")}
    if args.teacher_force_nll:
        result["mean_answer_nll_per_token"] = sum(
            x["answer_nll_per_token"] for x in results) / len(results)
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
    parser.add_argument("--protocol", choices=["v2", "v3"], default="v2")
    parser.add_argument("--teacher-force-nll", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
