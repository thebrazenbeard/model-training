"""Independent, one-shot R18 structural evaluation; never import from trainers.

Run *after* the chosen training receipt has been frozen and externally SHA-pinned.
The public receipt contains aggregate greedy exact-match and target+EOS NLL,
never evaluation IDs, prompts, answers, predictions, or individual scores.
Private 0600 outcomes support blinded, paired comparisons without rereading
answers. This is a development benchmark, not independent model qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
import unicodedata
from pathlib import Path

from trainingday.admission import AdmissionError, _hash_file
from trainingday.eval_structural_r16 import (
    BANK_ID_SHA256, CORPUS_SHA256, R16_PRODUCER_HEAD, R16_RECIPE,
    _old_ids, _r16_adapter,
)
from trainingday.numeric_prompt import NUMERIC_QUESTION_CONTRACT, numeric_question
from trainingday.structural_eval import (
    R15_SCORED_SIGNATURE_COUNT, preregister_r15_fresh_structural64,
    select_structural_cases,
)
from trainingday.train_neural import _parameter_digest, _required_base


PROTOCOL = "R18_NEW_STRUCTURAL64_AFTER_R15_16_AND_R16_64_SCORED_EXCLUSION_V1"
PLAN_SCHEMA = "VERA_TRAININGDAY_R18_BENCHMARK_PLAN_V1"
PRIVATE_BANK_SCHEMA = "VERA_TRAININGDAY_R18_PRIVATE_EVAL_BANK_V1"
EXCLUSIONS_SCHEMA = "VERA_TRAININGDAY_R18_PRIVATE_EVAL_EXCLUSIONS_V1"
ARM_SCHEMA = "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1"
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
R18_QLORA_SOURCE_HEAD = "9a6c3b72ef8ad425fa4df371ee10cf700972c8cd"
EVAL_SOURCE = "R18_NEW_SEALED_NOT_R16_FRESH64"
RECIPES = {
    "r18-mezo": "R18_MEZO_STYLE_FP32_LORA_ZO_ON_FROZEN_NF4_QWEN35_4B_V1",
    "r18-qlora": "VERA_R18_QLORA_NF4_ALL_LINEAR_FP32_ADAMW8BIT_V1",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CONTROLS = {"frozen", "zero", "correct", "corrupt"}
SOURCES = {"frozen", "r16-qlora", "r18-mezo", "r18-qlora"}
PLAN_KEYS = frozenset({
    "schema", "base_model", "base_revision", "base_binding_sha256",
    "tokenizer_binding_sha256", "corpus_manifest_sha256", "selected_id_sha256",
    "selected_count", "train_content_sha256", "source_admission_status",
    "eval_bank_source", "eval_custody_sha256", "eval_index_sha256",
    "eval_examples", "r16_eval_custody_sha256", "r16_holdout_id_sha256",
    "seed", "training_budget_seconds", "probe_budget_seconds", "fit_policy",
    "fit_candidates", "primary_training_config", "no_eval_access_before_arm_freeze",
    "preregistered_checkpoint_selection", "promotion_policy",
})
FIT_LENGTHS = (64, 128, 256, 384, 512, 768, 1024)


def _fit_candidates() -> list[dict]:
    return [{"micro_batch_size": batch, "actual_padded_sequence_length": length,
             "effective_batch_size": 8, "offload": False, "truncation_allowed": False}
            for batch in (1, 2) for length in FIT_LENGTHS]


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False)
                           .encode("utf-8")).hexdigest()


def _private_custody(path: Path, *, directory: bool) -> None:
    """Verify existing secret custody; never silently fix exposed eval bank ACLs."""
    if os.name == "nt":
        try:
            from trainingday.r18_benchmark import (
                BenchmarkError, _windows_owner_sid, _windows_verify_owner_only,
            )
        except ImportError as exc:
            raise AdmissionError("Windows private ACL verifier missing") from exc
        try:
            _windows_verify_owner_only(path, _windows_owner_sid(), directory=directory)
        except BenchmarkError as exc:
            raise AdmissionError("private evaluation custody has unsafe Windows ACL") from exc
    elif stat.S_IMODE(path.stat().st_mode) != (0o700 if directory else 0o600):
        raise AdmissionError("private evaluation custody is not owner-only")


def _json(path: Path, *, private: bool = False) -> dict:
    if (not path.is_file() or path.is_symlink() or
            any(parent.is_symlink() for parent in path.parents)):
        raise AdmissionError("missing/symlinked custody file")
    if private:
        _private_custody(path.parent, directory=True)
        _private_custody(path, directory=False)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdmissionError("custody JSON must be an object")
    return value


def _new_file(path: Path) -> None:
    if (path.exists() or path.is_symlink() or not path.parent.is_dir()
            or any(parent.is_symlink() for parent in path.parents)):
        raise AdmissionError("score output must be new in a non-symlinked directory")


def _private_write(path: Path, data: dict) -> None:
    _private_custody(path.parent, directory=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if os.name == "nt":
        # The file initially inherits ONLY the owner ACE from the already
        # protected parent. Make that ACE explicit and independently verify.
        try:
            from trainingday.r18_benchmark import BenchmarkError, _protect_private_path
        except ImportError as exc:
            raise AdmissionError("Windows private ACL protection unavailable") from exc
        try:
            _protect_private_path(path, directory=False)
        except BenchmarkError as exc:
            raise AdmissionError("unable to restrict private outcome Windows ACL") from exc
    _private_custody(path, directory=False)


def _prompt_sha(prompt: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _plan(path: Path, binding_path: Path, base: dict) -> tuple[dict, str]:
    plan = _json(path)
    if (plan.get("schema") != PLAN_SCHEMA
            or set(plan) != PLAN_KEYS
            or plan.get("eval_bank_source") != EVAL_SOURCE
            or plan.get("no_eval_access_before_arm_freeze") is not True
            or plan.get("preregistered_checkpoint_selection") != "END_OF_TIME_BUDGET_ONLY"
            or plan.get("promotion_policy") !=
            "NEVER_AUTOPROMOTE_R18_REQUIRES_INDEPENDENT_REPLICATION"
            or plan.get("eval_examples") != 64
            or plan.get("selected_count") != 1024
            or plan.get("source_admission_status") !=
            "ADMITTED_LABYRINTH_TRAIN_ONLY_NO_WEIGHT_UPDATE"
            or type(plan.get("seed")) is not int or plan["seed"] < 0
            or type(plan.get("training_budget_seconds")) not in (int, float)
            or type(plan.get("probe_budget_seconds")) not in (int, float)
            or not 0 < plan["probe_budget_seconds"] < plan["training_budget_seconds"] <= 86400
            or plan.get("fit_policy") !=
            "ASCENDING_LENGTH_BATCH1_THEN_BATCH2_UNTIL_FIRST_OOM"
            or plan.get("fit_candidates") != _fit_candidates()
            or plan.get("primary_training_config") not in _fit_candidates()[:7]
            or plan.get("base_model") != BASE_MODEL
            or plan.get("base_revision") != BASE_REVISION
            or plan.get("corpus_manifest_sha256") != CORPUS_SHA256
            or plan.get("base_binding_sha256") != _hash_file(binding_path)
            or plan.get("tokenizer_binding_sha256") !=
            base["files"]["tokenizer_config.json"]
            or any(not HEX64.fullmatch(str(plan.get(key))) for key in (
                "eval_index_sha256", "eval_custody_sha256", "r16_eval_custody_sha256",
                "r16_holdout_id_sha256", "selected_id_sha256", "train_content_sha256"))):
        raise AdmissionError("R18 preregistration/base/source contract changed")
    return plan, canonical_sha(plan)


def _adapter(args, plan: dict, plan_sha: str) -> tuple[dict | None, Path | None, str | None]:
    """Check frozen artifact receipts before touching the private evaluation bank."""
    if args.source not in SOURCES or args.control not in CONTROLS:
        raise AdmissionError("unsupported arm/source")
    if args.source == "frozen":
        if (args.control != "frozen" or args.adapter_receipt is not None
                or args.expected_adapter_receipt_sha256 is not None):
            raise AdmissionError("frozen base must not use a training adapter")
        return None, None, None
    if args.control == "frozen":
        raise AdmissionError("adapter cannot be called a frozen base")
    receipt_path = args.adapter_receipt
    if (receipt_path is None or not HEX64.fullmatch(
            args.expected_adapter_receipt_sha256 or "")
            or _hash_file(receipt_path) != args.expected_adapter_receipt_sha256):
        raise AdmissionError("external immutable training receipt SHA pin required")
    receipt = _json(receipt_path)
    if args.source == "r16-qlora":
        if receipt.get("recipe") != R16_RECIPE:
            raise AdmissionError("R16 adapter must use R16 pinned QLoRA recipe")
        receipt, adapter = _r16_adapter(
            receipt_path, args.base_binding,
            producer_head=R16_PRODUCER_HEAD, required_mode=args.control)
    else:
        if (receipt.get("schema") != ARM_SCHEMA
                or receipt.get("arm") != args.source.removeprefix("r18-")
                or receipt.get("control", receipt.get("mode")) != args.control
                or receipt.get("recipe") != RECIPES[args.source]
                or receipt.get("status") != "TRAINED_UNSCORED"
                or receipt.get("fit_status") not in {"FIT", "fits"}
                or receipt.get("plan_sha256") != plan_sha
                or receipt.get("question_contract") != NUMERIC_QUESTION_CONTRACT
                or receipt.get("eval_custody_sha256") != plan["eval_custody_sha256"]
                or receipt.get("eval_index_sha256") != plan["eval_index_sha256"]
                or receipt.get("training_budget_seconds") != plan["training_budget_seconds"]
                or receipt.get("eval_accuracy") is not None
                or receipt.get("eval_nll") is not None
                or receipt.get("eval_nll_sum") is not None
                or receipt.get("eval_case_receipt_sha256") is not None
                or receipt.get("eval_examples") != 0
                or receipt.get("eval_target_tokens") != 0
                or receipt.get("selected_id_sha256") != plan["selected_id_sha256"]
                or receipt.get("selected_count") != plan["selected_count"]
                or receipt.get("seed") != plan["seed"]
                or receipt.get("corpus_manifest_sha256") != CORPUS_SHA256
                or receipt.get("train_content_sha256") != plan["train_content_sha256"]
                or receipt.get("source_admission_status") != plan["source_admission_status"]
                or receipt.get("tokenizer_binding_sha256") != plan["tokenizer_binding_sha256"]
                or receipt.get("selected_fit_config") != plan["primary_training_config"]
                or (args.source == "r18-qlora" and (
                    receipt.get("source_head") != R18_QLORA_SOURCE_HEAD
                    or receipt.get("lora_target_modules") != "all-linear"
                    or receipt.get("lora_r") != 4
                    or receipt.get("lora_alpha") != 16
                    or receipt.get("quantization") != "NF4_DOUBLE_BF16"
                    or receipt.get("cpu_offload") is not False
                    or receipt.get("disk_offload") is not False
                    or receipt.get("truncation_allowed") is not False))
                or (args.source == "r18-mezo" and (
                    receipt.get("model_method") !=
                    "MeZO-style LoRA-only ZO; frozen NF4 base"
                    or receipt.get("zero_update") is not (args.control == "zero")))
                or not HEX64.fullmatch(str(receipt.get("training_input_sha256")))
                or not HEX64.fullmatch(str(receipt.get("fit_probe_sha256")))):
            raise AdmissionError("R18 arm not frozen, properly trained, or plan-matched")
        steps = receipt.get("optimizer_steps")
        before, after = receipt.get("trainable_digest_before"), receipt.get("trainable_digest_after")
        if (type(steps) is not int or steps < 0
                or (steps == 0) != (args.control == "zero")
                or not HEX64.fullmatch(str(before)) or not HEX64.fullmatch(str(after))
                or (before == after) != (args.control == "zero")
                or type(receipt.get("trainable_parameter_count")) is not int
                or receipt["trainable_parameter_count"] <= 0):
            raise AdmissionError("R18 adapter optimizer/FP32 delta evidence inconsistent")
        adapter = receipt_path.parent / "adapter"
        if not adapter.is_dir() or adapter.is_symlink():
            raise AdmissionError("R18 saved adapter directory missing or symlinked")
        for filename, key in (("adapter_model.safetensors", "adapter_model_sha256"),
                              ("adapter_config.json", "adapter_config_sha256")):
            path = adapter / filename
            if (not path.is_file() or path.is_symlink()
                    or _hash_file(path) != receipt.get(key)):
                raise AdmissionError("R18 adapter bytes fail receipt custody")
    if (receipt.get("base_model") != BASE_MODEL
            or receipt.get("base_revision") != BASE_REVISION
            or receipt.get("base_binding_sha256") != _hash_file(args.base_binding)):
        raise AdmissionError("adapter trained on wrong official base")
    return receipt, adapter, args.expected_adapter_receipt_sha256


def _bank(manifest: Path, old_base_eval: Path, exclusions_path: Path,
          private_bank_path: Path, plan: dict) -> list[dict]:
    """Re-derive all 80 prior cases and 64 new cases; fail on any signature reuse."""
    if _hash_file(manifest) != CORPUS_SHA256 or manifest.is_symlink():
        raise AdmissionError("R18 pinned v2 manifest changed")
    old_ids = _old_ids(old_base_eval)  # pinned scored R15 16-row evaluation receipt
    exclusions = _json(exclusions_path, private=True)
    if (exclusions.get("schema") != EXCLUSIONS_SCHEMA
            or exclusions.get("old_r15_scored16_ids") != old_ids
            or exclusions.get("r16_selected_id_sha256") != BANK_ID_SHA256
            or exclusions.get("excluded_signature_count") != 77):
        raise AdmissionError("historical scored16 exclusions changed")
    r16 = preregister_r15_fresh_structural64(
        manifest, previously_scored_16_ids=old_ids)
    if (r16.get("selected_count") != 64
            or r16.get("selected_id_sha256") != BANK_ID_SHA256):
        raise AdmissionError("R16 previously scored 64 cases cannot be reconstructed")
    r16_ids = [row["id"] for row in r16["cases"]]
    if (exclusions.get("r16_fresh64_ids") != r16_ids
            or plan["r16_holdout_id_sha256"] != canonical_sha(sorted(r16_ids))
            or plan["r16_eval_custody_sha256"] != canonical_sha({
                "manifest": CORPUS_SHA256, "cases": r16["cases"]})):
        raise AdmissionError("R16 scored bank signature/ID custody mismatch")
    excluded = old_ids + r16_ids
    digest = hashlib.sha256("\n".join(excluded).encode()).hexdigest()
    r18 = select_structural_cases(
        manifest, expected_manifest_sha256=CORPUS_SHA256,
        max_cases=64, previously_scored_ids=excluded,
        expected_scored_ids_sha256=digest,
        expected_scored_signature_count=R15_SCORED_SIGNATURE_COUNT + 64,
    )
    cases = r18["cases"]
    if (len(cases) != 64 or len({row["id"] for row in cases}) != 64
            or {row["id"] for row in cases}.intersection(excluded)):
        raise AdmissionError("insufficient never-scored R18 unique structures")
    private = _json(private_bank_path, private=True)
    index = sorted(({"id": row["id"], "prompt_sha256": _prompt_sha(row["prompt"])}
                    for row in cases), key=lambda item: item["id"])
    if (private.get("schema") != PRIVATE_BANK_SCHEMA
            or private.get("manifest_sha256") != CORPUS_SHA256
            or private.get("cases") != cases
            or private.get("eval_custody_sha256") != plan["eval_custody_sha256"]
            or private.get("eval_index_sha256") != plan["eval_index_sha256"]
            or canonical_sha({"manifest": CORPUS_SHA256, "cases": cases}) !=
            plan["eval_custody_sha256"]
            or canonical_sha(index) != plan["eval_index_sha256"]):
        raise AdmissionError("R18 sealed evaluation bank differs from preregistration")
    return cases


def _exact(generated: str, expected: str) -> bool:
    return bool(re.fullmatch(r"-?(?:0|[1-9]\d*)", generated)) and generated == expected


def _run_model(cases: list[dict], model_path: Path,
               adapter: Path | None, receipt: dict | None,
               plan_sha: str) -> list[dict]:
    """Only heavy path; compare physical FP32 LoRA tensors after disk reload."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA inference unavailable")
    if torch.cuda.mem_get_info(0)[0] < 3200 * 2**20:
        raise RuntimeError("less than 3200 MiB free CUDA memory")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if not tokenizer.chat_template or type(tokenizer.eos_token_id) is not int:
        raise RuntimeError("pinned tokenizer lacks chat or EOS contract")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path, quantization_config=quant, device_map=0,
        dtype=torch.bfloat16, local_files_only=True)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        digest, count = _parameter_digest(model, torch)
        if (digest != receipt["trainable_digest_after"]
                or count != receipt["trainable_parameter_count"]):
            raise RuntimeError("freshly loaded FP32 adapter digest/count mismatch")
    model.eval()
    results = []
    for case in cases:
        question = numeric_question(case["prompt"])
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        head = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        prefix_size = head["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **head, do_sample=False, max_new_tokens=24,
                pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(generated[0][prefix_size:],
                                   skip_special_tokens=True).strip()
        answer = str(case["expected"])
        # Tokenize the full prefix+completion as one string and require the
        # prefix token boundary to be unchanged. This avoids target shifting.
        whole_ids = tokenizer(prefix + answer + tokenizer.eos_token,
                              add_special_tokens=False)["input_ids"]
        original_ids = head["input_ids"][0].tolist()
        if (whole_ids[:prefix_size] != original_ids
                or len(whole_ids) <= prefix_size
                or whole_ids[-1] != tokenizer.eos_token_id):
            raise RuntimeError("answer+EOS token boundary changed under tokenizer")
        full = torch.tensor([whole_ids], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            logits = model(input_ids=full).logits
        target = full[0, prefix_size:]
        predicted = logits[0, prefix_size - 1:len(whole_ids) - 1]
        nll_sum = float(torch.nn.functional.cross_entropy(
            predicted.float(), target, reduction="sum").item())
        if not math.isfinite(nll_sum) or nll_sum < 0:
            raise RuntimeError("nonfinite answer+EOS NLL")
        blind = hashlib.sha256((plan_sha + "\x00" + case["id"]).encode()).hexdigest()
        results.append({"blind_case_sha256": blind,
                        "exact": _exact(decoded, answer),
                        "nll_sum": nll_sum, "target_tokens": int(target.numel())})
    return results


def _paired(rows: list[dict], peer_path: Path, plan_sha: str) -> dict:
    peer = _json(peer_path, private=True)
    if (peer.get("schema") != "VERA_R18_PRIVATE_CASE_OUTCOMES_V1"
            or peer.get("plan_sha256") != plan_sha
            or not isinstance(peer.get("outcomes"), list)
            or len(peer["outcomes"]) != 64
            or peer.get("outcome_sha256") != canonical_sha(peer["outcomes"])):
        raise AdmissionError("peer case outcome custody mismatch")
    left = {r["blind_case_sha256"]: r for r in rows}
    right = {r["blind_case_sha256"]: r for r in peer["outcomes"]}
    if len(left) != 64 or set(left) != set(right):
        raise AdmissionError("peer uses a different sealed case set")
    if any(type(r.get("exact")) is not bool for r in right.values()):
        raise AdmissionError("peer blinded exact-match observations invalid")
    pairs = [{"case": key, "this": left[key]["exact"], "peer": right[key]["exact"]}
             for key in sorted(left)]
    return {"peer_private_receipt_sha256": _hash_file(peer_path),
            "pairwise_outcome_sha256": canonical_sha(pairs),
            "this_only_correct": sum(r["this"] and not r["peer"] for r in pairs),
            "peer_only_correct": sum(r["peer"] and not r["this"] for r in pairs),
            "both_correct": sum(r["peer"] and r["this"] for r in pairs),
            "both_incorrect": sum(not r["peer"] and not r["this"] for r in pairs)}


def evaluate(args) -> dict:
    _new_file(args.output)
    if args.private_output is None or args.private_output == args.output:
        raise AdmissionError("separate new private outcome receipt required")
    _new_file(args.private_output)
    if _hash_file(args.manifest) != CORPUS_SHA256 or args.manifest.is_symlink():
        raise AdmissionError("pinned v2 source manifest changed")
    base, model_path = _required_base(args.base_binding)
    if base["model_id"] != BASE_MODEL or base["revision"] != BASE_REVISION:
        raise AdmissionError("not official pinned Qwen3.5-4B revision")
    plan, plan_sha = _plan(args.plan, args.base_binding, base)
    receipt, adapter, receipt_sha = _adapter(args, plan, plan_sha)
    if args.preflight_only:
        # Source-only path does not open the private evaluation bank or score.
        return {"status": "R18_ARM_PREFLIGHT_NO_BANK_OPENED_NO_MODEL_EXECUTION",
                "plan_sha256": plan_sha, "source": args.source, "control": args.control,
                "adapter_receipt_sha256": receipt_sha}
    if args.peer_private is not None and args.peer_private == args.private_output:
        raise AdmissionError("peer must be a previously frozen private score")
    # Opening private answers is expressly AFTER the arm's immutable source
    # receipt was checked (frozen base has no mutable adapter state).
    cases = _bank(args.manifest, args.old_base_eval, args.exclusions,
                  args.sealed_bank, plan)
    rows = _run_model(cases, model_path, adapter, receipt, plan_sha)
    if (len(rows) != 64 or len({r["blind_case_sha256"] for r in rows}) != 64
            or any(type(r["exact"]) is not bool
                   or type(r["target_tokens"]) is not int or r["target_tokens"] < 1
                   or not math.isfinite(r["nll_sum"]) or r["nll_sum"] < 0 for r in rows)):
        raise RuntimeError("incomplete or invalid physical R18 case scoring")
    # Verify the training receipt did not mutate during GPU execution.
    if receipt_sha and _hash_file(args.adapter_receipt) != receipt_sha:
        raise RuntimeError("training receipt changed during evaluation")
    count = sum(r["exact"] for r in rows)
    tokens = sum(r["target_tokens"] for r in rows)
    nll_sum = sum(r["nll_sum"] for r in rows)
    private = {"schema": "VERA_R18_PRIVATE_CASE_OUTCOMES_V1",
               "protocol": PROTOCOL, "plan_sha256": plan_sha,
               "eval_custody_sha256": plan["eval_custody_sha256"],
               "source": args.source, "control": args.control,
               "outcomes": rows, "outcome_sha256": canonical_sha(rows)}
    if args.peer_private is not None:
        private["pairwise"] = _paired(rows, args.peer_private, plan_sha)
    # Private scores are sealed before public aggregate score is created.
    _private_write(args.private_output, private)
    public = {"status": "R18_MODEL_SCORED_DEVELOPMENT_ONLY_NO_PROMOTION",
              "protocol": PROTOCOL, "plan_sha256": plan_sha,
              "eval_custody_sha256": plan["eval_custody_sha256"],
              "eval_index_sha256": plan["eval_index_sha256"],
              "corpus_manifest_sha256": CORPUS_SHA256,
              "excluded_previously_scored_cases": 80,
              "excluded_previously_scored_signatures": 77,
              "cases": 64, "source": args.source, "control": args.control,
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "adapter_training_receipt_sha256": receipt_sha,
              "fresh_adapter_fp32_reload_verified": adapter is not None,
              "greedy_exact_match": count, "greedy_accuracy": count / 64,
              "answer_eos_target_tokens": tokens, "answer_eos_nll_sum": nll_sum,
              "answer_eos_nll_per_token": nll_sum / tokens,
              "private_case_outcome_receipt_sha256": _hash_file(args.private_output),
              "private_blinded_case_outcomes_sha256": private["outcome_sha256"],
              "eval_unsealed_after_arm_freeze": True,
              "eval_unsealed_at_utc_ns": time.time_ns(),
              "automatic_promotion": False}
    if args.peer_private is not None:
        public["pairwise_digest_private_only"] = True
    with args.output.open("x", encoding="utf-8") as writer:
        json.dump(public, writer, sort_keys=True, indent=2, allow_nan=False)
        writer.write("\n")
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "old-base-eval", "exclusions", "sealed-bank",
                   "plan", "base-binding", "output", "private-output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--control", choices=sorted(CONTROLS), required=True)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--expected-adapter-receipt-sha256")
    parser.add_argument("--peer-private", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
