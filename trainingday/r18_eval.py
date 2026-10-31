"""Independent, one-shot R18 structural evaluation; never import from trainers.

Run *after* the chosen training receipt has been frozen and externally SHA-pinned.
The public receipt contains aggregate greedy exact-match and target+EOS NLL,
never evaluation IDs, prompts, answers, predictions, or individual scores.
Private 0600 outcomes support blinded, paired comparisons without rereading
answers. This is a development benchmark, not independent model qualification.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import gc
import functools
import hashlib
import itertools
import json
import math
import operator
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
from trainingday.structural_eval import preregister_r15_fresh_structural64
from trainingday.train_neural import _parameter_digest, _required_base


PROTOCOL = "R18_EXTERNAL_LONG_CHAIN64_OOD_V1"
PLAN_SCHEMA = "VERA_TRAININGDAY_R18_BENCHMARK_PLAN_V1"
PRIVATE_BANK_SCHEMA = "VERA_TRAININGDAY_R18_PRIVATE_EVAL_BANK_V1"
EXCLUSIONS_SCHEMA = "VERA_TRAININGDAY_R18_PRIVATE_EVAL_EXCLUSIONS_V1"
ARM_SCHEMA = "VERA_TRAININGDAY_R18_ARM_RECEIPT_V1"
FREEZE_SCHEMA = "VERA_TRAININGDAY_R18_PREUNSEAL_FREEZE_INVENTORY_V1"
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
R18_QLORA_SOURCE_HEAD = "9a6c3b72ef8ad425fa4df371ee10cf700972c8cd"
EVAL_SOURCE = "R18_NEW_SEALED_NOT_R16_FRESH64"
NOVEL_SOURCE_SCHEMA = "VERA_TRAININGDAY_R18_NOVEL_CHAIN6_EVAL_SOURCE_V1"
NOVEL_SEED = 18041
OPS = ("add", "subtract", "multiply")
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
    "eval_source_manifest_sha256",
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


def _replay_rendered_chain6(prompt: str) -> tuple[int, tuple[tuple[str, int], ...], int]:
    """Separate prompt-text oracle, with no reliance on generator operation objects."""
    match = re.fullmatch(
        r"Start with (-?(?:0|[1-9][0-9]*))\. (.+) What is the final integer\?",
        prompt,
    )
    if match is None or not match.group(2).endswith("."):
        raise AdmissionError("R18 rendered numeric question contract invalid")
    start = int(match.group(1))
    clauses = match.group(2)[:-1].split(". ")
    if len(clauses) != 6:
        raise AdmissionError("R18 rendered case is not exactly six operators")
    value = start
    operations: list[tuple[str, int]] = []
    states = {start}
    peak_absolute = abs(start)
    for clause in clauses:
        parsed = re.fullmatch(r"(add|subtract|multiply) (-?(?:0|[1-9][0-9]*))", clause)
        if parsed is None:
            raise AdmissionError("R18 rendered operator malformed")
        op, operand = parsed.group(1), int(parsed.group(2))
        if (op in {"add", "subtract"} and not 1 <= abs(operand) <= 9
                or op == "multiply" and operand not in (-3, -2, 2, 3)):
            raise AdmissionError("R18 rendered operand violates no-reset bounds")
        if op == "add":
            result = value + operand
        elif op == "subtract":
            result = value - operand
        else:
            result = value * operand
        if result in states or abs(result) > 100000:
            raise AdmissionError("R18 rendered case has no-op/inverse/reset/overflow")
        states.add(result)
        peak_absolute = max(peak_absolute, abs(result))
        value = result
        operations.append((op, operand))
    if len(prompt) > 220 or len(str(value)) > 8:
        raise AdmissionError("R18 rendered answer/prompt exceeds bounds")
    return value, tuple(operations), peak_absolute


def generate_novel_r18_cases(*, seed: int = NOVEL_SEED,
                             private_nonce_hex: str) -> list[dict]:
    """Independently reconstruct the custodian's disjoint 6-op OOD holdout.

    All pinned v2 reasoning chains have 2–5 operators; these 6-op *full*
    signatures are outside that training/eval universe. This is out-of-
    distribution compositional evaluation, not novel primitive operations.
    """
    if (type(seed) is not int or seed != NOVEL_SEED
            or not isinstance(private_nonce_hex, str)
            or not HEX64.fullmatch(private_nonce_hex)):
        raise AdmissionError("R18 private 256-bit generation nonce or seed invalid")
    rankings = []
    for sig in itertools.product(OPS, repeat=6):
        text = "\x1f".join(sig)
        priority = hashlib.sha256(
            f"R18_CH6_SIG_V2\x00{seed}\x00{private_nonce_hex}\x00{text}"
            .encode("utf-8")).hexdigest()
        rankings.append((priority, sig))
    cases = []
    start_values = (*range(-20, 0), *range(1, 21))
    arithmetic_operands = (*range(-9, 0), *range(1, 10))
    multiplication_operands = (-3, -2, 2, 3)
    fn_by_op = {"add": operator.add, "subtract": operator.sub,
                "multiply": operator.mul}
    for _, signature in sorted(rankings):
        sig_text = "\x1f".join(signature)

        def draw(position: int) -> int:
            value = hashlib.sha256(
                f"R18_CH6_VAL_V2\x00{seed}\x00{private_nonce_hex}\x00{sig_text}\x00{position}"
                .encode("utf-8")).digest()
            return int.from_bytes(value, "big")

        start = start_values[draw(0) % len(start_values)]
        operands = [
            (multiplication_operands if name == "multiply" else arithmetic_operands)[
                draw(j) % (len(multiplication_operands) if name == "multiply"
                           else len(arithmetic_operands))]
            for j, name in enumerate(signature, start=1)
        ]
        intermediate = start
        observed_states = {start}
        acceptable = True
        for name, operand in zip(signature, operands, strict=True):
            next_value = fn_by_op[name](intermediate, operand)
            if (next_value == intermediate or next_value in observed_states
                    or abs(next_value) > 100000):
                acceptable = False
                break
            observed_states.add(next_value)
            intermediate = next_value
        if not acceptable:
            continue
        # Different implementation from the loop above, using a functional
        # reduce over immutable operator pairs as an independent integer oracle.
        paired = tuple(zip(signature, operands, strict=True))
        independent = functools.reduce(
            lambda n, pair: fn_by_op[pair[0]](n, pair[1]), paired, start)
        if independent != intermediate:
            raise AdmissionError("R18 independent generated arithmetic oracle mismatch")
        answer = str(intermediate)
        prompt = (f"Start with {start}. " +
                  " ".join(f"{name} {operand}." for name, operand in paired) +
                  " What is the final integer?")
        if len(prompt) > 220 or len(answer) > 8:
            continue
        replayed, rendered_ops, _ = _replay_rendered_chain6(prompt)
        if replayed != intermediate or rendered_ops != paired:
            raise AdmissionError("rendered R18 evaluation prompt disagrees with arithmetic")
        case_sha = hashlib.sha256(
            f'R18_CH6_CASE_V2\x00{seed}\x00{private_nonce_hex}\x00{sig_text}\x00{start}\x00'
            f'{",".join(str(n) for n in operands)}\x00{answer}'.encode()).hexdigest()
        cases.append({"id": "r18-reasoning-" + case_sha,
                      "prompt": prompt, "expected": answer,
                      "signature_sha256": hashlib.sha256(sig_text.encode()).hexdigest()})
        if len(cases) == 64:
            break
    if (len(cases) != 64 or len({item["id"] for item in cases}) != 64
            or len({item["signature_sha256"] for item in cases}) != 64):
        raise AdmissionError("R18 generation did not yield 64 unique novel structures")
    return cases


def _source_manifest(path: Path, generator_source: Path) -> tuple[dict, str]:
    """Pin separate public OOD corpus source without opening private labels."""
    manifest = _json(path)
    if not generator_source.is_file() or generator_source.is_symlink():
        raise AdmissionError("R18 pinned six-chain generator source file missing")
    # Do not generate any answer-bearing cases here: this source-only gate
    # must also be safe for preflight before an adapter is frozen.
    if (manifest.get("schema") != NOVEL_SOURCE_SCHEMA
            or manifest.get("protocol") != PROTOCOL
            or manifest.get("evaluation_distribution") !=
            "EXTERNAL_OOD_SIX_OPERATOR_COMPOSITION_NOT_V2_HOLDOUT"
            or manifest.get("algorithm") !=
            "R18_SHA256_PRIORITY_SIX_STEP_PRIVATE_NONCE_DUAL_ORACLE_V2"
            or not HEX64.fullmatch(str(manifest.get("private_nonce_sha256")))
            or manifest.get("generator_code_sha256") != _hash_file(generator_source)
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("producer_git_head")))
            or manifest.get("seed") != NOVEL_SEED
            or manifest.get("count") != 64
            or manifest.get("chain_length") != 6
            or manifest.get("operator_vocabulary") != list(OPS)
            or manifest.get("start_values") != [-20, 20]
            or manifest.get("add_sub_values") != [-9, 9]
            or manifest.get("multiply_values") != [-3, -2, 2, 3]
            or manifest.get("max_abs_intermediate") != 100000
            or manifest.get("max_prompt_chars") != 220
            or manifest.get("max_answer_chars") != 8
            or manifest.get("older_v2_corpus_manifest_sha256") != CORPUS_SHA256
            or manifest.get("older_v2_chain_length_range") != [2, 5]
            or manifest.get("previously_scored_v2_signature_count") != 77
            or manifest.get("older_v2_unseen_signatures_after_r16") != 0
            or manifest.get("selected_unique_signature_count") != 64
            or any(not HEX64.fullmatch(str(manifest.get(key))) for key in (
                "selected_id_sha256", "selected_signature_sha256",
                "private_case_content_sha256"))):
        raise AdmissionError("R18 external OOD generator source or custody mismatch")
    return manifest, canonical_sha(manifest)


def _plan(path: Path, binding_path: Path, base: dict,
          source_manifest_sha: str | None = None) -> tuple[dict, str]:
    plan = _json(path)
    if (plan.get("schema") != PLAN_SCHEMA
            or set(plan) != PLAN_KEYS
            or plan.get("eval_bank_source") != EVAL_SOURCE
            or plan.get("no_eval_access_before_arm_freeze") is not True
            or plan.get("preregistered_checkpoint_selection") != "END_OF_TIME_BUDGET_ONLY"
            or plan.get("promotion_policy") !=
            "NEVER_AUTOPROMOTE_R18_REQUIRES_INDEPENDENT_REPLICATION"
            or plan.get("eval_examples") != 64
            or plan.get("eval_source_manifest_sha256") != source_manifest_sha
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
                "eval_source_manifest_sha256",
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


def _preunseal_inventory(args, plan: dict, plan_sha: str) -> tuple[str, dict, dict]:
    """Require all eight *unscored* arms to be frozen before reading labels.

    This is source/receipt custody, not attestation of independent GPU reload,
    the evaluator's host identity, or the physical private bank access log.
    """
    path = getattr(args, "freeze_inventory", None)
    expected = getattr(args, "expected_freeze_inventory_sha256", None)
    if (not isinstance(path, Path) or not HEX64.fullmatch(expected or "")
            or path.is_symlink() or not path.is_file()
            or _hash_file(path) != expected):
        raise AdmissionError("pre-unseal eight-arm freeze inventory and external SHA pin required")
    inventory = _json(path)
    entries = inventory.get("arms")
    if (set(inventory) != {"schema", "plan_sha256", "arms"}
            or inventory["schema"] != FREEZE_SCHEMA
            or inventory["plan_sha256"] != plan_sha
            or type(entries) is not list or len(entries) != 8):
        raise AdmissionError("pre-unseal inventory schema or plan mismatch")
    # Local import avoids coupling the source-only preflight to R18 heavy paths.
    from trainingday.r18_benchmark import BenchmarkError, validate_arm_receipt
    keyed: dict[tuple[str, str], dict] = {}
    pinned: dict[tuple[str, str], tuple[Path, str]] = {}
    saved_files: dict[Path, str] = {}
    run_ids = set()
    for item in entries:
        if (type(item) is not dict
                or set(item) != {"arm", "control", "receipt_path", "receipt_sha256"}
                or item["arm"] not in ("mezo", "qlora")
                or item["control"] not in CONTROLS
                or not isinstance(item["receipt_path"], str)
                or not HEX64.fullmatch(str(item["receipt_sha256"]))):
            raise AdmissionError("pre-unseal inventory entry invalid")
        key = (item["arm"], item["control"])
        receipt_path = Path(item["receipt_path"])
        if (key in keyed or not receipt_path.is_absolute()
                or receipt_path.is_symlink()
                or not receipt_path.is_file()
                or _hash_file(receipt_path) != item["receipt_sha256"]):
            raise AdmissionError("pre-unseal duplicate or changed arm receipt")
        receipt = _json(receipt_path)
        try:
            validate_arm_receipt(receipt, plan, allow_unscored=True)
        except (BenchmarkError, KeyError, TypeError, ValueError) as exc:
            raise AdmissionError("pre-unseal arm receipt failed independent contract") from exc
        if (receipt["arm"], receipt["control"]) != key or (
                receipt["status"] != "TRAINED_UNSCORED"
                or receipt.get("eval_access") is True
                or receipt.get("eval_unsealed_at_utc_ns") is not None
                or receipt["run_id"] in run_ids):
            raise AdmissionError("pre-unseal arm already scored or identity collision")
        if item["control"] != "frozen":
            adapter = receipt_path.parent / "adapter"
            if not adapter.is_dir() or adapter.is_symlink():
                raise AdmissionError("pre-unseal saved adapter directory absent")
            for filename, key_name in (
                    ("adapter_model.safetensors", "adapter_model_sha256"),
                    ("adapter_config.json", "adapter_config_sha256")):
                saved = adapter / filename
                if (not saved.is_file() or saved.is_symlink()
                        or _hash_file(saved) != receipt.get(key_name)):
                    raise AdmissionError("pre-unseal adapter bytes do not match receipt")
                saved_files[saved] = receipt[key_name]
        run_ids.add(receipt["run_id"])
        keyed[key], pinned[key] = receipt, (receipt_path, item["receipt_sha256"])
    if set(keyed) != {(arm, control) for arm in ("mezo", "qlora")
                     for control in CONTROLS}:
        raise AdmissionError("pre-unseal incomplete eight-arm freeze")
    initial = {r["trainable_digest_before"] for r in keyed.values()}
    if len(initial) != 1:
        raise AdmissionError("pre-unseal arms do not share initial adapter digest")
    for arm in ("mezo", "qlora"):
        controls = {mode: keyed[(arm, mode)] for mode in CONTROLS}
        if (controls["correct"]["training_input_sha256"] ==
                controls["corrupt"]["training_input_sha256"]
                or len({r["fit_probe_sha256"] for r in controls.values()}) != 1
                or controls["correct"]["optimizer_window_seconds"] <
                .85 * plan["training_budget_seconds"]
                or controls["corrupt"]["optimizer_window_seconds"] <
                .85 * plan["training_budget_seconds"]):
            raise AdmissionError("pre-unseal incomplete controls or run budget")
    left = keyed[("mezo", "correct")]
    right = keyed[("qlora", "correct")]
    if (left["training_input_sha256"] != right["training_input_sha256"]
            or abs(left["optimizer_window_seconds"] -
                   right["optimizer_window_seconds"]) >
            .05 * plan["training_budget_seconds"]):
        raise AdmissionError("pre-unseal unpaired training data or elapsed budget")
    if args.source in ("r18-mezo", "r18-qlora"):
        selected = pinned[(args.source.removeprefix("r18-"), args.control)]
        if (args.adapter_receipt != selected[0]
                or args.expected_adapter_receipt_sha256 != selected[1]):
            raise AdmissionError("pre-unseal selected adapter not bound to inventory")
    elif args.source != "frozen" or args.control != "frozen":
        raise AdmissionError("pre-unseal only R18 comparison arms may score")
    return expected, pinned, saved_files


def _require_independent_preunseal_attestation() -> None:
    """Hold the private bank shut until a real host-attester is integrated.

    Local SHA pins, arm receipts, and an evaluator-supplied freeze inventory
    prove internal byte consistency only. They cannot establish which trainer
    produced those bytes, that the eight runs occurred, or which principal
    will read the sealed bank. No independent attester, trust anchor, or
    bank-access audit is wired into this source-only evaluator yet.

    In particular, an environment flag, CLI-provided digest, self-signed
    document, or mockable source receipt must not authorize unsealing.
    The trusted host integration must replace this fail-closed boundary and
    qualify it independently before any real private evaluation is run.
    """
    raise AdmissionError(
        "independent pre-unseal attestation unavailable: "
        "R18 source receipt custody cannot authorize private evaluation")


def _cuda_preflight() -> None:
    """Reject an ineligible CUDA host before private answers are opened."""
    try:
        import torch
        if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
                or torch.cuda.mem_get_info(0)[0] < 3200 * 2**20):
            raise RuntimeError("R18 CUDA/BF16/3200MiB pre-unseal gate failed")
    except (ImportError, OSError) as exc:
        raise RuntimeError("R18 CUDA pre-unseal preflight unavailable") from exc


def _bank(manifest: Path, old_base_eval: Path, exclusions_path: Path,
          private_bank_path: Path, plan: dict, source_manifest: dict) -> list[dict]:
    """Verify 80 historic scores and rederive 64 new 6-op OOD structures."""
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
            or r16.get("selected_id_sha256") != BANK_ID_SHA256
            or r16.get("signature_count") != 64):
        raise AdmissionError("R16 previously scored 64 cases cannot be reconstructed")
    r16_ids = [row["id"] for row in r16["cases"]]
    if (exclusions.get("r16_fresh64_ids") != r16_ids
            or plan["r16_holdout_id_sha256"] != canonical_sha(sorted(r16_ids))
            or plan["r16_eval_custody_sha256"] != canonical_sha({
                "manifest": CORPUS_SHA256, "cases": r16["cases"]})):
        raise AdmissionError("R16 scored bank signature/ID custody mismatch")
    # Earlier 2–5-op v2 signatures are exhausted, so an additional v2
    # structure is impossible. The frozen external generator uses 6-op full
    # signatures and never opens/trains on either the prior v2 eval or train.
    # Historical `_replay` in the R16 selector independently verifies all
    # v2 reasoning rows are 2–5-op chains, hence no six-op train overlap.
    private = _json(private_bank_path, private=True)
    nonce = private.get("private_nonce_hex")
    if (not isinstance(nonce, str) or not HEX64.fullmatch(nonce)
            or hashlib.sha256(bytes.fromhex(nonce)).hexdigest() !=
            source_manifest.get("private_nonce_sha256")):
        raise AdmissionError("R18 private generation nonce differs from public commitment")
    cases = generate_novel_r18_cases(private_nonce_hex=nonce)
    if (source_manifest.get("selected_id_sha256") != hashlib.sha256(
            "\n".join(case["id"] for case in cases).encode()).hexdigest()
            or source_manifest.get("selected_signature_sha256") != hashlib.sha256(
                "\n".join(case["signature_sha256"] for case in cases).encode()).hexdigest()
            or source_manifest.get("private_case_content_sha256") != canonical_sha(cases)):
        raise AdmissionError("R18 generated case content does not match public source commitment")
    excluded = set(old_ids + r16_ids)
    if (len(cases) != 64 or len({row["id"] for row in cases}) != 64
            or len({row["signature_sha256"] for row in cases}) != 64
            or {row["id"] for row in cases}.intersection(excluded)):
        raise AdmissionError("R18 external long-chain source not fully fresh")
    index = sorted(({"id": row["id"], "prompt_sha256": _prompt_sha(row["prompt"])}
                    for row in cases), key=lambda item: item["id"])
    if (private.get("schema") != PRIVATE_BANK_SCHEMA
            or private.get("manifest_sha256") != CORPUS_SHA256
            or private.get("eval_source_manifest_sha256") !=
            plan["eval_source_manifest_sha256"]
            or private.get("cases") != cases
            or private.get("eval_custody_sha256") != plan["eval_custody_sha256"]
            or private.get("eval_index_sha256") != plan["eval_index_sha256"]
            or canonical_sha({"source_manifest_sha256":
                              plan["eval_source_manifest_sha256"], "cases": cases}) !=
            plan["eval_custody_sha256"]
            or canonical_sha(index) != plan["eval_index_sha256"]):
        raise AdmissionError("R18 sealed evaluation bank differs from preregistration")
    return cases


def _exact(generated: str, expected: str) -> bool:
    """Primary metric: canonical decimal answer after whitespace strip."""
    return bool(re.fullmatch(r"-?(?:0|[1-9]\d*)", generated)) and generated == expected


def _integer_semantic(generated: str, expected: str) -> bool:
    """Secondary metric: mathematically equal integer, irrespective of +/zeros."""
    return bool(re.fullmatch(r"[+-]?\d+", generated)) and int(generated) == int(expected)


class _InferenceSession:
    """One successfully loaded model/adapter, reused for scoring and then released."""

    def __init__(self, torch, tokenizer, model):
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.torch = None
        try:
            gc.collect()
        except Exception:
            # Closing while handling a bank-custody error must not mask the
            # original error with a best-effort collection failure.
            pass


def _load_model(model_path: Path, adapter: Path | None,
                receipt: dict | None) -> _InferenceSession:
    """Load the full pinned NF4 base and adapter BEFORE any private bank access."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    model = tokenizer = None
    try:
        _cuda_preflight()
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
    except BaseException:
        # A failed model/PEFT load occurs before closing() has a session to
        # manage. Drop partially loaded tensors and release unoccupied cache;
        # cleanup must not replace the original fail-closed load exception.
        model = tokenizer = None
        try:
            gc.collect()
        except Exception:
            # GC is an optimization during a failed load, not the failure to
            # report. Preserve the original tokenizer/model/adapter error.
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise
    return _InferenceSession(torch, tokenizer, model)


def _run_model(cases: list[dict], session: _InferenceSession,
               plan_sha: str) -> list[dict]:
    """Score with the exact model object loaded before private bank unseal."""
    torch, tokenizer, model = session.torch, session.tokenizer, session.model
    if model is None or tokenizer is None or torch is None:
        raise RuntimeError("model session was released before scoring")
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
                or len(whole_ids) > 1024
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
        _, operations, peak_absolute = _replay_rendered_chain6(case["prompt"])
        results.append({"blind_case_sha256": blind,
                        "exact": _integer_semantic(decoded, answer),
                        "canonical_format_exact": _exact(decoded, answer),
                        "prompt_tokens": prefix_size,
                        "multiply_count": sum(op == "multiply" for op, _ in operations),
                        "peak_abs_intermediate": peak_absolute,
                        "answer_digits": len(answer.lstrip("-")),
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
    source_manifest, source_sha = _source_manifest(
        args.eval_source_manifest, args.generator_source)
    plan, plan_sha = _plan(args.plan, args.base_binding, base, source_sha)
    receipt, adapter, receipt_sha = _adapter(args, plan, plan_sha)
    if args.preflight_only:
        # Source-only path does not open the private evaluation bank or score.
        return {"status": "R18_ARM_PREFLIGHT_NO_BANK_OPENED_NO_MODEL_EXECUTION",
                "plan_sha256": plan_sha, "source": args.source, "control": args.control,
                "adapter_receipt_sha256": receipt_sha}
    if args.peer_private is not None and args.peer_private == args.private_output:
        raise AdmissionError("peer must be a previously frozen private score")
    freeze_sha, pinned_receipts, pinned_adapters = _preunseal_inventory(
        args, plan, plan_sha)
    _require_independent_preunseal_attestation()
    _cuda_preflight()
    # Preload the actual pinned NF4 model and saved adapter (including physical
    # tensor digest) BEFORE private answers. Reuse this same session to score;
    # full-load failure cannot reach _bank(). This is not host attestation.
    with closing(_load_model(model_path, adapter, receipt)) as session:
        # A persistent on-disk swap while the loader reads the base must not
        # reach the private bank. _required_base rehashes every pinned file;
        # this narrows a race but cannot defeat a same-principal writer that
        # swaps and restores bytes between observations. Real custody needs
        # an immutable, independently controlled snapshot and bank broker.
        loaded_base, loaded_path = _required_base(args.base_binding)
        if loaded_base != base or loaded_path != model_path:
            raise AdmissionError("pinned base binding changed during model preload")
        cases = _bank(args.manifest, args.old_base_eval, args.exclusions,
                      args.sealed_bank, plan, source_manifest)
        rows = _run_model(cases, session, plan_sha)
    if (len(rows) != 64 or len({r["blind_case_sha256"] for r in rows}) != 64
            or any(type(r["exact"]) is not bool
                   or type(r["canonical_format_exact"]) is not bool
                   or (r["canonical_format_exact"] and not r["exact"])
                   or type(r["prompt_tokens"]) is not int or r["prompt_tokens"] < 1
                   or type(r["multiply_count"]) is not int
                   or not 0 <= r["multiply_count"] <= 6
                   or type(r["peak_abs_intermediate"]) is not int
                   or not 0 <= r["peak_abs_intermediate"] <= 100000
                   or type(r["answer_digits"]) is not int
                   or not 1 <= r["answer_digits"] <= 8
                   or type(r["target_tokens"]) is not int or r["target_tokens"] < 1
                   or not math.isfinite(r["nll_sum"]) or r["nll_sum"] < 0 for r in rows)):
        raise RuntimeError("incomplete or invalid physical R18 case scoring")
    # Verify the training receipt did not mutate during GPU execution.
    if receipt_sha and _hash_file(args.adapter_receipt) != receipt_sha:
        raise RuntimeError("training receipt changed during evaluation")
    if (_hash_file(args.freeze_inventory) != freeze_sha or any(
            _hash_file(path) != digest for path, digest in pinned_receipts.values())
            or any(_hash_file(path) != digest
                   for path, digest in pinned_adapters.items())):
        raise RuntimeError("pre-unseal inventory/arm/adapter bytes changed during scoring")
    count = sum(r["exact"] for r in rows)
    tokens = sum(r["target_tokens"] for r in rows)
    nll_sum = sum(r["nll_sum"] for r in rows)
    strata = {}
    for multiply_count in sorted({row["multiply_count"] for row in rows}):
        selected = [row for row in rows if row["multiply_count"] == multiply_count]
        strata[str(multiply_count)] = {
            "cases": len(selected), "integer_semantic_correct": sum(r["exact"] for r in selected),
            "canonical_format_correct": sum(r["canonical_format_exact"] for r in selected),
            "nll_per_answer_eos_token": sum(r["nll_sum"] for r in selected) /
            sum(r["target_tokens"] for r in selected),
        }
    private = {"schema": "VERA_R18_PRIVATE_CASE_OUTCOMES_V1",
               "protocol": PROTOCOL, "plan_sha256": plan_sha,
               "preunseal_freeze_inventory_sha256": freeze_sha,
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
              "eval_distribution": "EXTERNAL_OOD_SIX_OPERATION_COMPOSITION",
              "full_operator_signature_novel": True,
              "shorter_subchain_novelty": False,
              "eval_source_manifest_sha256": source_sha,
              "excluded_previously_scored_cases": 80,
              "excluded_previously_scored_signatures": 77,
              "cases": 64, "source": args.source, "control": args.control,
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "adapter_training_receipt_sha256": receipt_sha,
              "preunseal_freeze_inventory_sha256": freeze_sha,
              "fresh_adapter_fp32_reload_verified": adapter is not None,
              "greedy_exact_match": count, "greedy_accuracy": count / 64,
              "greedy_integer_semantic_correct": count,
              "greedy_canonical_format_correct": sum(r["canonical_format_exact"] for r in rows),
              "greedy_exact_metric_contract":
              "PARSED_SIGNED_INTEGER_VALUE_AFTER_WHITESPACE_STRIP_V1",
              "format_metric_contract": "CANONICAL_SIGNED_DECIMAL_STRING_V1",
              "answer_eos_target_tokens": tokens, "answer_eos_nll_sum": nll_sum,
              "answer_eos_nll_per_token": nll_sum / tokens,
              "multiplication_count_strata": strata,
              "prompt_tokens_min": min(r["prompt_tokens"] for r in rows),
              "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
              "answer_digits_distribution": {
                  str(d): sum(r["answer_digits"] == d for r in rows)
                  for d in sorted({r["answer_digits"] for r in rows})},
              "peak_abs_intermediate_max": max(r["peak_abs_intermediate"] for r in rows),
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
                   "plan", "eval-source-manifest", "base-binding", "output", "private-output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--generator-source", type=Path, default=Path(__file__).with_name(
        "r18_eval_source.py"))
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--control", choices=sorted(CONTROLS), required=True)
    parser.add_argument("--adapter-receipt", type=Path)
    parser.add_argument("--expected-adapter-receipt-sha256")
    parser.add_argument("--peer-private", type=Path)
    parser.add_argument("--freeze-inventory", type=Path)
    parser.add_argument("--expected-freeze-inventory-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
