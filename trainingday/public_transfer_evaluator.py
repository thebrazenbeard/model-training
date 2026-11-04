"""Public development transfer scorer. Never a sealed-bank or Q6 witness."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Mapping


ARMS = ("trained", "untrained", "sham", "shuffled_labels", "corrupted_labels")
SCORER_VERSION = "public-transfer-v1"
PROVENANCE_FIELDS = (
    "base_hf_revision", "physical_model_sha256", "tokenizer_revision",
    "chat_template_sha256", "dataset_generator", "dataset_version",
    "dataset_sha256", "evaluation_bank_id", "evaluation_custodian",
    "seeds", "timing",
)


class PublicTransferEvaluationError(ValueError):
    """Invalid fixture, incompatible controls, or train/eval leakage."""


def _norm(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicTransferEvaluationError("Expected nonempty text")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _digest(value: object) -> str:
    try:
        content = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PublicTransferEvaluationError("Input must be canonical JSON") from exc
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def evaluate_public_fixture(bundle: Mapping) -> dict:
    """Score matched arms by transfer stratum, returning prompt-free receipt.

    `predictions` must contain every fixture case exactly once in *each* arm.
    Public differential is strictly descriptive; all Q6 claims remain HOLD.
    """
    if not isinstance(bundle, Mapping):
        raise PublicTransferEvaluationError("Bundle must be an object")
    fixture = bundle.get("fixture")
    if not isinstance(fixture, Mapping):
        raise PublicTransferEvaluationError("Missing fixture")
    if (fixture.get("visibility"), fixture.get("purpose"), fixture.get("split")) != (
        "public", "development", "transfer"
    ) or any(fixture.get(key) for key in ("sealed", "protected", "frozen")):
        raise PublicTransferEvaluationError("Only explicitly public development transfer fixtures allowed")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10000:
        raise PublicTransferEvaluationError("Cases must be a bounded nonempty list")
    expected = {}
    strata = defaultdict(list)
    eval_prompts = set()
    for row in cases:
        if not isinstance(row, Mapping):
            raise PublicTransferEvaluationError("Case must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in expected:
            raise PublicTransferEvaluationError("Case IDs must be unique nonempty strings")
        prompt = _norm(row.get("prompt"))
        if prompt in eval_prompts:
            raise PublicTransferEvaluationError("Duplicate eval prompt")
        eval_prompts.add(prompt)
        _norm(row.get("expected_semantic"))
        stratum = _norm(row.get("stratum"))
        for field in ("family_id", "source_group_id", "template_id"):
            _norm(row.get(field))
        expected[case_id] = row
        strata[stratum].append(case_id)
    if len(strata) < 2:
        raise PublicTransferEvaluationError("At least two transfer strata required")

    manifest = bundle.get("train_manifest")
    if not isinstance(manifest, Mapping):
        raise PublicTransferEvaluationError("An identified training manifest is required")
    _norm(manifest.get("manifest_id"))
    training = bundle.get("train_rows")
    if not isinstance(training, list) or not 1 <= len(training) <= 10000:
        raise PublicTransferEvaluationError("Nonempty, bounded train_rows are required")
    for row in training:
        if not isinstance(row, Mapping):
            raise PublicTransferEvaluationError("Training row must be an object")
        for field in ("prompt", "family_id", "source_group_id", "template_id"):
            _norm(row.get(field))
        for case in cases:
            if _norm(row["prompt"]) == _norm(case["prompt"]):
                raise PublicTransferEvaluationError("Train/eval prompt leakage")
            for field in ("family_id", "source_group_id", "template_id"):
                if _norm(row[field]) == _norm(case[field]):
                    raise PublicTransferEvaluationError(f"Train/eval {field} leakage")

    predictions = bundle.get("predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != set(ARMS):
        raise PublicTransferEvaluationError("All five matched prediction arms required")
    correct = {}
    for arm in ARMS:
        rows = predictions[arm]
        if not isinstance(rows, list):
            raise PublicTransferEvaluationError("Prediction arm must be a list")
        observed = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("case_id"), str) or row["case_id"] not in expected:
                raise PublicTransferEvaluationError("Unknown prediction case")
            case_id = row["case_id"]
            if case_id in observed:
                raise PublicTransferEvaluationError("Duplicate prediction case")
            observed[case_id] = _norm(row.get("semantic_answer")) == _norm(expected[case_id]["expected_semantic"])
        if set(observed) != set(expected):
            raise PublicTransferEvaluationError("Prediction coverage must exactly match fixture")
        correct[arm] = observed

    counts = {
        arm: {
            "correct": sum(outcomes.values()), "total": len(cases),
            "by_stratum": {
                name: {"correct": sum(outcomes[i] for i in ids), "total": len(ids)}
                for name, ids in sorted(strata.items())
            },
        }
        for arm, outcomes in correct.items()
    }
    paired = {}
    for control in ARMS[1:]:
        wins = sum(correct["trained"][i] and not correct[control][i] for i in expected)
        losses = sum(not correct["trained"][i] and correct[control][i] for i in expected)
        paired[control] = {"trained_wins": wins, "trained_losses": losses,
                           "ties": len(expected) - wins - losses}
    differential = all(
        counts["trained"]["correct"] > counts[arm]["correct"]
        and all(counts["trained"]["by_stratum"][s]["correct"] > counts[arm]["by_stratum"][s]["correct"]
                for s in strata)
        for arm in ARMS[1:]
    )
    provenance = bundle.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise PublicTransferEvaluationError("provenance must be an object")
    receipt = {
        "scorer_version": SCORER_VERSION,
        "evidence_class": "PUBLIC_DEV_TEST",
        "fixture_sha256": _digest(fixture),
        "train_manifest_sha256": _digest(manifest),
        "train_rows_sha256": _digest(training),
        "input_sha256": _digest(bundle),
        "target_split": "public_development_transfer",
        "input_trust": {"gold": "UNVERIFIED_CALLER_INPUT", "predictions": "UNVERIFIED_CALLER_INPUT",
                        "provenance": "UNVERIFIED_CALLER_INPUT"},
        "control_types": list(ARMS),
        "case_count": len(cases),
        "scores": counts,
        "paired_against_trained": paired,
        "result": "PUBLIC_DEV_DIFFERENTIAL_ONLY" if differential else "NO_DIFFERENTIAL_SIGNAL",
        "q6_status": "HOLD",
        "qualifies_q6": False,
        "provenance": {key: ("UNKNOWN" if key in ("evaluation_bank_id", "evaluation_custodian")
                              else provenance.get(key, "UNKNOWN")) for key in PROVENANCE_FIELDS},
        "limitations": ["Public fixture only", "Caller-supplied gold and predictions are unverified",
                        "Scores do not establish statistical significance or model execution",
                        "Not a sealed bank", "No independent human or custodial witness"],
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt
