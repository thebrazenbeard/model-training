"""Offline, fail-closed classification of claimed neural-model evidence.

This inventory does not verify files, hashes, training execution or model quality.
"""
import json
import re
from pathlib import Path

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
STATES = ("CONFIGURED_ONLY", "WEIGHT_UPDATE_REPORTED", "ARTIFACT_REPORTED", "EVALUATION_REPORTED")


def classify(receipt):
    if not isinstance(receipt, dict) or receipt.get("schema") != "VERA_NEURAL_EVIDENCE_V1":
        raise ValueError("unsupported receipt schema")
    if receipt.get("training_kind") not in ("none", "full_finetune", "adapter_finetune"):
        raise ValueError("unknown training kind")
    claims = receipt.get("claims", {})
    if not isinstance(claims, dict) or set(claims) != {"weight_update", "artifact", "evaluation"}:
        raise ValueError("exact claims required")
    if any(type(v) is not bool for v in claims.values()):
        raise ValueError("claims must be boolean")
    if (claims["evaluation"] and not claims["artifact"]) or (claims["artifact"] and not claims["weight_update"]):
        raise ValueError("inconsistent evidence order")
    if receipt["training_kind"] == "none" and any(claims.values()):
        raise ValueError("no training cannot assert trained artifacts")
    evidence = receipt.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be an object")
    required = []
    if claims["weight_update"]:
        required += ["base_revision", "dataset_sha256", "recipe_sha256", "run_log_sha256"]
    if claims["artifact"]:
        required += ["weights_sha256"]
    if claims["evaluation"]:
        required += ["eval_manifest_sha256", "eval_results_sha256", "baseline_results_sha256"]
    for key in required:
        value = evidence.get(key)
        pattern = (re.fullmatch(r"[0-9a-f]{40}", value) if key == "base_revision"
                   else SHA256.fullmatch(value)) if isinstance(value, str) else None
        if not pattern:
            raise ValueError(f"missing or malformed {key}")
    if claims["evaluation"]:
        if receipt.get("held_out") is not True or receipt.get("independent_evaluator") is not True:
            raise ValueError("evaluation requires held-out and independent evaluator declarations")
    level = sum(claims.values())
    return {"classification": STATES[level], "qualification": "UNVERIFIED",
            "hashes_verified": False, "training_executed_verified": False, "model_qualified": False}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(classify(json.loads(args.receipt.read_text(encoding="utf-8"))), sort_keys=True))


if __name__ == "__main__":
    main()
