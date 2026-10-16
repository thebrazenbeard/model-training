"""Deterministic readback of public, synthetic local-model development receipts.

Proves file/internal consistency against frozen cases and prompt codebooks,
NOT model/host attestation, independent training, or semantic capability.
"""
from __future__ import annotations
import json
from pathlib import Path

from semantic_transfer_eval_v1 import (
    load_cases, score_predictions, digest, PREDICTION_SCHEMA,
    TransferEvidenceError,
)
from semantic_transfer_local_probe_v1 import _SYSTEM, _CODEBOOK, _MODEL, _interpret


def verify_receipt(path: str | Path, cases_path: str | Path) -> dict:
    path=Path(path)
    if not path.is_file() or path.stat().st_size > 250_000:
        raise TransferEvidenceError("Absent or oversized evidence artifact")
    try:
        receipt=json.loads(path.read_text(encoding="utf8"))
    except (ValueError,UnicodeError) as exc:
        raise TransferEvidenceError("Malformed model evidence artifact") from exc
    required = {
        "schema", "model_advertisement", "endpoint_scope",
        "model_artifact_sha256", "training_activity", "case_limit",
        "prompt_profile", "prompt_profile_sha256", "dataset_sha256",
        "prediction_receipt", "model_case_results", "report",
        "evaluation_scope",
    }
    if type(receipt) is not dict or set(receipt)!=required:
        raise TransferEvidenceError("Unexpected evidence artifact structure")
    if (receipt["schema"]!="VERA_SEMANTIC_TRANSFER_LOCAL_MODEL_PROBE_V1"
        or receipt["model_advertisement"]!=_MODEL
        or receipt["endpoint_scope"]!="LAPPY_LOOPBACK_ONLY"
        or receipt["training_activity"]!="NONE"
        or receipt["model_artifact_sha256"]!="UNVERIFIED"
        or receipt["evaluation_scope"]!="PUBLIC_DEVELOPMENT_SET_NOT_HELD_OUT"):
        raise TransferEvidenceError("Evidence scope changed or overstated")
    profile=receipt["prompt_profile"]
    if profile not in ("minimal","codebook"):
        raise TransferEvidenceError("Unknown fixed prompt profile")
    expected_prompt=_SYSTEM+(_CODEBOOK if profile=="codebook" else "")
    if receipt["prompt_profile_sha256"]!=digest({"system_prompt":expected_prompt}):
        raise TransferEvidenceError("Prompt profile not content bound")
    cases=load_cases(cases_path)
    count=receipt["case_limit"]
    if type(count) is not int or count!=len(cases):
        raise TransferEvidenceError("Partial evidence is not full evaluation")
    rows=receipt["model_case_results"]
    if type(rows) is not list or len(rows)!=count:
        raise TransferEvidenceError("Missing or extra observed response records")
    predictions={}
    for c,row in zip(cases,rows):
        if type(row) is not dict or row.get("id")!=c["id"]:
            raise TransferEvidenceError("Response rows reordered or identity drifted")
        if "error" in row:
            if "parsed" in row or "model_response" in row:
                raise TransferEvidenceError("An errored request cannot report a parsed response")
        else:
            content=row.get("model_response")
            if type(content) is not str or len(content)>4000:
                raise TransferEvidenceError("Missing or invalid raw response")
            observed=_interpret(content)
            if observed!=row.get("parsed"):
                raise TransferEvidenceError("Raw response and parsed value disagree")
            if observed is not None:
                predictions[c["id"]]=observed
    packet=receipt["prediction_receipt"]
    if (type(packet) is not dict or set(packet)!={"schema","predictions"}
        or packet["schema"]!=PREDICTION_SCHEMA
        or packet["predictions"]!=predictions):
        raise TransferEvidenceError("Submitted predictions differ from raw observed outputs")
    report=receipt["report"]
    if type(report) is not dict or report.get("submission_kind")!="LOCAL_MODEL_DEVELOPMENT":
        raise TransferEvidenceError("Development evidence type mismatch")
    rerun=score_predictions(cases,predictions,run_id=report.get("run_id"),
                            submission_kind="LOCAL_MODEL_DEVELOPMENT")
    if (receipt["dataset_sha256"]!=rerun.dataset_sha256 or
        report!=rerun.to_mapping()):
        raise TransferEvidenceError("Stored score or dataset digest differs from recomputation")
    return {
        "integrity":"CONSISTENT_UNATTESTED",
        "profile":profile,
        "cases":count,
        "valid_json_responses":len(predictions),
        "exact_matches":sum(p for _,p,_ in rerun.scores),
        "category_scores":list(rerun.scores),
        "dataset_sha256":rerun.dataset_sha256,
        "prediction_sha256":rerun.prediction_sha256,
        "model_advertisement":_MODEL,
        "qualification":"NONE",
    }


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--cases",type=Path,
        default=Path(__file__).parent/"training"/"semantic_transfer_devset_v1.json")
    opts=parser.parse_args()
    print(json.dumps(verify_receipt(opts.receipt,opts.cases),sort_keys=True))
