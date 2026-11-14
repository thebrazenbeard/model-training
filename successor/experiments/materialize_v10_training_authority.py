from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


INTENT_FILENAME = "V10_QWEN35_TRAINING_AUTHORITY_INTENT_V1.json"
AUTHORITY_FILENAME = "V10_QWEN35_TRAINING_AUTHORITY_V1.json"
CONTRACT_FILENAME = "V10_QWEN35_EXPERIMENT_CONTRACT_V2.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required:{path}")
    return value


def _validate_intent(intent: dict) -> None:
    expected = {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_INTENT_V1",
        "status": "USER_DIRECT_CONDITIONAL_ONE_RUN_AUTHORITY",
        "authority_actor_id": "PATRICK_USER_AUTHORITY",
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "max_training_runs": 1,
        "paid_compute_authorized": False,
        "merge_authorized": False,
        "install_activate_deploy_authorized": False,
    }
    for field, value in expected.items():
        if intent.get(field) != value:
            raise ValueError(f"intent:{field}")


def _require_verified_component(
    preflight: dict,
    field: str,
) -> dict:
    value = preflight.get(field)
    if not isinstance(value, dict):
        raise RuntimeError(f"preflight component missing:{field}")
    if value.get("status") != "VERIFIED":
        raise RuntimeError(
            f"preflight component not verified:{field}:{value.get('status')}"
        )
    return value


def build_training_authority_receipt(
    *,
    intent: dict,
    contract: dict,
    preflight: dict,
    output_namespace: str,
) -> dict:
    _validate_intent(intent)

    if preflight.get("schema") != "V10_QWEN35_PREFLIGHT_V1":
        raise ValueError("preflight:schema")
    if not isinstance(output_namespace, str) or not output_namespace.strip():
        raise ValueError("output_namespace")

    reasons = preflight.get("reasons")
    if not isinstance(reasons, list):
        raise RuntimeError("preflight reasons missing")
    non_authority = sorted(
        {
            reason
            for reason in reasons
            if reason != "patrick_exact_weight_change_authority"
        }
    )
    if non_authority:
        raise RuntimeError(
            "preflight has unresolved non-authority blockers: "
            + " | ".join(non_authority)
        )

    sealed = _require_verified_component(preflight, "sealed_final_bank")
    runtime = _require_verified_component(preflight, "training_runtime")
    execution = _require_verified_component(preflight, "training_execution")

    if preflight.get("training_authority", {}).get("status") not in {
        "ABSENT",
        None,
    }:
        raise RuntimeError("training authority is not absent")

    source = contract.get("source_subject")
    base = contract.get("base_model")
    recipe = contract.get("training_recipe")
    if not isinstance(source, dict):
        raise ValueError("contract:source_subject")
    if not isinstance(base, dict):
        raise ValueError("contract:base_model")
    if not isinstance(recipe, dict):
        raise ValueError("contract:training_recipe")

    value = {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_V1",
        "status": "AUTHORIZED",
        "authority_actor_id": "PATRICK_USER_AUTHORITY",
        "authorization_source": "EXPLICIT_CURRENT_USER_INSTRUCTION",
        "experiment_id": contract.get("experiment_id"),
        "effect": "TRAIN_ONE_FRESH_QLORA_ADAPTER",
        "base_model_revision": base.get("revision"),
        "training_corpus_id": source.get("training_corpus_id"),
        "train_sha256": source.get("train_sha256"),
        "validation_sha256": source.get("validation_sha256"),
        "training_recipe_sha256": hashlib.sha256(_canonical(recipe)).hexdigest(),
        "sealed_final_bank_commitment_sha256": sealed.get(
            "commitment_sha256"
        ),
        "training_runtime_binding_sha256": runtime.get("binding_sha256"),
        "training_execution_binding_sha256": execution.get("binding_sha256"),
        "max_training_runs": 1,
        "output_namespace": output_namespace,
        "paid_compute_authorized": False,
        "merge_authorized": False,
        "install_activate_deploy_authorized": False,
    }
    missing = [
        field
        for field in (
            "experiment_id",
            "base_model_revision",
            "training_corpus_id",
            "train_sha256",
            "validation_sha256",
            "sealed_final_bank_commitment_sha256",
            "training_runtime_binding_sha256",
            "training_execution_binding_sha256",
        )
        if not isinstance(value.get(field), str) or not value[field].strip()
    ]
    if missing:
        raise ValueError("authority subjects missing:" + ",".join(missing))

    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def materialize_from_repo(
    *,
    repo_root: Path,
    output_namespace: str,
    write: bool,
) -> dict:
    from successor.evaluate_successor import assess_v10_experiment_state

    experiment_dir = repo_root / "successor" / "experiments"
    intent = _read_json(experiment_dir / INTENT_FILENAME)
    contract = _read_json(experiment_dir / CONTRACT_FILENAME)
    preflight = assess_v10_experiment_state(repo_root)

    receipt = build_training_authority_receipt(
        intent=intent,
        contract=contract,
        preflight=preflight,
        output_namespace=output_namespace,
    )
    if not write:
        return {
            "schema": "V10_QWEN35_TRAINING_AUTHORITY_MATERIALIZATION_PLAN_V1",
            "status": "READY_TO_WRITE_EXACT_AUTHORITY",
            "effect": "READ_ONLY_NO_AUTHORITY_FILE_WRITTEN",
            "receipt": receipt,
        }

    path = experiment_dir / AUTHORITY_FILENAME
    if path.exists():
        raise RuntimeError(f"authority receipt already exists:{path}")
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "schema": "V10_QWEN35_TRAINING_AUTHORITY_MATERIALIZATION_V1",
        "status": "EXACT_AUTHORITY_WRITTEN",
        "effect": "AUTHORITY_RECEIPT_WRITE_ONLY_NO_WEIGHT_CHANGE",
        "path": str(path),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-namespace", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = materialize_from_repo(
            repo_root=args.repo_root,
            output_namespace=args.output_namespace,
            write=args.write,
        )
        code = 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        result = {
            "schema": "V10_QWEN35_TRAINING_AUTHORITY_MATERIALIZATION_V1",
            "status": "HOLD",
            "effect": "NO_AUTHORITY_FILE_WRITTEN_NO_WEIGHT_CHANGE",
            "reasons": [str(exc)],
        }
        code = 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
