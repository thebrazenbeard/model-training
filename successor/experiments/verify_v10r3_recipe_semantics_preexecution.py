from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.evaluate_v10r3_recipe_semantics import (
    load_train_holdout_panel,
    select_train_holdout_rows,
)
from successor.experiments.train_v10_qwen35_authorized import (
    canonical_bytes,
    load_verified_jsonl,
)
from successor.experiments.train_v10r2_dev import load_and_validate_dev_spec


TRAIN_SHA256 = "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
TRAIN_ROWS = 50000
ORDER_IDS_SHA256 = "904ac73c903a3878f950215d004a7d4a17af45a002d2637cba3b3a7995974142"
EXPECTED_INITIAL_DIGEST = (
    "134dc5a9fdbdff2a6edc7c1bae6e999fed112b81048ae9af6fd7298338079bfd"
)
HISTORICAL_RECEIPT_SHA256 = (
    "8c02cad57d1ec6f89139623310d9de8a0ac13a484701e5a371a868a0baa3bc38"
)


class PreexecutionHold(RuntimeError):
    """Fail-closed V10R3 recipe-semantics preexecution refusal."""


def committed_text_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_record_id(row: dict, index: int) -> str:
    for field in ("record_id", "case_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return f"bound-train-row-{index}"


def verify_receipt(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    claimed = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    actual = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if claimed != actual:
        raise PreexecutionHold("historical initialization receipt self-hash mismatch")
    return value


def run_preexecution_verification(
    repo_root: Path,
    *,
    train_jsonl: Path,
    historical_initialization_receipt: Path,
) -> dict:
    exp = repo_root / "successor" / "experiments"
    protocol_path = exp / "V10R3_STAGED_VS_CONTINUOUS_PROTOCOL_20261004_V2.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))

    rows = load_verified_jsonl(
        train_jsonl,
        expected_sha256=TRAIN_SHA256,
        expected_rows=TRAIN_ROWS,
    )
    ids = [stable_record_id(rows[index], index) for index in range(160)]
    ids_sha = hashlib.sha256(
        ("\n".join(ids) + "\n").encode("utf-8")
    ).hexdigest()
    if ids_sha != ORDER_IDS_SHA256:
        raise PreexecutionHold("frozen train order digest mismatch")

    manifest_path = exp / "V10R3_RECIPE_SEMANTICS_ORDER_ROWS0_159_20261004_V2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("record_ids_serialization") != "NEWLINE_UTF8_FINAL_LF":
        raise PreexecutionHold("order serialization binding mismatch")
    if manifest.get("row_indices") != list(range(160)):
        raise PreexecutionHold("order manifest indices mismatch")
    if manifest.get("record_ids") != ids:
        raise PreexecutionHold("order manifest record ids mismatch frozen corpus")
    if manifest.get("record_ids_sha256") != ids_sha:
        raise PreexecutionHold("order manifest record id hash mismatch")
    if len(set(ids)) != 160:
        raise PreexecutionHold("order manifest record ids are not unique")

    bindings = protocol.get("committed_file_bindings")
    if not isinstance(bindings, dict):
        raise PreexecutionHold("protocol committed-file bindings missing")
    binding_paths = {
        "control_stage1_spec": exp
        / "V10R3_STAGED_RESET_CONTROL_STAGE1_SPEC_20261004_V2.json",
        "continuous20_spec": exp
        / "V10R3_CONTINUOUS20_TRAINING_SPEC_20261004_V2.json",
        "order_manifest": manifest_path,
        "fresh_eval_panel": exp
        / "V10R3_RECIPE_SEMANTICS_PANEL_ROWS448_511_V1.json",
    }
    actual_file_hashes: dict[str, str] = {}
    for key, path in binding_paths.items():
        actual = committed_text_sha(path)
        expected = bindings.get(key, {}).get("sha256")
        if actual != expected:
            raise PreexecutionHold(
                f"committed text hash mismatch:{key}:{actual}!={expected}"
            )
        actual_file_hashes[key] = actual

    stage1 = load_and_validate_dev_spec(binding_paths["control_stage1_spec"])
    continuous = load_and_validate_dev_spec(binding_paths["continuous20_spec"])
    expected_init = protocol["initialization_equivalence_gate"][
        "expected_initial_trainable_parameter_digest"
    ]
    if expected_init != EXPECTED_INITIAL_DIGEST:
        raise PreexecutionHold("protocol expected initialization digest mismatch")
    for name, spec in (("stage1", stage1), ("continuous20", continuous)):
        actual_expected = spec["comparison"].get(
            "expected_initial_trainable_parameter_digest"
        )
        if actual_expected != EXPECTED_INITIAL_DIGEST:
            raise PreexecutionHold(
                f"{name} expected initialization digest mismatch"
            )
        if spec["trainer"].get("shuffle_dataset") is not False:
            raise PreexecutionHold(f"{name} shuffle must be disabled")
        if spec["trainer"].get("train_sampling_strategy") != "sequential":
            raise PreexecutionHold(f"{name} sampler must be sequential")
        output = Path(spec["output"]["namespace"])
        if output.exists():
            raise PreexecutionHold(f"{name} output already exists:{output}")

    historical = verify_receipt(historical_initialization_receipt)
    if historical.get("receipt_sha256") != HISTORICAL_RECEIPT_SHA256:
        raise PreexecutionHold("historical initialization receipt identity mismatch")
    if historical.get("runtime_binding_sha256") != protocol["runtime"]["binding_sha256"]:
        raise PreexecutionHold("historical initialization runtime mismatch")
    if historical.get("weight_digest_before") != EXPECTED_INITIAL_DIGEST:
        raise PreexecutionHold("historical initialization digest mismatch")
    if historical.get("train_sha256") != TRAIN_SHA256:
        raise PreexecutionHold("historical initialization train binding mismatch")

    panel_path = binding_paths["fresh_eval_panel"]
    panel = load_train_holdout_panel(panel_path)
    selected = select_train_holdout_rows(rows, panel)
    if len(selected) != 64:
        raise PreexecutionHold("fresh evaluation panel row count mismatch")

    return {
        "schema": "V10R3_RECIPE_SEMANTICS_PREEXECUTION_VERIFICATION_V2",
        "status": "PASS",
        "role": "LOCAL_FAIL_CLOSED_PREEXECUTION_BINDING_VERIFICATION",
        "train_jsonl": str(train_jsonl),
        "train_sha256": TRAIN_SHA256,
        "train_rows": TRAIN_ROWS,
        "ordered_record_ids_sha256": ids_sha,
        "ordered_record_count": len(ids),
        "committed_text_hashes": actual_file_hashes,
        "historical_initialization_receipt": {
            "path": str(historical_initialization_receipt),
            "receipt_sha256": historical["receipt_sha256"],
            "expected_initial_trainable_parameter_digest": EXPECTED_INITIAL_DIGEST,
        },
        "fresh_eval_panel": {
            "path": str(panel_path),
            "record_ids_sha256": panel["record_ids_sha256"],
            "row_count": len(selected),
        },
        "outputs_absent": True,
        "gpu_effect": "NONE",
        "claim_ceiling": (
            "PREEXECUTION_BINDING_VERIFICATION_ONLY_NOT_TRAINING_RESULT_"
            "NOT_FINAL_BANK"
        ),
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument(
        "--historical-initialization-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_preexecution_verification(
            args.repo_root,
            train_jsonl=args.train_jsonl,
            historical_initialization_receipt=args.historical_initialization_receipt,
        )
    except (PreexecutionHold, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "V10R3_RECIPE_SEMANTICS_PREEXECUTION_VERIFICATION_V2",
                    "status": "HOLD",
                    "reason": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    if args.output.exists():
        raise PreexecutionHold(f"verification output already exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())