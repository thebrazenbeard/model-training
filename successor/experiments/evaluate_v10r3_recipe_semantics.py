from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.evaluate_v10r2_heldout import (
    _paired_comparison,
    build_completion_example,
    evaluate_candidate,
)
from successor.experiments.train_v10_qwen35_authorized import (
    _observe_live_runtime,
    _read_json,
    load_training_stack,
    load_verified_jsonl,
    sha256_file,
    validate_runtime_observation,
)


TRAIN_SHA256 = "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
TRAIN_ROWS = 50000
PANEL_SCHEMA = "V10R3_RECIPE_SEMANTICS_TRAIN_HOLDOUT_PANEL_V1"
PANEL_ROLE = "ONE_TIME_DEVELOPMENT_RECIPE_SEMANTICS_DIAGNOSTIC_NOT_FINAL_BANK"


class RecipeEvalHold(RuntimeError):
    """Fail-closed refusal for malformed V10R3 recipe-semantics evaluation."""


def _record_id(row: dict) -> str:
    for field in ("record_id", "case_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise RecipeEvalHold("row missing stable record id")


def _ids_sha(record_ids: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(record_ids) + "\n").encode("utf-8")
    ).hexdigest()


def load_train_holdout_panel(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RecipeEvalHold(f"panel missing:{path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("schema") != PANEL_SCHEMA:
        raise RecipeEvalHold("panel schema mismatch")
    if value.get("role") != PANEL_ROLE:
        raise RecipeEvalHold("panel role mismatch")
    if value.get("train_sha256") != TRAIN_SHA256:
        raise RecipeEvalHold("panel train sha mismatch")
    if value.get("train_rows") != TRAIN_ROWS:
        raise RecipeEvalHold("panel train row count mismatch")

    count = value.get("row_count")
    indices = value.get("row_indices")
    ids = value.get("record_ids")
    if not isinstance(count, int) or count < 1:
        raise RecipeEvalHold("panel row count invalid")
    if (
        not isinstance(indices, list)
        or len(indices) != count
        or any(not isinstance(i, int) or i < 0 or i >= TRAIN_ROWS for i in indices)
    ):
        raise RecipeEvalHold("panel row indices invalid")
    if len(set(indices)) != count:
        raise RecipeEvalHold("panel row indices not unique")
    if (
        not isinstance(ids, list)
        or len(ids) != count
        or any(not isinstance(item, str) or not item for item in ids)
    ):
        raise RecipeEvalHold("panel record ids invalid")
    if len(set(ids)) != count:
        raise RecipeEvalHold("panel record ids not unique")
    if value.get("record_ids_sha256") != _ids_sha(ids):
        raise RecipeEvalHold("panel record ids digest mismatch")

    policy = value.get("selection_policy")
    if not isinstance(policy, dict):
        raise RecipeEvalHold("panel selection policy missing")
    if policy.get("one_time_recipe_selection_use") is not True:
        raise RecipeEvalHold("panel must be one-time recipe selection use")
    if policy.get("subsequent_checkpoint_selection_reuse") != "PROHIBITED":
        raise RecipeEvalHold("panel subsequent reuse must be prohibited")
    if policy.get("final_bank_use") != "PROHIBITED":
        raise RecipeEvalHold("panel final-bank use must be prohibited")
    if policy.get("external_qualification_use") != "PROHIBITED":
        raise RecipeEvalHold("panel external qualification use must be prohibited")
    return value


def select_train_holdout_rows(rows: list[dict], panel: dict) -> list[dict]:
    if len(rows) != TRAIN_ROWS:
        raise RecipeEvalHold(f"train row count mismatch:{len(rows)}!={TRAIN_ROWS}")
    selected = []
    for index, expected_id in zip(
        panel["row_indices"],
        panel["record_ids"],
        strict=True,
    ):
        row = rows[index]
        actual_id = _record_id(row)
        if actual_id != expected_id:
            raise RecipeEvalHold(
                f"panel record id mismatch at index {index}:{actual_id}!={expected_id}"
            )
        selected.append(row)
    return selected


def parse_candidate_specs(values: list[str]) -> list[tuple[str, Path]]:
    if len(values) < 2:
        raise RecipeEvalHold("at least two candidates are required")
    names: set[str] = set()
    result: list[tuple[str, Path]] = []
    for raw in values:
        if "=" not in raw:
            raise RecipeEvalHold(f"invalid candidate spec:{raw}")
        name, path_text = raw.split("=", 1)
        name = name.strip()
        path_text = path_text.strip()
        if not name or not path_text:
            raise RecipeEvalHold(f"invalid candidate spec:{raw}")
        if name in names:
            raise RecipeEvalHold(f"duplicate candidate name:{name}")
        names.add(name)
        result.append((name, Path(path_text)))
    return result


def run_recipe_semantics_eval(
    repo_root: Path,
    *,
    train_jsonl: Path,
    panel_path: Path,
    candidates: list[tuple[str, Path]],
) -> dict:
    runtime_path = (
        repo_root
        / "successor"
        / "experiments"
        / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    runtime_binding = _read_json(runtime_path)
    stack = load_training_stack()
    runtime_observation = _observe_live_runtime(runtime_binding, stack)
    runtime_check = validate_runtime_observation(
        runtime_binding,
        runtime_observation,
    )

    rows = load_verified_jsonl(
        train_jsonl,
        expected_sha256=TRAIN_SHA256,
        expected_rows=TRAIN_ROWS,
    )
    panel = load_train_holdout_panel(panel_path)
    selected = select_train_holdout_rows(rows, panel)

    base_path = Path(runtime_binding["target"]["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        build_completion_example(tokenizer, row, max_length=512)
        for row in selected
    ]

    torch = stack["torch"]
    quant_config = stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    results = [
        evaluate_candidate(
            stack=stack,
            tokenizer=tokenizer,
            examples=examples,
            base_path=base_path,
            quant_config=quant_config,
            name=name,
            adapter_dir=adapter_dir,
        )
        for name, adapter_dir in candidates
    ]
    comparisons = []
    for left_index in range(len(results)):
        for right_index in range(left_index + 1, len(results)):
            comparisons.append(
                _paired_comparison(results[left_index], results[right_index])
            )

    return {
        "schema": "V10R3_RECIPE_SEMANTICS_TRAIN_HOLDOUT_EVAL_V1",
        "status": "DEVELOPMENT_RECIPE_SEMANTICS_DIAGNOSTIC_COMPLETE",
        "role": "ONE_TIME_TRAIN_CORPUS_HOLDOUT_DIAGNOSTIC_NOT_FINAL_BANK",
        "train_sha256": sha256_file(train_jsonl),
        "train_rows": TRAIN_ROWS,
        "panel_path": str(panel_path),
        "panel_sha256": sha256_file(panel_path),
        "panel_record_ids_sha256": panel["record_ids_sha256"],
        "panel_row_indices": panel["row_indices"],
        "panel_record_ids": panel["record_ids"],
        "runtime_binding_sha256": runtime_binding["binding_sha256"],
        "runtime_observation": runtime_observation,
        "runtime_check": runtime_check,
        "candidates": results,
        "paired_comparisons": comparisons,
        "claim_ceiling": (
            "DEVELOPMENT RECIPE-SEMANTICS TRAIN-CORPUS HOLDOUT ONLY / "
            "ONE-TIME RECIPE SELECTION USE / NOT FINAL BANK / "
            "NOT EXTERNAL QUALIFICATION"
        ),
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        candidates = parse_candidate_specs(args.candidate)
        result = run_recipe_semantics_eval(
            args.repo_root,
            train_jsonl=args.train_jsonl,
            panel_path=args.panel,
            candidates=candidates,
        )
    except (RecipeEvalHold, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema": "V10R3_RECIPE_SEMANTICS_TRAIN_HOLDOUT_EVAL_V1",
                    "status": "HOLD",
                    "reason": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RecipeEvalHold(f"output already exists:{args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())