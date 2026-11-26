from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from successor.experiments.evaluate_v10r3_recipe_semantics import (
    RecipeEvalHold,
    load_train_holdout_panel,
    select_train_holdout_rows,
)


def _write_panel(path: Path) -> Path:
    ids = ["row-a", "row-b"]
    value = {
        "schema": "V10R3_RECIPE_SEMANTICS_TRAIN_HOLDOUT_PANEL_V1",
        "role": "ONE_TIME_DEVELOPMENT_RECIPE_SEMANTICS_DIAGNOSTIC_NOT_FINAL_BANK",
        "train_sha256": (
            "a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300"
        ),
        "train_rows": 50000,
        "row_start": 10,
        "row_count": 2,
        "row_indices": [10, 11],
        "record_ids": ids,
        "record_ids_sha256": hashlib.sha256(
            ("\n".join(ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "selection_policy": {
            "one_time_recipe_selection_use": True,
            "subsequent_checkpoint_selection_reuse": "PROHIBITED",
            "final_bank_use": "PROHIBITED",
            "external_qualification_use": "PROHIBITED",
            "rows_may_remain_in_eventual_full_training_corpus": True,
        },
        "claim_ceiling": (
            "DEVELOPMENT_TRAIN_CORPUS_HOLDOUT_ONLY_NOT_FRESH_FINAL_BANK_"
            "NOT_EXTERNAL_QUALIFICATION"
        ),
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_train_holdout_panel_accepts_exact_binding(tmp_path: Path) -> None:
    got = load_train_holdout_panel(_write_panel(tmp_path / "panel.json"))
    assert got["row_indices"] == [10, 11]
    assert got["record_ids"] == ["row-a", "row-b"]


def test_load_train_holdout_panel_rejects_reuse_policy(tmp_path: Path) -> None:
    path = _write_panel(tmp_path / "panel.json")
    value = json.loads(path.read_text())
    value["selection_policy"]["subsequent_checkpoint_selection_reuse"] = "ALLOWED"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RecipeEvalHold, match="subsequent reuse"):
        load_train_holdout_panel(path)


def test_select_train_holdout_rows_requires_index_and_id_match(
    tmp_path: Path,
) -> None:
    panel = load_train_holdout_panel(_write_panel(tmp_path / "panel.json"))
    rows = [{"record_id": f"row-{i}"} for i in range(50000)]
    rows[10]["record_id"] = "row-a"
    rows[11]["record_id"] = "row-b"
    selected = select_train_holdout_rows(rows, panel)
    assert [r["record_id"] for r in selected] == ["row-a", "row-b"]

    rows[11]["record_id"] = "wrong"
    with pytest.raises(RecipeEvalHold, match="record id mismatch"):
        select_train_holdout_rows(rows, panel)