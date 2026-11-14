from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "successor" / "experiments"
TEMPLATE = EXP / "V10R3R1_CONTINUOUS20_EXECUTABILITY_REPLICATION_SPEC_20261004_V1.json"
PROTOCOL = EXP / "V10R3R1_CONTINUOUS20_EXECUTABILITY_REPLICATION_PROTOCOL_20261004_V1.json"
VERIFIER = EXP / "verify_v10r3r1_execution_projection.py"


def _load_verifier():
    assert VERIFIER.is_file(), "V10R3R1 execution projection verifier is missing"
    spec = importlib.util.spec_from_file_location("v10r3r1_projection_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _authorized_execution_spec(tmp_path: Path) -> Path:
    value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    value["status"] = "AUTHORIZED_LOCAL_DEVELOPMENT_TRAINING"
    value["authority"] = {
        "actor_id": "PATRICK_USER_AUTHORITY",
        "source": "EXPLICIT_CURRENT_USER_INSTRUCTION",
        "scope": "CHANGE_TRAINING_METHOD_AND_CONTINUE_LOCAL_MODEL_TRAINING",
        "paid_compute_authorized": False,
        "merge_authorized": False,
        "deploy_activate_authorized": False,
    }
    path = tmp_path / "V10R3R1_EXECUTION_SPEC.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_execution_projection_accepts_authority_only_delta(tmp_path: Path) -> None:
    module = _load_verifier()
    execution = _authorized_execution_spec(tmp_path)

    result = module.verify_execution_projection(
        ROOT,
        template_path=TEMPLATE,
        protocol_path=PROTOCOL,
        execution_spec_path=execution,
    )

    assert result["status"] == "PASS"
    assert result["recipe_projection_match"] is True
    assert result["output_namespace_match"] is True


def test_execution_projection_rejects_recipe_mutation(tmp_path: Path) -> None:
    module = _load_verifier()
    execution = _authorized_execution_spec(tmp_path)
    value = json.loads(execution.read_text(encoding="utf-8"))
    value["trainer"]["learning_rate"] = 0.00003
    execution.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(module.ExecutionProjectionHold, match="execution projection mismatch"):
        module.verify_execution_projection(
            ROOT,
            template_path=TEMPLATE,
            protocol_path=PROTOCOL,
            execution_spec_path=execution,
        )
