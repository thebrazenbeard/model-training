import json
from pathlib import Path

from successor.vera_lab.smollm_adapter import (
    INVENTORY_DIGEST_ALGORITHM,
    SMOLLM3_CACHE_TREE_SHA256,
    SMOLLM3_FILE_INVENTORY_SHA256,
    SMOLLM3_TOTAL_BYTES,
)


def test_base_manifest_is_exactly_pinned_as_pilot_control_not_identity_target():
    path = Path("successor/base_model_manifest.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["repo_id"] == "HuggingFaceTB/SmolLM3-3B"
    assert data["revision"] == "a07cc9a04f16550a088caea529712d1d335b0ac1"
    assert data["local_path"].lower() == r"c:\vera\models\base\smollm3-3b"
    assert data["mutable_revision_allowed"] is False
    assert data["license"] == "apache-2.0"
    assert data["file_inventory_sha256"] == SMOLLM3_FILE_INVENTORY_SHA256
    assert data["inventory_digest_algorithm"] == INVENTORY_DIGEST_ALGORITHM
    assert data["total_bytes"] == SMOLLM3_TOTAL_BYTES
    assert data["observed_local_cache_tree_sha256"] == SMOLLM3_CACHE_TREE_SHA256
    assert data["training_role"] == "PILOT_CONTROL_SUBSTRATE"
    assert data["identity_package_binding"] == "SUBSTRATE_AGNOSTIC"
    assert data["successor_target_claim"] == "BEHAVIORAL_IDENTITY_PACKAGE_NOT_BASE_MODEL_IDENTITY"
