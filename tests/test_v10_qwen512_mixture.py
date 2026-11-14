from __future__ import annotations

import hashlib
import json


def test_qwen512_training_manifest_rebinds_subject() -> None:
    from successor import build_v4_1_qwen512_training_corpus as training

    old = {
        "schema": "VERA_V4_1_TRAINING_CORPUS_MANIFEST_V1",
        "corpus_id": "VERA_SUCCESSOR_V4_1_50K_20260930_V10",
        "train_rows": 50_000,
        "validation_rows": 2_500,
        "weight_change_performed": False,
        "training_authorization": "NOT_GRANTED",
        "custom_source": {
            "corpus_id": "VERA_SUCCESSOR_V4_1_10K_DIVERSE_CORE_20260930_V10"
        },
        "rehearsal_source": {"corpus_id": "old"},
        "manifest_sha256": "0" * 64,
    }
    rebound = training.rebind_manifest(old)
    assert rebound["schema"] == "VERA_V10_QWEN512_TRAINING_CORPUS_MANIFEST_V1"
    assert rebound["corpus_id"] == "VERA_SUCCESSOR_V10_QWEN512_50K_20261001_V1"
    assert rebound["parent_v10_custom_corpus_id"] == "VERA_SUCCESSOR_V4_1_10K_DIVERSE_CORE_20260930_V10"
    assert rebound["training_authorization"] == "NOT_GRANTED"
    assert rebound["weight_change_performed"] is False
    assert rebound["qwen_training_contract"]["max_length"] == 512
    assert rebound["qwen_training_contract"]["overflow"] == "error"
    assert rebound["qwen_training_contract"]["parent_adapter"] is None

    canonical = dict(rebound)
    digest = canonical.pop("manifest_sha256")
    expected = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest == expected
