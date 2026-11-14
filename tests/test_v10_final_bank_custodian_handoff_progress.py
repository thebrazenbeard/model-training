from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXP = ROOT / "successor" / "experiments"


def test_handoff_package_binds_current_synthetic_identity_receipts() -> None:
    package = json.loads(
        (EXP / "V10_FINAL_BANK_CUSTODIAN_HANDOFF_PACKAGE_V1.json").read_text(
            encoding="utf-8"
        )
    )
    pair = json.loads(
        (EXP / "V10_FINAL_BANK_SYNTHETIC_CUSTODIANS_V1.json").read_text(
            encoding="utf-8"
        )
    )
    receipts = package["synthetic_identity_receipts"]
    assert pair["status"] == "SYNTHETIC_CUSTODIANS_BOUND"
    assert receipts["A"]["identity_receipt_sha256"] == pair["custodians"]["A"][
        "identity_receipt_sha256"
    ]
    assert receipts["B"]["identity_receipt_sha256"] == pair["custodians"]["B"][
        "identity_receipt_sha256"
    ]
    assert package["currently_unresolved"] == [
        "human_author_identity",
        "human_reviewer_identity",
        "independent_custody_surface",
        "access_control_receipt",
    ]
    assert package["status"] == "SYNTHETIC_IDENTITIES_BOUND_HUMAN_CUSTODY_HOLD"
