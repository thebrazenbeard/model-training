"""Independent R199 arithmetic TRAIN candidate; never an R185/R18 substitute.

Only first-principles integer arithmetic is used. No chat exports, inherited
corpus rows, external dataset, model output, or protected evaluator data are
read. A separately held 32-byte nonce is needed to commit the hypothetical
private DEV split. Nothing in this module grants training or evaluation access.
"""

from __future__ import annotations

import hashlib
import hmac
import json


SCHEMA = "VERA_TRAININGDAY_R199_NEW_SYNTH_ARITHMETIC_V1"
TRAIN_COUNT = 1024
DEV_COUNT = 64
_DEV_DOMAIN = (SCHEMA + ":PRIVATE_DEV:\x00").encode("ascii")


class SyntheticArithmeticError(ValueError):
    """Invalid synthetic split construction, never an authority grant."""


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _problem(index: int, a: int, b: int, operation: int, split: str) -> dict:
    if operation == 0:
        prompt, answer = f"Compute {a} + {b}.", a + b
    elif operation == 1:
        prompt, answer = f"Compute {a} - {b}.", a - b
    elif operation == 2:
        prompt, answer = f"Compute {a} * {b}.", a * b
    else:
        # Exact integral division; no rounding or numerical approximation.
        prompt, answer = f"Compute {a * b} / {b}.", a
    return {"id": f"r199-{split}-{index:04d}", "prompt": prompt,
            "answer": str(answer), "split": split}


def train_rows() -> list[dict]:
    """Deterministic public TRAIN records; no private/copyrighted inputs."""
    rows = []
    for index in range(TRAIN_COUNT):
        seed = hashlib.sha256((SCHEMA + f":TRAIN:{index}").encode("ascii")).digest()
        # TRAIN operands use a small bounded domain. Each `a` is unique.
        a = 100 + index
        b = 2 + int.from_bytes(seed[:2], "big") % 97
        rows.append(_problem(index, a, b, index % 4, "train"))
    if len({row["prompt"] for row in rows}) != TRAIN_COUNT:
        raise SyntheticArithmeticError("TRAIN collision")
    return rows


def train_bytes() -> bytes:
    return _jsonl(train_rows())


def _private_dev_rows(nonce: bytes) -> list[dict]:
    if (type(nonce) is not bytes or len(nonce) != 32
            or nonce == b"\x00" * 32):
        raise SyntheticArithmeticError("private DEV custodian nonce missing")
    rows = []
    for index in range(DEV_COUNT):
        seed = hmac.new(nonce, _DEV_DOMAIN + index.to_bytes(4, "big"),
                        hashlib.sha256).digest()
        # DEV uses disjoint operand scale and divisor domain, not the TRAIN
        # index sequence. The externally kept nonce is never returned.
        a = 20000 + int.from_bytes(seed[:4], "big") % 50000
        b = 211 + int.from_bytes(seed[4:8], "big") % 500
        rows.append(_problem(index, a, b, seed[8] % 4, "dev"))
    if len({row["prompt"] for row in rows}) != DEV_COUNT:
        raise SyntheticArithmeticError("DEV collision")
    return rows


def commitment(nonce: bytes) -> dict:
    """Hash-only source candidate, with NO DEV rows, answers, or nonce output."""
    train = train_rows()
    dev = _private_dev_rows(nonce)
    if ({row["id"] for row in train} & {row["id"] for row in dev}
            or {row["prompt"] for row in train} & {row["prompt"] for row in dev}):
        raise SyntheticArithmeticError("TRAIN/DEV overlap")
    return {
        "schema": SCHEMA,
        "status": "NEW_SOURCE_CANDIDATE_UNAUTHENTICATED_HOLD",
        "corpus_id": "R199_NEW_SYNTHETIC_ARITHMETIC",
        "r185_corpus_replacement": False,
        "origin": "FIRST_PRINCIPLES_INTEGER_ARITHMETIC_ONLY",
        "train_count": len(train),
        "train_sha256": hashlib.sha256(_jsonl(train)).hexdigest(),
        "dev_count": len(dev),
        "private_dev_sha256": hashlib.sha256(_jsonl(dev)).hexdigest(),
        "private_dev_rows_returned": False,
        "nonce_entropy_independently_verified": False,
        "rights_holder_authenticated": False,
        "data_rights_granted": False,
        "protected_evaluator_custody_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
    }
