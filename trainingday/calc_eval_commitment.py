"""Source-only, nonce-derived mathematics DEV commitment for TrainingDay.

This deliberately does NOT unseal examples or authorize a training job. A
separately held 32-byte secret is needed to reproduce the private examples.
The exact integer power-rule oracle is independent from ``calc`` execution,
which lets a protected evaluator check Calc-generated TRAIN examples without
trusting the same runtime implementation on both sides.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Sequence


SCHEMA = "VERA_TRAININGDAY_R201_CALC_PRIVATE_DEV_COMMITMENT_V1"
CALC_SOURCE_REPO = "thebrazenbeard/calc"
CALC_SOURCE_REVISION = "17e12d0f659b42732fb98e5730732aba9ff1b488"
CALC_ALGEBRA_GIT_BLOB = "54b7bb78e94481d80b973f46edbaab48b9877b96"
CALC_ALGEBRA_SHA256 = "6a1b269d14ddb34fbe4e8e5a9f56ac98c401fa98636cd3b13ab423cba1bead08"
DEV_COUNT = 64
R201_TRAIN_COUNT = 1024
MATCHED_SCHEMA = "VERA_TRAININGDAY_R201_CALC_MATCHED_PRIVATE_DEV_COMMITMENT_V1"
MAX_PUBLIC_PROMPTS = 16_384
MAX_PUBLIC_PROMPT_CHARS = 2_048
_DOMAIN = (SCHEMA + ":" + CALC_SOURCE_REVISION + ":\x00").encode("ascii")


class CalcEvalCommitmentError(ValueError):
    """Reject an invalid proposed private DEV commitment or public comparison."""


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _valid_nonce(nonce: bytes) -> None:
    if type(nonce) is not bytes or len(nonce) != 32 or nonce == bytes(32):
        raise CalcEvalCommitmentError("separately protected 32-byte DEV nonce required")


def _draw(nonce: bytes, index: int, subdomain: bytes) -> bytes:
    return hmac.new(nonce, _DOMAIN + subdomain + index.to_bytes(4, "big"),
                    hashlib.sha256).digest()


def _private_dev_rows(nonce: bytes) -> list[dict]:
    """Private-only generator; never return this from ``commitment``.

    Degree 4--6 with |x| >= 11 differs structurally from the R201 proposed
    public cubic/small-x curriculum. That is a *distribution shift*, not a
    claim of independent held-out evaluator identity or semantic novelty.
    """
    _valid_nonce(nonce)
    rows: list[dict] = []
    for index in range(DEV_COUNT):
        seed = _draw(nonce, index, b"SPEC:\x00")
        degree = 4 + seed[0] % 3
        x = (11 + seed[2] % 43) * (-1 if seed[1] & 1 else 1)
        coefficients = []
        for power in range(degree + 1):
            cell = _draw(nonce, index, b"COEFF:" + power.to_bytes(1, "big"))
            coefficient = int.from_bytes(cell[:2], "big") % 101 - 50
            if power == degree and coefficient == 0:
                coefficient = 1
            coefficients.append(coefficient)
        answer = sum(power * coefficients[power] * x ** (power - 1)
                     for power in range(1, degree + 1))
        prompt = ("Evaluate P'(x) at x=" + str(x)
                  + " for the polynomial with ascending coefficients "
                  + json.dumps(coefficients, separators=(",", ":")) + ".")
        rows.append({"id": f"r201-calc-dev-{index:04d}",
                     "prompt": prompt, "answer": str(answer), "split": "dev"})
    if (len({row["id"] for row in rows}) != DEV_COUNT
            or len({row["prompt"] for row in rows}) != DEV_COUNT):
        raise CalcEvalCommitmentError("private DEV example collision")
    return rows


def _private_matched_dev_rows(nonce: bytes) -> list[dict]:
    """Private R201-format examples, HMAC-domain-separated from shifted DEV.

    Public TRAIN uses cubics with |a| 1..17, b -30..30, c/d -50..50,
    x -5..5. The same bounds and prompt form are reproduced here with a
    distinct secret-derived sample; this alone is NOT semantic independence.
    """
    _valid_nonce(nonce)
    rows: list[dict] = []
    for index in range(DEV_COUNT):
        seed = _draw(nonce, index, b"R201_MATCHED_CUBIC_V1:\x00")
        a = 1 + int.from_bytes(seed[0:2], "big") % 17
        if seed[2] & 1:
            a = -a
        b = int.from_bytes(seed[3:5], "big") % 61 - 30
        c = int.from_bytes(seed[5:7], "big") % 101 - 50
        d = int.from_bytes(seed[7:9], "big") % 101 - 50
        x = int.from_bytes(seed[9:11], "big") % 11 - 5
        prompt = f"f(x)=({a})x^3+({b})x^2+({c})x+({d}); f'({x})=?"
        answer = 3*a*x*x + 2*b*x + c
        rows.append({"id": f"r201-calc-matched-dev-{index:04d}",
                     "prompt": prompt, "answer": str(answer), "split": "dev"})
    if (len({row["id"] for row in rows}) != DEV_COUNT
            or len({row["prompt"] for row in rows}) != DEV_COUNT):
        raise CalcEvalCommitmentError("private matched DEV example collision")
    return rows


def _validate_public_prompts(prompts: Sequence[str] | None) -> set[str] | None:
    if prompts is None:
        return None
    if type(prompts) not in (tuple, list) or len(prompts) > MAX_PUBLIC_PROMPTS:
        raise CalcEvalCommitmentError("invalid bounded public TRAIN prompt sequence")
    if any(type(prompt) is not str or not prompt
           or len(prompt) > MAX_PUBLIC_PROMPT_CHARS
           for prompt in prompts):
        raise CalcEvalCommitmentError("unsafe public TRAIN prompt")
    if len(set(prompts)) != len(prompts):
        raise CalcEvalCommitmentError("duplicate public TRAIN prompt")
    return set(prompts)


def commitment(nonce: bytes, *, public_train_prompts: Sequence[str] | None = None) -> dict:
    """Return aggregate hashes and denied grants; never DEV rows/nonce.

    An optional public TRAIN comparison detects exact surface-string overlap.
    It does *not* authenticate caller-supplied TRAIN, prove semantic disjointness,
    establish secret nonce entropy, or enroll an independent evaluator.
    """
    public = _validate_public_prompts(public_train_prompts)
    rows = _private_dev_rows(nonce)
    if public is not None and public.intersection(row["prompt"] for row in rows):
        raise CalcEvalCommitmentError("public TRAIN/private DEV surface overlap")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_UNAUTHENTICATED_PRIVATE_DEV_HOLD",
        "corpus_lineage": "R201_CALC_PRIVATE_POLYNOMIAL_DEV_NOT_R18_R185",
        "calc_source_repo": CALC_SOURCE_REPO,
        "calc_source_revision": CALC_SOURCE_REVISION,
        "calc_algebra_git_blob": CALC_ALGEBRA_GIT_BLOB,
        "calc_algebra_sha256_declared": CALC_ALGEBRA_SHA256,
        "calc_source_runtime_authenticated": False,
        "oracle": "EXACT_INTEGER_POLYNOMIAL_POWER_RULE_SEPARATE_FROM_CALC_RUNTIME",
        "dev_count": DEV_COUNT,
        "private_dev_sha256": hashlib.sha256(_jsonl(rows)).hexdigest(),
        "private_dev_rows_returned": False,
        "private_dev_nonce_returned": False,
        "public_train_surface_overlap_checked": public is not None,
        "semantic_disjointness_independently_verified": False,
        "private_nonce_entropy_independently_verified": False,
        "source_rights_authenticated": False,
        "protected_evaluator_custody_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
        "authorizes_eval_unseal": False,
    }


def matched_commitment(nonce: bytes, *, public_train_prompts: Sequence[str]) -> dict:
    """Commit separately to R201-format DEV; reveal no examples or secret.

    Requires a full-size caller-supplied public TRAIN list for a surface
    collision check. The list is not independently source-authenticated;
    private evaluator identity/custody and TRAIN/DEV semantics remain HOLD.
    """
    public = _validate_public_prompts(public_train_prompts)
    if public is None or len(public) != R201_TRAIN_COUNT:
        raise CalcEvalCommitmentError("exact full-size R201 public TRAIN list required")
    rows = _private_matched_dev_rows(nonce)
    if public.intersection(row["prompt"] for row in rows):
        raise CalcEvalCommitmentError("public TRAIN/private DEV surface overlap")
    return {
        "schema": MATCHED_SCHEMA,
        "status": "SOURCE_ONLY_UNAUTHENTICATED_PRIVATE_DEV_HOLD",
        "corpus_lineage": "R201_CALC_MATCHED_PRIVATE_DEV_NOT_R18_R185",
        "calc_source_repo": CALC_SOURCE_REPO,
        "calc_source_revision": CALC_SOURCE_REVISION,
        "oracle": "EXACT_INTEGER_CUBIC_POWER_RULE_SEPARATE_FROM_CALC_RUNTIME",
        "distribution": "R201_CUBIC_SMALL_X_MATCHED",
        "dev_count": DEV_COUNT,
        "private_dev_sha256": hashlib.sha256(_jsonl(rows)).hexdigest(),
        "private_dev_rows_returned": False,
        "private_dev_nonce_returned": False,
        "public_train_surface_overlap_checked": True,
        "public_train_source_independently_authenticated": False,
        "semantic_disjointness_independently_verified": False,
        "private_nonce_entropy_independently_verified": False,
        "source_rights_authenticated": False,
        "protected_evaluator_custody_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
        "authorizes_eval_unseal": False,
    }
