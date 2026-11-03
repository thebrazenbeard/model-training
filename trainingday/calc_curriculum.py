"""R201 Calc-backed polynomial curriculum candidate, with no TRAIN grant.

The external Calc source is an optional *oracle dependency*, not a training
authority or a dataset license. The source checksum pins one reviewed module;
it does not attest the install owner, package environment, or runtime host.
This module generates public TRAIN rows only. Protected evaluation is separate.
"""

from __future__ import annotations

import hashlib
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
import json
import math
from pathlib import Path
import sys
from types import ModuleType


SCHEMA = "VERA_TRAININGDAY_R201_CALC_POLYNOMIAL_V1"
CALC_REPOSITORY = "thebrazenbeard/calc"
CALC_REVISION = "17e12d0f659b42732fb98e5730732aba9ff1b488"
CALC_SYMBOLIC_SHA256 = "aba705f0d4370afa35538280f3db00ecbe36da1b3f3ca083ee29ebc292113b0b"
CALC_INIT_SHA256 = "4bb35910f0f5897476911d7846498915e8a7150f7993d79a12360570c5199cf4"
ORACLE_FAMILY = "CALC_SYMBOLIC_BOUNDED_INTEGER_POLYNOMIAL_DERIVATIVE_AT_POINT"
TRAIN_COUNT = 1024


class CalcCurriculumError(ValueError):
    """Invalid source binding or mathematical oracle result."""


def _preflight_calc_files() -> tuple[Path, Path]:
    """Hash the package initializer and oracle *before* normal Python import.

    PathFinder avoids user-installed meta-path finders for the initial lookup
    of ordinary on-disk Python modules.  This is a static provenance check, not
    a sandbox or protection against concurrent filesystem replacement.
    """
    package = PathFinder.find_spec("calc")
    if (package is None or not isinstance(package.loader, SourceFileLoader)
            or type(package.origin) is not str
            or not package.submodule_search_locations):
        raise CalcCurriculumError(
            "Calc symbolic source unavailable; install the pinned local Calc checkout"
        )
    symbolic = PathFinder.find_spec("calc.symbolic", package.submodule_search_locations)
    if (symbolic is None or not isinstance(symbolic.loader, SourceFileLoader)
            or type(symbolic.origin) is not str):
        raise CalcCurriculumError(
            "Calc symbolic source unavailable; install the pinned local Calc checkout"
        )
    try:
        init_path = Path(package.origin).resolve(strict=True)
        symbolic_path = Path(symbolic.origin).resolve(strict=True)
        if (init_path.name != "__init__.py" or symbolic_path.name != "symbolic.py"
                or init_path.parent != symbolic_path.parent):
            raise CalcCurriculumError("Calc package/source identity mismatch")
        if hashlib.sha256(init_path.read_bytes()).hexdigest() != CALC_INIT_SHA256:
            raise CalcCurriculumError("Calc package initializer SHA256 differs from pinned source")
        if hashlib.sha256(symbolic_path.read_bytes()).hexdigest() != CALC_SYMBOLIC_SHA256:
            raise CalcCurriculumError("Calc symbolic module SHA256 differs from pinned source")
    except OSError as exc:
        raise CalcCurriculumError("Calc source unavailable during pre-import verification") from exc
    return init_path, symbolic_path


def _canonical_json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _load_verified_symbolic():
    """Execute only SHA-pinned Calc source bytes, without its Python .pyc cache.

    This deliberately does not import the external ``calc`` package: normal
    Python import can execute timestamp-valid but forged .pyc files despite
    matching .py hashes. A verified source module is loaded under an isolated
    private name so dataclasses can find their defining module in sys.modules.
    The interpreter and imported standard-library modules remain trusted;
    this is provenance checking, not sandboxing.
    """
    init_path, path = _preflight_calc_files()
    if init_path.name != "__init__.py" or path.name != "symbolic.py":
        raise CalcCurriculumError("Calc source filename changed after preflight")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CalcCurriculumError("Calc symbolic source disappeared") from exc
    if hashlib.sha256(raw).hexdigest() != CALC_SYMBOLIC_SHA256:
        raise CalcCurriculumError("Calc symbolic source changed before execution")

    name = "trainingday._verified_calc_symbolic"
    symbolic = ModuleType(name)
    symbolic.__file__ = str(path)
    symbolic.__package__ = "trainingday"
    symbolic.__loader__ = SourceFileLoader(name, str(path))
    symbolic.__spec__ = ModuleSpec(name, symbolic.__loader__, origin=str(path))
    previous = sys.modules.get(name)
    sys.modules[name] = symbolic
    try:
        # Compile the exact checked bytes, rather than loading a .pyc that may
        # have a valid timestamp and size but entirely different instructions.
        exec(compile(raw, str(path), "exec", dont_inherit=True), symbolic.__dict__)
    except Exception as exc:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise CalcCurriculumError("verified Calc symbolic execution failed") from exc

    if (hashlib.sha256(path.read_bytes()).hexdigest() != CALC_SYMBOLIC_SHA256
            or hashlib.sha256(init_path.read_bytes()).hexdigest() != CALC_INIT_SHA256):
        raise CalcCurriculumError("Calc source changed during execution")
    for required in ("Const", "Var"):
        if not hasattr(symbolic, required):
            raise CalcCurriculumError(f"Calc symbolic oracle missing {required}")
    return symbolic


def _parameters(index: int) -> tuple[int, int, int, int, int]:
    if type(index) is not int or not 0 <= index < TRAIN_COUNT:
        raise CalcCurriculumError("TRAIN index outside frozen range")
    seed = hashlib.sha256(f"{SCHEMA}:TRAIN:{index}".encode("ascii")).digest()
    a = 1 + int.from_bytes(seed[0:2], "big") % 17
    if seed[2] & 1:
        a = -a
    b = int.from_bytes(seed[3:5], "big") % 61 - 30
    c = int.from_bytes(seed[5:7], "big") % 101 - 50
    d = int.from_bytes(seed[7:9], "big") % 101 - 50
    x0 = int.from_bytes(seed[9:11], "big") % 11 - 5
    return (a, b, c, d, x0)


def _row(index: int, symbolic) -> dict:
    a, b, c, d, x0 = _parameters(index)
    x = symbolic.Var("x")
    expr = (symbolic.Const(a) * (x ** 3) + symbolic.Const(b) * (x ** 2)
            + symbolic.Const(c) * x + symbolic.Const(d))
    observed = expr.diff("x").eval({"x": x0})
    # Independent exact power-rule calculation is the cross-oracle check.
    expected = 3 * a * x0 * x0 + 2 * b * x0 + c
    if (type(observed) is not float or not math.isfinite(observed)
            or abs(expected) >= 2**53 or observed != float(expected)):
        raise CalcCurriculumError("Calc derivative disagrees with exact integer power rule")
    # Compact bounded prompt for the proposed seq64 low-memory token budget;
    # actual Qwen-tokenizer fit must still be measured by the training owner.
    prompt = f"f(x)=({a})x^3+({b})x^2+({c})x+({d}); f'({x0})=?"
    return {"id": f"r201-calc-train-{index:04d}", "prompt": prompt,
            "answer": str(expected), "split": "train"}


def train_rows() -> list[dict]:
    """Deterministic public TRAIN data; real pinned Calc oracle is mandatory."""
    symbolic = _load_verified_symbolic()
    rows = [_row(index, symbolic) for index in range(TRAIN_COUNT)]
    if len({row["prompt"] for row in rows}) != TRAIN_COUNT:
        raise CalcCurriculumError("duplicate polynomial TRAIN prompt")
    return rows


def train_bytes() -> bytes:
    return _jsonl(train_rows())


def source_commitment() -> dict:
    """Evidence of a reproducible source candidate, never an authorization."""
    data = train_bytes()
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_UNAUTHENTICATED_HOLD",
        "calc_repository": CALC_REPOSITORY,
        "calc_revision": CALC_REVISION,
        "calc_symbolic_sha256": CALC_SYMBOLIC_SHA256,
        "oracle_family": ORACLE_FAMILY,
        "corpus_id": "R201_NEW_CALC_POLYNOMIAL_TRAIN",
        "r185_corpus_replacement": False,
        "train_count": TRAIN_COUNT,
        "train_sha256": hashlib.sha256(data).hexdigest(),
        "public_train_rows_returned": False,
        "private_dev_included": False,
        "private_dev_commitment": None,
        "rights_holder_authenticated": False,
        "data_rights_granted": False,
        "protected_evaluator_custody_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
    }


if __name__ == "__main__":
    print(json.dumps(source_commitment(), sort_keys=True))
