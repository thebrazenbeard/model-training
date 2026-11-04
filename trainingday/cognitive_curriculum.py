"""Deterministic, original R202 multi-domain TRAIN candidate (hard HOLD).

This module is a source-data generator, not a license grant, evaluator,
training launcher, model, or evidence of learning. It has no external
dependencies, model access, private inputs, or network operations.
"""

from __future__ import annotations

import hashlib
import json


SCHEMA = "VERA_TRAININGDAY_R202_COGNITIVE_CURRICULUM_V1"
CORPUS_ID = "R202_NEW_ORIGINAL_COGNITIVE_TRAIN"
FAMILIES = (
    "math_logic",
    "language_tracking",
    "tool_selection",
    "dependency_planning",
    "error_correction",
    "evidence_uncertainty",
)
ROWS_PER_FAMILY = 256
TRAIN_COUNT = ROWS_PER_FAMILY * len(FAMILIES)
_NAMES = ("Ada", "Bryn", "Cato", "Dara", "Eli", "Faye", "Gus", "Hana")
_CITIES = ("Aster", "Bellwood", "Cedarbay", "Dunmere",
           "Eastvale", "Fairwater", "Glenport", "Highfield")
_OBJECTS = ("amber token", "blue key", "copper ring", "silver coin",
            "bronze bell", "green map", "ivory die", "red ticket",
            "glass bead", "striped scarf", "black folder", "gold seal")
_LIMIT = 512


class CognitiveCurriculumError(ValueError):
    """Reject an index, contract, or deterministic generation error."""


def _seed(family: str, index: int) -> bytes:
    if family not in FAMILIES or type(index) is not int or not 0 <= index < ROWS_PER_FAMILY:
        raise CognitiveCurriculumError("unknown family or index outside frozen TRAIN range")
    return hashlib.sha256(f"{SCHEMA}:TRAIN:{family}:{index}".encode("ascii")).digest()


def _rotate(values: tuple[str, ...], seed: bytes, count: int) -> tuple[str, ...]:
    """Select distinct identities using deterministic per-name sort keys."""
    return tuple(sorted(values, key=lambda value:
                        hashlib.sha256(seed + value.encode("ascii")).digest())[:count])


def _math_logic(seed: bytes) -> tuple[str, str]:
    start = seed[2] % 35 - 17
    add = 1 + seed[3] % 23
    multiplier = 2 + seed[4] % 8
    subtract = 1 + seed[5] % 37
    divisor = 2 + seed[6] % 8
    variant = seed[10] % 3
    if variant == 0:
        result = (start + add) * multiplier - subtract
        prompt = (f"Start at {start}; add {add}; multiply the result by {multiplier}; "
                  f"subtract {subtract}. Is the final integer divisible by {divisor}? "
                  "Answer YES or NO only.")
        holds = result % divisor == 0
    elif variant == 1:
        threshold = seed[11] % 91 - 45
        result = 2 * (start + add) - subtract
        prompt = (f"Let x={start}. Replace x by 2*(x+{add})-{subtract}. "
                  f"Is x strictly greater than {threshold}? Answer YES or NO only.")
        holds = result > threshold
    else:
        result = start + add - subtract
        prompt = (f"Begin with {start}; increase it by {add}, then decrease it "
                  f"by {subtract}. Is the answer even? Answer YES or NO only.")
        holds = result % 2 == 0
    return prompt, "YES" if holds else "NO"


def _language_tracking(seed: bytes) -> tuple[str, str]:
    a, b, c = _rotate(_NAMES, seed, 3)
    object_a = _OBJECTS[seed[2] % len(_OBJECTS)]
    object_b = _OBJECTS[(seed[2] + 1 + seed[3] % (len(_OBJECTS) - 1)) % len(_OBJECTS)]
    # Two independent objects, with one or two transfers of the queried item.
    if seed[4] & 1:
        prompt = (f"{a} holds the {object_a}. {b} holds the {object_b}. "
                  f"{a} gives the {object_a} to {c}. "
                  f"{b} gives the {object_b} to {a}. "
                  f"Who now holds the {object_a}? Answer with one name.")
        answer = c
    else:
        prompt = (f"{a} holds the {object_a}. {b} holds the {object_b}. "
                  f"{a} gives the {object_a} to {c}. "
                  f"{c} gives the {object_a} to {b}. "
                  f"Who now holds the {object_a}? Answer with one name.")
        answer = b
    return prompt, answer


def _tool_selection(seed: bytes) -> tuple[str, str]:
    a = seed[2] % 90 - 40
    b = seed[3] % 90 - 40
    c = seed[4] % 90 - 40
    mode = seed[5] % 3
    prefix = ("Available tools: sum_ints(a,b), sort_ints(values), or no_tool. "
              "Return only compact JSON with keys tool and args. ")
    if mode == 0:
        claim = a + b + seed[9] % 9 - 4
        prompt = (prefix + f"Select a tool to add integers {a} and {b} "
                  f"to check whether the claimed sum {claim} is correct.")
        answer = {"tool": "sum_ints", "args": {"a": a, "b": b}}
    elif mode == 1:
        prompt = prefix + f"Select a tool to sort the integers [{a},{b},{c}] ascending."
        answer = {"tool": "sort_ints", "args": {"values": [a, b, c]}}
    else:
        day = 1 + int.from_bytes(seed[7:9], "big") % 366
        hour = seed[9] % 24
        prompt = prefix + (f"Select a tool to retrieve weather for hour {hour} on day {day} of the year "
                           f"in city {_CITIES[seed[6] % len(_CITIES)]}. No weather tool is available.")
        answer = {"tool": "no_tool", "args": {}}
    return prompt, json.dumps(answer, sort_keys=True, separators=(",", ":"))


def _dependency_planning(seed: bytes) -> tuple[str, str]:
    # Acyclic by construction: each task may depend only on earlier tasks.
    names = _rotate(tuple("VWXYZ"), seed, 5)
    prereqs: dict[str, tuple[str, ...]] = {}
    durations = {name: 1 + seed[10 + n] % 9 for n, name in enumerate(names)}
    for n, name in enumerate(names):
        deps = tuple(names[k] for k in range(n) if seed[4 + n] & (1 << k))
        prereqs[name] = deps
    policy = "shortest" if seed[18] & 1 else "longest"
    spec = "; ".join(f"{name}:[{','.join(prereqs[name])}]/{durations[name]}"
                     for name in sorted(names))
    ready: list[str] = []
    pending = set(names)
    while pending:
        available = sorted((name for name in pending if all(d in ready for d in prereqs[name])),
                           key=lambda name: ((durations[name] if policy == "shortest"
                                              else -durations[name]), name))
        if not available:
            raise CognitiveCurriculumError("cyclic planning input")
        chosen = available[0]
        ready.append(chosen)
        pending.remove(chosen)
    prompt = ("Schedule tasks V,W,X,Y,Z after dependencies; pick the currently "
              f"available task with {policy} duration (alphabetical ties). "
              f"Each task:[prerequisites]/duration: {spec}. Reply comma-separated order.")
    return prompt, ",".join(ready)


def _error_correction(seed: bytes) -> tuple[str, str]:
    start = seed[2] % 29 - 14
    add = 2 + seed[3] % 13
    mult = 2 + seed[4] % 7
    sub = 1 + seed[5] % 19
    variant = seed[10] % 2
    if variant == 0:
        operations = f"+{add}, *{mult}, -{sub}"
        correct = (start + add, (start + add) * mult,
                   (start + add) * mult - sub)
    else:
        operations = f"*{mult}, -{sub}, +{add}"
        correct = (start * mult, start * mult - sub,
                   start * mult - sub + add)
    error_step = 1 + seed[6] % 3
    wrong = correct[error_step - 1] + (1 + seed[7] % 7)
    shown = list(correct)
    shown[error_step - 1] = wrong
    prompt = (f"Starting value {start}; independent arithmetic checks are "
              f"{operations}. A log lists step1={shown[0]}, step2={shown[1]}, "
              f"step3={shown[2]} (each step computed from the intended correct "
              "prior value). Identify the first wrong entry and its correct value. "
              "Reply 'STEP N: VALUE'.")
    return prompt, f"STEP {error_step}: {correct[error_step - 1]}"


def _evidence_uncertainty(seed: bytes) -> tuple[str, str]:
    level_mode = seed[2] % 3  # 0 absent, 1 below threshold, 2 meets threshold
    training_mode = seed[3] % 3  # 0 absent, 1 incomplete, 2 complete
    orders_mode = seed[4] % 3  # 0 absent, 1 beyond limit, 2 within limit
    threshold = 3 + seed[5] % 8
    max_orders = 11 + seed[6] % 12
    level = ("unknown" if level_mode == 0 else
             str(threshold - 1 - seed[7] % 3 if level_mode == 1
                 else threshold + seed[7] % 3))
    training = ("unknown", "incomplete", "complete")[training_mode]
    orders = ("unknown" if orders_mode == 0 else
              str(max_orders + 1 + seed[8] % 6 if orders_mode == 1
                  else max_orders - seed[8] % 6))
    module = _NAMES[seed[9] % len(_NAMES)]
    if level_mode == 1 or training_mode == 1 or orders_mode == 1:
        answer = "REJECT"
    elif level_mode == 2 and training_mode == 2 and orders_mode == 2:
        answer = "ACCEPT"
    else:
        answer = "INSUFFICIENT"
    prompt = (f"Access requires level>={threshold}, completed {module} training, "
              f"and open_orders<={max_orders}. A known failure means REJECT; "
              "otherwise missing data means INSUFFICIENT. "
              f"Record: level={level}; training={training}; open_orders={orders}. "
              "Reply ACCEPT/REJECT/INSUFFICIENT.")
    return prompt, answer


_GENERATORS = {
    "math_logic": _math_logic,
    "language_tracking": _language_tracking,
    "tool_selection": _tool_selection,
    "dependency_planning": _dependency_planning,
    "error_correction": _error_correction,
    "evidence_uncertainty": _evidence_uncertainty,
}


def train_row(family: str, index: int) -> dict[str, str]:
    """Return one bounded, reproducible public TRAIN row, never DEV data."""
    seed = _seed(family, index)
    # Source-label balance prevents a majority-class shortcut from scoring
    # 76% on divisibility or >69% on three-valued access. The public rejection
    # loop is deterministic, bounded, and NOT a protected evaluation sampler.
    target = ("YES" if index % 2 else "NO") if family == "math_logic" else None
    if family == "evidence_uncertainty":
        target = ("ACCEPT", "REJECT", "INSUFFICIENT")[index % 3]
    for attempt in range(8192):
        proposed = (hashlib.sha256(seed + attempt.to_bytes(4, "big")).digest()
                    if target is not None else seed)
        prompt, answer = _GENERATORS[family](proposed)
        if target is None or answer == target:
            break
    else:
        raise CognitiveCurriculumError("bounded class-balanced generation exhausted")
    if (type(prompt) is not str or type(answer) is not str or not prompt or not answer
            or len(prompt) > _LIMIT or len(answer) > _LIMIT):
        raise CognitiveCurriculumError("invalid or overlong generated row")
    return {"id": f"r202-{family}-train-{index:04d}", "prompt": prompt,
            "answer": answer, "split": "train"}


def train_rows() -> list[dict[str, str]]:
    rows = [train_row(family, i) for family in FAMILIES
            for i in range(ROWS_PER_FAMILY)]
    if (len(rows) != TRAIN_COUNT or len({r["id"] for r in rows}) != TRAIN_COUNT
            or len({r["prompt"] for r in rows}) != TRAIN_COUNT):
        raise CognitiveCurriculumError("TRAIN row collision")
    return rows


def train_bytes() -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                               sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
                    for row in train_rows())


def source_commitment() -> dict:
    """Integrity metadata: all legal and runtime qualification gates remain HOLD."""
    return {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "status": "SOURCE_ONLY_UNAUTHENTICATED_HOLD",
        "train_count": TRAIN_COUNT,
        "families": {family: ROWS_PER_FAMILY for family in FAMILIES},
        "train_sha256": hashlib.sha256(train_bytes()).hexdigest(),
        "generator_provenance": "ORIGINAL_R202_SOURCE_ONLY",
        "external_copyright_text_ingested": False,
        "private_dev_included": False,
        "rights_holder_authenticated": False,
        "data_rights_granted": False,
        "evaluation_custody_verified": False,
        "semantic_disjointness_independently_verified": False,
        "model_tokenizer_fit_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
    }


if __name__ == "__main__":
    print(json.dumps(source_commitment(), sort_keys=True))
