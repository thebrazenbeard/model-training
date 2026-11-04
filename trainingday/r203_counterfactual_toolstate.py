"""Original R203 counterfactual tool-state TRAIN candidate; HARD HOLD.

Each of 64 independent initial cases has a three-way, minimally perturbed
state-update contrast: add, sort or no_tool. Public examples are not protected
DEV, a license, an evaluator, a tool invocation, or evidence of model learning.
"""

from __future__ import annotations

import hashlib
import json


SCHEMA = "VERA_TRAININGDAY_R203_COUNTERFACTUAL_TOOLSTATE_V1"
CORPUS_ID = "R203_NEW_ORIGINAL_COUNTERFACTUAL_TOOLSTATE_TRAIN"
TRIPLE_COUNT = 64
TRAIN_COUNT = TRIPLE_COUNT * 3
SURFACE_TEMPLATES = (
    ("Apply both changes to the stored state before selecting an action. "
     "Afterward, if A+B is below floor, add the updated readings; "
     "otherwise sort Q when it is not ascending; otherwise take no action."),
    ("Reconstruct the state after the events first. If updated A+B < floor, "
     "sum the readings. If not, sort a nonascending Q; if Q is already "
     "ascending, abstain."),
    ("Select by the *post-update* state: an updated reading total below floor "
     "requires a sum; failing that, a disordered queue requires sorting; "
     "otherwise do nothing."),
)


class R203ToolstateError(ValueError):
    """Reject impossible generator inputs or broken local invariants."""


def _case_seed(index: int) -> bytes:
    if type(index) is not int or not 0 <= index < TRIPLE_COUNT:
        raise R203ToolstateError("case index outside frozen range")
    return hashlib.sha256(f"{SCHEMA}:TRAIN:{index}".encode("ascii")).digest()


def _rotate_left(values: tuple[int, int, int], positions: int) -> list[int]:
    slot = positions % len(values)
    return list(values[slot:] + values[:slot])


def contrastive_triple(index: int) -> tuple[dict[str, str], ...]:
    """Return 3 public TRAIN contrasts in sum/sort/no-tool order.

    SUM versus SORT changes only A_delta; SORT versus NO_TOOL changes only
    Q_rotate_left. The initial case, tool catalog, and all wording are fixed
    within the triple. They are *public*, not secret evaluation samples.
    """
    seed = _case_seed(index)
    a, b = seed[0] % 61 - 30, seed[1] % 61 - 30
    base = seed[2] % 35 - 17
    ascending = (base, base + 1 + seed[3] % 9,
                 base + 2 + seed[3] % 9 + seed[4] % 9)
    initial_rotation = seed[5] % 3
    initial_queue = tuple(_rotate_left(ascending, initial_rotation))
    restoring_shift = (-initial_rotation) % 3
    disorder_shift = (restoring_shift + 1 + seed[6] % 2) % 3
    floor = seed[7] % 65 - 32
    boundary_delta = floor - (a + b)
    low_delta = boundary_delta - (1 + seed[8] % 9)
    high_delta = boundary_delta + seed[9] % 9
    if not (a + low_delta + b < floor <= a + high_delta + b):
        raise R203ToolstateError("counterfactual threshold not separated")
    variants = ((low_delta, disorder_shift),
                (high_delta, disorder_shift),
                (high_delta, restoring_shift))
    rows = []
    for delta, rotation in variants:
        updated_a = a + delta
        updated_queue = _rotate_left(initial_queue, rotation)
        if updated_a + b < floor:
            tool, args = "sum_ints", {"a": updated_a, "b": b}
            result = updated_a + b
        elif updated_queue != sorted(updated_queue):
            tool, args = "sort_ints", {"values": updated_queue}
            result = sorted(updated_queue)
        else:
            tool, args, result = "no_tool", {}, None
        prompt = (
            "Tools: sum_ints(a,b), sort_ints(values), no_tool. "
            + SURFACE_TEMPLATES[seed[10] % len(SURFACE_TEMPLATES)] + " "
            + f"Initial: A={a}; B={b}; Q={json.dumps(initial_queue, separators=(',', ':'))}; "
            + f"floor={floor}. Events: A_delta={delta}; Q_rotate_left={rotation}. "
            + "Reply compact JSON with tool, args and result only."
        )
        answer = json.dumps({"tool": tool, "args": args, "result": result},
                            sort_keys=True, separators=(",", ":"))
        public_id = hashlib.sha256((SCHEMA + "\x00" + prompt).encode("utf-8")).hexdigest()[:20]
        rows.append({"id": "r203-toolstate-" + public_id,
                     "prompt": prompt, "answer": answer, "split": "train"})
    if [json.loads(row["answer"])["tool"] for row in rows] != [
        "sum_ints", "sort_ints", "no_tool"
    ] or any(len(row["prompt"]) > 1024 for row in rows):
        raise R203ToolstateError("triple semantics or prompt cap violated")
    return tuple(rows)


def train_rows() -> list[dict[str, str]]:
    # Shuffle the public *row order* by a domain-separated content hash rather
    # than emitting a SUM/SORT/NO_TOOL class cycle. This does not hide public
    # labels, but avoids trivial ordinal-modulo leakage into training metadata.
    rows = sorted(
        (row for index in range(TRIPLE_COUNT) for row in contrastive_triple(index)),
        key=lambda row: hashlib.sha256((SCHEMA + ":ORDER:" + row["id"])
                                       .encode("utf-8")).digest(),
    )
    if (len(rows) != TRAIN_COUNT or len({r["id"] for r in rows}) != TRAIN_COUNT
            or len({r["prompt"] for r in rows}) != TRAIN_COUNT):
        raise R203ToolstateError("TRAIN row collision")
    return rows


def train_bytes() -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8") + b"\n"
                    for row in train_rows())


def source_commitment() -> dict[str, object]:
    """Public integrity proof only; no autonomous rights or training grant."""
    return {
        "schema": SCHEMA, "corpus_id": CORPUS_ID,
        "status": "SOURCE_ONLY_UNAUTHENTICATED_HOLD",
        "train_count": TRAIN_COUNT, "triples": TRIPLE_COUNT,
        "train_sha256": hashlib.sha256(train_bytes()).hexdigest(),
        "provenance": "NEW_ORIGINAL_R203_GENERATOR",
        "private_dev_included": False,
        "data_rights_granted": False,
        "independent_evaluator_qualified": False,
        "model_tokenizer_fit_verified": False,
        "gpu_lease_granted": False,
        "authorizes_training": False,
    }


if __name__ == "__main__":
    print(json.dumps(source_commitment(), sort_keys=True))
