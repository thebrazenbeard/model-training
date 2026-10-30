"""Independent, deterministic R18 six-operation arithmetic evaluation source.

This is a NEW external OOD composition benchmark, not a fresh subset of the
exhausted R16 v2 evaluation structures. It uses the same arithmetic primitives
but longer six-step compositions. It must only be opened in the separate R18
evaluation custodian, never in neural training or max-fit probes.

No model, tokenizer, GPU, network, RNG, or private data is loaded here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import operator
import re
from functools import reduce
from pathlib import Path


SOURCE_SCHEMA = "VERA_TRAININGDAY_R18_NOVEL_CHAIN6_EVAL_SOURCE_V1"
ALGORITHM = "R18_SHA256_PRIORITY_SIX_STEP_PRIVATE_NONCE_DUAL_ORACLE_V2"
OPS = ("add", "subtract", "multiply")
SEED = 18041
COUNT = 64
CHAIN_LENGTH = 6
START_VALUES = tuple(range(-20, 0)) + tuple(range(1, 21))
ADD_SUB_VALUES = tuple(range(-9, 0)) + tuple(range(1, 10))
MULTIPLY_VALUES = (-3, -2, 2, 3)
MAX_ABS_INTERMEDIATE = 100000
MAX_PROMPT_CHARS = 220
MAX_ANSWER_CHARS = 8


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":"))
                           .encode("utf-8")).hexdigest()


def _value(seed: int, private_nonce_hex: str, sig_text: str,
           position: int, values: tuple[int, ...]) -> int:
    material = f"R18_CH6_VAL_V2\x00{seed}\x00{private_nonce_hex}\x00{sig_text}\x00{position}"
    raw = int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest(), "big")
    return values[raw % len(values)]


def _calculate(start: int, steps: list[tuple[str, int]]) -> tuple[int, list[int]]:
    state = start
    states = [start]
    for name, operand in steps:
        if name == "add":
            state += operand
        elif name == "subtract":
            state -= operand
        elif name == "multiply":
            state *= operand
        else:
            raise ValueError("unknown operation")
        states.append(state)
    # Independently interpret the same steps through operator functions.
    actions = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul}
    independent = reduce(lambda value, step: actions[step[0]](value, step[1]),
                         steps, start)
    if state != independent:
        raise ValueError("independent arithmetic oracles disagree")
    return state, states


def _replay_rendered_prompt(prompt: str) -> tuple[int, list[tuple[str, int]], int]:
    """Independent parse-and-execute oracle starting from rendered text bytes."""
    match = re.fullmatch(
        r"Start with (-?(?:0|[1-9]\d*))\. (.+) What is the final integer\?",
        prompt,
    )
    if match is None:
        raise ValueError("external evaluation prompt grammar changed")
    start = int(match.group(1))
    commands = match.group(2)
    parsed = re.findall(r"(add|subtract|multiply) (-?(?:0|[1-9]\d*))\.", commands)
    if (len(parsed) != CHAIN_LENGTH
            or " ".join(f"{op} {value}." for op, value in parsed) != commands):
        raise ValueError("rendered evaluation instruction boundary changed")
    steps = [(name, int(value)) for name, value in parsed]
    actions = {"add": operator.add, "subtract": operator.sub, "multiply": operator.mul}
    result = reduce(lambda state, step: actions[step[0]](state, step[1]),
                    steps, start)
    return start, steps, result


def generate_novel_r18_cases(*, private_nonce_hex: str, seed: int = SEED,
                             count: int = COUNT) -> list[dict]:
    """Enumerate 729 six-op signatures and accept first 64 safe distinct ones.

    The exact seed, encoding, sorting, candidate skips and prompt grammar are
    immutable. A rejection consumes only that signature; never tune this list
    on any model's answers. Returned case answers are EVALUATOR-PRIVATE.
    """
    if type(seed) is not int or seed != SEED or type(count) is not int or count != COUNT:
        raise ValueError("R18 eval seed/count changed after preregistration")
    if (not isinstance(private_nonce_hex, str) or len(private_nonce_hex) != 64
            or any(c not in "0123456789abcdef" for c in private_nonce_hex)):
        raise ValueError("R18 evaluator private 256-bit nonce absent or malformed")
    ranked = []
    for signature in itertools.product(OPS, repeat=CHAIN_LENGTH):
        sig_text = "\x1f".join(signature)
        priority = _sha(f"R18_CH6_SIG_V2\x00{seed}\x00{private_nonce_hex}\x00{sig_text}")
        ranked.append((priority, signature, sig_text))
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    cases = []
    for _, signature, sig_text in ranked:
        start = _value(seed, private_nonce_hex, sig_text, 0, START_VALUES)
        steps = [(name, _value(seed, private_nonce_hex, sig_text, pos,
                               MULTIPLY_VALUES if name == "multiply" else ADD_SUB_VALUES))
                 for pos, name in enumerate(signature, start=1)]
        answer, states = _calculate(start, steps)
        prompt = (f"Start with {start}. "
                  + " ".join(f"{name} {operand}." for name, operand in steps)
                  + " What is the final integer?")
        parsed_start, parsed_steps, independently_replayed = _replay_rendered_prompt(prompt)
        if (parsed_start != start or parsed_steps != steps
                or independently_replayed != answer):
            raise ValueError("rendered six-chain prompt disagrees with oracle")
        if (any(before == after for before, after in zip(states, states[1:]))
                or len(set(states)) != CHAIN_LENGTH + 1
                or any(abs(state) > MAX_ABS_INTERMEDIATE for state in states)
                or len(prompt) > MAX_PROMPT_CHARS
                or len(str(answer)) > MAX_ANSWER_CHARS):
            continue
        operands = ",".join(str(operand) for _, operand in steps)
        case_id = "r18-reasoning-" + _sha(
            f"R18_CH6_CASE_V2\x00{seed}\x00{private_nonce_hex}\x00{sig_text}\x00{start}\x00{operands}\x00{answer}")
        cases.append({"id": case_id, "prompt": prompt, "expected": str(answer),
                      "signature_sha256": _sha(sig_text)})
        if len(cases) == count:
            break
    if len(cases) != count or len({case["id"] for case in cases}) != count or len(
            {case["signature_sha256"] for case in cases}) != count or len(
            {case["prompt"] for case in cases}) != count:
        raise ValueError("insufficient fresh, safe six-operation signature cases")
    return cases


def generator_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_manifest(cases: list[dict], *, private_nonce_hex: str,
                    v2_manifest_sha256: str, producer_git_head: str) -> dict:
    """Answer-free public source manifest; custody digests commit private cases."""
    if (cases != generate_novel_r18_cases(private_nonce_hex=private_nonce_hex)
            or len(v2_manifest_sha256) != 64 or len(producer_git_head) != 40
            or any(c not in "0123456789abcdef" for c in
                   v2_manifest_sha256 + producer_git_head)):
        raise ValueError("unverified external evaluation source provenance")
    return {
        "schema": SOURCE_SCHEMA, "algorithm": ALGORITHM,
        "protocol": "R18_EXTERNAL_LONG_CHAIN64_OOD_V1",
        "evaluation_distribution": "EXTERNAL_OOD_SIX_OPERATOR_COMPOSITION_NOT_V2_HOLDOUT",
        "generator_code_sha256": generator_code_sha256(),
        "private_nonce_sha256": hashlib.sha256(bytes.fromhex(private_nonce_hex)).hexdigest(),
        "producer_git_head": producer_git_head,
        "seed": SEED, "count": COUNT, "chain_length": CHAIN_LENGTH,
        "operator_vocabulary": list(OPS),
        "start_values": [min(START_VALUES), max(START_VALUES)],
        "add_sub_values": [min(ADD_SUB_VALUES), max(ADD_SUB_VALUES)],
        "multiply_values": list(MULTIPLY_VALUES),
        "max_abs_intermediate": MAX_ABS_INTERMEDIATE,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "max_answer_chars": MAX_ANSWER_CHARS,
        "older_v2_corpus_manifest_sha256": v2_manifest_sha256,
        "older_v2_chain_length_range": [2, 5],
        "previously_scored_v2_signature_count": 77,
        "older_v2_unseen_signatures_after_r16": 0,
        "selected_unique_signature_count": COUNT,
        "selected_id_sha256": hashlib.sha256(
            "\n".join(case["id"] for case in cases).encode()).hexdigest(),
        "selected_signature_sha256": hashlib.sha256(
            "\n".join(case["signature_sha256"] for case in cases).encode()).hexdigest(),
        "private_case_content_sha256": _canonical_sha(cases),
    }
