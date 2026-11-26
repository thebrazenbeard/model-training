from __future__ import annotations

import json
from typing import Any


def check_instruction_contract(text: str, contract: dict) -> bool:
    if not isinstance(text, str) or not isinstance(contract, dict):
        return False
    kind = contract.get("type")
    if kind == "token_exact_count":
        token = contract["token"]
        return text.count(token) == int(contract["count"])
    if kind == "prefix_suffix":
        return text.startswith(contract["prefix"]) and text.endswith(contract["suffix"])
    if kind == "line_prefixes":
        lines = text.splitlines()
        prefixes = list(contract["prefixes"])
        return (
            len(lines) == len(prefixes)
            and all(line for line in lines)
            and all(line.startswith(prefix) for line, prefix in zip(lines, prefixes))
        )
    if kind == "json_object_exact_keys":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(value, dict) or set(value) != set(contract["keys"]):
            return False
        return all(
            item is None or isinstance(item, (str, int, float, bool))
            for item in value.values()
        )
    if kind == "forbidden_character_and_required_token":
        return (
            contract["forbidden"] not in text
            and contract["required_token"] in text
        )
    raise ValueError(f"unknown instruction contract type: {kind}")


def reference_instruction_output(contract: dict) -> str:
    kind = contract["type"]
    if kind == "token_exact_count":
        token = contract["token"]
        count = int(contract["count"])
        return " ".join([token] * count + ["measurement"])
    if kind == "prefix_suffix":
        return contract["prefix"] + "verified" + contract["suffix"]
    if kind == "line_prefixes":
        return "\n".join(prefix + "value" for prefix in contract["prefixes"])
    if kind == "json_object_exact_keys":
        return json.dumps(
            {key: index for index, key in enumerate(contract["keys"])},
            separators=(",", ":"),
            sort_keys=True,
        )
    if kind == "forbidden_character_and_required_token":
        token = contract["required_token"]
        forbidden = contract["forbidden"]
        text = f"verified {token} measurement"
        if forbidden in text:
            text = token
        return text
    raise ValueError(f"unknown instruction contract type: {kind}")


def invalid_instruction_output(contract: dict) -> str:
    kind = contract["type"]
    if kind == "token_exact_count":
        return "measurement without required token"
    if kind == "prefix_suffix":
        return "wrong"
    if kind == "line_prefixes":
        return "wrong"
    if kind == "json_object_exact_keys":
        return "{}"
    if kind == "forbidden_character_and_required_token":
        return contract["required_token"] + contract["forbidden"]
    raise ValueError(f"unknown instruction contract type: {kind}")


def reference_coding_source(family: str, function_name: str) -> str:
    if family == "rotate_left":
        body = "return [] if not values else list(values[k % len(values):]) + list(values[:k % len(values)])"
        args = "values, k"
    elif family == "count_vowels":
        body = "return sum(ch.lower() in 'aeiou' for ch in text)"
        args = "text"
    elif family == "unique_preserve":
        body = "return list(dict.fromkeys(values))"
        args = "values"
    elif family == "digit_sum":
        body = "return sum(int(ch) for ch in str(value))"
        args = "value"
    elif family == "reverse_words":
        body = "return ' '.join(reversed(text.split()))"
        args = "text"
    elif family == "clamp_values":
        body = "return [min(high, max(low, item)) for item in values]"
        args = "values, low, high"
    elif family == "pairwise_sums":
        body = "return [a + b for a, b in zip(values, values[1:])]"
        args = "values"
    elif family == "flatten_once":
        body = "return [item for group in groups for item in group]"
        args = "groups"
    elif family == "chunk_list":
        body = "return [values[i:i+n] for i in range(0, len(values), n)]"
        args = "values, n"
    elif family == "frequency_map":
        body = "from collections import Counter\n    return dict(Counter(values))"
        args = "values"
    else:
        raise ValueError(f"unknown coding family: {family}")
    return f"def {function_name}({args}):\n    {body}\n"


def run_python_test_contract(source: str, tests: list[str]) -> None:
    namespace: dict[str, Any] = {}
    exec(source, namespace, namespace)
    for test in tests:
        exec(test, namespace, namespace)
