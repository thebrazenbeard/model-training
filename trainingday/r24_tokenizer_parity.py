"""CPU-only witness for R18 MeZO/QLoRA token-sequence parity.

Use a caller-supplied tokenizer and admitted/public examples only.  This
compares the actual preparation functions; it neither authenticates corpus
membership nor grants permission to train, evaluate, or disclose examples.
No token IDs, prompt strings, or responses are included in the report.
"""

from __future__ import annotations

from trainingday.r18_mezo import tokenize_cases
from trainingday.r18_qlora import _tensorize


SCHEMA = "VERA_TRAININGDAY_R24_TOKENIZER_PARITY_WITNESS_V1"


def inspect_tokenizer_parity(tokenizer, cases: list[dict],
                             max_seq_length: int) -> dict:
    """Require exact per-example input IDs and causal labels for both arms.

    A `PASS_SOURCE_PARITY` result is meaningful only for the supplied cases
    and tokenizer instance. It is always a non-authorizing, source-level
    observation: no independent base/tokenizer hash, hardware state, sampling
    schedule or corpus custody is established here.
    """
    result = {
        "schema": SCHEMA,
        "status": "HOLD",
        "authorizes_training": False,
        "physical_tokenizer_attested": False,
        "corpus_membership_attested": False,
        "sampling_schedule_attested": False,
        "example_count": 0,
        "supervised_target_tokens": 0,
        "mismatched_examples": 0,
        "reason": "INVALID_INPUT",
    }
    if (type(max_seq_length) is not int or max_seq_length < 2
            or not isinstance(cases, list) or not cases):
        return result
    seen_ids = set()
    for row in cases:
        if (not isinstance(row, dict)
                or not all(k in row for k in ("id", "prompt", "response"))
                or type(row["id"]) is not str or not row["id"]
                or row["id"] in seen_ids
                or type(row["prompt"]) is not str
                or type(row["response"]) is not str
                or not row["response"]):
            return result
        seen_ids.add(row["id"])

    result["example_count"] = len(cases)
    # Freeze the caller's selected case values before either preparer runs.
    # A stateful tokenizer must not mutate the scope being certified.
    selected = [(row["id"], row["prompt"], row["response"]) for row in cases]

    def fresh_rows():
        return [{"id": case_id, "prompt": prompt, "response": response}
                for case_id, prompt, response in selected]

    try:
        mezo, _ = tokenize_cases(tokenizer, fresh_rows(), max_seq_length)
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, OverflowError, UnicodeError):
        result["reason"] = "MEZO_PREPARATION_REJECTED"
        return result
    try:
        qlora = _tensorize(tokenizer, fresh_rows())
    except (RuntimeError, ValueError, TypeError, KeyError, IndexError,
            AttributeError, OverflowError, UnicodeError):
        result["reason"] = "QLORA_PREPARATION_REJECTED"
        return result

    if (not isinstance(mezo, list) or not isinstance(qlora, list)
            or len(mezo) != len(qlora) or len(mezo) != result["example_count"]):
        result["reason"] = "PREPARER_EXAMPLE_COUNT_MISMATCH"
        return result

    if any(not isinstance(row, dict)
           or not isinstance(row.get("input_ids"), list)
           or not isinstance(row.get("labels"), list)
           or len(row["input_ids"]) != len(row["labels"])
           or type(row.get("target_tokens")) is not int
           or type(row.get("id")) is not str
           for row in mezo + qlora):
        result["reason"] = "MALFORMED_PREPARER_OUTPUT"
        return result

    # The runner's _tensorize deliberately leaves max-length fitting to the
    # caller. A parity witness must refuse that unverified fit condition.
    if any(len(row["input_ids"]) > max_seq_length for row in qlora):
        result["reason"] = "QLORA_NO_TRUNCATION_LIMIT_EXCEEDED"
        return result

    mismatches = 0
    total_targets = 0
    for left, right in zip(mezo, qlora, strict=True):
        left_labels = left["labels"]
        right_labels = right["labels"]
        left_targets = sum(label != -100 for label in left_labels)
        right_targets = sum(label != -100 for label in right_labels)
        total_targets += left_targets
        if (left["id"] != right["id"]
                or left["input_ids"] != right["input_ids"]
                or left_labels != right_labels
                or left_targets != right_targets
                or left["target_tokens"] != right["target_tokens"]):
            mismatches += 1

    result["supervised_target_tokens"] = total_targets
    result["mismatched_examples"] = mismatches
    result["reason"] = ("TOKEN_SEQUENCE_OR_MASK_MISMATCH" if mismatches
                        else "MATCHED_SUPPLIED_TOKEN_IDS_AND_LABELS")
    if mismatches == 0:
        result["status"] = "PASS_SOURCE_PARITY"
    return result
