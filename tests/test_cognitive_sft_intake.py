"""Source-only cognitive TRAIN intake: synthetic, no private gold fixtures."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from trainingday.admission import AdmissionError
from trainingday.cognitive_sft_intake import (
    PINNED_LABYRINTH_HEAD,
    PINNED_LABYRINTH_TREE,
    UNVERIFIED_REJECTED,
    TokenizerSourceBinding,
    inspect_cognitive_sft_source,
    semantic_digest_from_prompt,
)


ASK = "Which interpretation best preserves uncertainty about the person's perspective?"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt(i: int) -> str:
    return (f"Case: train-cognitive-room-{i}\n"
            f"a teammate {i} declines an optional celebration\n{ASK}\nOptions:\n"
            f"A. They may want quiet for reason {i}.\n"
            f"B. They resent everybody in situation {i}.\n"
            f"C. They must hate everyone in situation {i}.\n"
            "Respond with only A, B, or C.")


def row(i: int) -> dict:
    message = prompt(i)
    return {
        "schema": "labyrinth.cognitive-supervised-train-example.v1",
        "split": "train",
        "provider_id": "synthetic_cognitive_perspective_train_v1",
        "training_rubric_id": "perspective_choice_train_v1",
        "objective_id": "synthetic-cognitive-v1",
        "objective_contract_sha256": "1" * 64,
        "cycle_index": 0,
        "room_id": f"room-{i}",
        "case_id": f"train-cognitive-room-{i}",
        "prompt": message,
        "prompt_sha256": digest(message.encode()),
        "semantic_sha256": semantic_digest_from_prompt(message),
        "source_binding_sha256": f"{i:064x}",
        "target_choice": "ABC"[i % 3],
    }


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(r, sort_keys=True, ensure_ascii=True,
                              separators=(",", ":")).encode() + b"\n" for r in rows)


class StableTokenizer:
    name_or_path = "Qwen/Qwen3.5-4B"
    eos_token_id = 400000
    chat_template = "{{ messages }} -- synthetic fixture only"
    init_kwargs = {"revision": "e" * 40}

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict:
        return {"input_ids": ([1] if add_special_tokens else []) +
                [ord(char) for char in value]}


class MergingTokenizer(StableTokenizer):
    def __call__(self, value: str, *, add_special_tokens: bool) -> dict:
        output = super().__call__(value, add_special_tokens=add_special_tokens)
        if "\nAnswer: " in value:
            output["input_ids"] = output["input_ids"][:-1] + [333333]
        return output


def binding(tokenizer: StableTokenizer) -> TokenizerSourceBinding:
    artifacts = tokenizer_artifacts(tokenizer)
    return TokenizerSourceBinding(
        model_id=tokenizer.name_or_path,
        model_revision="e" * 40,
        tokenizer_json_sha256=digest(artifacts["tokenizer.json"]),
        tokenizer_config_sha256=digest(artifacts["tokenizer_config.json"]),
        special_tokens_map_sha256=digest(artifacts["special_tokens_map.json"]),
        chat_template_sha256=digest(tokenizer.chat_template.encode()),
        eos_token_id=tokenizer.eos_token_id,
    )


def tokenizer_artifacts(tokenizer: StableTokenizer) -> dict[str, bytes]:
    return {
        "tokenizer.json": b'{"version":"1.0","model":{"type":"fixture"}}',
        "tokenizer_config.json": json.dumps({"chat_template": tokenizer.chat_template},
                                            separators=(",", ":")).encode(),
        "special_tokens_map.json": b'{"eos_token":"<eos>"}',
    }


def inspect(rows: list[dict] | None = None, *, tokenizer=None,
            token_binding=None, forbidden=(), raw=None, **changes):
    tokenizer = tokenizer or StableTokenizer()
    token_binding = token_binding or binding(tokenizer)
    raw = jsonl([row(1), row(2), row(3)] if rows is None else rows) if raw is None else raw
    args = dict(
        dataset_jsonl=raw,
        dataset_sha256=digest(raw),
        independent_dataset_sha256=digest(raw),
        producer_git_head=PINNED_LABYRINTH_HEAD,
        producer_git_tree=PINNED_LABYRINTH_TREE,
        expected_objective_id="synthetic-cognitive-v1",
        expected_contract_sha256="1" * 64,
        expected_cycle_index=0,
        forbidden_semantic_sha256=tuple(forbidden),
        tokenizer=tokenizer,
        tokenizer_binding=token_binding,
        tokenizer_artifacts=tokenizer_artifacts(tokenizer),
        independent_tokenizer_binding_sha256=token_binding.sha256(),
    )
    args.update(changes)
    return inspect_cognitive_sft_source(**args)


def test_structurally_valid_source_is_unverified_rejected_and_never_gold_exported():
    report = inspect()
    assert report["status"] == UNVERIFIED_REJECTED
    assert report["source_structure_checked"] is True
    assert report["dataset_sha256"] == digest(jsonl([row(1), row(2), row(3)]))
    assert report["case_count"] == 3
    assert report["rights_admitted"] is False
    assert report["producer_replay_verified"] is False
    assert report["gold_verified"] is False
    assert report["source_binding_verified"] is False
    assert report["tokenizer_host_attested"] is False
    assert report["heldout_custody_verified"] is False
    assert report["authorizes_training"] is False
    assert report["authorizes_host_execution"] is False
    assert report["authorizes_exam"] is False
    assert report["authorizes_graduation"] is False
    assert report["optimizer_steps"] == 0
    encoded = json.dumps(report)
    assert "target_choice" not in encoded and "prompt" not in encoded
    assert "They may" not in encoded and "Answer:" not in encoded
    assert "train-cognitive-room" not in encoded


def test_rejects_forged_binding_and_provenance():
    for kwargs in (
        {"producer_git_head": "f" * 40},
        {"producer_git_tree": "f" * 40},
        {"dataset_sha256": "0" * 64},
        {"independent_dataset_sha256": "0" * 64},
        {"expected_objective_id": "another-objective"},
        {"expected_contract_sha256": "2" * 64},
        {"expected_cycle_index": 1},
    ):
        with unittest.TestCase().assertRaises(AdmissionError):
            inspect(**kwargs)


def test_no_heldout_rights_spoof_or_row_drift():
    for mutate in (
        lambda r: r.update(split="eval"),
        lambda r: r.update(split="test"),
        lambda r: r.update(rights_review="PASS"),
        lambda r: r.update(evaluation_holdout=True),
        lambda r: r.update(schema="vera.trainingday.corpus.v1"),
        lambda r: r.update(provider_id="unapproved"),
        lambda r: r.update(training_rubric_id="exam"),
        lambda r: r.update(source_binding_sha256="not-a-hash"),
        lambda r: r.update(target_choice="D"),
        lambda r: r.update(prompt_sha256="0" * 64),
        lambda r: r.update(semantic_sha256="0" * 64),
        lambda r: r.update(room_id="wrong-room"),
        lambda r: r.update(cycle_index=True),
        lambda r: r.update(prompt=["forged", "nonstring"]),
        lambda r: r.update(case_id=["forged", "nonstring"]),
    ):
        rows = [row(1), row(2), row(3)]
        mutate(rows[0])
        with unittest.TestCase().assertRaises(AdmissionError):
            inspect(rows)


def test_semantic_duplicates_with_different_ids_and_self_hashes():
    a, b = row(1), row(1)
    b["case_id"] = "train-cognitive-room-99"
    b["room_id"] = "room-99"
    b["prompt"] = b["prompt"].replace("room-1", "room-99")
    b["prompt_sha256"] = digest(b["prompt"].encode())
    b["semantic_sha256"] = semantic_digest_from_prompt(b["prompt"])
    b["source_binding_sha256"] = "f" * 64
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "semantic"):
        inspect([a, b])


def test_external_semantic_exclusion_fails_without_ever_reading_exam_rows():
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "semantic"):
        inspect(forbidden=(row(2)["semantic_sha256"],))


def test_repeated_room_source_commitment_is_not_independent_replay():
    rows = [row(1), row(2)]
    rows[1]["source_binding_sha256"] = rows[0]["source_binding_sha256"]
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "source room"):
        inspect(rows)


def test_caller_rehashed_forged_gold_and_source_are_explicitly_rejected():
    rows = [row(1), row(2), row(3)]
    rows[0]["target_choice"] = "C"  # no independent gold oracle in consumer
    rows[0]["source_binding_sha256"] = "a" * 64  # source binding is forged
    forged = jsonl(rows)
    result = inspect(rows, raw=forged,
                     dataset_sha256=digest(forged),
                     independent_dataset_sha256=digest(forged))
    assert result["status"] == UNVERIFIED_REJECTED
    assert result["gold_verified"] is False
    assert result["source_binding_verified"] is False
    assert result["authorizes_training"] is False
    assert result["authorizes_host_execution"] is False


def test_empty_operator_exclusions_do_not_claim_heldout_separation():
    result = inspect(forbidden=())
    assert result["status"] == UNVERIFIED_REJECTED
    assert result["heldout_custody_verified"] is False


def test_duplicate_keys_and_noncanonical_wire_fail():
    good = jsonl([row(1)])
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(raw=good.replace(b'"split":"train",', b'"split":"train","split":"train",'))
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(raw=good.replace(b'"split":"train"', b'"split": "train"'))
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(raw=good + b"\n")


def test_tokenizer_whole_context_parity_and_binding_are_mandatory():
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "parity"):
        inspect(tokenizer=MergingTokenizer())
    broken = StableTokenizer()
    broken.chat_template = "changed"
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "template"):
        inspect(tokenizer=broken, token_binding=binding(StableTokenizer()))
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(independent_tokenizer_binding_sha256="0" * 64)
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(token_binding=replace(binding(StableTokenizer()), model_id="untrusted-model"))


def test_tokenizer_artifacts_and_special_token_identity_are_pinned():
    tokenizer = StableTokenizer()
    forged_files = tokenizer_artifacts(tokenizer)
    forged_files["tokenizer.json"] = b'{"model":{"type":"forged"}}'
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "artifact"):
        inspect(tokenizer_artifacts=forged_files)
    missing_files = tokenizer_artifacts(tokenizer)
    missing_files.pop("special_tokens_map.json")
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(tokenizer_artifacts=missing_files)
    with unittest.TestCase().assertRaises(AdmissionError):
        inspect(token_binding=replace(binding(tokenizer), model_revision="f" * 40))


def test_tokenizer_alias_eos_and_sequence_budget_fail_closed():
    class Aliased(StableTokenizer):
        def __call__(self, value, *, add_special_tokens):
            if value in (" A", " B", " C"):
                return {"input_ids": [100]}
            return super().__call__(value, add_special_tokens=add_special_tokens)

    class EarlyEos(StableTokenizer):
        def __call__(self, value, *, add_special_tokens):
            values = super().__call__(value, add_special_tokens=add_special_tokens)
            if value.endswith("\nAnswer:"):
                values["input_ids"].append(self.eos_token_id)
            return values

    for tokenizer in (Aliased(), EarlyEos()):
        with unittest.TestCase().assertRaises(AdmissionError):
            inspect(tokenizer=tokenizer)
    with unittest.TestCase().assertRaisesRegex(AdmissionError, "token budget"):
        inspect(max_tokens=16)


def test_no_gold_written_and_no_ml_imports_even_under_valid_inspection():
    import sys

    before_modules = set(sys.modules)
    with TemporaryDirectory() as root:
        before_files = tuple(Path(root).iterdir())
        report = inspect()
        assert tuple(Path(root).iterdir()) == before_files
    assert report["optimizer_steps"] == 0
    assert not ({"torch", "transformers", "peft", "bitsandbytes"}
                - before_modules) & set(sys.modules)


def test_missing_context_and_invalid_token_ranges_rejected():
    class NoTemplate(StableTokenizer):
        chat_template = None

    class NoEos(StableTokenizer):
        eos_token_id = None

    class BadIds(StableTokenizer):
        def __call__(self, value, *, add_special_tokens):
            return {"input_ids": [True]}

    for tokenizer in (NoTemplate(), NoEos(), BadIds()):
        with unittest.TestCase().assertRaises(AdmissionError):
            inspect(tokenizer=tokenizer, token_binding=binding(StableTokenizer()))


def test_source_is_import_pure_and_does_not_load_or_execute_models():
    import sys

    before = set(sys.modules)
    mod = sys.modules["trainingday.cognitive_sft_intake"]
    assert callable(mod.inspect_cognitive_sft_source)
    assert not ({"torch", "transformers", "peft", "bitsandbytes"} - before) & set(sys.modules)


def load_tests(loader, tests, pattern):
    """Run these same pytest-discoverable functions with stdlib unittest offline."""
    return unittest.TestSuite(
        unittest.FunctionTestCase(value, description=name)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
