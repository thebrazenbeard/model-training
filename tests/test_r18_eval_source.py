"""Independent source-only checks for the new external R18 OOD bank."""

import copy
import hashlib
import json
import operator
import re

import pytest

from trainingday.r18_eval_source import (
    ALGORITHM, CHAIN_LENGTH, COUNT, MAX_ABS_INTERMEDIATE,
    MAX_ANSWER_CHARS, MAX_PROMPT_CHARS, SEED, SOURCE_SCHEMA,
    generate_novel_r18_cases, generator_code_sha256, source_manifest,
)
from trainingday.r18_benchmark import canonical_sha256


TEST_NONCE = "ab" * 32
EXPECTED_CASES_SHA256 = "3d3355c03ab267c56197fd8bf6d07c8518c00c146cf72204030186630b834d19"
EXPECTED_FIRST_SIGNATURE_SHA256 = (
    "34e730fa3d11043820ebfffc55f3b3856da9e6756c5b965e1927cb926a83c323"
)


def test_fresh64_has_unique_six_step_signatures_and_independent_prompt_oracle():
    cases = generate_novel_r18_cases(private_nonce_hex=TEST_NONCE)
    assert len(cases) == COUNT == 64
    assert len({r["id"] for r in cases}) == 64
    assert len({r["signature_sha256"] for r in cases}) == 64
    assert len({r["prompt"] for r in cases}) == 64
    assert canonical_sha256(cases) == EXPECTED_CASES_SHA256
    assert cases[0]["signature_sha256"] == EXPECTED_FIRST_SIGNATURE_SHA256

    for case in cases:
        prompt = case["prompt"]
        assert len(prompt) <= MAX_PROMPT_CHARS
        assert len(case["expected"]) <= MAX_ANSWER_CHARS
        assert re.fullmatch(r"r18-reasoning-[0-9a-f]{64}", case["id"])
        match = re.fullmatch(
            r"Start with (-?(?:0|[1-9]\d*))\. (.+) What is the final integer\?",
            prompt)
        assert match is not None
        steps = re.findall(r"(add|subtract|multiply) (-?(?:0|[1-9]\d*))\.", match.group(2))
        assert len(steps) == CHAIN_LENGTH == 6  # v2 max was five, disjoint by construction.
        value = int(match.group(1))
        seen = {value}
        functions = {"add": operator.add, "subtract": operator.sub,
                     "multiply": operator.mul}
        for name, value_text in steps:
            operand = int(value_text)
            assert operand != 0
            if name == "multiply":
                assert operand not in {-1, 0, 1}
            new_value = functions[name](value, operand)
            assert new_value != value and new_value not in seen
            assert abs(new_value) <= MAX_ABS_INTERMEDIATE
            value = new_value
            seen.add(value)
        assert str(value) == case["expected"]


def test_frozen_generator_seed_and_case_order_cannot_be_tuned():
    assert generate_novel_r18_cases(private_nonce_hex=TEST_NONCE) == generate_novel_r18_cases(
        private_nonce_hex=TEST_NONCE)
    assert generate_novel_r18_cases(private_nonce_hex="cd"*32) != generate_novel_r18_cases(
        private_nonce_hex=TEST_NONCE)
    with pytest.raises(ValueError):
        generate_novel_r18_cases(private_nonce_hex=TEST_NONCE, seed=SEED + 1)
    with pytest.raises(ValueError):
        generate_novel_r18_cases(private_nonce_hex=TEST_NONCE, count=63)
    with pytest.raises(ValueError):
        generate_novel_r18_cases(private_nonce_hex="public-short")


def test_public_source_manifest_has_commit_provenance_no_private_cases():
    cases = generate_novel_r18_cases(private_nonce_hex=TEST_NONCE)
    manifest = source_manifest(cases, private_nonce_hex=TEST_NONCE,
                               v2_manifest_sha256="a" * 64,
                               producer_git_head="b" * 40)
    assert manifest["schema"] == SOURCE_SCHEMA
    assert manifest["algorithm"] == ALGORITHM
    assert manifest["protocol"] == "R18_EXTERNAL_LONG_CHAIN64_OOD_V1"
    assert manifest["generator_code_sha256"] == generator_code_sha256()
    assert manifest["private_nonce_sha256"] == hashlib.sha256(bytes.fromhex(TEST_NONCE)).hexdigest()
    assert TEST_NONCE not in json.dumps(manifest)
    assert manifest["selected_unique_signature_count"] == 64
    assert manifest["older_v2_chain_length_range"] == [2, 5]
    assert manifest["previously_scored_v2_signature_count"] == 77
    assert manifest["older_v2_unseen_signatures_after_r16"] == 0
    assert manifest["private_case_content_sha256"] == EXPECTED_CASES_SHA256
    assert manifest["selected_signature_sha256"] == hashlib.sha256(
        "\n".join(row["signature_sha256"] for row in cases).encode()).hexdigest()
    serialized = json.dumps(manifest)
    assert "Start with" not in serialized and '"expected"' not in serialized
    mutated = copy.deepcopy(cases)
    mutated[0]["expected"] = "wrong"
    with pytest.raises(ValueError):
        source_manifest(mutated, private_nonce_hex=TEST_NONCE,
                        v2_manifest_sha256="a" * 64,
                        producer_git_head="b" * 40)
