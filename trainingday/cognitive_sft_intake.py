"""Bounded, read-only cognitive TRAIN source inspection; never launch authority.

This R154 contract is intentionally separate from the frozen numeric R16/R18
corpus and QLoRA paths.  The input contains synthetic TRAIN gold, so callers
must keep it on an operator-controlled side of the evaluation boundary. No
gold, prompt, token IDs, or sample labels are returned or printed here.

An exact Git head and local SHA-256 are *claims*, not source authenticity or
rights-holder approval.  The result is never admissible to a physical trainer.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from trainingday.admission import AdmissionError, _require, _strict_json


SCHEMA = "vera.trainingday.cognitive-sft-source-intake.v1"
PRODUCER_SCHEMA = "labyrinth.cognitive-supervised-train-example.v1"
PROVIDER_ID = "synthetic_cognitive_perspective_train_v1"
RUBRIC_ID = "perspective_choice_train_v1"
PINNED_LABYRINTH_HEAD = "b69502c4718333c6055a69817110e2bd061a6346"
PINNED_LABYRINTH_TREE = "0c7038397f7fe65fbf781be2237a34cb5a93633f"
MODEL_ID = "Qwen/Qwen3.5-4B"
SOURCE_PREPARED_NOT_ADMITTED_FOR_PHYSICAL_TRAINING = (
    "SOURCE_PREPARED_NOT_ADMITTED_FOR_PHYSICAL_TRAINING"
)
# Reserved for a future, externally authenticated replay/rights-custody
# integration. Caller-supplied SHA values can never reach SOURCE_PREPARED.
UNVERIFIED_REJECTED = "UNVERIFIED_REJECTED"
MAX_DATASET_BYTES = 262_144
MAX_ROOMS = 15
MAX_SEQUENCE_TOKENS = 8192
MAX_TOKENIZER_ARTIFACT_BYTES = 64 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_CASE = re.compile(r"Case: (train-cognitive-[a-z0-9-]+)\n", re.ASCII)
_OPTION = re.compile(r"([ABC])\. (.+)\Z", re.ASCII)
_ROOM = re.compile(r"[a-z0-9-]{1,128}\Z", re.ASCII)
_ASK = "Which interpretation best preserves uncertainty about the person's perspective?"
_INSTRUCTION = "Respond with only A, B, or C."
_VIGNETTE_TAIL = ". No one has supplied a complete account of that person's reasons."
_SETTINGS = (
    "during a community project", "after an unexpectedly busy week",
    "while schedules are being rearranged", "during a shared planning session",
    "when several priorities compete", "after a change in routine",
    "as a deadline approaches", "while information is incomplete",
)
_FIELDS = frozenset({
    "schema", "split", "provider_id", "training_rubric_id", "objective_id",
    "objective_contract_sha256", "cycle_index", "room_id", "case_id",
    "prompt", "prompt_sha256", "semantic_sha256", "source_binding_sha256",
    "target_choice",
})
_TOKEN_FILES = frozenset({
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":")).encode("utf-8")


def _digest(value: object, name: str) -> str:
    _require(type(value) is str and _HEX64.fullmatch(value) is not None,
             f"{name} must be a canonical SHA-256 digest")
    return value


def _normalized(value: str) -> str:
    _require(type(value) is str and 0 < len(value) <= 4096,
             "invalid bounded semantic scenario text")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    _require(bool(normalized), "empty normalized semantic scenario text")
    return normalized


def _core_scenario(vignette: str) -> str:
    """Mirror the producer's *specified* cosmetic-setting normalization."""
    normalized = _normalized(vignette)
    suffix = _normalized(_VIGNETTE_TAIL + " " + _ASK)
    if normalized.endswith(suffix):
        for setting in _SETTINGS:
            prefix = _normalized(f"At {setting},") + " "
            if normalized.startswith(prefix):
                return normalized[len(prefix):-len(suffix)].strip() + " " + _normalized(_ASK)
    return normalized


def semantic_digest_from_prompt(prompt: str) -> str:
    """Independently recompute R90 scenario+unordered-choice fingerprint.

    This detects deterministic duplicates/renames; it is *not* an NLP proof
    that all paraphrases are semantically different.
    """
    _require(type(prompt) is str and len(prompt) <= 4096,
             "invalid bounded cognitive TRAIN prompt")
    matched = _CASE.match(prompt)
    _require(matched is not None, "missing cognitive TRAIN case prefix")
    lines = prompt[matched.end():].splitlines()
    _require(len(lines) == 7 and lines[1] == _ASK and lines[2] == "Options:"
             and lines[6] == _INSTRUCTION, "unexpected cognitive TRAIN prompt format")
    options = tuple(_OPTION.fullmatch(line) for line in lines[3:6])
    _require(all(option is not None for option in options), "invalid cognitive choices")
    _require({option.group(1) for option in options} == {"A", "B", "C"},
             "repeated/missing cognitive choice")
    choices = sorted(_normalized(option.group(2)) for option in options)
    _require(len(set(choices)) == 3, "semantic duplicate choice text")
    payload = {
        "schema": "labyrinth.cognitive-train-semantic.v2",
        "scenario": _normalized(_core_scenario(lines[0] + " " + _ASK)),
        "choices": choices,
    }
    return _sha(_canonical_json(payload))


@dataclass(frozen=True, slots=True)
class TokenizerSourceBinding:
    """Source-file identity claim, not a signed model or tokenizer attestation."""

    model_id: str
    model_revision: str
    tokenizer_json_sha256: str
    tokenizer_config_sha256: str
    special_tokens_map_sha256: str
    chat_template_sha256: str
    eos_token_id: int

    def sha256(self) -> str:
        return _sha(_canonical_json({
            "schema": "vera.trainingday.cognitive-tokenizer-source-binding.v1",
            "model_id": self.model_id, "model_revision": self.model_revision,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "special_tokens_map_sha256": self.special_tokens_map_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "eos_token_id": self.eos_token_id,
        }))


def _token_ids(tokenizer: Any, value: str, *, special: bool) -> tuple[int, ...]:
    try:
        encoded = tokenizer(value, add_special_tokens=special)
    except Exception:
        # A tokenizer's arbitrary exception text could echo TRAIN gold.
        raise AdmissionError("operator tokenizer rejected a bounded TRAIN string") from None
    _require(isinstance(encoded, Mapping), "tokenizer must return an input_ids mapping")
    ids = encoded.get("input_ids")
    _require(type(ids) in (tuple, list) and bool(ids)
             and all(type(token) is int and 0 <= token <= 2**31 - 1 for token in ids),
             "tokenizer yielded invalid flat token IDs")
    return tuple(ids)


def _check_tokenizer_binding(
    tokenizer: Any, binding: TokenizerSourceBinding,
    artifacts: Mapping[str, bytes], independent_sha256: str,
) -> int:
    _require(type(binding) is TokenizerSourceBinding,
             "exact source-only tokenizer binding is required")
    _require(binding.model_id == MODEL_ID
             and type(binding.model_revision) is str
             and _HEX40.fullmatch(binding.model_revision) is not None,
             "unsupported/unpinned tokenizer model identity")
    _require(type(binding.eos_token_id) is int
             and 0 <= binding.eos_token_id <= 2**31 - 1,
             "invalid bound tokenizer EOS")
    _require(callable(tokenizer) and getattr(tokenizer, "name_or_path", None) == binding.model_id
             and getattr(tokenizer, "eos_token_id", None) == binding.eos_token_id,
             "injected tokenizer identity/EOS mismatch")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    _require(isinstance(init_kwargs, Mapping)
             and init_kwargs.get("revision") == binding.model_revision,
             "injected tokenizer revision is unbound")
    template = getattr(tokenizer, "chat_template", None)
    _require(type(template) is str and bool(template)
             and _sha(template.encode("utf-8")) == _digest(
                 binding.chat_template_sha256, "chat template"),
             "tokenizer chat template missing or mismatch")
    _require(isinstance(artifacts, Mapping) and set(artifacts) == _TOKEN_FILES,
             "exact tokenizer source artifacts required")
    for filename, attribute in (
        ("tokenizer.json", "tokenizer_json_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("special_tokens_map.json", "special_tokens_map_sha256"),
    ):
        raw = artifacts[filename]
        _require(type(raw) is bytes and 0 < len(raw) <= MAX_TOKENIZER_ARTIFACT_BYTES
                 and _sha(raw) == _digest(getattr(binding, attribute), filename),
                 f"tokenizer source artifact {filename} mismatch")
    config = _strict_json(artifacts["tokenizer_config.json"])
    _require(type(config) is dict and config.get("chat_template") == template,
             "tokenizer config/chat template mismatch")
    _require(binding.sha256() == _digest(independent_sha256, "independent tokenizer"),
             "independent tokenizer source binding mismatch")
    return binding.eos_token_id


def _parse_train_rows(
    payload: bytes, *, expected_objective_id: str,
    expected_contract_sha256: str, expected_cycle_index: int,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str],
) -> tuple[dict, ...]:
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_DATASET_BYTES,
             "bounded cognitive TRAIN JSONL required")
    _require(payload.endswith(b"\n"), "noncanonical TRAIN JSONL termination")
    lines = payload.split(b"\n")[:-1]
    _require(1 <= len(lines) <= MAX_ROOMS and all(lines),
             "empty/oversized/blank cognitive TRAIN corpus")
    _require(type(expected_objective_id) is str
             and bool(re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", expected_objective_id)),
             "independent objective ID missing")
    _digest(expected_contract_sha256, "independent objective contract")
    _require(type(expected_cycle_index) is int and 0 <= expected_cycle_index <= 1024,
             "invalid expected cognitive TRAIN cycle")
    _require(type(forbidden_semantic_sha256) in (tuple, frozenset)
             and len(forbidden_semantic_sha256) <= 4096,
             "forbidden semantic commitment must be a bounded immutable set")
    excluded = set(forbidden_semantic_sha256)
    _require(len(excluded) == len(forbidden_semantic_sha256),
             "duplicate forbidden semantic commitment")
    for fingerprint in excluded:
        _digest(fingerprint, "forbidden semantic")
    cases, semantics, prompts, sources, result = set(), set(), set(), set(), []
    for line in lines:
        row = _strict_json(line)
        _require(type(row) is dict and set(row) == _FIELDS,
                 "unsupported cognitive row field, rights assertion, or split leakage")
        _require(_canonical_json(row) == line, "noncanonical cognitive TRAIN JSONL row")
        _require(row["schema"] == PRODUCER_SCHEMA and row["split"] == "train"
                 and row["provider_id"] == PROVIDER_ID
                 and row["training_rubric_id"] == RUBRIC_ID,
                 "unapproved cognitive TRAIN producer, rubric, or split")
        _require(row["objective_id"] == expected_objective_id
                 and row["objective_contract_sha256"] == expected_contract_sha256
                 and type(row["cycle_index"]) is int
                 and row["cycle_index"] == expected_cycle_index,
                 "cognitive objective/cycle identity mismatch")
        case_id, room_id, prompt = row["case_id"], row["room_id"], row["prompt"]
        _require(type(prompt) is str and type(case_id) is str
                 and type(room_id) is str and _ROOM.fullmatch(room_id) is not None
                 and case_id == "train-cognitive-" + room_id
                 and prompt.startswith("Case: " + case_id + "\n"),
                 "cognitive case/room identity mismatch")
        _require(type(prompt) is str and 0 < len(prompt) <= 4096
                 and _sha(prompt.encode("utf-8")) == _digest(row["prompt_sha256"], "prompt"),
                 "cognitive prompt commitment mismatch")
        semantic = semantic_digest_from_prompt(prompt)
        _require(semantic == _digest(row["semantic_sha256"], "semantic"),
                 "forged cognitive semantic commitment")
        source_sha = _digest(row["source_binding_sha256"], "source room")
        _require(row["target_choice"] in ("A", "B", "C"),
                 "invalid cognitive TRAIN supervised choice")
        _require(case_id not in cases and row["prompt_sha256"] not in prompts,
                 "duplicate cognitive TRAIN case or prompt")
        _require(source_sha not in sources, "duplicate cognitive source room binding")
        _require(semantic not in semantics and semantic not in excluded,
                 "semantic duplicate or held-out overlap in TRAIN")
        cases.add(case_id)
        semantics.add(semantic)
        prompts.add(row["prompt_sha256"])
        sources.add(source_sha)
        result.append(row)
    return tuple(result)


def inspect_cognitive_sft_source(
    *, dataset_jsonl: bytes, dataset_sha256: str,
    independent_dataset_sha256: str, producer_git_head: str,
    producer_git_tree: str, expected_objective_id: str,
    expected_contract_sha256: str, expected_cycle_index: int,
    forbidden_semantic_sha256: tuple[str, ...] | frozenset[str],
    tokenizer: Any, tokenizer_binding: TokenizerSourceBinding,
    tokenizer_artifacts: Mapping[str, bytes],
    independent_tokenizer_binding_sha256: str,
    max_tokens: int = 2048,
) -> dict[str, str | int | bool]:
    """Inspect TRAIN source and tokenizer parity, return explicit rejection.

    SHA values and forbidden semantic digests are caller-provided and cannot
    authenticate their own origin, especially gold labels and room-source
    digests. Without independently authenticated canonical producer replay,
    even byte-consistent source MUST be UNVERIFIED_REJECTED; there is no
    caller-controlled override or positive admission path in this version.
    This function never imports ML libraries, writes, trains, or reveals gold.
    """
    _require(producer_git_head == PINNED_LABYRINTH_HEAD
             and producer_git_tree == PINNED_LABYRINTH_TREE,
             "Labyrinth source head/tree differs from exact pinned producer")
    _require(type(dataset_jsonl) is bytes
             and _sha(dataset_jsonl) == _digest(dataset_sha256, "dataset")
             and dataset_sha256 == _digest(independent_dataset_sha256, "operator dataset"),
             "producer/independent TRAIN dataset digest mismatch")
    _require(type(max_tokens) is int and 16 <= max_tokens <= MAX_SEQUENCE_TOKENS,
             "invalid cognitive SFT token budget")
    rows = _parse_train_rows(
        dataset_jsonl, expected_objective_id=expected_objective_id,
        expected_contract_sha256=expected_contract_sha256,
        expected_cycle_index=expected_cycle_index,
        forbidden_semantic_sha256=forbidden_semantic_sha256,
    )
    eos = _check_tokenizer_binding(tokenizer, tokenizer_binding,
                                   tokenizer_artifacts,
                                   independent_tokenizer_binding_sha256)
    choice_ids = {choice: _token_ids(tokenizer, " " + choice, special=False)
                  for choice in "ABC"}
    _require(len(set(choice_ids.values())) == 3
             and all(eos not in ids for ids in choice_ids.values()),
             "ambiguous or EOS-containing cognitive choice tokenization")
    for row in rows:
        prefix_text = row["prompt"] + "\nAnswer:"
        prefix = _token_ids(tokenizer, prefix_text, special=True)
        _require(eos not in prefix, "EOS present before supervised choice")
        for choice in "ABC":
            full = _token_ids(tokenizer, prefix_text + " " + choice, special=True)
            _require(full == prefix + choice_ids[choice],
                     "whole-context tokenizer parity mismatch")
            _require(len(full) + 1 <= max_tokens,
                     "cognitive SFT sequence exceeds token budget")
        # Never construct/return gold labels or encoded TRAIN samples. The
        # checked target choice remains solely in the private input bytes.
    semantic_commitment = _sha(_canonical_json(sorted(
        row["semantic_sha256"] for row in rows
    )))
    return {
        "schema": SCHEMA,
        "status": UNVERIFIED_REJECTED,
        "source_structure_checked": True,
        "producer_git_head": PINNED_LABYRINTH_HEAD,
        "producer_git_tree": PINNED_LABYRINTH_TREE,
        "dataset_sha256": dataset_sha256,
        "tokenizer_binding_sha256": tokenizer_binding.sha256(),
        "semantic_commitment_sha256": semantic_commitment,
        "case_count": len(rows),
        "cycle_index": expected_cycle_index,
        "rights_admitted": False,
        "producer_replay_verified": False,
        "gold_verified": False,
        "source_binding_verified": False,
        "heldout_custody_verified": False,
        "tokenizer_host_attested": False,
        "authorizes_training": False,
        "authorizes_host_execution": False,
        "authorizes_exam": False,
        "authorizes_graduation": False,
        "optimizer_steps": 0,
    }
