"""R18 private DEV/exclusion hash-only bridge; not an authority or a bank reader.

This function must be called *inside* an independently controlled evaluator
process after it has authenticated its own OS identity, ACLs, input origin,
signing key, and replay log. Those are deliberately absent from this module.
The returned object is never an ADMIT, a signature, or proof of such custody.

No private content is logged, written, imported at module load, or returned.
The source-only unit tests use synthetic strings unrelated to protected gold.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata


SCHEMA = "VERA_R18_PRIVATE_DEV_EXCLUSION_HASH_BRIDGE_V1"
SELECTED_ID_V2_SCHEMA = "VERA_TRAININGDAY_SELECTED_ID_DIGEST_V2"
STATUS = "HASHES_ONLY_UNAUTHENTICATED_HOLD"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_SUBJECT_FIELDS = frozenset({
    "corpus_id", "manifest_sha256", "train_sha256", "selected_id_v2_sha256",
    "generator_source_sha256", "train_content_sha256",
})
_EXCLUSION_BANKS = frozenset({"R16_PROTECTED", "R18_EXTERNAL_OOD"})


class CustodyInputError(ValueError):
    """Reject private-source declaration without disclosing its contents."""


def _require(condition: bool) -> None:
    if not condition:
        raise CustodyInputError("private DEV/exclusion commitment input rejected")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    normalized = set()
    for key, value in pairs:
        _require(type(key) is str)
        alias = unicodedata.normalize("NFKC", key).casefold()
        _require(alias not in normalized and not any(
            unicodedata.category(c) in {"Cc", "Cf"} for c in key))
        normalized.add(alias)
        result[key] = value
    return result


def _decode(raw: bytes) -> object:
    _require(type(raw) is bytes and raw and len(raw) <= 16 * 1024 * 1024)
    try:
        return json.loads(raw.decode("utf-8", errors="strict"),
                          object_pairs_hook=_strict_object,
                          parse_constant=lambda _: _require(False))
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise CustodyInputError("private DEV/exclusion commitment input rejected") from exc


def _case_id(value: object) -> bool:
    return (type(value) is str and 1 <= len(value) <= 256
            and value.strip() == value and
            not any(unicodedata.category(c) in {"Cc", "Cf"} for c in value))


def _prompt_hash(value: str) -> str:
    _require(type(value) is str and 1 <= len(value) <= 4096)
    normalized = " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
    _require(bool(normalized))
    return _sha(normalized.encode("utf-8"))


def _index_entry(item: object) -> tuple[str, str]:
    _require(type(item) is dict and set(item) == {"id", "prompt_sha256"})
    _require(_case_id(item["id"]) and type(item["prompt_sha256"]) is str
             and _SHA.fullmatch(item["prompt_sha256"]) is not None)
    return item["id"], item["prompt_sha256"]


def _train_index_entry(item: object) -> tuple[str, str, str]:
    _require(type(item) is dict
             and set(item) == {"id", "prompt_sha256", "answer_sha256"})
    case_id, prompt_digest = _index_entry({"id": item["id"],
                                           "prompt_sha256": item["prompt_sha256"]})
    answer_digest = item["answer_sha256"]
    _require(type(answer_digest) is str and _SHA.fullmatch(answer_digest) is not None)
    return case_id, prompt_digest, answer_digest


def _unique(entries: list[tuple[str, ...]]) -> None:
    _require(len({unicodedata.normalize("NFKC", entry[0]).casefold()
                  for entry in entries}) == len(entries)
             and len({entry[1] for entry in entries}) == len(entries))


def build_source_only_private_commitment(*, subject: dict,
                                         train_index: list[dict],
                                         dev_jsonl: bytes,
                                         protected_exclusions_json: bytes) -> dict:
    """Cross-check disjoint private DEV + protected index; release only hashes.

    Raw bytes are hashed exactly; input reordering changes the commitment.
    The output never authenticates the caller, source rights, or physical ACLs.
    It is suitable only for a future separately privileged signer to inspect.
    """
    _require(type(subject) is dict and set(subject) == _SUBJECT_FIELDS)
    _require(type(subject["corpus_id"]) is str
             and subject["corpus_id"] == "R18_ARITHMETIC")
    for key in _SUBJECT_FIELDS - {"corpus_id"}:
        _require(type(subject[key]) is str and _SHA.fullmatch(subject[key]) is not None)

    _require(type(train_index) is list and len(train_index) == 1024)
    train = [_train_index_entry(item) for item in train_index]
    _unique(train)
    ids = [entry[0] for entry in train]
    selected_digest = _sha(
        SELECTED_ID_V2_SCHEMA.encode("ascii") + b"\x00"
        + json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    _require(selected_digest == subject["selected_id_v2_sha256"])
    # Mirrors r18_qlora.train_content_digest, including the ordered answers.
    # This checks only consistency with a caller declaration, not its provenance.
    train_content_digest = _sha(_json_canonical([
        {"id": case_id, "prompt_sha256": prompt_digest,
         "answer_sha256": answer_digest}
        for case_id, prompt_digest, answer_digest in train
    ]))
    _require(train_content_digest == subject["train_content_sha256"])

    _require(type(dev_jsonl) is bytes and 0 < len(dev_jsonl) <= 16 * 1024 * 1024
             and dev_jsonl.endswith(b"\n") and b"\r" not in dev_jsonl)
    dev_raw_lines = dev_jsonl[:-1].split(b"\n")
    _require(1 <= len(dev_raw_lines) <= 65536 and all(dev_raw_lines))
    dev_index = []
    for raw_line in dev_raw_lines:
        row = _decode(raw_line)
        _require(type(row) is dict and set(row) == {"id", "prompt", "answer", "split"})
        _require(_case_id(row["id"]) and row["split"] == "dev"
                 and type(row["answer"]) is str
                 and len(row["answer"]) <= 8
                 and _INTEGER.fullmatch(row["answer"]) is not None)
        _require(raw_line == _json_canonical(row))
        dev_index.append((row["id"], _prompt_hash(row["prompt"])))
    _unique(dev_index)

    exclusion = _decode(protected_exclusions_json)
    _require(type(exclusion) is list and len(exclusion) == 128
             and protected_exclusions_json == _json_canonical(exclusion))
    exclusion_index: list[tuple[str, str]] = []
    counts = {bank: 0 for bank in _EXCLUSION_BANKS}
    for item in exclusion:
        _require(type(item) is dict
                 and set(item) == {"id", "prompt_sha256", "bank"}
                 and type(item["bank"]) is str and item["bank"] in counts)
        counts[item["bank"]] += 1
        exclusion_index.append(_index_entry({"id": item["id"],
                                             "prompt_sha256": item["prompt_sha256"]}))
    _require(all(count == 64 for count in counts.values()))
    _unique(exclusion_index)

    # Reject ambiguous case/NFKC aliases across splits as well as byte-exact
    # duplicates. Keep original case IDs unchanged in the V2 source digest;
    # this only strengthens the protected overlap rejection.
    all_ids = [unicodedata.normalize("NFKC", item[0]).casefold()
               for item in train + dev_index + exclusion_index]
    all_prompts = [item[1] for item in train + dev_index + exclusion_index]
    _require(len(all_ids) == len(set(all_ids))
             and len(all_prompts) == len(set(all_prompts)))
    return {
        "schema": SCHEMA, "status": STATUS,
        "corpus_id": "R18_ARITHMETIC",
        "manifest_sha256": subject["manifest_sha256"],
        "train_sha256": subject["train_sha256"],
        "train_content_sha256": train_content_digest,
        "selected_id_v2_sha256": selected_digest,
        "generator_source_sha256": subject["generator_source_sha256"],
        "dev_sha256": _sha(dev_jsonl),
        "protected_exclusion_sha256": _sha(protected_exclusions_json),
        "dev_count": len(dev_index),
        "protected_exclusion_count": len(exclusion_index),
        "private_content_returned": False,
        "independent_custody_authenticated": False,
        "train_content_commitment_authenticated": False,
        "owner_identity_authenticated": False,
        "source_use_rights_verified": False,
        "host_ratification_authenticated": False,
        "authorizes_training": False,
        "authorizes_eval_unseal": False,
    }
