"""Small, deterministic, source-authored training *candidate* curriculum.

No external corpus is read. Generation is not a rights-admission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DOMAINS = ("logic", "math", "language", "tool_decisions", "uncertainty", "generalization")
GENERATOR = "trainingday.original_curriculum:v1"
FAMILIES_PER_DOMAIN = 32
TRAIN_FAMILIES = 24
VARIANTS = 3


def example(domain: str, family: int, variant: int) -> tuple[str, str]:
    n = 11 + family * 5 + variant
    if domain == "logic":
        prompt = f"For integer {n}, is it divisible by both 2 and 3? Reply true or false."
        return prompt, "true" if n % 6 == 0 else "false"
    if domain == "math":
        a, x, b = 2 + family % 7, n - 9, 3 + variant * 4
        return f"Solve for integer x: {a}x + {b} = {a * x + b}.", str(x)
    if domain == "language":
        stem = f"nava{family:02d}{variant}"
        suffix, transformed = (("ka", "zi") if family % 2 == 0 else ("tu", "vo"))
        return (f"In this invented code, replace final '{suffix}' with '{transformed}'. "
                f"Transform '{stem}{suffix}'."), stem + transformed
    if domain == "tool_decisions":
        choices = (
            ("A user asks to multiply two supplied integers", "calculator"),
            ("A user requests a current fact absent from the supplied evidence", "lookup"),
            ("A user asks to send a message but gives no recipient", "clarify"),
        )
        situation, answer = choices[(family + variant) % len(choices)]
        return (f"Case {n}: {situation}. Select one next step: calculator, lookup, clarify."), answer
    if domain == "uncertainty":
        states = (
            ("The sole provided statement says the lamp is blue. Claim: the lamp is blue.", "supported"),
            ("The sole provided statement says the lamp is red. Claim: the lamp is blue.", "contradicted"),
            ("The sole provided statement gives the lamp's weight only. Claim: the lamp is blue.", "insufficient"),
        )
        text, answer = states[(family + variant) % len(states)]
        return f"Case {n}: {text} Reply supported, contradicted, or insufficient.", answer
    if domain == "generalization":
        a, b = 2 + family % 8, 1 + variant + family
        return f"A linear rule has f(1)={a + b} and f(2)={2 * a + b}. Find f(4).", str(4 * a + b)
    raise ValueError(f"unknown domain: {domain}")


def build_rows() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        for family in range(FAMILIES_PER_DOMAIN):
            split = "train" if family < TRAIN_FAMILIES else "dev"
            for variant in range(VARIANTS):
                prompt, answer = example(domain, family, variant)
                rows.append({
                    "id": f"oc-v1-{domain}-{family:02d}-{variant}",
                    "case_id": f"oc-v1-{domain}-{family:02d}-{variant}",
                    "domain": domain,
                    "family_id": f"oc-v1-{domain}-{family:02d}",
                    # Isolation IDs are family-scoped, but all families in a
                    # domain still share one rule archetype. This is NOT an
                    # unseen-archetype transfer evaluation.
                    "template_id": f"oc-v1-{domain}-{family:02d}",
                    "source_id": f"{GENERATOR}/{domain}/{family:02d}",
                    "archetype_id": f"oc-v1-{domain}",
                    "variant": variant,
                    "split": split,
                    "prompt": prompt,
                    "answer": answer,
                    "rights_status": "NOT_ADMITTED",
                    "training_authorized": False,
                    "provenance": {"kind": "source_authored", "generator": GENERATOR,
                                   "external_corpus": False},
                })
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    ids, prompts = set(), set()
    split_groups = {key: {} for key in ("family_id", "template_id", "source_id")}
    counts = Counter()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("duplicate ID")
        if row["prompt"] in prompts:
            raise ValueError("duplicate prompt")
        ids.add(row["id"])
        prompts.add(row["prompt"])
        for key, groups in split_groups.items():
            groups.setdefault(row[key], set()).add(row["split"])
        if row["rights_status"] != "NOT_ADMITTED" or row["training_authorized"] is not False:
            raise ValueError("candidate rights status changed")
        if row["provenance"]["external_corpus"] is not False:
            raise ValueError("external corpus provenance not permitted")
        counts[(row["domain"], row["split"])] += 1
    for key, groups in split_groups.items():
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError(f"{key} leaked across splits")
    for domain in DOMAINS:
        if counts[(domain, "train")] != TRAIN_FAMILIES * VARIANTS:
            raise ValueError(f"invalid train count for {domain}")
        if counts[(domain, "dev")] != (FAMILIES_PER_DOMAIN - TRAIN_FAMILIES) * VARIANTS:
            raise ValueError(f"invalid dev count for {domain}")
    if len(rows) != len(DOMAINS) * FAMILIES_PER_DOMAIN * VARIANTS:
        raise ValueError("unexpected domain or split")


def write_candidate(output_dir: Path) -> dict:
    rows = build_rows()
    names = ("train.jsonl", "dev.jsonl", "manifest.json")
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("candidate output already exists; choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "dev"):
        payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          + "\n" for row in rows if row["split"] == split).encode("utf-8")
        filename = f"{split}.jsonl"
        (output_dir / filename).write_bytes(payload)
        files[filename] = {"rows": sum(r["split"] == split for r in rows),
                           "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "schema_version": "original-curriculum-candidate-v1", "generator": GENERATOR,
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": None, "determinism": "closed-form/no-rng", "domains": list(DOMAINS),
        "split_policy": "family/template/source-id-disjoint: per domain families 0-23 train, 24-31 dev",
        "evaluation_scope": "same-archetype diagnostic only; NOT unseen-archetype transfer",
        "archetype_overlap": True,
        "source_id_semantics": "generator/family synthetic provenance partition; all rows share one source-authored generator",
        "rights_status": "NOT_ADMITTED", "training_authorized": False,
        "rights_note": "Source-authored synthetic examples; requires independent rights/admission review.",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                                 encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_candidate(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
