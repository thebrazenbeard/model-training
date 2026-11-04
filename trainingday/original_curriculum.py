"""Small, deterministic, source-authored training *candidate* curriculum.

No external corpus is read. Generation is not a rights-admission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
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


def _canonical_row(domain: str, family: int, variant: int) -> dict:
    prompt, answer = example(domain, family, variant)
    case = f"oc-v1-{domain}-{family:02d}-{variant}"
    family_id = f"oc-v1-{domain}-{family:02d}"
    return {
        "id": case,
        "case_id": case,
        "domain": domain,
        "family_id": family_id,
        # Family-scoped IDs separate split members, but rule archetypes overlap.
        "template_id": family_id,
        "source_id": f"{GENERATOR}/{domain}/{family:02d}",
        "archetype_id": f"oc-v1-{domain}",
        "variant": variant,
        "split": "train" if family < TRAIN_FAMILIES else "dev",
        "prompt": prompt,
        "answer": answer,
        "rights_status": "NOT_ADMITTED",
        "training_authorized": False,
        "provenance": {"kind": "source_authored", "generator": GENERATOR,
                       "external_corpus": False},
    }


def build_rows() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        for family in range(FAMILIES_PER_DOMAIN):
            for variant in range(VARIANTS):
                rows.append(_canonical_row(domain, family, variant))
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
    for row in rows:
        domain, variant = row["domain"], row["variant"]
        # The family index is encoded in the generated ID, never trusted from
        # an optional caller field that might be altered independently.
        if domain not in DOMAINS or type(variant) is not int or not 0 <= variant < VARIANTS:
            raise ValueError("noncanonical domain or variant")
        try:
            family = int(row["family_id"].removeprefix(f"oc-v1-{domain}-"))
        except (ValueError, TypeError):
            raise ValueError("noncanonical family identity") from None
        if not 0 <= family < FAMILIES_PER_DOMAIN:
            raise ValueError("noncanonical family index")
        expected = _canonical_row(domain, family, variant)
        if row != expected:
            differences = sorted(key for key in row.keys() | expected.keys()
                                 if row.get(key) != expected.get(key))
            raise ValueError(f"noncanonical generated row: {', '.join(differences)}")


def _inode(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _verify_directory(output_dir: Path, dir_fd: int) -> None:
    try:
        path_info = os.stat(output_dir, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("output directory path changed; publication indeterminate") from exc
    if not stat.S_ISDIR(path_info.st_mode) or _inode(path_info) != _inode(os.fstat(dir_fd)):
        raise RuntimeError("output directory identity changed; publication indeterminate")


def _verify_published(dir_fd: int, name: str, staged_fd: int, data: bytes) -> None:
    """Refuse symlinks, changed inode or bytes at a published final name."""
    try:
        published_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as exc:
        raise RuntimeError(f"{name} unsafe or missing; publication indeterminate") from exc
    try:
        info = os.fstat(published_fd)
        if not stat.S_ISREG(info.st_mode) or _inode(info) != _inode(os.fstat(staged_fd)):
            raise RuntimeError(f"{name} inode changed; publication indeterminate")
        digest = hashlib.sha256()
        while chunk := os.read(published_fd, 1024 * 1024):
            digest.update(chunk)
        if digest.digest() != hashlib.sha256(data).digest():
            raise RuntimeError(f"{name} hash changed; publication indeterminate")
    finally:
        os.close(published_fd)


def _publish_exclusive(output_dir: Path, payloads: dict[str, bytes]) -> None:
    """Stage all bytes; link final names without following or replacing links.

    Files are independently linked, so this is not a filesystem transaction.
    Manifest is last. Ordinary failures roll back only names created here;
    after process death, an incomplete directory requires manual inspection.
    The destination directory and its ancestors must be owner-controlled:
    same-UID hostile modification after readback cannot be prevented here.
    """
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("exclusive curriculum publication requires POSIX no-follow directory support")
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _verify_directory(output_dir, dir_fd)
        for name in payloads:
            try:
                os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(f"candidate output already exists: {name}")

        staged: list[tuple[str, str, int, bytes]] = []
        published: list[tuple[str, tuple[int, int]]] = []
        try:
            for name, data in payloads.items():
                tmp_name = f".oc-stage-{secrets.token_hex(16)}"
                file_fd = os.open(tmp_name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                  0o600, dir_fd=dir_fd)
                staged.append((tmp_name, name, file_fd, data))
                with os.fdopen(os.dup(file_fd), "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            _verify_directory(output_dir, dir_fd)
            for tmp_name, name, file_fd, data in staged:
                source_info = os.stat(tmp_name, dir_fd=dir_fd, follow_symlinks=False)
                if not stat.S_ISREG(source_info.st_mode) or _inode(source_info) != _inode(os.fstat(file_fd)):
                    raise RuntimeError(f"{name} staged inode changed; publication indeterminate")
                os.link(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd,
                        follow_symlinks=False)
                published.append((name, _inode(os.fstat(file_fd))))
                _verify_published(dir_fd, name, file_fd, data)
                _verify_directory(output_dir, dir_fd)
                # Ensure earlier final names still hold their original bytes
                # before the last manifest link can mark completion.
                for prior_name, _, prior_fd, prior_data in staged[:len(published)]:
                    _verify_published(dir_fd, prior_name, prior_fd, prior_data)
            os.fsync(dir_fd)
            for name, _, file_fd, data in staged:
                _verify_published(dir_fd, name, file_fd, data)
            _verify_directory(output_dir, dir_fd)
        except BaseException as exc:
            rollback_errors = []
            for name, expected_inode in reversed(published):
                try:
                    info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                    if _inode(info) == expected_inode:
                        os.unlink(name, dir_fd=dir_fd)
                    else:
                        rollback_errors.append(f"{name}: identity changed; left untouched")
                except OSError as rollback_exc:
                    rollback_errors.append(f"{name}: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError("incomplete publication; manually inspect output directory: "
                                   + "; ".join(rollback_errors)) from exc
            raise
        finally:
            for tmp_name, _, file_fd, _ in staged:
                try:
                    info = os.stat(tmp_name, dir_fd=dir_fd, follow_symlinks=False)
                    if _inode(info) == _inode(os.fstat(file_fd)):
                        os.unlink(tmp_name, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass
                finally:
                    os.close(file_fd)
    finally:
        os.close(dir_fd)


def write_candidate(output_dir: Path) -> dict:
    rows = build_rows()
    payloads = {}
    files = {}
    for split in ("train", "dev"):
        payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          + "\n" for row in rows if row["split"] == split).encode("utf-8")
        filename = f"{split}.jsonl"
        payloads[filename] = payload
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
        "publication_note": "Manifest linked last after no-follow inode/hash checks, but multi-file publication is not crash-atomic. Output directory and ancestors must be owner-controlled; concurrent same-UID attackers cannot be excluded after readback. Verify hashes and manually inspect incomplete directories before retrying.",
        "files": files,
    }
    payloads["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _publish_exclusive(output_dir, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_candidate(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
