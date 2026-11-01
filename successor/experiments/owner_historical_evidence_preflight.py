"""Read-only R9–R13 historical evidence admission on the owning Windows host.

This is *not* the fresh TrainingDay R14 training preflight. It neither loads
weights nor grants training, release, protected evaluation, or merge authority.
Run only where the owner has admitted the existing private historical files.
Output contains symbolic IDs and status codes, never file paths or contents.
Protected heldout admission is exact-byte hash, stable size, and raw-line
count ONLY. Without independently authenticated evaluation data rights, this
tool never decodes heldout plaintext, inspects its JSON markers, or compares
its prompts to training source. Such semantic checks remain NOT_AUTHORIZED.
The `protected_plaintext_opened=false` report field means no textual
decode/parse; it does not deny raw binary reads required for hash/count.

    py -m successor.experiments.owner_historical_evidence_preflight \
       --r9-train D:\\VERA\\path-to-existing-r9-train.jsonl

The R9 corpus path is explicitly supplied because the R10 builder pins its
digest but does not bind one canonical filesystem path. Other paths and pins
are inherited from the frozen historical builders. No private material is
created or copied, and unavailable evidence blocks with exit code 42.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


R13_BINDING_SHA256 = "af27fa2a9fadfa5c097fe8a6f82d6210d610b00dacc48f47a2840f4b711b5377"
_BLOCKED = {
    "BLOCKED_MISSING_OWNER_EVIDENCE",
    "BLOCKED_CRLF_BYTE_PROVENANCE",
    "BLOCKED_UNSUPPORTED_HOST",
}


@dataclass(frozen=True)
class EvidencePin:
    id: str
    path: Path | None
    expected_sha256: str | None = None
    rows: int | None = None
    kind: str = "file"
    receipt_self_sha256: str | None = None
    fields: tuple[tuple[str, object], ...] = ()
    crlf_historical: bool = False


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_receipt_sha(receipt: dict[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    payload = json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _field(receipt: dict[str, Any], dotted_path: str) -> Any:
    current: Any = receipt
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _count_and_validate_rows(path: Path, pin: EvidencePin) -> tuple[str, set[str] | None]:
    count = 0
    prompts: set[str] = set()
    axis_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    with path.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            count += 1
            # The protected heldout is byte-opaque. Only the separately
            # public R13 source may be parsed for source-row provenance.
            if pin.kind != "r13_source":
                continue
            try:
                item = json.loads(raw_line)
            except (UnicodeError, ValueError):
                return "FAIL_FORMAT", None
            if not isinstance(item, dict) or not isinstance(item.get("user"), str):
                return "FAIL_FORMAT", None
            prompt = item["user"].strip().casefold()
            if not prompt or prompt in prompts:
                return "FAIL_SOURCE_PROVENANCE", None
            prompts.add(prompt)
            identifier = item.get("id")
            if (not isinstance(identifier, str) or not identifier
                    or identifier in seen_ids
                    or item.get("privacy") != "NON_PERSONAL"
                    or item.get("training_use") != "targeted_behavioral_sft"
                    or item.get("source_class") != "NEW_ORIGINAL_BEHAVIOR_NOT_HELDOUT"
                    or item.get("axis") not in ("identity", "judgment")):
                return "FAIL_SOURCE_PROVENANCE", None
            seen_ids.add(identifier)
            axis = item["axis"]
            axis_counts[axis] = axis_counts.get(axis, 0) + 1
    if pin.rows is not None and count != pin.rows:
        return "FAIL_ROW_COUNT", None
    if pin.kind == "r13_source" and pin.rows == 128 and (
        axis_counts.get("identity") != 64 or axis_counts.get("judgment") != 64
    ):
        return "FAIL_SOURCE_PROVENANCE", None
    return "PASS", prompts if pin.kind == "r13_source" else None


def _verify_pin(pin: EvidencePin) -> tuple[str, set[str] | None]:
    path = pin.path
    if path is None:
        return "BLOCKED_MISSING_OWNER_EVIDENCE", None
    try:
        if path.is_symlink():
            return "FAIL_UNSAFE_LINK", None
        if not path.is_file():
            return "BLOCKED_MISSING_OWNER_EVIDENCE", None
        before = path.stat()
        if pin.expected_sha256 is not None:
            actual = _sha_file(path)
            if actual != pin.expected_sha256:
                if pin.crlf_historical:
                    raw = path.read_bytes()
                    if b"\r" not in raw and hashlib.sha256(
                        raw.replace(b"\n", b"\r\n")
                    ).hexdigest() == pin.expected_sha256:
                        # Git LF transport is explainable, but NOT byte-identical
                        # to the completed historical Windows receipt's spec.
                        return "BLOCKED_CRLF_BYTE_PROVENANCE", None
                return "FAIL_DIGEST", None
        if pin.receipt_self_sha256 is not None:
            try:
                receipt = json.loads(path.read_text(encoding="utf8"))
            except (ValueError, UnicodeError):
                return "FAIL_FORMAT", None
            if not isinstance(receipt, dict):
                return "FAIL_FORMAT", None
            self_sha = receipt.get("receipt_sha256")
            if (self_sha != pin.receipt_self_sha256
                    or _canonical_receipt_sha(receipt) != self_sha):
                return "FAIL_RECEIPT", None
            for key, expected in pin.fields:
                actual = _field(receipt, key)
                if type(actual) is not type(expected) or actual != expected:
                    return "FAIL_RECEIPT", None
        if pin.rows is not None or pin.kind in ("heldout", "r13_source"):
            status, prompts = _count_and_validate_rows(path, pin)
            if status != "PASS":
                return status, None
        else:
            prompts = None
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if identity(before) != identity(after):
            return "FAIL_MUTABLE_EVIDENCE", None
        if pin.expected_sha256 is not None and _sha_file(path) != pin.expected_sha256:
            return "FAIL_MUTABLE_EVIDENCE", None
        return "PASS", prompts
    except (OSError, UnicodeError, ValueError):
        # Exception strings may contain owner paths or protected text. Never
        # export them through the public/reporting boundary.
        return "FAIL_READ", None


def evaluate(pins: Iterable[EvidencePin]) -> dict[str, Any]:
    """Verify only explicit known artifacts; return a sanitized status receipt."""
    pins = tuple(pins)
    if not pins:
        raise ValueError("an empty inventory cannot qualify historical evidence")
    ids = [pin.id for pin in pins]
    if len(ids) != len(set(ids)) or any(
        not identifier or not identifier.replace("_", "").isalnum()
        for identifier in ids
    ):
        raise ValueError("invalid or duplicate symbolic evidence IDs")
    checks = []
    for pin in pins:
        status, _ = _verify_pin(pin)
        checks.append({"id": pin.id, "status": status})
    if any(check["status"] != "PASS" and check["status"] not in _BLOCKED
           for check in checks):
        status = "HISTORICAL_FAIL"
    elif any(check["status"] in _BLOCKED for check in checks):
        status = "HISTORICAL_BLOCKED_MISSING_EVIDENCE"
    else:
        status = "HISTORICAL_PASS"
    return {"status": status, "checks": checks,
            "scope": "SELECTED_HISTORICAL_ARTIFACT_PINS_ONLY",
            "training_qualified": False,
            "protected_plaintext_opened": False,
            "protected_semantic_qualification": "NOT_AUTHORIZED_NOT_RUN"}


def exit_code(report: dict[str, Any]) -> int:
    if report.get("status") == "HISTORICAL_PASS":
        return 0
    if str(report.get("status", "")).startswith("HISTORICAL_BLOCKED"):
        return 42
    return 1


def historical_pins(repo_root: Path, *, r9_train: Path | None) -> tuple[EvidencePin, ...]:
    """Bind existing exact historical producer pins. Not the fresh R14 lane."""
    from successor.experiments import build_v10r3r10_vera_voice as r10
    from successor.experiments import build_v10r3r11_vera_voice as r11
    from successor.experiments import build_r12_voice_corrective as r12
    from successor.experiments import build_r13_behavior_corrective as r13

    binding_path = repo_root / "successor/experiments/V10R3R13_BEHAVIOR_CORPUS_BINDING_20261016_V1.json"
    # A frozen source-controlled binding must itself still have exact bytes.
    # If it drifts, never use its altered hashes to redefine admission.
    if _sha_file(binding_path) != R13_BINDING_SHA256:
        raise ValueError("R13 frozen binding source mismatch")
    r13_binding = json.loads(binding_path.read_text(encoding="utf8"))
    return (
        EvidencePin("R9_CORPUS", Path(r9_train) if r9_train else None,
                    r10.R9_TRAIN_SHA256, rows=50000),
        EvidencePin("R9_SPEC", repo_root / r10.R9_SPEC_REL, r10.R9_SPEC_SHA256),
        EvidencePin("R9_ADAPTER", r10.R9_ADAPTER / "adapter_model.safetensors",
                    r10.R9_ADAPTER_SHA256),
        EvidencePin("R9_RECEIPT", r10.R9_RECEIPT,
                    receipt_self_sha256=r10.R9_RECEIPT_SHA256,
                    fields=(("status", "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"),
                            ("weight_digest_changed", True),
                            ("cumulative_optimizer_steps", 28),
                            ("spec_sha256", r10.R9_SPEC_SHA256))),
        EvidencePin("R10_CORPUS", r11.R10_CORPUS, r11.R10_CORPUS_SHA, rows=50000),
        EvidencePin("R10_SPEC", repo_root / r11.R10_SPEC_REL,
                    r11.R10_SPEC_SHA, crlf_historical=True),
        EvidencePin("R10_ADAPTER", r11.R10_ADAPTER / "adapter_model.safetensors",
                    r11.R10_ADAPTER_SHA),
        EvidencePin("R10_RECEIPT", r11.R10_RECEIPT,
                    receipt_self_sha256=r11.R10_RECEIPT_SELF_SHA,
                    fields=(("status", "DEVELOPMENT_OPTIMIZER_STEP_COMPLETE"),
                            ("weight_digest_changed", True),
                            ("cumulative_optimizer_steps", 36),
                            ("spec_sha256", r11.R10_SPEC_SHA),
                            ("train_sha256", r11.R10_CORPUS_SHA))),
        EvidencePin("R11_CORPUS", r12.SOURCE, r12.SOURCE_SHA, rows=50000),
        EvidencePin("R11_SPEC", repo_root / r12.R11_SPEC_REL, r12.R11_SPEC_SHA),
        EvidencePin("R11_ADAPTER", r12.R11_PATH / "adapter/adapter_model.safetensors",
                    r12.R11_SHA),
        EvidencePin("R11_RECEIPT", r12.R11_PATH / "DEV_TRAINING_COMPLETE.json",
                    receipt_self_sha256=r12.R11_RECEIPT_SELF,
                    fields=(("cumulative_optimizer_steps", 44),
                            ("weight_digest_changed", True))),
        EvidencePin("R12_CORPUS", r13.R12_CORPUS, r13.R12_CORPUS_SHA, rows=50000),
        EvidencePin("R12_SPEC", repo_root / r13.R12_SPEC_REL, r13.R12_SPEC_SHA),
        EvidencePin("R12_ADAPTER", r13.R12_ADAPTER / "adapter/adapter_model.safetensors",
                    r13.R12_ADAPTER_SHA),
        EvidencePin("R12_RECEIPT", r13.R12_ADAPTER / "DEV_TRAINING_COMPLETE.json",
                    receipt_self_sha256=r13.R12_RECEIPT_SHA,
                    fields=(("cumulative_optimizer_steps", 52),
                            ("adapter_precision_final.tensor_dtype", "float32"),
                            ("deployment_status", "NOT_DEPLOYED"))),
        EvidencePin("HELDOUT", r13.HOLDOUT, r13.HOLDOUT_SHA,
                    rows=25, kind="heldout"),
        EvidencePin("R13_SOURCE", r13.SOURCE, r13.SOURCE_SHA,
                    rows=128, kind="r13_source"),
        EvidencePin("R13_SPEC", r13.SPEC,
                    r13_binding["new_spec_sha256"]),
        EvidencePin("R13_CORPUS", r13.OUT_DIR / "train.jsonl",
                    r13_binding["new_corpus_sha256"], rows=50000),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r9-train", type=Path,
                        help="Existing owner-held R9 corpus (required to PASS)")
    args = parser.parse_args(argv)
    if os.name != "nt":
        report = {"status": "HISTORICAL_BLOCKED_UNSUPPORTED_HOST",
                  "checks": [{"id": "HOST", "status": "BLOCKED_UNSUPPORTED_HOST"}],
                  "scope": "SELECTED_HISTORICAL_ARTIFACT_PINS_ONLY",
                  "training_qualified": False,
                  "protected_plaintext_opened": False,
                  "protected_semantic_qualification": "NOT_AUTHORIZED_NOT_RUN"}
    else:
        try:
            report = evaluate(historical_pins(
                Path(__file__).resolve().parents[2], r9_train=args.r9_train
            ))
        except (OSError, ValueError, KeyError, UnicodeError):
            # Even source-binding failures must not disclose exception text.
            report = {"status": "HISTORICAL_FAIL",
                      "checks": [{"id": "SOURCE_BINDING", "status": "FAIL_SOURCE_BINDING"}],
                      "scope": "SELECTED_HISTORICAL_ARTIFACT_PINS_ONLY",
                      "training_qualified": False,
                      "protected_plaintext_opened": False,
                      "protected_semantic_qualification": "NOT_AUTHORIZED_NOT_RUN"}
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
