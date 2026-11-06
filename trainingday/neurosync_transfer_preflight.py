"""Source-only compatibility preflight for TRAININGDAY -> Neuro-Sync.

This module checks *declared* manifest fields. It does not inspect the model
artifacts, authenticate rights, execute tests, authorize a checkpoint write,
or provide a path around TRAININGDAY's physical-launch HARD HOLD.

The repository SHAs in the manifest identify the source revisions the author
claims to use; only an independent fetch/verification can establish that the
actual checked-out sources or artifacts match those values. In particular,
success here is never a TRAININGDAY launch or StateBus commit permit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any


SCHEMA = "trainingday.neurosync.transfer-preflight.v1"
WEIGHT_PATHWAYS = frozenset({"adapter_copy", "delta_transfer"})
PATHWAYS = WEIGHT_PATHWAYS | {"context_only"}
EVIDENCE_CLASSES = frozenset(
    {"PROPOSED", "SOURCE_INSPECTION", "TEST", "BUILD", "INSTALL",
     "RUNTIME_PROBE", "FORMAL_VERIFY"}
)
MANDATORY_CONTROLS = frozenset({"untouched", "context_only", "shuffled_donor"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_TARGET = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_SOURCE_KEYS = frozenset({"model_training_git_sha", "neuro_sync_git_sha"})
_DATASET_KEYS = frozenset(
    {"train_split_sha256", "eval_split_sha256", "rights_receipt_sha256"}
)
_MODEL_KEYS = frozenset(
    {"architecture", "base_revision", "base_sha256", "tokenizer_sha256",
     "chat_template_sha256", "adapter_targets", "adapter_rank",
     "adapter_alpha", "adapter_scaling_rule", "adapter_scaling", "format"}
)
_EVAL_KEYS = frozenset(
    {"donor_disconnected", "recipient_cold_restart", "control_arms",
     "training_evidence_class"}
)
_ROOT_KEYS = frozenset(
    {"schema", "source_revisions", "pathway", "donor", "recipient",
     "dataset", "evaluation"}
)


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """A purely descriptive result; both authority fields always remain false."""

    compatible: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def physical_launch_authorized(self) -> bool:
        return False

    @property
    def state_commit_authorized(self) -> bool:
        return False

    @property
    def status(self) -> str:
        return "SOURCE_COMPATIBLE_HOLD" if self.compatible else "INCOMPATIBLE_HOLD"

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable result, including the explicit authority holds."""
        return {
            "status": self.status,
            "compatible": self.compatible,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "physical_launch_authorized": False,
            "state_commit_authorized": False,
            "evidence_class": "SOURCE_INSPECTION",
            "rights_verified": False,
            "artifact_bytes_verified": False,
            "evaluation_executed_or_verified": False,
        }


def _schema_keys(
    value: object, *, path: str, required: frozenset[str],
    blockers: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        blockers.append(f"{path}:NOT_OBJECT")
        return None
    if not all(isinstance(key, str) for key in value):
        blockers.append(f"{path}:INVALID_KEY_TYPE")
    actual = set(value)
    for missing in sorted(required - actual):
        blockers.append(f"{path}.{missing}:MISSING")
    for unknown in sorted((actual - required), key=str):
        blockers.append(f"{path}.{unknown}:UNKNOWN_FIELD")
    return value


def _digest(
    container: Mapping[str, Any], key: str, path: str,
    blockers: list[str], *, git_sha: bool = False,
) -> str | None:
    value = container.get(key)
    pattern = _HEX40 if git_sha else _HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value):
        blockers.append(f"{path}.{key}:INVALID_{'GIT_SHA' if git_sha else 'SHA256'}")
        return None
    return value


def _model(
    raw: object, path: str, blockers: list[str],
) -> dict[str, Any] | None:
    model = _schema_keys(raw, path=path, required=_MODEL_KEYS, blockers=blockers)
    if model is None:
        return None
    for key in ("base_sha256", "tokenizer_sha256", "chat_template_sha256"):
        _digest(model, key, path, blockers)
    _digest(model, "base_revision", path, blockers, git_sha=True)

    architecture = model.get("architecture")
    if (not isinstance(architecture, str) or not architecture.strip()
            or len(architecture) > 160):
        blockers.append(f"{path}.architecture:INVALID")
    fmt = model.get("format")
    if fmt not in ("peft_adapter", "gguf") or not isinstance(fmt, str):
        blockers.append(f"{path}.format:UNSUPPORTED")

    raw_targets = model.get("adapter_targets")
    if (not isinstance(raw_targets, list) or not raw_targets
            or any(not isinstance(t, str) or not _TARGET.fullmatch(t)
                   or len(t) > 256 for t in raw_targets)):
        blockers.append(f"{path}.adapter_targets:INVALID")
    elif len(set(raw_targets)) != len(raw_targets):
        blockers.append(f"{path}.adapter_targets:DUPLICATE")
    elif raw_targets != sorted(raw_targets):
        # Stable canonical order prevents different manifests describing the
        # same target list without exposing the difference at review time.
        blockers.append(f"{path}.adapter_targets:NOT_SORTED")

    rank = model.get("adapter_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or rank > 65536:
        blockers.append(f"{path}.adapter_rank:INVALID")
        rank = None

    def positive_number(key: str) -> float | None:
        value = model.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            blockers.append(f"{path}.{key}:INVALID")
            return None
        try:
            num = float(value)
        except (OverflowError, ValueError):
            blockers.append(f"{path}.{key}:INVALID")
            return None
        if not math.isfinite(num) or num <= 0:
            blockers.append(f"{path}.{key}:INVALID")
            return None
        return num

    alpha = positive_number("adapter_alpha")
    scaling = positive_number("adapter_scaling")
    rule = model.get("adapter_scaling_rule")
    if rule not in ("alpha_over_rank", "alpha_over_sqrt_rank") or not isinstance(rule, str):
        blockers.append(f"{path}.adapter_scaling_rule:UNSUPPORTED")
    elif rank is not None and alpha is not None and scaling is not None:
        expected = alpha / (rank if rule == "alpha_over_rank" else math.sqrt(rank))
        if not math.isclose(scaling, expected, rel_tol=1e-12, abs_tol=0.0):
            blockers.append(f"{path}.adapter_scaling:INCONSISTENT_WITH_RULE")

    return dict(model)


def validate_transfer_manifest(manifest: Mapping[str, Any]) -> PreflightResult:
    """Check declared compatibility and retain a permanent source-only HOLD.

    A true ``compatible`` means only that the claimed source metadata is
    internally coherent. It does *not* establish weights, rights, training,
    independent model evaluation, persistence, donor disconnection, or any
    runtime/effect authority. The caller must never branch on ``compatible``
    to launch GPU work or commit state.
    """
    blockers: list[str] = []
    warnings: list[str] = [
        "SOURCE_ONLY_NO_ARTIFACT_BYTE_VERIFICATION",
        "SOURCE_GIT_SHAS_NOT_INDEPENDENTLY_VERIFIED",
        "RIGHTS_RECEIPT_NOT_AUTHENTICATED",
        "EVALUATION_ATTESTATIONS_UNVERIFIED",
        "CAUSAL_COMPARISON_UNVERIFIED",
        "TRAININGDAY_PHYSICAL_LAUNCH_HARD_HOLD",
        "PERSISTENT_STATE_COMMIT_HARD_HOLD",
    ]

    root = _schema_keys(manifest, path="manifest", required=_ROOT_KEYS,
                        blockers=blockers)
    if root is None:
        return PreflightResult(False, tuple(blockers), tuple(warnings))

    if root.get("schema") != SCHEMA or not isinstance(root.get("schema"), str):
        blockers.append("manifest.schema:UNSUPPORTED")
    pathway = root.get("pathway")
    if not isinstance(pathway, str) or pathway not in PATHWAYS:
        blockers.append("manifest.pathway:UNSUPPORTED")

    sources = _schema_keys(root.get("source_revisions"), path="source_revisions",
                           required=_SOURCE_KEYS, blockers=blockers)
    if sources is not None:
        for key in sorted(_SOURCE_KEYS):
            _digest(sources, key, "source_revisions", blockers, git_sha=True)

    dataset = _schema_keys(root.get("dataset"), path="dataset",
                           required=_DATASET_KEYS, blockers=blockers)
    if dataset is not None:
        hashes = {key: _digest(dataset, key, "dataset", blockers)
                  for key in sorted(_DATASET_KEYS)}
        if (hashes["train_split_sha256"] is not None
                and hashes["train_split_sha256"] == hashes["eval_split_sha256"]):
            blockers.append("dataset:TRAIN_EVAL_IDENTICAL_DIGEST")
        warnings.append("TRAIN_EVAL_DISJOINTNESS_NOT_PROVEN_BY_DIGESTS")

    donor = _model(root.get("donor"), "donor", blockers)
    recipient = _model(root.get("recipient"), "recipient", blockers)

    evaluation = _schema_keys(root.get("evaluation"), path="evaluation",
                              required=_EVAL_KEYS, blockers=blockers)
    if evaluation is not None:
        for key in ("donor_disconnected", "recipient_cold_restart"):
            if not isinstance(evaluation.get(key), bool):
                blockers.append(f"evaluation.{key}:NOT_BOOLEAN")
        arms = evaluation.get("control_arms")
        if (not isinstance(arms, list)
                or any(not isinstance(arm, str) or not arm or len(arm) > 80
                       for arm in arms)):
            blockers.append("evaluation.control_arms:INVALID")
        elif len(set(arms)) != len(arms):
            blockers.append("evaluation.control_arms:DUPLICATE")
        elif not MANDATORY_CONTROLS.issubset(arms):
            blockers.append("evaluation.control_arms:MISSING_REQUIRED_CONTROLS")
        evidence_class = evaluation.get("training_evidence_class")
        if (not isinstance(evidence_class, str)
                or evidence_class not in EVIDENCE_CLASSES):
            blockers.append("evaluation.training_evidence_class:UNRECOGNIZED")
        if evaluation.get("donor_disconnected") is False:
            if pathway in WEIGHT_PATHWAYS:
                blockers.append("evaluation.donor_disconnected:REQUIRED_FOR_WEIGHT_PATH")
            else:
                warnings.append("DONOR_DISCONNECT_NOT_REPORTED")
        if evaluation.get("recipient_cold_restart") is False:
            if pathway in WEIGHT_PATHWAYS:
                blockers.append("evaluation.recipient_cold_restart:REQUIRED_FOR_WEIGHT_PATH")
            else:
                warnings.append("RECIPIENT_COLD_RESTART_NOT_REPORTED")

    if donor is not None and recipient is not None and pathway in WEIGHT_PATHWAYS:
        for key in (
            "architecture", "base_revision", "base_sha256", "tokenizer_sha256",
            "chat_template_sha256", "adapter_targets", "adapter_rank",
            "adapter_alpha", "adapter_scaling_rule", "adapter_scaling",
        ):
            if donor.get(key) != recipient.get(key):
                blockers.append(f"compatibility.{key}:MISMATCH")
        if donor.get("format") != "peft_adapter":
            blockers.append("donor.format:WEIGHT_PATH_REQUIRES_PEFT_ADAPTER")
        if recipient.get("format") != "peft_adapter":
            blockers.append("recipient.format:WEIGHT_PATH_REQUIRES_PEFT_ADAPTER")
        warnings.append("SAME_BASE_MANIFEST_NOT_WEIGHT_TRANSFER_EVIDENCE")
    elif pathway == "context_only":
        warnings.append("CONTEXT_ONLY_IS_NOT_PERSISTENT_WEIGHT_TRANSFER")

    return PreflightResult(
        compatible=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["SCHEMA", "PreflightResult", "validate_transfer_manifest"]
