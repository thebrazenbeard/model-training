from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from successor.v3_readiness_verifier import verify_local_v3_readiness


V3_TRAINING_SCHEMA = "VERA_SUCCESSOR_V3_TRAINING_CONFIG_V1"
TASK9_READY_SCHEMA = "VERA_SUCCESSOR_V3_TRAINING_READY_RECEIPT_V1"
TRAINING_AUTH_SCHEMA = "VERA_SUCCESSOR_V3_TRAINING_AUTHORIZATION_V1"

_CANONICAL_V3_SPEC_REL = "successor/v3_training_config.json"
_CANONICAL_AUTHORITY_BUS_REMOTE = "https://github.com/thebrazenbeard/chat-communication-bus"
_CANONICAL_AUTHORITY_BUS_BRANCH_REF = "origin/bus/vera-v2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class V3TrainingPreflight:
    runnable: bool
    status: str
    code_commit: str
    parent_adapter_sha256: str | None
    parent_adapter_config_sha256: str | None
    parent_candidate_subject_digest: str | None
    base_tree_sha256: str | None
    base_inventory_sha256: str | None
    train_sha256: str | None
    validation_sha256: str | None
    corpus_manifest_sha256: str | None
    task9_ready_receipt_sha256: str | None
    training_authorization_receipt_sha256: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_show_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
        stderr=subprocess.STDOUT,
    )


def _read_committed_v3_spec(repo_root: Path, commit: str) -> dict[str, Any]:
    raw = _git_show_bytes(repo_root, commit, _CANONICAL_V3_SPEC_REL)
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("committed V3 training config must be a JSON object")
    return value


def _normalize_github_remote(value: str) -> str:
    remote = value.strip()
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote[len("git@github.com:"):]
    if remote.endswith(".git"):
        remote = remote[:-4]
    return remote.rstrip("/").lower()


def _authority_bus_remote_is_canonical(repo_root: Path) -> bool:
    try:
        remote = subprocess.check_output(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return _normalize_github_remote(remote) == _normalize_github_remote(
        _CANONICAL_AUTHORITY_BUS_REMOTE
    )


def _refresh_authority_bus(repo_root: Path) -> None:
    subprocess.check_call(
        ["git", "-C", str(repo_root), "fetch", "--quiet", "origin", "bus/vera-v2"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _verify_task9_external(
    spec: dict[str, Any],
    task9_receipt: dict[str, Any],
    reasons: list[str],
) -> str | None:
    task9_repo = Path(spec["task9_repo_root"])
    if not task9_repo.is_dir():
        reasons.append("task9_repo_root_missing")
        return None
    try:
        observed_task9_head = _git_head(task9_repo)
    except (OSError, subprocess.CalledProcessError):
        reasons.append("task9_repo_head_unreadable")
        return None
    if observed_task9_head != spec["task9_source_commit"]:
        reasons.append("task9_repo_head_mismatch")

    receipt_fields = (
        ("task9_source_review_receipt_path", "task9_source_review_receipt_sha256"),
        ("task9_behavior_review_receipt_path", "task9_behavior_review_receipt_sha256"),
        ("task9_radical_registration_receipt_path", "task9_radical_registration_receipt_sha256"),
        ("task9_pragmatic_registration_receipt_path", "task9_pragmatic_registration_receipt_sha256"),
    )
    observed_paths: dict[str, Path] = {}
    for path_field, sha_field in receipt_fields:
        path = Path(spec[path_field])
        observed_paths[path_field] = path
        if not path.is_file():
            reasons.append(f"task9_external_receipt_missing:{path_field}")
            continue
        if _sha256_file(path) != spec[sha_field]:
            reasons.append(f"task9_external_receipt_sha256_mismatch:{path_field}")

    if any(reason.startswith("task9_external_receipt_") for reason in reasons):
        return None

    try:
        verification = verify_local_v3_readiness(
            repo_root=task9_repo,
            readiness_dir=spec["task9_readiness_dir"],
            blind_dir=spec["task9_blind_dir"],
            training_lane_key=spec["task9_training_lane_key"],
            vera_lab_source_review_receipt=observed_paths["task9_source_review_receipt_path"],
            vera_lab_behavior_review_receipt=observed_paths["task9_behavior_review_receipt_path"],
            radical_registration_receipt=observed_paths["task9_radical_registration_receipt_path"],
            pragmatic_registration_receipt=observed_paths["task9_pragmatic_registration_receipt_path"],
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        reasons.append("task9_external_verification_error:" + type(exc).__name__)
        return None

    if not verification.verified or verification.status != "VERIFIED":
        reasons.append("task9_external_verification_not_verified")
        reasons.extend(
            f"task9_external:{reason}" for reason in verification.reasons
        )
    if task9_receipt.get("external_verification_digest") != verification.evidence_digest:
        reasons.append("task9_external_verification_digest_mismatch")
    return verification.evidence_digest


def _verify_training_authority_bus(
    spec: dict[str, Any],
    authority: dict[str, Any],
    *,
    code_commit: str,
    task9_ready_receipt_sha256: str | None,
    reasons: list[str],
) -> None:
    if authority.get("training_code_commit") != code_commit:
        reasons.append("training_authorization_code_commit_mismatch")
    if authority.get("resource_execution") != "LOCAL_ONLY":
        reasons.append("training_authorization_resource_scope_mismatch")
    if authority.get("paid_compute") != "NOT_AUTHORIZED":
        reasons.append("training_authorization_paid_compute_mismatch")
    if authority.get("replay_policy") != "EXACT_SUBJECT_NO_OUTPUT_NAMESPACE_REUSE":
        reasons.append("training_authorization_replay_policy_mismatch")
    if authority.get("excluded_effects") != "MERGE|DEPLOY|RUNTIME_ACTIVATION|SD1_INSTALL|PROVIDER_CREDENTIAL_MUTATION|CANDIDATE_PROMOTION":
        reasons.append("training_authorization_excluded_effects_mismatch")

    bus_commit = authority.get("authority_bus_commit")
    bus_path = authority.get("authority_bus_path")
    bus_file_sha = authority.get("authority_bus_file_sha256")
    user_instruction_sha = authority.get("user_instruction_sha256")
    if not _is_commit(bus_commit):
        reasons.append("training_authority_bus_commit_invalid")
    if not isinstance(bus_path, str) or not bus_path.strip():
        reasons.append("training_authority_bus_path_missing")
    if not _is_sha256(bus_file_sha):
        reasons.append("training_authority_bus_file_sha256_invalid")
    if not _is_sha256(user_instruction_sha):
        reasons.append("training_authority_user_instruction_sha256_invalid")
    if any(
        reason in {
            "training_authority_bus_commit_invalid",
            "training_authority_bus_path_missing",
            "training_authority_bus_file_sha256_invalid",
            "training_authority_user_instruction_sha256_invalid",
        }
        for reason in reasons
    ):
        return

    bus_repo = Path(spec["training_authority_bus_repo_path"])
    branch_ref = spec["training_authority_bus_branch_ref"]
    if not bus_repo.is_dir():
        reasons.append("training_authority_bus_repo_missing")
        return
    if branch_ref != _CANONICAL_AUTHORITY_BUS_BRANCH_REF:
        reasons.append("training_authority_bus_branch_not_canonical")
        return
    if not _authority_bus_remote_is_canonical(bus_repo):
        reasons.append("training_authority_bus_not_canonical")
        return
    try:
        _refresh_authority_bus(bus_repo)
    except (OSError, subprocess.CalledProcessError):
        reasons.append("training_authority_bus_refresh_failed")
        return
    if not _is_ancestor(bus_repo, bus_commit, _CANONICAL_AUTHORITY_BUS_BRANCH_REF):
        reasons.append("training_authority_bus_commit_not_on_fresh_branch")
        return
    try:
        payload = _git_show_bytes(bus_repo, bus_commit, bus_path)
    except (OSError, subprocess.CalledProcessError):
        reasons.append("training_authority_bus_receipt_unreadable")
        return
    if _sha256_bytes(payload) != bus_file_sha:
        reasons.append("training_authority_bus_file_sha256_mismatch")
        return
    if authority.get("authority_ref") != f"bus:{bus_commit}:{bus_path}":
        reasons.append("training_authority_ref_not_bound_to_bus_receipt")

    text = payload.decode("utf-8")
    expected = {
        "authority_kind": "PATRICK_EXPLICIT",
        "effect": "WEIGHT_CHANGING_TRAINING",
        "training_code_commit": code_commit,
        "run_id": spec["run_id"],
        "task9_source_commit": spec["task9_source_commit"],
        "task9_ready_receipt_sha256": task9_ready_receipt_sha256,
        "parent_adapter_sha256": spec["parent_adapter_sha256"],
        "parent_adapter_config_sha256": spec["parent_adapter_config_sha256"],
        "parent_candidate_subject_digest": spec["parent_candidate_subject_digest"],
        "corpus_manifest_sha256": spec["corpus_manifest_sha256"],
        "train_sha256": spec["train_sha256"],
        "validation_sha256": spec["validation_sha256"],
        "user_instruction_sha256": user_instruction_sha,
        "resource_execution": "LOCAL_ONLY",
        "paid_compute": "NOT_AUTHORIZED",
        "replay_policy": "EXACT_SUBJECT_NO_OUTPUT_NAMESPACE_REUSE",
        "excluded_effects": "MERGE|DEPLOY|RUNTIME_ACTIVATION|SD1_INSTALL|PROVIDER_CREDENTIAL_MUTATION|CANDIDATE_PROMOTION",
    }
    for key, value in expected.items():
        marker = f"AUTHORITY_FIELD {key}={value}"
        if marker not in text:
            reasons.append(f"training_authority_bus_binding_mismatch:{key}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_COMMIT_RE.fullmatch(value))


def validate_v3_training_config(spec: dict[str, Any], base_manifest: dict[str, Any]) -> None:
    required = {
        "schema",
        "run_id",
        "task9_source_commit",
        "task9_ready_receipt_path",
        "task9_ready_receipt_sha256",
        "task9_repo_root",
        "task9_readiness_dir",
        "task9_blind_dir",
        "task9_training_lane_key",
        "task9_external_verifier_source_commit",
        "task9_source_review_receipt_path",
        "task9_source_review_receipt_sha256",
        "task9_behavior_review_receipt_path",
        "task9_behavior_review_receipt_sha256",
        "task9_radical_registration_receipt_path",
        "task9_radical_registration_receipt_sha256",
        "task9_pragmatic_registration_receipt_path",
        "task9_pragmatic_registration_receipt_sha256",
        "training_authorization_receipt_path",
        "training_authority_bus_repo_path",
        "training_authority_bus_branch_ref",
        "base_repo_id",
        "base_revision",
        "base_tree_sha256",
        "base_inventory_sha256",
        "parent_adapter_path",
        "parent_adapter_sha256",
        "parent_adapter_config_sha256",
        "parent_candidate_subject_digest",
        "parent_selection_basis",
        "train_path",
        "train_sha256",
        "validation_path",
        "validation_sha256",
        "corpus_manifest_path",
        "corpus_manifest_sha256",
        "output_dir",
        "seed",
        "learning_rate",
        "gradient_accumulation",
        "assistant_only_loss",
        "neutral_system_override",
        "general_rehearsal_fraction",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError("missing V3 training fields: " + ", ".join(missing))
    if spec["schema"] != V3_TRAINING_SCHEMA:
        raise ValueError("unsupported V3 training schema")
    run_id = spec["run_id"]
    if not isinstance(run_id, str) or not run_id.startswith("VERA_SUCCESSOR_V3_DEV_"):
        raise ValueError("run_id must be a new V3 development run id")
    if base_manifest.get("mutable_revision_allowed") is not False:
        raise ValueError("base manifest must forbid mutable revisions")
    if spec["base_repo_id"] != base_manifest.get("repo_id"):
        raise ValueError("base repository does not match manifest")
    if spec["base_revision"] != base_manifest.get("revision"):
        raise ValueError("base revision does not match manifest")
    for field in ("task9_source_commit", "task9_external_verifier_source_commit"):
        if not _is_commit(spec[field]):
            raise ValueError(f"{field} must be exact 40-hex commit")
    for field in (
        "task9_ready_receipt_sha256",
        "task9_source_review_receipt_sha256",
        "task9_behavior_review_receipt_sha256",
        "task9_radical_registration_receipt_sha256",
        "task9_pragmatic_registration_receipt_sha256",
        "parent_adapter_sha256",
        "parent_adapter_config_sha256",
        "parent_candidate_subject_digest",
        "base_tree_sha256",
        "base_inventory_sha256",
        "train_sha256",
        "validation_sha256",
        "corpus_manifest_sha256",
    ):
        if not _is_sha256(spec[field]):
            raise ValueError(f"{field} must be exact SHA-256")
    for field in (
        "task9_ready_receipt_path",
        "task9_repo_root",
        "task9_readiness_dir",
        "task9_blind_dir",
        "task9_training_lane_key",
        "task9_source_review_receipt_path",
        "task9_behavior_review_receipt_path",
        "task9_radical_registration_receipt_path",
        "task9_pragmatic_registration_receipt_path",
        "training_authorization_receipt_path",
        "training_authority_bus_repo_path",
        "training_authority_bus_branch_ref",
        "parent_adapter_path",
        "train_path",
        "validation_path",
        "corpus_manifest_path",
        "output_dir",
        "parent_selection_basis",
    ):
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ValueError(f"{field} must be non-empty")
    if spec["assistant_only_loss"] is not True:
        raise ValueError("assistant_only_loss must be true")
    if spec["neutral_system_override"] is not True:
        raise ValueError("neutral_system_override must be true")
    fraction = spec["general_rehearsal_fraction"]
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0.50 <= float(fraction) <= 1.0
    ):
        raise ValueError("general_rehearsal_fraction must be between 0.50 and 1.0")
    if not isinstance(spec["seed"], int) or isinstance(spec["seed"], bool):
        raise ValueError("seed must be a deterministic integer")
    if float(spec["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if int(spec["gradient_accumulation"]) < 1:
        raise ValueError("gradient_accumulation must be positive")


def preflight_v3_training(
    spec_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> V3TrainingPreflight:
    spec_path = Path(spec_path)
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    base_manifest = _read_json(repo / "successor" / "base_model_manifest.json")
    spec = _read_json(spec_path)
    validate_v3_training_config(spec, base_manifest)

    reasons: list[str] = []
    code_commit = _git_head(repo)
    canonical_spec_path = (repo / _CANONICAL_V3_SPEC_REL).resolve()
    if spec_path.resolve() != canonical_spec_path:
        reasons.append("training_spec_not_canonical")
    try:
        committed_spec = _read_committed_v3_spec(repo, code_commit)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError, subprocess.CalledProcessError):
        reasons.append("training_spec_git_readback_failed")
    else:
        if _canonical_json(committed_spec) != _canonical_json(spec):
            reasons.append("training_spec_not_committed")

    task9_source_commit = spec["task9_source_commit"]
    if not _is_ancestor(repo, task9_source_commit, code_commit):
        reasons.append("task9_source_not_ancestor_of_training_code")
    if not _is_ancestor(
        repo, spec["task9_external_verifier_source_commit"], code_commit
    ):
        reasons.append("task9_external_verifier_not_ancestor_of_training_code")

    output_dir = Path(spec["output_dir"])
    if output_dir.exists():
        reasons.append("output_dir_already_exists")

    parent_dir = Path(spec["parent_adapter_path"])
    parent_path = parent_dir / "adapter_model.safetensors"
    parent_config_path = parent_dir / "adapter_config.json"
    train_path = Path(spec["train_path"])
    validation_path = Path(spec["validation_path"])
    corpus_path = Path(spec["corpus_manifest_path"])

    observed_parent = _sha256_file(parent_path) if parent_path.is_file() else None
    observed_parent_config = (
        _sha256_file(parent_config_path) if parent_config_path.is_file() else None
    )
    observed_base_tree = base_manifest.get("observed_local_cache_tree_sha256")
    observed_base_inventory = base_manifest.get("file_inventory_sha256")
    observed_train = _sha256_file(train_path) if train_path.is_file() else None
    observed_validation = _sha256_file(validation_path) if validation_path.is_file() else None
    observed_corpus = _sha256_file(corpus_path) if corpus_path.is_file() else None

    if observed_parent is None:
        reasons.append("parent_adapter_missing")
    elif observed_parent != spec["parent_adapter_sha256"]:
        reasons.append("parent_adapter_sha256_mismatch")
    if observed_parent_config is None:
        reasons.append("parent_adapter_config_missing")
    elif observed_parent_config != spec["parent_adapter_config_sha256"]:
        reasons.append("parent_adapter_config_sha256_mismatch")
    if observed_base_tree != spec["base_tree_sha256"]:
        reasons.append("base_tree_sha256_mismatch")
    if observed_base_inventory != spec["base_inventory_sha256"]:
        reasons.append("base_inventory_sha256_mismatch")

    observed_parent_subject = None
    if (
        observed_parent is not None
        and observed_parent_config is not None
        and isinstance(observed_base_tree, str)
        and isinstance(observed_base_inventory, str)
    ):
        observed_parent_subject = _sha256_json({
            "adapter_config_sha256": observed_parent_config,
            "adapter_sha256": observed_parent,
            "base_revision": spec["base_revision"],
            "base_tree_sha256": observed_base_tree,
            "base_inventory_sha256": observed_base_inventory,
        })
        if observed_parent_subject != spec["parent_candidate_subject_digest"]:
            reasons.append("parent_candidate_subject_digest_mismatch")

    if observed_train is None:
        reasons.append("train_file_missing")
    elif observed_train != spec["train_sha256"]:
        reasons.append("train_sha256_mismatch")
    if observed_validation is None:
        reasons.append("validation_file_missing")
    elif observed_validation != spec["validation_sha256"]:
        reasons.append("validation_sha256_mismatch")
    if observed_corpus is None:
        reasons.append("corpus_manifest_missing")
    elif observed_corpus != spec["corpus_manifest_sha256"]:
        reasons.append("corpus_manifest_sha256_mismatch")

    if observed_corpus is not None:
        corpus = _read_json(corpus_path)
        if corpus.get("readiness_source_commit") != task9_source_commit:
            reasons.append("corpus_task9_source_commit_mismatch")
        if corpus.get("train_file_sha256") != observed_train:
            reasons.append("corpus_train_file_binding_mismatch")
        if corpus.get("validation_file_sha256") != observed_validation:
            reasons.append("corpus_validation_file_binding_mismatch")
        declared_general = corpus.get("general_fraction")
        if (
            not isinstance(declared_general, (int, float))
            or isinstance(declared_general, bool)
            or float(declared_general) < float(spec["general_rehearsal_fraction"])
        ):
            reasons.append("corpus_general_rehearsal_fraction_below_config")
        if corpus.get("train_validation_overlap") != 0:
            reasons.append("corpus_train_validation_overlap_nonzero")
        if corpus.get("blind_plaintext_included") is not False:
            reasons.append("blind_plaintext_included_in_training_corpus")
        if corpus.get("weight_change_performed") is not False:
            reasons.append("readiness_corpus_reports_weight_change")

    task9_path = Path(spec["task9_ready_receipt_path"])
    task9_sha = _sha256_file(task9_path) if task9_path.is_file() else None
    task9_receipt: dict[str, Any] = {}
    if task9_sha is None:
        reasons.append("task9_ready_receipt_missing")
    else:
        if task9_sha != spec["task9_ready_receipt_sha256"]:
            reasons.append("task9_ready_receipt_sha256_mismatch")
        task9_receipt = _read_json(task9_path)
        if task9_receipt.get("schema") != TASK9_READY_SCHEMA:
            reasons.append("task9_ready_receipt_schema_invalid")
        if task9_receipt.get("ready") is not True or task9_receipt.get("status") != "READY":
            reasons.append("task9_not_ready")
        if task9_receipt.get("readiness_source_commit") != task9_source_commit:
            reasons.append("task9_ready_receipt_source_mismatch")
        for field in ("subject_digest", "evidence_digest", "external_verification_digest"):
            if not _is_sha256(task9_receipt.get(field)):
                reasons.append(f"task9_ready_receipt_invalid:{field}")
        if (
            task9_receipt.get("external_verifier_commit")
            != spec["task9_external_verifier_source_commit"]
        ):
            reasons.append("task9_ready_receipt_verifier_commit_mismatch")
        _verify_task9_external(spec, task9_receipt, reasons)

    authority_path = Path(spec["training_authorization_receipt_path"])
    authority_sha = _sha256_file(authority_path) if authority_path.is_file() else None
    if authority_sha is None:
        reasons.append("training_authorization_receipt_missing")
    else:
        authority = _read_json(authority_path)
        if authority.get("schema") != TRAINING_AUTH_SCHEMA:
            reasons.append("training_authorization_schema_invalid")
        if authority.get("authorized") is not True or authority.get("effect") != "WEIGHT_CHANGING_TRAINING":
            reasons.append("training_not_authorized")
        if authority.get("authority_kind") != "PATRICK_EXPLICIT":
            reasons.append("training_authority_not_explicit")
        authority_ref = authority.get("authority_ref")
        if not isinstance(authority_ref, str) or not authority_ref.strip():
            reasons.append("training_authority_ref_missing")
        if authority.get("task9_ready_receipt_sha256") != task9_sha:
            reasons.append("training_authorization_task9_receipt_mismatch")
        if authority.get("task9_source_commit") != task9_source_commit:
            reasons.append("training_authorization_source_mismatch")
        if authority.get("run_id") != spec["run_id"]:
            reasons.append("training_authorization_run_id_mismatch")
        if authority.get("parent_adapter_sha256") != spec["parent_adapter_sha256"]:
            reasons.append("training_authorization_parent_mismatch")
        if authority.get("parent_adapter_config_sha256") != spec["parent_adapter_config_sha256"]:
            reasons.append("training_authorization_parent_config_mismatch")
        if authority.get("parent_candidate_subject_digest") != spec["parent_candidate_subject_digest"]:
            reasons.append("training_authorization_parent_subject_mismatch")
        if authority.get("corpus_manifest_sha256") != spec["corpus_manifest_sha256"]:
            reasons.append("training_authorization_corpus_mismatch")
        if authority.get("train_sha256") != spec["train_sha256"]:
            reasons.append("training_authorization_train_mismatch")
        if authority.get("validation_sha256") != spec["validation_sha256"]:
            reasons.append("training_authorization_validation_mismatch")
        _verify_training_authority_bus(
            spec,
            authority,
            code_commit=code_commit,
            task9_ready_receipt_sha256=task9_sha,
            reasons=reasons,
        )

    reasons = list(dict.fromkeys(reasons))
    if not reasons:
        status = "RUNNABLE"
        runnable = True
    elif all(
        reason in {"task9_ready_receipt_missing", "training_authorization_receipt_missing"}
        for reason in reasons
    ):
        status = "HOLD"
        runnable = False
    else:
        status = "BLOCKED"
        runnable = False

    return V3TrainingPreflight(
        runnable=runnable,
        status=status,
        code_commit=code_commit,
        parent_adapter_sha256=observed_parent,
        parent_adapter_config_sha256=observed_parent_config,
        parent_candidate_subject_digest=observed_parent_subject,
        base_tree_sha256=observed_base_tree if isinstance(observed_base_tree, str) else None,
        base_inventory_sha256=(
            observed_base_inventory if isinstance(observed_base_inventory, str) else None
        ),
        train_sha256=observed_train,
        validation_sha256=observed_validation,
        corpus_manifest_sha256=observed_corpus,
        task9_ready_receipt_sha256=task9_sha,
        training_authorization_receipt_sha256=authority_sha,
        reasons=tuple(reasons),
    )
