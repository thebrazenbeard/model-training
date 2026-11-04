"""Source-only, immutable Contract-B public-fit request for TRAININGDAY.

This module creates a candidate request for a future *host-owned* broker. It
cannot sign a grant, launch a worker, load weights, use CUDA, or mutate Q6.
The existing physical_launch_authority/lowmem_experiment HARD HOLD remains.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


SCHEMA = "TRAININGDAY_TIER1_PUBLIC_FIT_PLAN_V1"
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
KERNEL_RELATIVE = "trainingday/lowmem_experiment.py"
MAX_KERNEL_BYTES = 512 * 1024
MAX_SECONDS = 30 * 60
MAX_VRAM_MIB = 3072
MIN_DRIVER_HEADROOM_MIB = 256
SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")

# Independently published official public-base bytes; the older repository
# artifact for rodrigomt/*Uncensored-Aggressive* is explicitly NOT this model.
OFFICIAL_FILE_SHA256 = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "model.safetensors-00001-of-00002.safetensors":
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors":
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}


class PublicFitPlanError(ValueError):
    """Candidate plan is malformed or source identity is unavailable."""


def _canonical(data: dict) -> bytes:
    """Unambiguous ASCII JSON, independent of insertion order or whitespace."""
    if type(data) is not dict:
        raise PublicFitPlanError("plan must be a JSON object")
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_public_fit_plan(*, executable_sha256: str) -> dict:
    """Build an unpromotable request; a caller-supplied digest grants nothing.

    An independent installed Windows broker must bind the signed plan hash to
    its independently measured kernel bytes, official model byte hashes and
    host-owned process identity. Source-only output is not authorization.
    """
    if (type(executable_sha256) is not str
            or SHA256_HEX.fullmatch(executable_sha256) is None):
        raise PublicFitPlanError("kernel SHA256 must be exact lowercase hex")
    # Immutable allowlist bytes, NOT a hash declaration selected by a caller.
    official_manifest_sha256 = _sha(_canonical(OFFICIAL_FILE_SHA256))
    return {
        "schema": SCHEMA,
        "tier": "TIER1_PUBLIC_SYNTHETIC",
        "host": "LAPPY",
        "policy": "WDDM_BOUNDED_COEXISTENCE",
        "foreign_wddm_clients_permitted": True,
        "gpu_physical_exclusivity_claimed": False,
        "max_seconds": MAX_SECONDS,
        "vram_target_mib": MAX_VRAM_MIB,
        "vram_limit_semantics": "OBSERVATIONAL_TARGET_NOT_HARD_WDDM_QUOTA",
        "memory_telemetry_missing_or_unsafe": "HOLD",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "official_public_file_sha256": dict(OFFICIAL_FILE_SHA256),
        "manifest_sha256": official_manifest_sha256,
        "remaining_snapshot_files_require_separate_signed_binding": True,
        "executable_relative_path": KERNEL_RELATIVE,
        "executable_sha256": executable_sha256,
        "candidate_entrypoint": "trainingday.lowmem_experiment.one_step_kernel",
        "one_optimizer_step_max": 1,
        "synthetic_input_only": True,
        "batch": 1,
        "sequence_length": 64,
        "pretrained_weights_read_only": True,
        "weights_download": "VN1_OR_VN2_ON_DEMAND_ONLY_AFTER_ADMITTED_JOB",
        "lappy_permanent_weights": False,
        "protected_r201_access": False,
        "protected_dev_access": False,
        "protected_access": "NONE",
        "nas_q6_write": False,
        "checkpoint_promotion": False,
        "outputs": "UNPROMOTED_ONLY",
        "output_class": "TIER1_FIT_OBSERVATION",
        "broker_owner_signed_grant_required": True,
        "broker_host_local_replay_epoch_required": True,
        "linux_worker_process_group_or_cgroup_required": True,
        "targeted_wsl_orphan_kill_and_readback_required": True,
        "wsl_cleanup": "TARGETED_WSL_PIDTREE_REQUIRED",
        "blanket_wsl_distro_termination_allowed": False,
        "linux_orphan_cleanup_qualified": False,
        "launch_path_implemented": False,
        "status": "SOURCE_ONLY_CANDIDATE_NOT_AUTHORIZED_TO_LAUNCH",
    }


def candidate_receipt(*, executable_sha256: str) -> dict:
    """Produce a reproducible broker input digest, never a launch permit."""
    plan = build_public_fit_plan(executable_sha256=executable_sha256)
    canonical = _canonical(plan)
    return {
        "plan": plan,
        "plan_sha256": _sha(canonical),
        "plan_canonical_ascii_bytes": len(canonical),
        "status": "SOURCE_ONLY_NO_HOST_OR_GPU_EFFECT",
        "authorized": False,
    }


def assess_observed_vram(*, total_mib: int, free_mib: int,
                         telemetry_complete: bool) -> dict:
    """Screen one WDDM observation; a PASS is never a GPU lease.

    Other Windows clients can allocate memory after this observation. The
    broker must keep monitoring and abort on unsafe/unknown device telemetry.
    """
    if (type(total_mib) is not int or type(free_mib) is not int
            or type(telemetry_complete) is not bool
            or total_mib <= 0 or free_mib < 0 or free_mib > total_mib):
        raise PublicFitPlanError("invalid device-wide WDDM memory observation")
    if not telemetry_complete:
        status = "HOLD_TELEMETRY_INCOMPLETE"
    elif free_mib < MAX_VRAM_MIB + MIN_DRIVER_HEADROOM_MIB:
        status = "HOLD_INSUFFICIENT_OBSERVED_FREE_VRAM"
    else:
        status = "RESOURCE_PREFLIGHT_CANDIDATE_ONLY_NO_GPU_LEASE"
    return {
        "status": status,
        "total_mib": total_mib,
        "free_mib": free_mib,
        "target_mib": MAX_VRAM_MIB,
        "driver_headroom_mib": MIN_DRIVER_HEADROOM_MIB,
        "wddm_foreign_clients_allowed": True,
        "physical_exclusivity_proved": False,
        "launch_authorized": False,
    }


def source_kernel_sha256() -> str:
    """Hash this checkout's fixed existing kernel, never an arbitrary path."""
    path = Path(__file__).with_name("lowmem_experiment.py")
    if path.is_symlink():
        raise PublicFitPlanError("candidate kernel symlink rejected")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_KERNEL_BYTES:
                raise PublicFitPlanError("candidate kernel not a bounded regular file")
            pieces = []
            total = 0
            while True:
                block = os.read(fd, 64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_KERNEL_BYTES:
                    raise PublicFitPlanError("candidate kernel too large")
                pieces.append(block)
        finally:
            os.close(fd)
    except OSError as exc:
        raise PublicFitPlanError("candidate kernel unavailable") from exc
    if not pieces:
        raise PublicFitPlanError("empty candidate kernel")
    return _sha(b"".join(pieces))


def main() -> int:
    """Read-only source receipt; no executable launch switch exists."""
    if len(sys.argv) != 1:
        raise SystemExit("TRAININGDAY Tier1 plan CLI accepts no launch or override flags")
    print(json.dumps(candidate_receipt(executable_sha256=source_kernel_sha256()),
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
