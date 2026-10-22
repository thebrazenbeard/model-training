"""Reproduce and freeze the EXACT native Mono/Runner producer boundary suite.

Research only: authentic provider issuer/route/model and cross-root nonce
authority remain ABSENT. No provider, backend, memory or protected effect.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

MONO = "8fd64c97998a8d24943439a96409f2114646905e"
PORTAL = "6ef585f36194a538a01b751d0802a9e876815f0a"
PARENT = "b0f5b8f2c8e4ca050b409be591b7b76843a52c9d"
TEST_FILES = [
    "test_mono_runner_untrusted_producer_v1.py",
    "test_mono_runner_claim_crash_v1.py",
    "test_mono_runner_producer_claim_boundary_v1.py",
    "test_mono_runner_post_claim_task_drift_v1.py",
]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, timeout=12
    ).strip()


def evaluate() -> dict:
    root = Path(__file__).resolve().parent
    mono = Path(os.environ["TRAINING_MONO_ROOT"]).resolve()
    portal = Path(os.environ["TRAINING_PORTAL_ROOT"]).resolve()
    observed = {"training": git(root, "rev-parse", "HEAD"),
                "mono": git(mono, "rev-parse", "HEAD"),
                "portal": git(portal, "rev-parse", "HEAD")}
    if observed != {"training": PARENT, "mono": MONO, "portal": PORTAL}:
        raise RuntimeError("EXACT_SOURCE_HEAD_MISMATCH:" + repr(observed))
    env = dict(os.environ)
    roots = [str(root), str(portal)]
    roots.extend(str(p) for p in sorted((mono / "packages").glob("*/src")))
    env["PYTHONPATH"] = os.pathsep.join(roots)
    paths = [str(root / "tests" / name) for name in TEST_FILES]
    cmd = [sys.executable, "-B", "-m", "pytest", "-q",
           "-p", "no:cacheprovider", *paths]
    task = subprocess.run(
        cmd, cwd=root, env=env, text=True, capture_output=True,
        timeout=75, check=False,
    )
    output = task.stdout + task.stderr
    summary = re.search(r"(\d+) passed(?:,\s*(\d+) skipped)?", output)
    passed = int(summary.group(1)) if summary else 0
    skipped = int(summary.group(2) or 0) if summary else 0
    failed = int(re.search(r"(\d+) failed", output).group(1)) \
        if re.search(r"(\d+) failed", output) else 0
    if task.returncode != 0 or passed != 13 or skipped or failed:
        raise RuntimeError("SOURCE_SUITE_FAILED_OR_SKIPPED:" + output[-6500:])
    evidence_dir = root / "docs" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    log = evidence_dir / "TRAINING_MONO_RUNNER_PRODUCER_NATIVE_20261018.log"
    log.write_text(output, encoding="utf-8")
    test_blobs = {name: git(root, "hash-object", "tests/" + name)
                  for name in TEST_FILES}
    return {
        "schema": "TRAINING_MONO_RUNNER_NATIVE_SOURCE_EXECUTION_V1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "LAPPY_REAL_WINDOWS_PYTHON_DISPOSABLE_SQLITE",
        "source_heads": observed,
        "source_test_git_blobs": test_blobs,
        "command": cmd,
        "command_exit": task.returncode,
        "native_pytest": {"passed": passed, "failed": failed,
                          "skipped": skipped,
                          "warnings": "PYTEST_ASYNCIO_DEPRECATION"},
        "raw_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "raw_log_file": str(log.relative_to(root)).replace("\\", "/"),
        "cases": {
            "literal_unapproved_marker_rejected_before_local_write": True,
            "unmarked_forged_caller_claim_locally_accepted": True,
            "own_task_A_stale_before_local_claim_still_local_admit": True,
            "unrelated_B_only_changes_global_journal": True,
            "split_runner_roots_locally_accept_same_unverified_nonce": True,
            "forged_task_source_issuer_model_route_untrusted": True,
            "cold_before_claim_leaves_no_attempt": True,
            "cold_after_claim_preserves_unknown_unresolved_attempt": True,
            "post_claim_cold_mono_A_checkpoint_leaves_runner_attempt": True,
            "post_claim_local_replay_rejected": True,
            "post_claim_split_root_still_locally_claims": True,
        },
        "real_receiver_issuer_authenticated": False,
        "process_loaded_model_verified": False,
        "global_cross_root_nonce_consumed": False,
        "trusted_mono_to_runner_producer_selected": False,
        "provider_result_accepted": False,
        "canonical_memory_written": False,
        "protected_effect_performed": False,
        "disposition": "LOCAL_RUNNER_CONTROL_PASS_AUTHENTIC_PRODUCER_HOLD",
    }


if __name__ == "__main__":
    evidence = evaluate()
    root = Path(__file__).resolve().parent
    path = root / "docs/evidence/TRAINING_MONO_RUNNER_PRODUCER_NATIVE_20261018.json"
    path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"status": evidence["disposition"],
                      "passed": evidence["native_pytest"]["passed"],
                      "raw_log_sha256": evidence["raw_log_sha256"],
                      "json_path": str(path)}, sort_keys=True))
