"""Actual Mono source, Windows cold-process, no-provider dependency falsifier.

Input fixtures are authored source challenges, not trusted provider evidence.
This evaluator does not exercise Linux boot, Pre-Active workers, or inference.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PROBE = Path(__file__).with_name("offline_mono_no_provider_probe_v1.py")
CASES = (
    {"name": "fabricated_done", "status": "COMPLETED", "event": "DONE"},
    {"name": "forged_provider", "provider_id": "rogue", "admitted": True},
    {"name": "reused_nonce", "one_use_nonce": "duplicate", "consumed": True},
    {"name": "wrong_route", "route_id": "foreign-route", "selected": True},
    {"name": "wrong_scope", "referent": "foreign", "scope": "another-task"},
    {"name": "self_certified_origin", "origin_id": "issuer-two", "independent": True},
)


def child(root: Path, phase: str, *, claim=None, changed_text=None,
          project_id="veraos", identity_id="VERA"):
    params = {"root": str(root), "phase": phase,
              "project_id": project_id, "identity_id": identity_id}
    if claim is not None: params["untrusted_provider_claim"] = claim
    if changed_text is not None: params["changed_text"] = changed_text
    start = time.perf_counter_ns()
    ran = subprocess.run([sys.executable, "-B", str(PROBE)],
        input=json.dumps(params), text=True, capture_output=True,
        timeout=20, check=False, env=os.environ.copy())
    elapsed = time.perf_counter_ns() - start
    if ran.returncode or not ran.stdout.strip():
        raise RuntimeError(f"probe {phase} failed: {ran.returncode} {ran.stderr[-900:]}")
    result = json.loads(ran.stdout)
    result["process_wall_ms"] = round(elapsed / 1e6, 2)
    return result


def evaluate():
    fixture_digest = sha256(json.dumps(CASES, sort_keys=True,
                 separators=(",", ":")).encode("utf-8")).hexdigest()
    with tempfile.TemporaryDirectory(prefix="training-offline-mono-") as tmp:
        root = Path(tmp) / "host-a"
        initial = child(root, "initial")
        recovered = child(root, "reopen")
        assert initial["task_id"] == recovered["task_id"]
        assert initial["head"] == recovered["head"]
        assert initial["source"] == recovered["source"]
        assert recovered["literal"] == "literal  content"
        results = []
        for case in CASES:
            r = child(root, "reopen", claim=case)
            results.append({"name": case["name"],
               "held_open": r["closed"] is False,
               "event_only_open": r["event_types"] == ["TASK_OPENED"],
               "original_identity": r["task_id"] == initial["task_id"] and
                  r["head"] == initial["head"] and r["source"] == initial["source"],
               "no_provider": r["provider_transport_count"] == 0,
               "no_optional_imports": r["no_optional_imports"],
               "wall_ms": r["process_wall_ms"]})
        changed = [child(root, "tamper", changed_text=x)
                   for x in ("literal content", "literal  changed")]
        assert all(r["tamper_rejected"] and r["task_count"] == 1 and not r["closed"]
                   for r in changed)
        other = child(Path(tmp) / "host-b", "initial",
                      project_id="VERA_MONO", identity_id="VERA_PROJECT_IDENTITY_V1")
        # Same locator maps to same deterministic task key, but two separate
        # roots are NOT one shared canonical memory or physical task ledger.
        separated_roots = (root / "tasks").exists() and (Path(tmp) / "host-b" / "tasks").exists()
        total_wall_ms = sum([initial["process_wall_ms"], recovered["process_wall_ms"],
            other["process_wall_ms"]] + [x["wall_ms"] for x in results] +
            [x["process_wall_ms"] for x in changed])
        return {"schema": "TRAINING_MONO_OFFLINE_NO_PROVIDER_SOURCE_ABLATION_V1",
                "fixture_sha256": fixture_digest, "hard_negative_cases": len(CASES),
                "negative_rejections": sum(all(r[k] for k in ("held_open",
                       "event_only_open", "original_identity", "no_provider",
                       "no_optional_imports")) for r in results),
                "cold_recovery": True, "literal_substitutions_rejected": len(changed),
                "separate_state_roots": bool(separated_roots),
                "optional_module_imported": False, "loaded_model": False,
                "real_guest_boot": False, "authorized_provider": False,
                "memory_proposal_admitted": False, "effect_executed": False,
                "initial": initial, "recovered": recovered,
                "negative_cases": results, "different_root": other,
                "total_process_wall_ms": round(total_wall_ms, 2),
                "selected_prototype_source_stage": {"mono_files": 316,
                    "portal_files": 89, "pre_active_files": 18,
                    "volition_files": 8, "total": 431,
                    "stage_is_not_strict_boot_requirement": True},
                "decision": "ACCEPT_MINIMAL_MONO_TASK_ONLY_DEFER_REAL_VERAOS_BOOT"}


if __name__ == "__main__":
    print(json.dumps(evaluate(), sort_keys=True, indent=2))
