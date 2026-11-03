"""Short-retention GitHub artifact witness for synthetic tiny Qwen CPU state.

The writer and reader are DIFFERENT GitHub Actions jobs. The saved adapter and
AdamW state exist between those jobs, unlike an in-run TemporaryDirectory.
This is an unauthenticated public test artifact, never official model custody.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

from trainingday import qwen35_full32_trl_integrated_cpu as integrated
from trainingday import qwen35_full32_trl_optimizer_resume_cpu as witness


SCHEMA = "TRAININGDAY_QWEN35_TINY_PUBLIC_CROSS_JOB_ARTIFACT_V1"
RECEIPT = "cross-job-expected-v1.json"
DIAGNOSTIC = "direct-step000002.safetensors"
# Predeclared diagnostic envelope, not an acceptance replacement for strict SHA.
DRIFT_ABS_LIMIT = 1e-5
DRIFT_REL_LIMIT = 1e-4
LOGIT_SAMPLES = 4096
_HEX = set("0123456789abcdef")



def _snapshot_map(model, ids, torch):
    """Read-only final synthetic trainable tensors plus sampled probe logits."""
    values = {"param/" + n: p.detach().cpu().contiguous().clone()
              for n, p in model.named_parameters() if p.requires_grad}
    if len(values) != 304:
        raise AssertionError("diagnostic expected 304 LoRA trainables")
    logits = integrated._probe_logits(model, ids, torch).detach().cpu().flatten()
    stride = max(1, logits.numel() // LOGIT_SAMPLES)
    values["logits_sample"] = logits[::stride][:LOGIT_SAMPLES].contiguous().clone()
    return values


def _scalar_drift(reference, observed, *, abs_limit=DRIFT_ABS_LIMIT,
                  rel_limit=DRIFT_REL_LIMIT) -> dict:
    """Symmetric deterministic scalar envelope; does not change strict SHA gate."""
    if len(reference) != len(observed) or not reference:
        raise ValueError("diagnostic shape or empty tensor mismatch")
    if abs_limit < 0 or rel_limit < 0:
        raise ValueError("negative diagnostic tolerance")
    max_abs = max_rel = sum_abs = 0.0
    outside = differing = 0
    for expected, actual in zip(reference, observed):
        a, b = float(expected), float(actual)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("nonfinite diagnostic tensor value")
        delta = abs(a - b)
        scale = max(abs(a), abs(b))
        max_abs = max(max_abs, delta)
        max_rel = max(max_rel, delta / max(scale, 1e-12))
        sum_abs += delta
        differing += int(delta != 0.0)
        outside += int(delta > abs_limit + rel_limit * scale)
    return {"elements": len(reference), "changed_elements": differing,
            "outside_predeclared_envelope": outside,
            "max_abs": max_abs, "max_relative": max_rel,
            "mean_abs": sum_abs / len(reference),
            "abs_limit": abs_limit, "rel_limit": rel_limit}


def _tensor_drift(expected: dict, actual: dict) -> dict:
    if set(expected) != set(actual):
        raise ValueError("diagnostic tensor names differ")
    if "logits_sample" not in expected:
        raise ValueError("sampled logits missing from diagnostic")
    params_expected, params_actual = [], []
    for name in sorted(expected):
        if tuple(expected[name].shape) != tuple(actual[name].shape):
            raise ValueError("diagnostic tensor shape differs: " + name)
        if name == "logits_sample":
            continue
        if not name.startswith("param/"):
            raise ValueError("unexpected diagnostic tensor identity")
        params_expected.extend(expected[name].flatten().tolist())
        params_actual.extend(actual[name].flatten().tolist())
    if len(expected) != 305:
        raise ValueError("diagnostic expected 304 parameters plus logits sample")
    return {"trainable_tensors": len(expected) - 1,
            "trainable": _scalar_drift(params_expected, params_actual),
            "sampled_logits": _scalar_drift(
                expected["logits_sample"].flatten().tolist(),
                actual["logits_sample"].flatten().tolist()),
            "logit_sampling": "even_stride_at_most_4096_not_full_logits",
            "strict_sha_acceptance_unchanged": True}


def _cpu_signature() -> dict[str, str]:
    """Diagnostic only: runtime fingerprint alone does not bind CPU ISA."""
    cpuinfo = Path("/proc/cpuinfo")
    first = cpuinfo.read_text(encoding="utf-8", errors="replace").split("\n\n", 1)[0] if cpuinfo.exists() else ""
    fields = {}
    for line in first.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return {"model": fields.get("model name", platform.machine()),
            "flags_sha256": hashlib.sha256(fields.get("flags", "").encode("utf-8")).hexdigest()}


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _read_receipt(path: Path) -> dict:
    raw = (path / RECEIPT).read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > 4096:
        raise ValueError("cross-job receipt oversized")
    receipt = json.loads(raw)
    expected = {"schema", "writer_run_id", "writer_job", "writer_pid", "writer_cpu", "checkpoint_manifest_sha256",
                "direct_adapter_sha256", "direct_optimizer_sha256", "direct_logits_sha256",
                "diagnostic_snapshot_sha256", "first_step_loss", "second_step_loss", "training_authorized"}
    if not isinstance(receipt, dict) or set(receipt) != expected or receipt["schema"] != SCHEMA:
        raise ValueError("cross-job receipt schema changed")
    if receipt["training_authorized"] is not False:
        raise ValueError("cross-job receipt cannot grant training")
    if (receipt["writer_job"] != "writer" or
            not isinstance(receipt["writer_run_id"], str) or
            not receipt["writer_run_id"].isdigit() or
            type(receipt["writer_pid"]) is not int or receipt["writer_pid"] <= 0):
        raise ValueError("cross-job writer identity invalid")
    if (not isinstance(receipt["writer_cpu"], dict) or
            set(receipt["writer_cpu"]) != {"model", "flags_sha256"} or
            not isinstance(receipt["writer_cpu"]["model"], str) or
            not _is_sha(receipt["writer_cpu"]["flags_sha256"])):
        raise ValueError("cross-job writer CPU diagnostic malformed")
    for key in ("checkpoint_manifest_sha256", "direct_adapter_sha256",
                "direct_optimizer_sha256", "direct_logits_sha256", "diagnostic_snapshot_sha256"):
        if not _is_sha(receipt[key]):
            raise ValueError("cross-job digest malformed")
    if not all(type(receipt[k]) is float for k in ("first_step_loss", "second_step_loss")):
        raise ValueError("cross-job loss receipt malformed")
    if witness._sha(path / witness.MANIFEST) != receipt["checkpoint_manifest_sha256"]:
        raise ValueError("cross-job checkpoint manifest corrupted")
    if witness._sha(path / DIAGNOSTIC) != receipt["diagnostic_snapshot_sha256"]:
        raise ValueError("cross-job diagnostic snapshot corrupted")
    witness._validate_checkpoint(path)
    return receipt


def write(path: Path) -> dict:
    if os.getenv("GITHUB_JOB") != "writer" or not os.getenv("GITHUB_RUN_ID", "").isdigit():
        raise ValueError("artifact writer requires GitHub Actions writer job")
    integrated._offline_cpu()
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    path.mkdir(parents=True, exist_ok=False)
    model, trainable, batches, frozen_sha = witness._context(torch)
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=witness.LR)
    first_loss = witness._step(torch, model, optimizer, batches[0])
    witness._save(path, model, optimizer, batches[1], frozen_sha,
                  [name for name, _ in trainable], torch)
    second_loss = witness._step(torch, model, optimizer, batches[1])
    ids = torch.tensor([integrated._synthetic_rows(integrated._load_tokenizer(
        Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])))[0]["input_ids"]], dtype=torch.long)
    from safetensors.torch import save_file
    save_file(_snapshot_map(model, ids, torch), str(path / DIAGNOSTIC))
    receipt = {"schema": SCHEMA, "writer_run_id": os.environ["GITHUB_RUN_ID"],
               "writer_job": "writer", "writer_cpu": _cpu_signature(),
               "writer_pid": os.getpid(),
               "checkpoint_manifest_sha256": witness._sha(path / witness.MANIFEST),
               "diagnostic_snapshot_sha256": witness._sha(path / DIAGNOSTIC),
               "direct_adapter_sha256": witness._trainable_digest(model),
               "direct_optimizer_sha256": witness._optimizer_digest(optimizer),
               "direct_logits_sha256": integrated._sha_tensor(integrated._probe_logits(model, ids, torch)),
               "first_step_loss": first_loss, "second_step_loss": second_loss,
               "training_authorized": False}
    (path / RECEIPT).write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    if _read_receipt(path) != receipt:
        raise AssertionError("writer cross-job receipt round trip changed")
    return {"status": "CROSS_JOB_PUBLIC_SYNTHETIC_CHECKPOINT_EXPORTED", **receipt,
            "official_pretrained_weights_loaded": False, "cuda_initialized": torch.cuda.is_initialized()}


def read(path: Path) -> dict:
    # Validate source fixture and SHA file set before any torch.load.
    receipt = _read_receipt(path)
    if (os.getenv("GITHUB_RUN_ID") != receipt["writer_run_id"] or
            os.getenv("GITHUB_JOB") != "reader"):
        raise ValueError("reader must be a distinct job in the SAME Actions run")
    # Backend/thread binding remains strict in witness._resume. These jobs
    # must not silently claim cross-hardware numerical portability.
    import torch
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    from safetensors.torch import load_file
    numeric = {}

    def observe(model, ids, torch):
        numeric.update(_tensor_drift(load_file(str(path / DIAGNOSTIC), device="cpu"),
                                     _snapshot_map(model, ids, torch)))

    result = witness._resume(path, _after_step_diagnostic=observe)
    print(json.dumps({"diagnostic": "CROSS_JOB_NUMERICAL_COMPARISON_ONLY",
                      "predeclared_tensor_drift": numeric,
                      "writer_cpu": receipt["writer_cpu"], "reader_cpu": _cpu_signature(),
                      "writer_digests": {k: receipt[k] for k in (
                          "direct_adapter_sha256", "direct_optimizer_sha256", "direct_logits_sha256")},
                      "reader_digests": {k: result[k] for k in (
                          "adapter_sha256", "optimizer_state_sha256", "logits_sha256")},
                      "writer_second_step_loss": receipt["second_step_loss"],
                      "reader_second_step_loss": result["second_step_loss"]}, sort_keys=True), file=sys.stderr)
    for key, source in (("adapter_sha256", "direct_adapter_sha256"),
                        ("optimizer_state_sha256", "direct_optimizer_sha256"),
                        ("logits_sha256", "direct_logits_sha256")):
        if result[key] != receipt[source]:
            raise AssertionError(f"cross-job {key} differs: writer={receipt[source]} "
                                 f"reader={result[key]} writer_cpu={receipt['writer_cpu']} "
                                 f"reader_cpu={_cpu_signature()}")
    if (result["after_resume_optimizer_steps"] != [2] or
            result["cuda_initialized"] or
            abs(result["second_step_loss"] - receipt["second_step_loss"]) > 1e-7):
        raise AssertionError("cross-job optimizer continuation not equivalent")
    return {"status": "CROSS_JOB_PUBLIC_SYNTHETIC_ADAMW_RESTORE_PASS",
            "schema": SCHEMA, "writer_run_id": receipt["writer_run_id"],
            "reader_pid": result["pid"], "writer_pid": receipt["writer_pid"],
            "adapter_sha256": result["adapter_sha256"],
            "optimizer_sha256": result["optimizer_state_sha256"],
            "logits_sha256": result["logits_sha256"],
            "training_authorized": False, "official_pretrained_weights_loaded": False,
            "persistence_scope": "GITHUB_ACTIONS_ARTIFACT_RETENTION_1_DAY_NOT_PERMANENT"}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("--write", "--read"):
        raise SystemExit("Usage: python -m trainingday.qwen35_full32_trl_cross_job_artifact_cpu --write|--read DIR")
    print(json.dumps(write(Path(sys.argv[2])) if sys.argv[1] == "--write"
                     else read(Path(sys.argv[2])), sort_keys=True))
