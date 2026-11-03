"""Short-retention GitHub artifact witness for synthetic tiny Qwen CPU state.

The writer and reader are DIFFERENT GitHub Actions jobs. The saved adapter and
AdamW state exist between those jobs, unlike an in-run TemporaryDirectory.
This is an unauthenticated public test artifact, never official model custody.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from trainingday import qwen35_full32_trl_integrated_cpu as integrated
from trainingday import qwen35_full32_trl_optimizer_resume_cpu as witness


SCHEMA = "TRAININGDAY_QWEN35_TINY_PUBLIC_CROSS_JOB_ARTIFACT_V1"
RECEIPT = "cross-job-expected-v1.json"
_HEX = set("0123456789abcdef")


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _read_receipt(path: Path) -> dict:
    raw = (path / RECEIPT).read_text(encoding="utf-8")
    if len(raw.encode("utf-8")) > 4096:
        raise ValueError("cross-job receipt oversized")
    receipt = json.loads(raw)
    expected = {"schema", "writer_run_id", "writer_job", "writer_pid", "checkpoint_manifest_sha256",
                "direct_adapter_sha256", "direct_optimizer_sha256", "direct_logits_sha256",
                "first_step_loss", "second_step_loss", "training_authorized"}
    if not isinstance(receipt, dict) or set(receipt) != expected or receipt["schema"] != SCHEMA:
        raise ValueError("cross-job receipt schema changed")
    if receipt["training_authorized"] is not False:
        raise ValueError("cross-job receipt cannot grant training")
    if (receipt["writer_job"] != "writer" or
            not isinstance(receipt["writer_run_id"], str) or
            not receipt["writer_run_id"].isdigit() or
            type(receipt["writer_pid"]) is not int or receipt["writer_pid"] <= 0):
        raise ValueError("cross-job writer identity invalid")
    for key in ("checkpoint_manifest_sha256", "direct_adapter_sha256",
                "direct_optimizer_sha256", "direct_logits_sha256"):
        if not _is_sha(receipt[key]):
            raise ValueError("cross-job digest malformed")
    if not all(type(receipt[k]) is float for k in ("first_step_loss", "second_step_loss")):
        raise ValueError("cross-job loss receipt malformed")
    if witness._sha(path / witness.MANIFEST) != receipt["checkpoint_manifest_sha256"]:
        raise ValueError("cross-job checkpoint manifest corrupted")
    witness._validate_checkpoint(path)
    return receipt


def write(path: Path) -> dict:
    if os.getenv("GITHUB_JOB") != "writer" or not os.getenv("GITHUB_RUN_ID", "").isdigit():
        raise ValueError("artifact writer requires GitHub Actions writer job")
    integrated._offline_cpu()
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    path.mkdir(parents=True, exist_ok=False)
    model, trainable, batches, frozen_sha = witness._context(torch)
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=witness.LR)
    first_loss = witness._step(torch, model, optimizer, batches[0])
    witness._save(path, model, optimizer, batches[1], frozen_sha,
                  [name for name, _ in trainable], torch)
    second_loss = witness._step(torch, model, optimizer, batches[1])
    ids = torch.tensor([integrated._synthetic_rows(integrated._load_tokenizer(
        Path(os.environ["TRAININGDAY_QWEN_TOKENIZER_JSON"])))[0]["input_ids"]], dtype=torch.long)
    receipt = {"schema": SCHEMA, "writer_run_id": os.environ["GITHUB_RUN_ID"],
               "writer_job": "writer",
               "writer_pid": os.getpid(),
               "checkpoint_manifest_sha256": witness._sha(path / witness.MANIFEST),
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
    result = witness._resume(path)
    for key, source in (("adapter_sha256", "direct_adapter_sha256"),
                        ("optimizer_state_sha256", "direct_optimizer_sha256"),
                        ("logits_sha256", "direct_logits_sha256")):
        if result[key] != receipt[source]:
            raise AssertionError(f"cross-job {key} not equivalent to writer direct step")
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
