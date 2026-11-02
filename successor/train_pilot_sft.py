from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

from trainingday.physical_launch_authority import (
    PhysicalLaunchAuthorityUnavailable,
    require_physical_launch_authority,
)

from successor.pilot_sft import (
    build_run_manifest,
    cosine_floor_scale,
    sha256_file,
    validate_run_spec,
)
from successor.training_data import assistant_only_labels
from successor.v3_training import V3_TRAINING_SCHEMA, preflight_v3_training

SYSTEM = {"role": "system", "content": "/no_think /system_override"}


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
            raise ValueError("training rows require string prompt and response")
    return rows


def render(tokenizer, row: dict):
    prompt_text = tokenizer.apply_chat_template(
        [SYSTEM, {"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [SYSTEM, {"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= len(full_ids):
        raise ValueError("row has no supervised assistant tokens")
    labels = assistant_only_labels(full_ids, len(prompt_ids))
    return full_ids, labels


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def prepare(spec_path: Path) -> tuple[dict, dict, list[dict], list[dict], Path]:
    repo_root = Path(__file__).resolve().parents[1]
    base_manifest = json.loads((repo_root / "successor" / "base_model_manifest.json").read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") == V3_TRAINING_SCHEMA:
        decision = preflight_v3_training(spec_path, repo_root=repo_root)
        if not decision.runnable:
            raise RuntimeError(
                "V3 training preflight blocked: " + ", ".join(decision.reasons)
            )
    validate_run_spec(spec, base_manifest)
    train_path = Path(spec["train_path"])
    validation_path = Path(spec["validation_path"])
    parent = Path(spec["parent_adapter_path"])
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("private train/validation file missing")
    if not (parent / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("parent adapter_model.safetensors missing")
    return spec, base_manifest, load_jsonl(train_path), load_jsonl(validation_path), repo_root


def dry_run(spec_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    raw_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if raw_spec.get("schema") == V3_TRAINING_SCHEMA:
        decision = preflight_v3_training(spec_path, repo_root=repo_root)
        payload = decision.to_dict()
        payload["run_id"] = raw_spec.get("run_id")
        return payload

    spec, _base_manifest, train_rows, validation_rows, repo_root = prepare(spec_path)
    parent = Path(spec["parent_adapter_path"])
    train_path = Path(spec["train_path"])
    validation_path = Path(spec["validation_path"])
    return {
        "run_id": spec["run_id"],
        "code_commit": git_head(repo_root),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "parent_adapter_sha256": sha256_file(parent / "adapter_model.safetensors"),
    }


def train(spec_path: Path) -> dict:
    # The V3/spec preflight cannot mint a host-owned exclusive GPU lease.
    # Require the independent physical-launch boundary before even prepare()
    # reads private data or any model framework is imported.
    require_physical_launch_authority(operation="legacy pilot SFT Qwen training")
    spec, base_manifest, train_rows, validation_rows, repo_root = prepare(spec_path)

    import torch
    import transformers
    import peft
    import bitsandbytes
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    seed = int(spec["seed"])
    accum = int(spec["gradient_accumulation"])
    lr = float(spec["learning_rate"])
    torch.manual_seed(seed)
    random.seed(seed)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_manifest["local_path"], local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_manifest["local_path"],
        local_files_only=True,
        quantization_config=quant,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        base, spec["parent_adapter_path"], is_trainable=True
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.train()

    optimizer = PagedAdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    optimizer.zero_grad(set_to_none=True)
    total_steps = math.ceil(len(train_rows) / accum)
    warmup_steps = max(1, round(total_steps * float(spec.get("warmup_fraction", 0.08))))
    floor = float(spec.get("lr_floor", 0.2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_floor_scale(
            step, total_steps, warmup_steps=warmup_steps, floor=floor
        ),
    )

    def row_loss(row: dict):
        full_ids, labels = render(tokenizer, row)
        ids = torch.tensor([full_ids], device="cuda")
        target = torch.tensor([labels], device="cuda")
        return model(input_ids=ids, labels=target).loss

    def eval_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for row in validation_rows:
                loss = row_loss(row)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite validation loss")
                losses.append(float(loss))
        model.train()
        return sum(losses) / len(losses)

    output_root = Path(spec["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    parent = Path(spec["parent_adapter_path"])
    train_path = Path(spec["train_path"])
    validation_path = Path(spec["validation_path"])
    source_commit = git_head(repo_root)
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    parent_sha = sha256_file(parent / "adapter_model.safetensors")
    package_versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    }
    parent_validation = eval_loss()
    start = time.time()
    order = list(range(len(train_rows)))
    random.Random(seed + 31).shuffle(order)
    half_step = max(1, math.ceil(total_steps / 2))
    milestones = {half_step: "half", total_steps: "full"}
    running = 0.0
    micro = 0
    step = 0
    results = {}

    for pos, idx in enumerate(order, 1):
        loss = row_loss(train_rows[idx])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at row {idx}")
        (loss / accum).backward()
        running += float(loss.detach())
        micro += 1
        if micro != accum and pos != len(order):
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        micro = 0
        print(json.dumps({
            "row": pos,
            "step": step,
            "loss_window": running / max(1, min(accum, pos)),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(time.time() - start, 1),
        }), flush=True)
        running = 0.0

        if step not in milestones:
            continue
        tag = milestones[step]
        validation = eval_loss()
        save_dir = output_root / tag
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        adapter_sha = sha256_file(save_dir / "adapter_model.safetensors")
        manifest = build_run_manifest(
            spec,
            code_commit=source_commit,
            train_sha256=train_sha,
            validation_sha256=validation_sha,
            parent_adapter_sha256=parent_sha,
            output_adapter_sha256=adapter_sha,
            gpu_type=torch.cuda.get_device_name(0),
            package_versions=package_versions,
        )
        manifest.update({
            "checkpoint": tag,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "optimizer_steps": step,
            "parent_validation_loss": parent_validation,
            "validation_loss": validation,
            "elapsed_s": round(time.time() - start, 1),
            "objective": "assistant_only_sft",
            "scheduler": "warmup_cosine_floor",
            "warmup_steps": warmup_steps,
            "lr_floor": floor,
        })
        if spec.get("schema") == V3_TRAINING_SCHEMA:
            manifest["v3_gate"] = {
                "task9_source_commit": spec["task9_source_commit"],
                "task9_ready_receipt_sha256": sha256_file(
                    spec["task9_ready_receipt_path"]
                ),
                "training_authorization_receipt_sha256": sha256_file(
                    spec["training_authorization_receipt_path"]
                ),
                "corpus_manifest_sha256": sha256_file(
                    spec["corpus_manifest_path"]
                ),
                "base_tree_sha256": spec["base_tree_sha256"],
                "base_inventory_sha256": spec["base_inventory_sha256"],
                "parent_adapter_config_sha256": spec[
                    "parent_adapter_config_sha256"
                ],
                "parent_candidate_subject_digest": spec[
                    "parent_candidate_subject_digest"
                ],
                "general_rehearsal_fraction": float(
                    spec["general_rehearsal_fraction"]
                ),
            }
        manifest_path = output_root / f"run_{tag}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        results[tag] = manifest
        print("CHECKPOINT", json.dumps(manifest), flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec_path = Path(args.spec)
    if args.dry_run:
        print(json.dumps(dry_run(spec_path), indent=2))
        return
    try:
        result = train(spec_path)
    except PhysicalLaunchAuthorityUnavailable as exc:
        print("TRAINING_HOLD", json.dumps({"status": "HOLD", "reason": str(exc)}), flush=True)
        raise SystemExit(2) from None
    print("TRAINING_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
