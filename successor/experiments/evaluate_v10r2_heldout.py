from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from successor.experiments.train_v10_qwen35_authorized import (
    _generation_prefix,
    _observe_live_runtime,
    _read_json,
    load_training_stack,
    load_verified_jsonl,
    sha256_file,
    validate_runtime_observation,
)


class HeldoutEvalHold(RuntimeError):
    pass


def deterministic_sample(rows: list[dict], *, count: int, salt: str) -> list[dict]:
    if count < 1 or count > len(rows):
        raise ValueError("invalid sample count")

    def rank(row: dict) -> str:
        record_id = row.get("record_id") or row.get("case_id") or row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("row missing stable identifier")
        return hashlib.sha256((salt + "\0" + record_id).encode("utf-8")).hexdigest()

    return sorted(rows, key=rank)[:count]


def build_completion_example(tokenizer, row: dict, *, max_length: int) -> dict:
    prompt = _generation_prefix(tokenizer, row["prompt"])
    completion = row["response"] + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        raise ValueError(
            f"example exceeds max_length:{len(full_ids)}>{max_length}"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("completion produced no tokens")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "record_id": row.get("record_id") or row.get("case_id") or row.get("id"),
        "input_ids": full_ids,
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(full_ids) - len(prompt_ids),
    }


def _model_candidate(
    *,
    stack: dict,
    base_path: Path,
    quant_config,
    adapter_dir: Path | None,
):
    torch = stack["torch"]
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        base_path,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if adapter_dir is not None:
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise HeldoutEvalHold(f"adapter model missing:{adapter_dir}")
        model = stack["peft"].PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
    model.eval()
    return model


def evaluate_candidate(
    *,
    stack: dict,
    tokenizer,
    examples: list[dict],
    base_path: Path,
    quant_config,
    name: str,
    adapter_dir: Path | None,
) -> dict:
    torch = stack["torch"]
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _model_candidate(
        stack=stack,
        base_path=base_path,
        quant_config=quant_config,
        adapter_dir=adapter_dir,
    )
    load_seconds = time.perf_counter() - started

    rows: list[dict] = []
    total_nll = 0.0
    total_tokens = 0
    eval_started = time.perf_counter()
    with torch.inference_mode():
        for index, item in enumerate(examples):
            input_ids = torch.tensor(
                [item["input_ids"]],
                device="cuda",
                dtype=torch.long,
            )
            labels = torch.tensor(
                [item["labels"]],
                device="cuda",
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = float(output.loss.detach().float().cpu().item())
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens < 1 or not math.isfinite(loss):
                raise HeldoutEvalHold(
                    f"invalid loss/tokens for {name} row {index}"
                )
            nll = loss * valid_tokens
            total_nll += nll
            total_tokens += valid_tokens
            rows.append(
                {
                    "record_id": item["record_id"],
                    "loss": loss,
                    "valid_completion_tokens": valid_tokens,
                    "nll": nll,
                }
            )
            del output, input_ids, labels, attention_mask

    eval_seconds = time.perf_counter() - eval_started
    mean_nll = total_nll / total_tokens
    result = {
        "name": name,
        "adapter_dir": None if adapter_dir is None else str(adapter_dir),
        "adapter_model_sha256": (
            None
            if adapter_dir is None
            else sha256_file(adapter_dir / "adapter_model.safetensors")
        ),
        "sample_count": len(rows),
        "completion_token_count": total_tokens,
        "token_weighted_completion_nll": mean_nll,
        "completion_perplexity": math.exp(mean_nll),
        "mean_case_loss": statistics.fmean(row["loss"] for row in rows),
        "median_case_loss": statistics.median(row["loss"] for row in rows),
        "model_load_seconds": round(load_seconds, 3),
        "eval_seconds": round(eval_seconds, 3),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024**2), 3
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / (1024**2), 3
        ),
        "cases": rows,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _paired_comparison(left: dict, right: dict) -> dict:
    left_map = {row["record_id"]: row for row in left["cases"]}
    right_map = {row["record_id"]: row for row in right["cases"]}
    if set(left_map) != set(right_map):
        raise HeldoutEvalHold("candidate case sets differ")
    deltas = [
        right_map[key]["loss"] - left_map[key]["loss"]
        for key in sorted(left_map)
    ]
    return {
        "left": left["name"],
        "right": right["name"],
        "right_minus_left_mean_case_loss": statistics.fmean(deltas),
        "right_minus_left_median_case_loss": statistics.median(deltas),
        "right_case_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "left_case_wins": sum(delta > 0 for delta in deltas),
    }


def run_heldout_eval(
    repo_root: Path,
    *,
    validation_jsonl: Path,
    sample_count: int,
    salt: str,
    candidates: list[tuple[str, Path | None]],
) -> dict:
    runtime_path = (
        repo_root
        / "successor"
        / "experiments"
        / "V10_QWEN35_TRAINING_RUNTIME_BINDING_V1.json"
    )
    runtime_binding = _read_json(runtime_path)
    stack = load_training_stack()
    runtime_observation = _observe_live_runtime(runtime_binding, stack)
    runtime_check = validate_runtime_observation(
        runtime_binding,
        runtime_observation,
    )

    validation_rows = load_verified_jsonl(
        validation_jsonl,
        expected_sha256=(
            "ccc57ad20e8dfbc826ce49f064e692e0eab3fa396ed602dfba54ad9e252bd6d7"
        ),
        expected_rows=2500,
    )
    selected = deterministic_sample(
        validation_rows,
        count=sample_count,
        salt=salt,
    )

    base_path = Path(runtime_binding["target"]["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [
        build_completion_example(tokenizer, row, max_length=512)
        for row in selected
    ]

    torch = stack["torch"]
    quant_config = stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    results = [
        evaluate_candidate(
            stack=stack,
            tokenizer=tokenizer,
            examples=examples,
            base_path=base_path,
            quant_config=quant_config,
            name=name,
            adapter_dir=adapter_dir,
        )
        for name, adapter_dir in candidates
    ]
    comparisons = []
    for left_index in range(len(results)):
        for right_index in range(left_index + 1, len(results)):
            comparisons.append(
                _paired_comparison(results[left_index], results[right_index])
            )

    sample_ids = [item["record_id"] for item in examples]
    sample_sha = hashlib.sha256(
        json.dumps(
            sample_ids,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "V10R2_HELDOUT_DIAGNOSTIC_V1",
        "status": "HELDOUT_DIAGNOSTIC_COMPLETE",
        "role": "POST_TRAIN_DEVELOPMENT_DIAGNOSTIC_NOT_FINAL_BANK",
        "validation_sha256": sha256_file(validation_jsonl),
        "validation_rows": 2500,
        "sample_method": "SHA256_RANK",
        "sample_salt": salt,
        "sample_count": sample_count,
        "sample_record_ids": sample_ids,
        "sample_record_ids_sha256": sample_sha,
        "runtime_binding_sha256": runtime_binding["binding_sha256"],
        "runtime_observation": runtime_observation,
        "runtime_check": runtime_check,
        "candidates": results,
        "paired_comparisons": comparisons,
        "claim_ceiling": (
            "DEVELOPMENT HELDOUT LOSS DIAGNOSTIC ONLY / "
            "NOT FINAL BANK / NOT EXTERNAL QUALIFICATION / "
            "NOT CHECKPOINT-SELECTION AUTHORITY FOR V10 FINAL"
        ),
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--salt", default="v10r2-heldout-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    candidates = [
        ("base", None),
        (
            "step4",
            Path(
                r"D:\VERA\models\adapters\v10r2-dev-bnb8-step4-20261003\adapter"
            ),
        ),
        (
            "step12",
            Path(
                r"D:\VERA\models\adapters\v10r2-dev-bnb8-cont4-plus8-20261003\adapter"
            ),
        ),
    ]
    result = run_heldout_eval(
        args.repo_root,
        validation_jsonl=args.validation_jsonl,
        sample_count=args.sample_count,
        salt=args.salt,
        candidates=candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
