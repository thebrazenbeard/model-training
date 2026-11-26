from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from successor.experiments.evaluate_v10r2_heldout import (
    build_completion_example,
    deterministic_sample,
    _paired_comparison,
)
from successor.experiments.train_v10_qwen35_authorized import (
    _observe_live_runtime,
    _read_json,
    load_training_stack,
    load_verified_jsonl,
    sha256_file,
    validate_runtime_observation,
)
from successor.experiments.v10r2_lane_b_runtime import apply_qwen35_acceleration


FROZEN_VALIDATION_SHA256 = (
    "ccc57ad20e8dfbc826ce49f064e692e0eab3fa396ed602dfba54ad9e252bd6d7"
)
FROZEN_VALIDATION_ROWS = 2500
DEFAULT_FIXED_EVAL_LENGTH = 512


class EvalConfigHold(RuntimeError):
    """Fail-closed refusal for malformed Lane B evaluation configuration."""


def parse_candidate_specs(values: list[str]) -> list[tuple[str, Path | None]]:
    if not values:
        raise EvalConfigHold("at least one candidate is required")
    result: list[tuple[str, Path | None]] = []
    names: set[str] = set()
    for raw in values:
        if raw == "base":
            name, path = "base", None
        else:
            if "=" not in raw:
                raise EvalConfigHold(f"invalid candidate spec:{raw}")
            name, path_text = raw.split("=", 1)
            name = name.strip()
            path_text = path_text.strip()
            if not name or not path_text:
                raise EvalConfigHold(f"invalid candidate spec:{raw}")
            path = Path(path_text)
        if name in names:
            raise EvalConfigHold(f"duplicate candidate name:{name}")
        names.add(name)
        result.append((name, path))
    return result


def resolve_runtime_binding_path(
    repo_root: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit
    return (
        repo_root
        / "successor"
        / "experiments"
        / "V10R2_LANE_B_RUNTIME_BINDING_V2.json"
    )


def _checkpoint_record_id(row: dict) -> str:
    for field in ("record_id", "case_id", "id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    raise EvalConfigHold("checkpoint panel row missing record id")


def _record_ids_sha256(record_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            record_ids,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def load_checkpoint_panel_manifest(path: Path) -> dict:
    if not path.is_file():
        raise EvalConfigHold(f"checkpoint panel manifest missing:{path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("schema") != "V10R2_HELDOUT_CHECKPOINT_PANEL_V1":
        raise EvalConfigHold("checkpoint panel schema mismatch")
    if (
        value.get("role")
        != "DEVELOPMENT_CHECKPOINT_SELECTION_DIAGNOSTIC_NOT_FINAL_BANK"
    ):
        raise EvalConfigHold("checkpoint panel role mismatch")
    policy = value.get("selection_policy")
    if not isinstance(policy, dict):
        raise EvalConfigHold("checkpoint panel selection policy missing")
    if policy.get("final_bank_use") != "PROHIBITED":
        raise EvalConfigHold("checkpoint panel must prohibit final-bank use")
    validation_sha = value.get("validation_sha256")
    if (
        not isinstance(validation_sha, str)
        or len(validation_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in validation_sha)
    ):
        raise EvalConfigHold("checkpoint panel validation sha invalid")
    if not isinstance(value.get("validation_rows"), int):
        raise EvalConfigHold("checkpoint panel validation row count invalid")

    panels = {}
    for name in ("anchor_panel", "confirmatory_panel", "combined_panel"):
        panel = value.get(name)
        if not isinstance(panel, dict):
            raise EvalConfigHold(f"checkpoint panel section missing:{name}")
        count = panel.get("count")
        ids = panel.get("record_ids")
        digest = panel.get("record_ids_sha256")
        if not isinstance(count, int) or count < 1:
            raise EvalConfigHold(f"checkpoint panel count invalid:{name}")
        if (
            not isinstance(ids, list)
            or len(ids) != count
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise EvalConfigHold(f"checkpoint panel ids invalid:{name}")
        if len(set(ids)) != len(ids):
            raise EvalConfigHold(f"checkpoint panel ids not unique:{name}")
        if digest != _record_ids_sha256(ids):
            raise EvalConfigHold(f"checkpoint panel digest mismatch:{name}")
        panels[name] = ids

    if set(panels["anchor_panel"]) & set(panels["confirmatory_panel"]):
        raise EvalConfigHold("checkpoint anchor/confirmatory panels must be disjoint")
    if (
        panels["combined_panel"]
        != panels["anchor_panel"] + panels["confirmatory_panel"]
    ):
        raise EvalConfigHold("checkpoint combined panel ordering mismatch")
    return value


def select_checkpoint_panel_rows(
    validation_rows: list[dict],
    manifest: dict,
    panel_name: str = "combined_panel",
) -> list[dict]:
    panel = manifest.get(panel_name)
    if not isinstance(panel, dict):
        raise EvalConfigHold(f"checkpoint panel section missing:{panel_name}")
    wanted = panel.get("record_ids")
    if not isinstance(wanted, list):
        raise EvalConfigHold(f"checkpoint panel record ids missing:{panel_name}")

    by_id = {}
    for row in validation_rows:
        record_id = _checkpoint_record_id(row)
        if record_id in by_id:
            raise EvalConfigHold(f"duplicate validation record id:{record_id}")
        by_id[record_id] = row

    missing = [record_id for record_id in wanted if record_id not in by_id]
    if missing:
        raise EvalConfigHold(
            "checkpoint panel validation records missing:"
            + ",".join(missing[:5])
        )
    return [by_id[record_id] for record_id in wanted]


def pad_completion_example(
    item: dict,
    *,
    pad_token_id: int,
    fixed_length: int,
) -> dict:
    input_ids = list(item["input_ids"])
    labels = list(item["labels"])
    if len(input_ids) != len(labels):
        raise EvalConfigHold("input/label length mismatch")
    if len(input_ids) > fixed_length:
        raise EvalConfigHold(
            f"example exceeds fixed eval length:{len(input_ids)}>{fixed_length}"
        )
    pad = fixed_length - len(input_ids)
    result = dict(item)
    result["input_ids"] = input_ids + [pad_token_id] * pad
    result["labels"] = labels + [-100] * pad
    result["attention_mask"] = [1] * len(input_ids) + [0] * pad
    result["fixed_length"] = fixed_length
    return result


def load_retention_shadow_manifest(path: Path) -> dict:
    if not path.is_file():
        raise EvalConfigHold(f"retention shadow manifest missing:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "V10_RETENTION_SHADOW_BENCHMARK_MANIFEST_V1":
        raise EvalConfigHold("retention shadow manifest schema mismatch")
    if value.get("status") != "PINNED_SHADOW_DIAGNOSTICS_NOT_FINAL_EVIDENCE":
        raise EvalConfigHold("retention shadow manifest status mismatch")
    policy = value.get("use_policy")
    if not isinstance(policy, dict):
        raise EvalConfigHold("retention shadow use policy missing")
    if policy.get("final_bank_use") is not False:
        raise EvalConfigHold("retention shadow manifest must forbid final-bank use")
    if policy.get("promotion_gate_use") is not False:
        raise EvalConfigHold("retention shadow manifest must forbid promotion-gate use")
    if policy.get("candidate_selection_use") is not False:
        raise EvalConfigHold("retention shadow manifest must forbid candidate selection")
    if policy.get("paired_base_candidate_diagnostic") is not True:
        raise EvalConfigHold("retention shadow manifest must allow paired diagnostics")

    sources = value.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise EvalConfigHold("retention shadow sources missing")
    for name, source in sorted(sources.items()):
        if not isinstance(source, dict):
            raise EvalConfigHold(f"retention source invalid:{name}")
        for field in ("dataset", "revision", "file", "file_sha256", "rows"):
            if source.get(field) in (None, ""):
                raise EvalConfigHold(f"retention source missing {field}:{name}")
        digest = source["file_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise EvalConfigHold(f"retention source sha invalid:{name}")
        if not isinstance(source["rows"], int) or source["rows"] < 1:
            raise EvalConfigHold(f"retention source rows invalid:{name}")
    return value


def retention_shadow_binding(repo_root: Path) -> dict:
    path = (
        repo_root
        / "research"
        / "measurement"
        / "V10_RETENTION_SHADOW_BENCHMARKS_V1.json"
    )
    value = load_retention_shadow_manifest(path)
    return {
        "manifest_path": str(path.relative_to(repo_root)),
        "manifest_sha256": sha256_file(path),
        "status": value["status"],
        "sources": value["sources"],
        "use_policy": value["use_policy"],
        "claim_ceiling": value.get("claim_ceiling"),
    }


def _model_candidate(
    *,
    stack: dict,
    base_path: Path,
    quant_config,
    adapter_dir: Path | None,
    acceleration_backend: str,
):
    torch = stack["torch"]
    model = stack["Qwen3_5ForCausalLM"].from_pretrained(
        base_path,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    acceleration = apply_qwen35_acceleration(
        model,
        backend=acceleration_backend,
    )
    if adapter_dir is not None:
        if not (adapter_dir / "adapter_model.safetensors").is_file():
            raise EvalConfigHold(f"adapter model missing:{adapter_dir}")
        model = stack["peft"].PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
    model.eval()
    return model, acceleration


def evaluate_candidate(
    *,
    stack: dict,
    examples: list[dict],
    base_path: Path,
    quant_config,
    name: str,
    adapter_dir: Path | None,
    acceleration_backend: str,
) -> dict:
    torch = stack["torch"]
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, acceleration = _model_candidate(
        stack=stack,
        base_path=base_path,
        quant_config=quant_config,
        adapter_dir=adapter_dir,
        acceleration_backend=acceleration_backend,
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
            attention_mask = torch.tensor(
                [item["attention_mask"]],
                device="cuda",
                dtype=torch.long,
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = float(output.loss.detach().float().cpu().item())
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            if valid_tokens < 1 or not math.isfinite(loss):
                raise EvalConfigHold(
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
        "acceleration": acceleration,
        "sample_count": len(rows),
        "completion_token_count": total_tokens,
        "token_weighted_completion_nll": mean_nll,
        "completion_perplexity": math.exp(mean_nll),
        "mean_case_loss": statistics.fmean(row["loss"] for row in rows),
        "median_case_loss": statistics.median(row["loss"] for row in rows),
        "model_load_seconds": round(load_seconds, 3),
        "eval_seconds": round(eval_seconds, 3),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / (1024**2),
            3,
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / (1024**2),
            3,
        ),
        "cases": rows,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_lane_b_heldout_eval(
    repo_root: Path,
    *,
    validation_jsonl: Path,
    sample_count: int,
    salt: str,
    candidates: list[tuple[str, Path | None]],
    runtime_binding_path: Path | None = None,
    acceleration_backend: str = "fla_triton",
    fixed_eval_length: int = DEFAULT_FIXED_EVAL_LENGTH,
    checkpoint_panel_path: Path | None = None,
    checkpoint_panel_section: str = "combined_panel",
) -> dict:
    binding_path = resolve_runtime_binding_path(
        repo_root,
        runtime_binding_path,
    )
    runtime_binding = _read_json(binding_path)
    stack = load_training_stack()
    runtime_observation = _observe_live_runtime(runtime_binding, stack)
    runtime_check = validate_runtime_observation(
        runtime_binding,
        runtime_observation,
    )

    validation_rows = load_verified_jsonl(
        validation_jsonl,
        expected_sha256=FROZEN_VALIDATION_SHA256,
        expected_rows=FROZEN_VALIDATION_ROWS,
    )
    checkpoint_panel_binding = None
    if checkpoint_panel_path is not None:
        checkpoint_panel = load_checkpoint_panel_manifest(
            checkpoint_panel_path,
        )
        if checkpoint_panel.get("validation_sha256") != FROZEN_VALIDATION_SHA256:
            raise EvalConfigHold(
                "checkpoint panel validation sha does not match frozen validation"
            )
        if checkpoint_panel.get("validation_rows") != FROZEN_VALIDATION_ROWS:
            raise EvalConfigHold(
                "checkpoint panel validation row count mismatch"
            )
        selected = select_checkpoint_panel_rows(
            validation_rows,
            checkpoint_panel,
            checkpoint_panel_section,
        )
        sample_count = len(selected)
        sample_method = "FROZEN_RECORD_ID_PANEL"
        sample_salt = None
        checkpoint_panel_binding = {
            "path": str(checkpoint_panel_path),
            "sha256": sha256_file(checkpoint_panel_path),
            "section": checkpoint_panel_section,
            "record_ids_sha256": checkpoint_panel[
                checkpoint_panel_section
            ]["record_ids_sha256"],
            "claim_ceiling": checkpoint_panel.get("claim_ceiling"),
        }
    else:
        selected = deterministic_sample(
            validation_rows,
            count=sample_count,
            salt=salt,
        )
        sample_method = "SHA256_RANK"
        sample_salt = salt

    base_path = Path(runtime_binding["target"]["base_path"])
    tokenizer = stack["AutoTokenizer"].from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise EvalConfigHold("tokenizer has no pad or eos token")

    raw_examples = [
        build_completion_example(
            tokenizer,
            row,
            max_length=fixed_eval_length,
        )
        for row in selected
    ]
    examples = [
        pad_completion_example(
            item,
            pad_token_id=int(tokenizer.pad_token_id),
            fixed_length=fixed_eval_length,
        )
        for item in raw_examples
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
            examples=examples,
            base_path=base_path,
            quant_config=quant_config,
            name=name,
            adapter_dir=adapter_dir,
            acceleration_backend=acceleration_backend,
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
        "schema": "V10R2_LANE_B_COMPARABLE_EVAL_V1",
        "status": "DEVELOPMENT_HELDOUT_DIAGNOSTIC_COMPLETE",
        "role": "PAIRED_DEVELOPMENT_DIAGNOSTIC_NOT_FINAL_BANK",
        "validation_sha256": sha256_file(validation_jsonl),
        "validation_rows": FROZEN_VALIDATION_ROWS,
        "sample_method": sample_method,
        "sample_salt": sample_salt,
        "sample_count": sample_count,
        "checkpoint_panel_binding": checkpoint_panel_binding,
        "sample_record_ids": sample_ids,
        "sample_record_ids_sha256": sample_sha,
        "fixed_eval_length": fixed_eval_length,
        "runtime_binding_path": str(binding_path),
        "runtime_binding_sha256": runtime_binding["binding_sha256"],
        "runtime_observation": runtime_observation,
        "runtime_check": runtime_check,
        "acceleration_backend": acceleration_backend,
        "candidates": results,
        "paired_comparisons": comparisons,
        "retention_shadow_binding": retention_shadow_binding(repo_root),
        "claim_ceiling": (
            "DEVELOPMENT HELDOUT + PINNED PUBLIC RETENTION-SHADOW "
            "COMPARABILITY ONLY / NOT FINAL BANK / NOT EXTERNAL "
            "QUALIFICATION / NOT PROMOTION OR CHECKPOINT-SELECTION AUTHORITY"
        ),
    }


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--salt", default="v10r2-heldout-v1")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--runtime-binding", type=Path)
    parser.add_argument("--checkpoint-panel", type=Path)
    parser.add_argument(
        "--checkpoint-panel-section",
        choices=("anchor_panel", "confirmatory_panel", "combined_panel"),
        default="combined_panel",
    )
    parser.add_argument(
        "--acceleration-backend",
        choices=("torch_reference", "fla_triton"),
        default="fla_triton",
    )
    parser.add_argument(
        "--fixed-eval-length",
        type=int,
        default=DEFAULT_FIXED_EVAL_LENGTH,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    candidates = parse_candidate_specs(args.candidate)
    result = run_lane_b_heldout_eval(
        args.repo_root,
        validation_jsonl=args.validation_jsonl,
        sample_count=args.sample_count,
        salt=args.salt,
        candidates=candidates,
        runtime_binding_path=args.runtime_binding,
        acceleration_backend=args.acceleration_backend,
        fixed_eval_length=args.fixed_eval_length,
        checkpoint_panel_path=args.checkpoint_panel,
        checkpoint_panel_section=args.checkpoint_panel_section,
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
