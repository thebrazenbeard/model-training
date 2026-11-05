"""Propose a namespace-safe hybrid Qwen3.5 LoRA target list, source-only.

Pinned official checkpoint names are projected exactly as the text-only
Qwen3_5ForCausalLM loader's ``text_key_mapping`` specifies. PEFT's documented
list-form target_modules exact/suffix matching is simulated over the resulting
parameter names; no PEFT import, model construction, weight read or optimizer
step occurs. Module existence, type/support and 4 GB fit remain unverified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainingday.qwen35_lora_target_coverage import inspect_lowmem_source
from trainingday.qwen35_memory_source_admission import (
    MAX_CONFIG_BYTES, MAX_INDEX_BYTES, MetadataAdmissionError, _read_bounded_file,
    _strict_json, admit_pinned_official_metadata,
)
from trainingday.qwen_text_checkpoint import IGNORED_PREFIXES, TEXT_PREFIX


SCHEMA = "TRAININGDAY_QWEN35_HYBRID_LORA_TARGET_CANDIDATE_V1"
TEXT_LOADER_PREFIX = "model."
LAYER_COUNT = 32
# The older q/v or three-GDN-projection *reference subset* is insufficient to
# describe all known nn.Linear attention-branch modules. The pinned index and
# Qwen3.5 Transformers source expose four full-attention projections and five
# Gated DeltaNet projections. This is still not an all-model LoRA promise.
HYBRID_LINEAR_PROJECTIONS = {
    "full_attention": ("self_attn.q_proj", "self_attn.k_proj",
                       "self_attn.v_proj", "self_attn.o_proj"),
    "linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                         "linear_attn.in_proj_b", "linear_attn.in_proj_a",
                         "linear_attn.out_proj"),
}
EXPECTED_TARGETS = 24 * 5 + 8 * 4


class HybridLoRACandidateError(ValueError):
    """Static target matching cannot establish an exact projected inventory."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HybridLoRACandidateError(reason)


def _peft_list_suffix_match(module: str, target: str) -> bool:
    """Bounded simulation of the current PEFT list-form exact/suffix policy."""
    return module == target or module.endswith("." + target)


def _layer_types(config: dict) -> list[str]:
    text = config.get("text_config") if type(config) is dict else None
    types = text.get("layer_types") if type(text) is dict else None
    _require(type(types) is list and len(types) == LAYER_COUNT
             and all(type(t) is str and t in HYBRID_LINEAR_PROJECTIONS for t in types)
             and types.count("linear_attention") == 24
             and types.count("full_attention") == 8,
             "pinned hybrid decoder types absent or inconsistent")
    return types


def _project_index_modules(index: dict) -> tuple[set[str], set[str]]:
    """Source map only; exact model.named_modules() must be verified later."""
    weight_map = index.get("weight_map") if type(index) is dict else None
    _require(type(weight_map) is dict and 100 <= len(weight_map) <= 16_000
             and all(type(k) is str and type(v) is str
                     for k, v in weight_map.items()),
             "bounded text checkpoint index absent")
    projected = set()
    excluded = set()
    for tensor in weight_map:
        if tensor.startswith(TEXT_PREFIX):
            mapped = TEXT_LOADER_PREFIX + tensor[len(TEXT_PREFIX):]
            _require(mapped.startswith("model.") and len(mapped) > len("model."),
                     "empty projected text tensor name")
            if mapped.endswith(".weight"):
                projected.add(mapped[:-len(".weight")])
        elif tensor.startswith(IGNORED_PREFIXES):
            if tensor.endswith(".weight"):
                excluded.add(tensor[:-len(".weight")])
        else:
            raise HybridLoRACandidateError("unexpected non-text checkpoint namespace")
    return projected, excluded


def propose_and_check(config: dict, index: dict, *,
                      candidate_targets: list[str] | None = None) -> dict:
    """Shape-only helper. Caller inputs never attest official metadata custody.

    ``candidate_targets`` exists only for deterministic negative tests and
    alternate proposal review; the production CLI uses the 152 derived
    attention-branch projection names, not every decoder nn.Linear module.
    """
    _require(TEXT_PREFIX == "model.language_model."
             and IGNORED_PREFIXES == ("model.visual.", "mtp."),
             "text projection source mapping changed")
    types = _layer_types(config)
    projected, excluded = _project_index_modules(index)
    exact = [f"model.layers.{layer}.{suffix}"
             for layer, kind in enumerate(types)
             for suffix in HYBRID_LINEAR_PROJECTIONS[kind]]
    _require(len(exact) == EXPECTED_TARGETS and len(set(exact)) == EXPECTED_TARGETS
             and all(item in projected for item in exact),
             "source index missing an expected hybrid projection")
    targets = exact if candidate_targets is None else candidate_targets
    _require(type(targets) is list and 1 <= len(targets) <= EXPECTED_TARGETS
             and len(set(str(item) for item in targets)) == len(targets)
             and all(type(item) is str and item.startswith("model.layers.")
                     and item in exact for item in targets),
             "candidate targets must be exact, namespace-qualified decoder projections")
    # Full checkpoint intentionally includes MTP and vision. Any broad suffix
    # that hits these would be unsafe if a future caller switched loader types.
    collision = sorted(module for module in excluded
                       if any(_peft_list_suffix_match(module, target) for target in targets))
    _require(not collision, "candidate would accidentally match MTP/vision namespace")
    matches: dict[str, list[str]] = {
        target: sorted(module for module in projected
                       if _peft_list_suffix_match(module, target))
        for target in targets
    }
    _require(all(len(hit) == 1 and hit[0] == target for target, hit in matches.items()),
             "candidate PEFT suffix target ambiguous or absent in text projection")
    all_matches = {name for names in matches.values() for name in names}
    complete = all_matches == set(exact)
    gdn = sorted(int(target.split(".")[2]) for target in all_matches
                 if ".linear_attn." in target)
    attention = sorted(int(target.split(".")[2]) for target in all_matches
                       if ".self_attn." in target)
    gdn_layers = sorted(set(gdn))
    attention_layers = sorted(set(attention))
    baseline_broad = {"q_proj", "v_proj"}
    baseline_auxiliary_collision = sorted(module for module in excluded
        if any(_peft_list_suffix_match(module, suffix) for suffix in baseline_broad))
    return {
        "schema": SCHEMA,
        "status": ("SOURCE_ONLY_ALL_32_ATTENTION_BRANCHES_TARGETED_UNQUALIFIED"
                   if complete else "SOURCE_ONLY_PARTIAL_HYBRID_COVERAGE_HOLD"),
        "candidate_origin": "CALLER_SUPPLIED_SHAPE_ONLY",
        "pinned_official_metadata_bytes_verified": False,
        "text_checkpoint_projection": "model.language_model.* -> model.*",
        "text_loader_class": "Qwen3_5ForCausalLM",
        "proposed_lora_config": {
            "r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
            "bias": "none", "task_type": "CAUSAL_LM",
            "target_modules": targets,
        },
        "peft_match_method": "LIST_FORM_EXACT_OR_DOT_SUFFIX_SIMULATION_NOT_RUNTIME_PROOF",
        "target_module_count": len(targets),
        "projected_matched_module_count": len(all_matches),
        "expected_hybrid_projection_module_count": len(exact),
        "full_attention_targeted_layers": attention_layers,
        "gdn_targeted_layers": gdn_layers,
        "full_attention_linear_projections_complete": len(attention) == 32,
        "gdn_linear_projections_complete": len(gdn) == 120,
        "projected_text_coverage_complete": complete,
        "potential_mtp_vision_target_collisions": collision,
        "baseline_qv_would_match_auxiliary_modules_if_loaded": baseline_auxiliary_collision,
        "source_only_limits": [
            "TARGETS_ARE_NOT_INSTALLED_IN_PROTECTED_LOWMEM_KERNEL",
            "NO_ACTUAL_MODEL_NAMED_MODULES_OR_PEFT_INJECTION_OBSERVED",
            "GDN_NF4_BNB_PEFT_COMPATIBILITY_UNPROVEN",
            "OPTIONAL_CAUSAL_CONV1D_OR_FLASH_LINEAR_ATTENTION_FALLBACK_MEMORY_UNKNOWN",
            "ACTIVATION_AND_QUANTIZATION_STORAGE_PEAKS_UNKNOWN",
            "BF16_CUDA_SUPPORT_AND_FOUR_GB_FIT_UNKNOWN",
            "SIXTY_GB_PAGEFILE_IS_NOT_GPU_MEMORY",
        ],
        "hardware_fit": "UNKNOWN",
        "physical_launch_authorized": False,
        "training_authorized": False,
        "optimizer_steps_observed": 0,
    }


def audit_pinned_hybrid_candidate(config_bytes: bytes, index_bytes: bytes) -> dict:
    """Real source entry: independently pinned *metadata*, never model weights."""
    admit_pinned_official_metadata(config_bytes, index_bytes)
    current = inspect_lowmem_source()
    _require(current["target_modules"] == ["q_proj", "v_proj"]
             and current["physical_authority_guard_precedes_model_path"] is True,
             "protected q/v low-memory pilot has changed")
    report = propose_and_check(_strict_json(config_bytes), _strict_json(index_bytes))
    report["candidate_origin"] = "PINNED_METADATA_AND_CHECKED_IN_LOWMEM_SOURCE"
    report["pinned_official_metadata_bytes_verified"] = True
    report["protected_kernel_qv_only_unchanged"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Source-only hybrid LoRA expansion")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if not args.config or not args.index:
        print(json.dumps({"schema": SCHEMA, "status": "OFFICIAL_METADATA_MISSING_HARD_HOLD",
                          "training_authorized": False, "hardware_fit": "UNKNOWN"},
                         sort_keys=True))
        return 2
    try:
        out = audit_pinned_hybrid_candidate(
            _read_bounded_file(args.config, MAX_CONFIG_BYTES),
            _read_bounded_file(args.index, MAX_INDEX_BYTES))
    except (MetadataAdmissionError, HybridLoRACandidateError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "SOURCE_REJECTED_HARD_HOLD",
                          "reason": str(exc), "training_authorized": False,
                          "hardware_fit": "UNKNOWN"}, sort_keys=True))
        return 1
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
