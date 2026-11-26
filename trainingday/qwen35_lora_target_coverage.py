"""Static Qwen3.5 hybrid-decoder LoRA coverage check for the low-memory pilot.

This inspects *source syntax* plus pinned config/index bytes, never executes the
kernel, loads weights, probes the host, or grants physical training authority.
The checkpoint index lists tensors; it does not prove PEFT runtime module
injection or support for every hybrid attention implementation.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import re
import textwrap

from trainingday.lowmem_experiment import one_step_kernel, source_plan
from trainingday.qwen35_memory_source_admission import (
    MAX_CONFIG_BYTES, MAX_INDEX_BYTES, MetadataAdmissionError, _read_bounded_file,
    _strict_json, admit_pinned_official_metadata,
)


SCHEMA = "TRAININGDAY_QWEN35_STATIC_LORA_COVERAGE_V1"
LAYER_TYPES = {"full_attention", "linear_attention"}
REFERENCE_PROJECTIONS = {
    "full_attention": ("self_attn.q_proj", "self_attn.v_proj"),
    "linear_attention": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                         "linear_attn.out_proj"),
}
MODULE_NAME = re.compile(r"[A-Za-z][A-Za-z_0-9]*(?:\.[A-Za-z][A-Za-z_0-9]*)*\Z")
DECODER_KEY = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)\Z")


class StaticCoverageError(ValueError):
    """The source candidate or index cannot support a bounded coverage claim."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise StaticCoverageError(reason)


def _call_named(root: ast.AST, name: str) -> ast.Call:
    hits = [node for node in ast.walk(root) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
            and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == name]
    _require(len(hits) == 1, f"expected exactly one static {name} call")
    return hits[0]


def _keyword(call: ast.Call, name: str) -> ast.AST:
    matches = [item.value for item in call.keywords if item.arg == name]
    _require(len(matches) == 1 and not any(item.arg is None for item in call.keywords),
             f"{name} must be a single static keyword without **kwargs")
    return matches[0]


def _literal(call: ast.Call, name: str):
    try:
        return ast.literal_eval(_keyword(call, name))
    except (ValueError, TypeError, MemoryError, RecursionError) as exc:
        raise StaticCoverageError(f"{name} must be a source literal") from exc


def extract_kernel_source_contract(source: str) -> dict:
    """Reject nonliteral/dangerous static settings instead of guessing effects.

    Caller-supplied text is useful only for testing this parser. Production
    reads the checked-in, hard-held ``one_step_kernel`` source itself.
    """
    _require(type(source) is str and len(source) <= 80_000, "kernel source invalid")
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise StaticCoverageError("unparseable low-memory kernel source") from exc
    quant = _call_named(tree, "BitsAndBytesConfig")
    lora = _call_named(tree, "LoraConfig")
    loader = _call_named(tree, "from_pretrained")
    preparation = _call_named(tree, "prepare_model_for_kbit_training")
    guard = _call_named(tree, "require_physical_launch_authority")
    _require(guard.lineno < min(quant.lineno, lora.lineno, loader.lineno,
                              preparation.lineno),
             "physical launch authority guard no longer precedes the model path")
    _require(_literal(quant, "load_in_4bit") is True, "4-bit quantization not statically bound")
    _require(_literal(quant, "bnb_4bit_quant_type") == "nf4", "NF4 source binding changed")
    _require(_literal(quant, "bnb_4bit_use_double_quant") is True,
             "nested quantization source binding changed")
    dtype = _keyword(quant, "bnb_4bit_compute_dtype")
    _require(isinstance(dtype, ast.Attribute) and dtype.attr == "bfloat16"
             and isinstance(dtype.value, ast.Name) and dtype.value.id == "torch",
             "BF16 source binding changed; hardware support remains unknown")
    _require(type(_literal(loader, "device_map")) is int
             and _literal(loader, "device_map") == 0,
             "device_map must bind a single GPU, not inferred CPU offload")
    _require(_literal(loader, "local_files_only") is True,
             "loader would no longer remain local-only")
    _require(_literal(preparation, "use_gradient_checkpointing") is True,
             "gradient checkpointing source binding changed")
    _require(type(_literal(lora, "r")) is int and _literal(lora, "r") > 0,
             "LoRA rank must be positive")
    targets = _literal(lora, "target_modules")
    _require(type(targets) is list and 1 <= len(targets) <= 16
             and len(targets) == len(set(str(x) for x in targets))
             and all(type(value) is str and MODULE_NAME.fullmatch(value)
                     for value in targets),
             "LoRA target_modules must be a unique, bounded literal suffix list")
    return {
        "target_modules": targets,
        "lora_rank": _literal(lora, "r"),
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "torch.bfloat16",
        "single_device_map": 0,
        "local_files_only": True,
        "gradient_checkpointing_requested": True,
        "physical_authority_guard_precedes_model_path": True,
    }


def inspect_lowmem_source() -> dict:
    """Cross-check the advertised source plan against the actual kernel AST."""
    result = extract_kernel_source_contract(inspect.getsource(one_step_kernel))
    plan = source_plan()
    candidate = plan.get("candidates", {}).get("nf4_minimal_gpu", {})
    _require(candidate.get("target_modules") == result["target_modules"]
             and candidate.get("device_map") == result["single_device_map"]
             and candidate.get("lora_r") == result["lora_rank"]
             and candidate.get("activation_checkpointing") is True
             and candidate.get("cpu_offload") is False,
             "advertised low-memory plan differs from protected kernel source")
    _require(plan.get("authorizes_training") is False
             and plan.get("optimizer_steps_observed") == 0,
             "low-memory source-only hold declaration changed")
    return result


def assess_projection_inventory(config: dict, index: dict, contract: dict) -> dict:
    """Pure index-based potential suffix match; does NOT load PEFT modules.

    When called with synthetic dictionaries this is only a shape test. Only
    ``audit_pinned_lowmem_source`` can attest exact pinned metadata bytes.
    """
    _require(type(config) is dict and type(index) is dict and type(contract) is dict,
             "source inventory must be dictionaries")
    text_config = config.get("text_config")
    _require(type(text_config) is dict, "text decoder configuration unavailable")
    types = text_config.get("layer_types")
    weight_map = index.get("weight_map")
    targets = contract.get("target_modules")
    _require(type(types) is list and len(types) == 32
             and all(t in LAYER_TYPES for t in types)
             and types.count("linear_attention") == 24
             and types.count("full_attention") == 8,
             "hybrid layer-type inventory unavailable")
    _require(type(weight_map) is dict and len(weight_map) >= 100
             and all(type(k) is str and type(v) is str
                     for k, v in weight_map.items()),
             "weight-map inventory unavailable")
    _require(type(targets) is list and 1 <= len(targets) <= 16
             and all(type(t) is str and MODULE_NAME.fullmatch(t) for t in targets)
             and len(set(targets)) == len(targets),
             "unsafe or nonliteral target list")
    selected = set()
    seen_targets = set()
    for key in weight_map:
        if not key.endswith(".weight"):
            continue
        module = key[:-len(".weight")]
        matches = [target for target in targets
                   if module == target or module.endswith("." + target)]
        if matches:
            seen_targets.update(matches)
            selected.add(module)
    _require(seen_targets == set(targets),
             "unmatched LoRA suffix in declared checkpoint tensor index")
    by_layer: dict[int, set[str]] = {layer: set() for layer in range(32)}
    non_decoder = []
    other_decoder = []
    for module in sorted(selected):
        match = DECODER_KEY.fullmatch(module)
        if match is None:
            non_decoder.append(module)
            continue
        layer = int(match.group(1))
        if not 0 <= layer < 32:
            other_decoder.append(module)
            continue
        projection = match.group(2)
        if projection not in REFERENCE_PROJECTIONS[types[layer]]:
            other_decoder.append(module)
            continue
        by_layer[layer].add(projection)
    _require(not other_decoder,
             "LoRA target matches unexpected or out-of-family decoder module")
    full = [i for i, kind in enumerate(types) if kind == "full_attention"]
    gdn = [i for i, kind in enumerate(types) if kind == "linear_attention"]
    any_full = [i for i in full if by_layer[i]]
    any_gdn = [i for i in gdn if by_layer[i]]
    complete_full = [i for i in full
                     if set(REFERENCE_PROJECTIONS[types[i]]).issubset(by_layer[i])]
    complete_gdn = [i for i in gdn
                    if set(REFERENCE_PROJECTIONS[types[i]]).issubset(by_layer[i])]
    decoder_count = len(any_full) + len(any_gdn)
    complete_count = len(complete_full) + len(complete_gdn)
    return {
        "schema": SCHEMA,
        "status": ("SOURCE_ONLY_REFERENCE_PROJECTIONS_COMPLETE_HARDWARE_UNKNOWN"
                   if complete_count == 32 else
                   "SOURCE_ONLY_HYBRID_DECODER_COVERAGE_INADEQUATE"),
        "audit_scope": "POTENTIAL_PEFT_SUFFIX_MATCHES_AGAINST_CHECKPOINT_TENSOR_INDEX_ONLY",
        "kernel_source_contract": contract,
        "contract_origin": "CALLER_SUPPLIED_SHAPE_TEST_ONLY",
        "checked_in_kernel_source_inspected": False,
        # Shape-only helper cannot be used to self-attest source custody.
        "pinned_official_metadata_bytes_verified": False,
        "official_origin_authenticated": False,
        "official_weight_shards_authenticated": False,
        "candidate_decoder_layers": 32,
        "full_attention_layers": len(full),
        "gated_deltanet_layers": len(gdn),
        "full_attention_targeted_layers": any_full,
        "gdn_targeted_layers": any_gdn,
        "targeted_decoder_layers": decoder_count,
        "targeted_decoder_fraction": decoder_count / 32,
        "reference_projection_complete_decoder_layers": complete_count,
        "reference_projection_complete_full_attention_layers": complete_full,
        "reference_projection_complete_gdn_layers": complete_gdn,
        "matched_decoder_projection_modules": sum(len(modules) for modules in by_layer.values()),
        "potential_non_decoder_suffix_matches": non_decoder,
        "quantization_static_limits": [
            "NF4_DOES_NOT_PROVE_ALL_TENSORS_ARE_QUANTIZED",
            "QUANTIZATION_METADATA_AND_ACTIVATION_PEAKS_NOT_MEASURED",
            "BF16_HARDWARE_CAPABILITY_UNVERIFIED",
            "PEFT_RUNTIME_MODULE_INJECTION_UNVERIFIED",
            "FOUR_GB_GPU_MEMORY_FIT_UNVERIFIED",
            "SIXTY_GB_PAGEFILE_DOES_NOT_INCREASE_VRAM",
        ],
        "hardware_fit": "UNKNOWN",
        "physical_launch_authorized": False,
        "training_authorized": False,
        "optimizer_steps_observed": 0,
    }


def audit_pinned_lowmem_source(config_bytes: bytes, index_bytes: bytes) -> dict:
    """Only production entry: require exact official metadata digests first."""
    admit_pinned_official_metadata(config_bytes, index_bytes)
    contract = inspect_lowmem_source()
    result = assess_projection_inventory(_strict_json(config_bytes),
                                         _strict_json(index_bytes), contract)
    # The only production route to this field has already matched the fixed
    # independent raw official metadata SHA256 pins above.
    result["pinned_official_metadata_bytes_verified"] = True
    result["contract_origin"] = "CHECKED_IN_KERNEL_AST_AND_SOURCE_PLAN"
    result["checked_in_kernel_source_inspected"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Source-only Qwen3.5 LoRA coverage")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if not args.config or not args.index:
        print(json.dumps({"schema": SCHEMA, "status": "OFFICIAL_METADATA_MISSING_HARD_HOLD",
                          "training_authorized": False, "hardware_fit": "UNKNOWN"},
                         sort_keys=True))
        return 2
    try:
        result = audit_pinned_lowmem_source(
            _read_bounded_file(args.config, MAX_CONFIG_BYTES),
            _read_bounded_file(args.index, MAX_INDEX_BYTES))
    except (MetadataAdmissionError, StaticCoverageError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "SOURCE_REJECTED_HARD_HOLD",
                          "reason": str(exc), "training_authorized": False,
                          "hardware_fit": "UNKNOWN"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
