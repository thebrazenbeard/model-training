"""CPU-only name/adapter coverage gate for the minimal Qwen3.5 LoRA pilot.

This verifies *instantiated* module and complete parameter names, including
frozen adapter subtrees, plus trainable-parameter names. A successful
result is not proof of official model custody, executable CUDA fit or training.
The pilot deliberately covers q/v in 8 full-attention decoder layers and no
Gated DeltaNet layers. No torch/PEFT imports or model construction occur here.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


class AdapterCoverageError(ValueError):
    """Unrecognized decoder shape, missing adapters, or overclaimed coverage."""


_MODULE = re.compile(
    r"(?:^|.*\.)model\.layers\.(0|[1-9][0-9]*)\."
    r"(self_attn\.(?:q_proj|v_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\Z"
)
_ADAPTER_COMPONENT = re.compile(r"(?:^|\.)lora_[A-Za-z0-9_]+(?=\.|$)")
_GDN = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
_FULL = ("self_attn.q_proj", "self_attn.v_proj")
_LORA_WEIGHTS = ("lora_A.default.weight", "lora_B.default.weight")


def audit_minimal_lora_coverage(
    *, layer_types: Iterable[str], module_names: Iterable[str],
    trainable_parameter_names: Iterable[str],
    all_parameter_names: Iterable[str],
    claimed_scope: str = "MINIMAL_QV_ONLY",
) -> dict:
    """Fail closed on exact 32-layer structure and 16 adapted q/v projections.

    This is a name-level preflight, not a proof that gradient tensors changed.
    The caller must supply complete names from the real *loaded* model after PEFT.
    Synthetic names only validate this function's logic, not the model.
    """
    if claimed_scope != "MINIMAL_QV_ONLY":
        raise AdapterCoverageError("minimal q/v pilot cannot claim full-decoder coverage")
    types = tuple(layer_types)
    if (len(types) != 32
            or any(type(kind) is not str or kind not in
                   ("linear_attention", "full_attention") for kind in types)
            or Counter(types) != {"linear_attention": 24, "full_attention": 8}):
        raise AdapterCoverageError("official hybrid 24 GDN / 8 full layer types missing")

    expected = {
        f"{idx}.{suffix}"
        for idx, kind in enumerate(types)
        for suffix in (_GDN if kind == "linear_attention" else _FULL)
    }
    names = tuple(module_names)
    observed: dict[str, str] = {}
    for name in names:
        if type(name) is not str:
            raise AdapterCoverageError("non-string model module name")
        match = _MODULE.fullmatch(name)
        if match is None:
            continue
        canonical = f"{int(match.group(1))}.{match.group(2)}"
        if canonical in observed:
            raise AdapterCoverageError("duplicate decoder projection identity")
        observed[canonical] = name
    if set(observed) != expected:
        raise AdapterCoverageError("actual decoder modules differ from pinned hybrid shape")

    adapted = {
        observed[f"{idx}.{suffix}"]
        for idx, kind in enumerate(types)
        if kind == "full_attention"
        for suffix in _FULL
    }
    expected_trainable = {
        f"{name}.{weight}" for name in adapted for weight in _LORA_WEIGHTS
    }
    # Inspect the *whole* module graph, not just modules with trainable weights.
    # A frozen GDN LoRA would otherwise be missed and falsely reported absent.
    for name in names:
        marker = _ADAPTER_COMPONENT.search(name)
        if marker is not None and name[:marker.start()].rstrip(".") not in adapted:
            raise AdapterCoverageError("adapter module exists outside minimal q/v targets")
    expected_adapter_modules = {
        f"{name}.lora_{side}.default"
        for name in adapted for side in ("A", "B")
    }
    if not expected_adapter_modules.issubset(set(names)):
        raise AdapterCoverageError("target LoRA A/B adapter module subtrees missing")

    trainables = tuple(trainable_parameter_names)
    parameters = tuple(all_parameter_names)
    if (any(type(name) is not str for name in trainables)
            or len(trainables) != len(set(trainables))
            or set(trainables) != expected_trainable):
        raise AdapterCoverageError(
            "trainable LoRA A/B tensors missing, unexpected, or outside q/v pilot"
        )
    if (any(type(name) is not str for name in parameters)
            or len(parameters) != len(set(parameters))
            or not set(trainables).issubset(set(parameters))
            or {name for name in parameters if _ADAPTER_COMPONENT.search(name)}
            != expected_trainable):
        raise AdapterCoverageError(
            "frozen, missing, or foreign adapter parameter inventory detected"
        )
    return {
        "status": "INSTANTIATED_MODULE_NAME_SHAPE_VALIDATED_NOT_AUTHENTICATED",
        "adapter_scope": "MINIMAL_QV_ONLY",
        "text_decoder_layers_seen": 32,
        "full_attention_layers_with_qv_adapters": 8,
        "gdn_layers_with_adapters": 0,
        "target_projection_modules": len(adapted),
        "trainable_lora_a_b_tensors": len(expected_trainable),
        "frozen_or_foreign_adapter_names_rejected": True,
        "architecture_full_coverage": False,
        "official_weights_authenticated": False,
        "physical_training_authorized": False,
        "optimizer_steps_observed": 0,
    }


def audit_loaded_minimal_lora_model(model, layer_types: Iterable[str]) -> dict:
    """Probe an actual PEFT model's named modules and trainable parameters."""
    return audit_minimal_lora_coverage(
        layer_types=layer_types,
        module_names=(name for name, _ in model.named_modules()),
        trainable_parameter_names=(
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        all_parameter_names=(name for name, _ in model.named_parameters()),
    )
