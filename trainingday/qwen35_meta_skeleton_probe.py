"""No-weights, CPU/meta-only official Qwen3.5-4B PEFT name qualification.

Use a separately downloaded SHA-pinned official config.json, never model
shards. All model and adapter tensors must remain on the PyTorch meta device.
This is a structural compatibility probe, not GPU fit or optimizer evidence.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import stat

from trainingday.qwen35_adapter_coverage import audit_loaded_minimal_lora_model


OFFICIAL_CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_CONFIG_BYTES = 64 * 1024


class MetaSkeletonProbeError(ValueError):
    """Insufficient official metadata or incompatible meta-only PEFT structure."""


def verified_config(path: Path) -> dict:
    """Read only bounded regular raw config bytes with a fixed official pin."""
    path = Path(path)
    if path.is_symlink():
        raise MetaSkeletonProbeError("official config path must not be symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        try:
            record = os.fstat(fd)
            if not stat.S_ISREG(record.st_mode) or record.st_size > MAX_CONFIG_BYTES:
                raise MetaSkeletonProbeError("official config must be bounded regular file")
            raw = os.read(fd, MAX_CONFIG_BYTES + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise MetaSkeletonProbeError("official config inaccessible") from exc
    if (len(raw) > MAX_CONFIG_BYTES
            or hashlib.sha256(raw).hexdigest() != OFFICIAL_CONFIG_SHA256):
        raise MetaSkeletonProbeError("raw config bytes differ from pinned official SHA256")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MetaSkeletonProbeError("invalid official config JSON") from exc
    if (type(payload) is not dict or payload.get("model_type") != "qwen3_5"
            or payload.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or type(payload.get("text_config")) is not dict
            or payload["text_config"].get("model_type") != "qwen3_5_text"):
        raise MetaSkeletonProbeError("official nested text config mismatch")
    return payload["text_config"]


def run_meta_probe(config_path: Path) -> dict:
    """Instantiate *empty* official text skeleton plus minimal PEFT adapter."""
    text = verified_config(config_path)  # Fail before importing heavy dependencies.
    import torch
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if torch.version.cuda is not None:
        raise MetaSkeletonProbeError("probe requires CPU-only PyTorch build")
    config = Qwen3_5TextConfig(**text)
    if (config.model_type != "qwen3_5_text"
            or tuple(config.layer_types) != tuple(text["layer_types"])):
        raise MetaSkeletonProbeError("text config changed during Transformers parsing")
    with init_empty_weights(include_buffers=True):
        skeleton = Qwen3_5ForCausalLM(config)
        model = get_peft_model(
            skeleton,
            LoraConfig(r=4, lora_alpha=8, lora_dropout=0, bias="none",
                       task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]),
        )

    parameters = tuple(model.named_parameters())
    if not parameters or any(parameter.device.type != "meta" for _, parameter in parameters):
        raise MetaSkeletonProbeError("real parameter tensors allocated outside meta")
    # This actual named_modules/named_parameters probe is the novel evidence
    # beyond metadata checks in #237 and #239. No forward or optimizer calls.
    coverage = audit_loaded_minimal_lora_model(model, config.layer_types)
    if (coverage["target_projection_modules"] != 16
            or coverage["gdn_layers_with_adapters"] != 0
            or coverage["trainable_lora_a_b_tensors"] != 32):
        raise MetaSkeletonProbeError("official meta adapter inventory mismatch")
    return {
        "status": "PINNED_OFFICIAL_QWEN35_TEXT_META_PEFT_NAMES_VERIFIED_ONLY",
        "official_config_sha256": OFFICIAL_CONFIG_SHA256,
        "official_revision": OFFICIAL_REVISION,
        "frameworks": {
            name: metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "peft")
        },
        "model_class": type(skeleton).__name__,
        "meta_parameters_inspected": len(parameters),
        "meta_only": True,
        "adapter_name_coverage": coverage,
        "pretrained_weights_loaded": False,
        "weight_shards_authenticated": False,
        "cuda_used": False,
        "optimizer_steps_observed": 0,
        "physical_training_authorized": False,
        "hardware_fit": "UNKNOWN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_meta_probe(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
