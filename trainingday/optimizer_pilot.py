"""Bounded, CPU-only synthetic rehearsal of an official Qwen3.5 text LoRA plan.

No official model bytes are loaded here. Synthetic SGD on four-byte float
fixtures is *not* a pretrained optimizer step, a GPU memory test, a lease,
or an evaluation of learning. The physical boundary remains fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import stat
import struct
from typing import Mapping

from trainingday.lowmem_experiment import (
    LowMemoryExperimentError, evaluate_optimizer_witness, validate_declaration,
)
from trainingday.official_public_base import (
    expected_source_record, validate_official_source_record,
)
from trainingday.physical_launch_authority import require_physical_launch_authority


SCHEMA = "TRAININGDAY_QWEN35_QLORA_SYNTHETIC_REPLAY_V1"
LAYER = re.compile(r"model\.layers\.([0-9]{1,3})\.self_attn\.(q_proj|v_proj)\Z")
CHECKPOINT_LIMIT = 16 * 1024
# Pinned upstream config.json at BASE_REVISION: text_config.num_hidden_layers=32,
# layer_types=(linear, linear, linear, full_attention) * 8. This is SOURCE
# METADATA, not locally authenticated config bytes or an inspected model graph.
SOURCE_TEXT_LAYER_TYPES = ("linear_attention", "linear_attention",
                           "linear_attention", "full_attention") * 8


@dataclass(frozen=True)
class PilotBounds:
    """Maximum candidate dimensions, not an empirically confirmed fit."""

    steps: int = 3
    sequence_length: int = 64
    rank: int = 4
    batch_size: int = 1
    seed: int = 73504
    device_total_mib: int = 4096
    reserve_mib: int = 512

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("steps", 1, 3), ("sequence_length", 1, 64),
            ("rank", 1, 4), ("batch_size", 1, 1),
            ("seed", 0, 2**32 - 1),
            ("device_total_mib", 1, 4096),
            ("reserve_mib", 256, 2048),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise LowMemoryExperimentError(f"unsafe pilot bound: {name}")
        if self.reserve_mib >= self.device_total_mib:
            raise LowMemoryExperimentError("reserved VRAM exceeds available capacity")


def validate_official_plan(
    declaration: Mapping[str, object], source_record: object,
    module_names: object,
) -> tuple[str, ...]:
    """Check exact *declared* identity and text module inventory shape only.

    A caller-provided list of module names is not a verified model graph.
    At physical launch, the trusted loader must independently inspect the
    pinned local config/index, actually loaded module graph, and frozen base.
    """
    validate_declaration(declaration)
    validate_official_source_record(source_record)
    if (type(module_names) not in (tuple, list)
            or not 2 <= len(module_names) <= 256
            or any(type(name) is not str for name in module_names)
            or len(set(module_names)) != len(module_names)):
        raise LowMemoryExperimentError("untrusted or duplicate LoRA target inventory")
    layers: dict[str, set[str]] = {}
    for name in module_names:
        match = LAYER.fullmatch(name)
        if match is None:
            raise LowMemoryExperimentError("LoRA target outside official text attention")
        index = int(match[1])
        if (index >= len(SOURCE_TEXT_LAYER_TYPES)
                or SOURCE_TEXT_LAYER_TYPES[index] != "full_attention"):
            raise LowMemoryExperimentError("target has no source-declared full-attention layer")
        layers.setdefault(match[1], set()).add(match[2])
    if any(targets != {"q_proj", "v_proj"} for targets in layers.values()):
        raise LowMemoryExperimentError("unpaired text attention LoRA targets")
    return tuple(sorted(module_names))


def memory_preflight(
    *, bounds: PilotBounds, free_mib: int, estimated_peak_mib: int,
) -> dict:
    """Evaluate caller-declared numbers, never infer a measured CUDA fit."""
    if (type(free_mib) is not int or type(estimated_peak_mib) is not int
            or not 0 <= free_mib <= bounds.device_total_mib
            or estimated_peak_mib <= 0):
        raise LowMemoryExperimentError("invalid/unbounded memory estimate")
    estimated_fit = (estimated_peak_mib + bounds.reserve_mib
                     <= min(free_mib, bounds.device_total_mib))
    return {
        "estimated_fit_from_untrusted_inputs": estimated_fit,
        "estimated_peak_mib": estimated_peak_mib,
        "declared_free_mib": free_mib,
        "reserve_mib": bounds.reserve_mib,
        "input_trust": "UNVERIFIED_DECLARATION",
        "physical_headroom_measured": False,
        "load_and_optimizer_peak_measured": False,
        "gpu_lease_issued": False,
        "physical_training_authorized": False,
    }


def _tensor_hash(tensors: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        payload = tensors[key]
        digest.update(len(key).to_bytes(4, "big"))
        digest.update(key.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_fixture_tensors(tensors: object, targets: tuple[str, ...]) -> None:
    expected = {name + ".lora_B.default.weight" for name in targets}
    if (type(tensors) is not dict or set(tensors) != expected
            or any(type(value) is not bytes or len(value) != 4
                   or not math.isfinite(struct.unpack("<f", value)[0])
                   for value in tensors.values())):
        raise LowMemoryExperimentError("incomplete/nonfinite synthetic adapter tensor set")


def persist_reload_fixture(
    path: str | Path, tensors: dict[str, bytes], targets: tuple[str, ...],
) -> dict[str, bytes]:
    """Exclusive local fixture checkpoint, read back and verify exact bytes.

    No arbitrary deletion/overwrite. Callers own any temporary directory.
    """
    _validate_fixture_tensors(tensors, targets)
    output = Path(path)
    if any(part == ".." for part in output.parts) or not output.name:
        raise LowMemoryExperimentError("unsafe checkpoint path")
    output = Path(os.path.abspath(output))
    document = {
        "schema": SCHEMA,
        "tensors": {key: value.hex() for key, value in sorted(tensors.items())},
        "sha256": _tensor_hash(tensors),
    }
    serialized = (json.dumps(document, sort_keys=True, separators=(",", ":"))
                  + "\n").encode("utf-8")
    if len(serialized) > CHECKPOINT_LIMIT:
        raise LowMemoryExperimentError("fixture checkpoint too large")
    dir_fd = file_fd = read_fd = None
    temporary_name = None
    try:
        # Traverse *each* ancestor with O_NOFOLLOW. Keep the final directory
        # FD throughout creation/publish/reload to prevent ancestor swaps.
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        dir_fd = os.open("/", flags)
        for component in output.parent.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        temporary_name = ".trainingday-fixture-" + secrets.token_hex(12)
        file_fd = os.open(temporary_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                          | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
        cursor = 0
        while cursor < len(serialized):
            written = os.write(file_fd, serialized[cursor:])
            if written <= 0:
                raise OSError("short checkpoint write")
            cursor += written
        os.fsync(file_fd)
        # Atomic no-clobber publication: hard-link refuses any existing name.
        # Unlike os.replace/rename, this cannot overwrite a user file.
        os.link(temporary_name, output.name, src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd, follow_symlinks=False)
        os.fsync(dir_fd)
        read_fd = os.open(output.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        if (not stat.S_ISREG(os.fstat(read_fd).st_mode)
                or os.fstat(read_fd).st_ino != os.fstat(file_fd).st_ino):
            raise LowMemoryExperimentError("checkpoint inode changed on reload")
        raw = os.read(read_fd, CHECKPOINT_LIMIT + 1)
        if len(raw) > CHECKPOINT_LIMIT:
            raise LowMemoryExperimentError("reloaded checkpoint too large")
        loaded = json.loads(raw.decode("utf-8"))
        if (type(loaded) is not dict or set(loaded) != {"schema", "tensors", "sha256"}
                or loaded["schema"] != SCHEMA or type(loaded["tensors"]) is not dict):
            raise LowMemoryExperimentError("checkpoint schema mismatch")
        restored = {key: bytes.fromhex(value) for key, value in loaded["tensors"].items()}
        _validate_fixture_tensors(restored, targets)
        if loaded["sha256"] != _tensor_hash(restored) or restored != tensors:
            raise LowMemoryExperimentError("checkpoint reload mismatch")
        return restored
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        raise LowMemoryExperimentError("checkpoint save/reload failed") from exc
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if file_fd is not None:
            os.close(file_fd)
        if temporary_name is not None and dir_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
        if dir_fd is not None:
            os.close(dir_fd)


def replay_synthetic_adapter(
    *, declaration: dict, source_record: dict, module_names: tuple[str, ...],
    bounds: PilotBounds, free_mib: int, estimated_peak_mib: int,
    checkpoint_path: str | Path,
) -> dict:
    """Pure-CPU reproducible toy SGD; never loads the 4B model or tokenizer."""
    targets = validate_official_plan(declaration, source_record, module_names)
    memory = memory_preflight(bounds=bounds, free_mib=free_mib,
                              estimated_peak_mib=estimated_peak_mib)
    if not memory["estimated_fit_from_untrusted_inputs"]:
        raise LowMemoryExperimentError("dry-run estimated VRAM budget exceeded")
    rng = random.Random(bounds.seed)
    params = {target + ".lora_B.default.weight": struct.pack("<f", 0.0)
              for target in targets}
    initial_hash = _tensor_hash(params)
    witnesses = []
    for step in range(bounds.steps):
        old = params.copy()
        gradients = {key: 0.1 + rng.random() for key in sorted(old)}
        norm = math.sqrt(sum(value * value for value in gradients.values()))
        params = {
            key: struct.pack("<f", struct.unpack("<f", old[key])[0]
                             - 0.01 * gradients[key])
            for key in sorted(old)
        }
        witness = evaluate_optimizer_witness(
            before=old, after=params, loss=1.0 / (step + 1),
            gradient_norm=norm, optimizer_step_returned=True,
        )
        witnesses.append(witness)
    _validate_fixture_tensors(params, targets)
    restored = persist_reload_fixture(checkpoint_path, params, targets)
    final_hash = _tensor_hash(restored)
    if initial_hash == final_hash or witnesses[-1]["after_sha256"] != final_hash:
        raise LowMemoryExperimentError("missing final synthetic adapter update")
    source = expected_source_record()["identity"]
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_HARD_HOLD",
        "identity": {"base_id": source["repository"], "base_revision": source["revision"],
                     "checkpoint_architecture": source["checkpoint_architecture"],
                     "text_loader": source["text_loader"],
                     "tokenizer_sha256_declared_unverified": declaration["files"]["tokenizer.json"],
                     "tokenizer_digest_trust": "UNVERIFIED_DECLARATION",
                     "trusted_tokenizer_pin_matched": False},
        "source_identity_trust": "UNVERIFIED_DECLARATION",
        "module_inventory_trust": "UNVERIFIED_DECLARATION",
        "adapter_targets": list(targets),
        "rank": bounds.rank,
        "sequence_length": bounds.sequence_length,
        "batch_size": bounds.batch_size,
        "seed": bounds.seed,
        "data_provenance": "SYNTHETIC_FIXED_SEED_NO_PROTECTED_DATA",
        "control_type": "CPU_TOY_SGD_NOT_QWEN_PRETRAINED_OPTIMIZER",
        "gradient_source": "SYNTHETIC_PRNG_NOT_MODEL_AUTOGRAD",
        "loss_source": "SYNTHETIC_SCHEDULE_NOT_MODEL_FORWARD",
        "synthetic_updates": len(witnesses),
        "physical_pretrained_optimizer_steps": 0,
        "initial_adapter_sha256": initial_hash,
        "final_adapter_sha256": final_hash,
        "checkpoint_reload_equal": restored == params,
        "gradient_norms_finite_positive": all(w["gradient_norm"] > 0 for w in witnesses),
        "step_witnesses": witnesses,
        "resource": memory,
        "gpu_lease_id": None,
        "rights_verified": False,
        "base_bytes_verified": False,
        "tokenizer_bytes_verified": False,
        "official_graph_verified": False,
        "physical_training_authorized": False,
        "model_quality_proved": False,
    }


def physical_optimizer_launch(*_args: object, **_kwargs: object) -> None:
    """A separate hard hold; never dispatch a process or touch GPU resources."""
    require_physical_launch_authority(operation="Qwen3.5 low-memory optimizer pilot")
