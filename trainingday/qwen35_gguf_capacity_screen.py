"""Read-only capacity screen for a *proposed* official Qwen3.5 GGUF conversion.

This is a coarse rejection screen, never process admission. Input capacity
figures are caller-declared observations, not authenticated host telemetry.
The converter, quantizer, model and checkpoint are never opened or executed.
"""

from __future__ import annotations

import argparse
import json

from .official_qwen35_gguf_workflow import (
    BASE_ID, BASE_REVISION, CANONICAL_FILES, LLAMA_COMMIT, R16_RUN,
)


SHARD_BYTES = sum(size for name, size, _ in CANONICAL_FILES if name.endswith(".safetensors"))
GIB = 1024 ** 3
SCHEMA = "TRAININGDAY_QWEN35_GGUF_CAPACITY_SCREEN_V1"


def command_templates() -> tuple[tuple[str, ...], ...]:
    """Display inert argv templates, never shell strings or executable grants."""
    return (
        ("python", "convert_hf_to_gguf.py", "<PINNED_OFFICIAL_HF_DIR>",
         "--outfile", "<OWNER_BF16_GGUF>", "--outtype", "bf16", "--no-mtp"),
        ("llama-quantize", "<OWNER_BF16_GGUF>", "<OWNER_Q4_K_M_GGUF>", "Q4_K_M"),
        ("python", "convert_lora_to_gguf.py", "<PINNED_CORRECT8_LORA_DIR>",
         "--base", "<PINNED_OFFICIAL_HF_DIR>", "--outfile",
         "<OWNER_CORRECT8_LORA_GGUF>", "--outtype", "bf16"),
    )


def screen(available_ram_bytes: int, free_scratch_bytes: int) -> dict:
    """Fail low-headroom telemetry, but do not approve adequate-looking values.

    Pinned llama.cpp quantizer guidance warns models may be fully loaded into
    memory. The original two safetensors total is only a planning proxy for
    the unmeasured BF16 GGUF footprint; it is NOT a measured peak-RSS bound.
    The scratch screen allows two such copies (BF16 and quantized output),
    still excluding temp files, cache, logs, and external disk ownership.
    """
    for label, value in (("available_ram_bytes", available_ram_bytes),
                         ("free_scratch_bytes", free_scratch_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative exact integer")

    ram_short = max(0, SHARD_BYTES - available_ram_bytes)
    scratch_proxy = 2 * SHARD_BYTES
    disk_short = max(0, scratch_proxy - free_scratch_bytes)
    return {
        "schema": SCHEMA,
        "source": {"model": BASE_ID, "revision": BASE_REVISION,
                   "converter_commit": LLAMA_COMMIT, "adapter_run": R16_RUN},
        "capacity_is_caller_declared": True,
        "available_ram_bytes": available_ram_bytes,
        "free_scratch_bytes": free_scratch_bytes,
        "original_shard_bytes": SHARD_BYTES,
        "ram_screen_proxy_bytes": SHARD_BYTES,
        "scratch_screen_proxy_bytes": scratch_proxy,
        "ram_shortfall_against_proxy_bytes": ram_short,
        "scratch_shortfall_against_proxy_bytes": disk_short,
        "capacity_screen": "SHORTFALL" if ram_short or disk_short else "NO_SHORTFALL_PROVEN",
        "capacity_qualified": False,
        "official_gguf_created": False,
        "physical_launch_authorized": False,
        "host_process_attempted": False,
        "inert_command_templates": [list(cmd) for cmd in command_templates()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-ram-bytes", required=True, type=int)
    parser.add_argument("--free-scratch-bytes", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        report = screen(args.available_ram_bytes, args.free_scratch_bytes)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 1  # Even ample untrusted capacity never grants conversion or serving.


if __name__ == "__main__":
    raise SystemExit(main())
