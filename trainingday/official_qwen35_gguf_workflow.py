"""Source-only, fail-closed Qwen3.5-4B conversion and inference contract.

This file does not launch a process, open a port, read model weights, or grant
trusted host, rights, artifact-custody, or evaluation authority. A model file
matching a caller's SHA is not independently established conversion lineage.
"""

from __future__ import annotations

import argparse
import copy
import json


SCHEMA = "TRAININGDAY_QWEN35_GGUF_SOURCE_WORKFLOW_V1"
BASE_ID = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_HOST = "VERA-NODE-1"  # Historical artifact receipt, not a live lease.
EXPECTED_SNAPSHOT_FILES = 26
EXPECTED_SNAPSHOT_BYTES = 9_342_822_940
CUSTODY_SIDECAR_SHA256 = (
    "06e730744a578cc2168d1b587711f82a8b7e691545182ff1b814dff95e450fca"
)

# Values from the independent 2026-11-04 VN1 custody readback. These are
# source-side expected bytes, not an authenticated runtime readback by this CLI.
CANONICAL_FILES = (
    ("chat_template.jinja", 7_756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    ("config.json", 3_161, "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
    ("merges.txt", 3_353_259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
    ("model.safetensors-00001-of-00002.safetensors", 5_329_398_688, "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"),
    ("model.safetensors-00002-of-00002.safetensors", 3_990_429_408, "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"),
    ("model.safetensors.index.json", 76_196, "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"),
    ("preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    ("tokenizer.json", 12_807_982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    ("tokenizer_config.json", 16_710, "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"),
    ("video_preprocessor_config.json", 385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    ("vocab.json", 6_722_759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
)

# The remaining 15 files are Hugging Face cache/readback metadata and the
# source receipt. Their per-file SHA is not independently pinned here.
AUXILIARY_FILES = (
    ".cache/huggingface/.gitignore",
    ".cache/huggingface/CACHEDIR.TAG",
    *(f".cache/huggingface/download/{name}.metadata" for name, _, _ in CANONICAL_FILES),
    f".cache/huggingface/trees/{BASE_REVISION}.json",
    "TRAININGDAY_PUBLIC_SOURCE_RECEIPT.json",
)

R16_RUN = "r16-v2-correct8-20261030-b"
R16_FILES = (
    ("adapter_model.safetensors", 32_531_344, "322d91e034cd2b2cf63b077e98061559478a4271c0bf5d4abd8520dca195fc11"),
    ("adapter_config.json", None, "d9c52b766f0eea6bb6e1cbce52ecafd024385822b0e75644ec4dd50b11024816"),
)

LLAMA_COMMIT = "149f5641b931aaf33697d70f9abc90c3e5300247"
LLAMA_RELEASE = "b12098"
LLAMA_WINDOWS_ZIP_SHA256 = "8d142d409515ce7a0f873654170ef0071d945acd1918fd774d374840917bf211"
LLAMA_SERVER_SHA256 = "a34c390204c49a5df796580c7e08962411f3f52f3187d7ef4353717b8ecdb767"
LLAMA_QUANTIZE_SHA256 = "3ede9649dcea3020df0fe1b1f2a1d436e68acdf33d7277c7d6cdf1b3e12555df"
LLAMA_LINUX_TAR_SHA256 = "7e6f56a0940e5db3744a766ba5480ab5e0ba30b61b656b682f3919666da6aae7"


class WorkflowHold(ValueError):
    """Source evidence cannot authorize execution or promote model identity."""


def _strict_equal(actual: object, expected: object) -> bool:
    """Avoid Python's bool/int equality and unrecognized manifest keys."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(item, value) for item, value in zip(actual, expected)
        )
    return actual == expected


def expected_source_declaration() -> dict:
    """Pinned 26-file source inventory with 11 canonical byte hashes."""
    assert len(CANONICAL_FILES) == 11
    assert len(AUXILIARY_FILES) == 15
    assert len(CANONICAL_FILES) + len(AUXILIARY_FILES) == EXPECTED_SNAPSHOT_FILES
    assert sum(size for _, size, _ in CANONICAL_FILES) < EXPECTED_SNAPSHOT_BYTES
    return {
        "schema": SCHEMA,
        "model": {"id": BASE_ID, "revision": BASE_REVISION,
                  "checkpoint_architecture": "Qwen3_5ForConditionalGeneration"},
        "custody_reported_host": SOURCE_HOST,
        "custody_sidecar_sha256": CUSTODY_SIDECAR_SHA256,
        "snapshot_file_count": EXPECTED_SNAPSHOT_FILES,
        "snapshot_total_bytes": EXPECTED_SNAPSHOT_BYTES,
        "canonical_files": [
            {"file": name, "bytes": size, "sha256": digest}
            for name, size, digest in CANONICAL_FILES
        ],
        "auxiliary_file_names_without_pinned_sha": list(AUXILIARY_FILES),
        "r16_correct_label": {
            "run": R16_RUN,
            "files": [{"file": name, "bytes": size, "sha256": digest}
                      for name, size, digest in R16_FILES],
        },
    }


def inspect_untrusted_declaration(value: object) -> dict:
    """Compare untrusted values to static pins; *never* attest physical bytes.

    A matching declaration can still be fabricated, stale, swapped on disk,
    outside rights, or lacking an independently controlled process identity.
    """
    if not _strict_equal(value, expected_source_declaration()):
        raise WorkflowHold("source declaration fails exact typed 26-file pin")
    return {
        "declaration_matches_source_pins": True,
        "source_files_freshly_hashed": False,
        "custody_sidecar_independently_authenticated": False,
        "conversion_lineage_verified": False,
        "physical_launch_allowed": False,
    }


def plan() -> dict:
    """An inspectable plan, not a command queue or trusted launch receipt."""
    return copy.deepcopy({
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_HARD_HOLD",
        "ready": False,
        "source": expected_source_declaration(),
        "toolchain": {
            "repo": "ggml-org/llama.cpp",
            "release": LLAMA_RELEASE,
            "commit": LLAMA_COMMIT,
            "windows_cpu_zip_sha256": LLAMA_WINDOWS_ZIP_SHA256,
            "windows_server_exe_sha256": LLAMA_SERVER_SHA256,
            "windows_quantize_exe_sha256": LLAMA_QUANTIZE_SHA256,
            "linux_cpu_tar_sha256": LLAMA_LINUX_TAR_SHA256,
            "version_execution_qualified_by_this_source": False,
            "model_execution_qualified": False,
            "arbitrary_manifest_executable_accepted": False,
        },
        "conversion": {
            "base": {"program": "convert_hf_to_gguf.py",
                     "input": "authenticated official HF snapshot",
                     "outtype": "bf16", "exclude_nextn": "--no-mtp",
                     "text_projection": "Qwen3_5ForCausalLM",
                     "vision_mmproj_required_for_multimodal": True},
            "quantize": {"program": "llama-quantize", "source_format": "BF16 GGUF",
                         "output_format": "Q4_K_M", "output_sha256": None},
            "correct8_lora": {"program": "convert_lora_to_gguf.py",
                              "source_run": R16_RUN,
                              "base": "same authenticated official HF snapshot",
                              "outtype": "bf16", "output_sha256": None},
            "official_derived_gguf_verified": False,
            "correct8_adapter_derived_gguf_verified": False,
        },
        "inference_identities": [
            {"state": "base", "model": BASE_ID, "adapter": None,
             "backend": "verified Q4_K_M GGUF", "runtime_verified": False},
            {"state": "trained", "model": BASE_ID, "adapter": R16_RUN,
             "backend": "verified Q4_K_M GGUF + verified GGUF LoRA", "runtime_verified": False},
            {"state": "interface", "model": BASE_ID, "adapter": R16_RUN,
             "backend": "owner-held local serving process via Vera interface", "runtime_verified": False},
        ],
        "nonnegotiable_gates": [
            "fresh 26-file host readback and independently pinned 11 canonical hashes",
            "admitted R16 correct-label adapter, tokenizer, and provenance",
            "owner-idle physical RAM/CPU/IO, process identity and resource cap",
            "independently qualified converter and executable dependency set",
            "private input files bound against check-then-use mutation",
            "protected output digest and independent destination readback",
            "fresh process, listener PID binding and owned process-tree teardown",
            "base/trained/interface responses with latency, memory, stops and context receipts",
        ],
        "network_or_process_effects": False,
        "physical_launch_attempted": False,
    })


def execute(*_args, **_kwargs) -> None:
    """Fail closed even if an untrusted caller supplies forged qualification."""
    raise WorkflowHold("source-only plan cannot launch converter, quantizer or model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print source-only plan")
    args = parser.parse_args(argv)
    result = plan()
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(f"{BASE_ID}@{BASE_REVISION}: SOURCE_ONLY_HARD_HOLD")
        print("11 canonical digests pinned, 26-file inventory specified; no model launch")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
