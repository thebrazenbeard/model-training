"""Verify selected real GGUF tensors against the merged synthetic HF checkpoint.

This is an exact post-cast tensor check, not an inference-logit equivalence
test. Two known representation changes (zero-centered RMSNorm and grouped-to-
tiled GDN V heads) are checked by an independent explicit NumPy oracle. The
official pretrained Qwen3.5-4B shards are not involved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


# The pinned converter's conversion/qwen.py intentionally maps HF zero-centered
# RMSNorm weights to GGUF 1+weight and reorders the GDN V heads from grouped to
# tiled. Both a LoRA-updated GDN target and a full-attention MLP are inspected.
TENSOR_MAP = (
    ("model.embed_tokens.weight", "token_embd.weight", "float16", "identity"),
    ("model.norm.weight", "output_norm.weight", "float32", "rmsnorm_plus_one"),
    ("model.layers.0.linear_attn.in_proj_qkv.weight", "blk.0.attn_qkv.weight", "float16", "gdn_v_tiled"),
    ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight", "float16", "identity"),
    ("model.layers.3.mlp.down_proj.weight", "blk.3.ffn_down.weight", "float16", "identity"),
    ("model.layers.31.mlp.down_proj.weight", "blk.31.ffn_down.weight", "float16", "identity"),
)


def transform_hf_values(source, operation: str):
    """Independent fixture-specific representation math, before F16/F32 cast."""
    import numpy as np

    if operation == "identity":
        return source
    if operation == "rmsnorm_plus_one":
        # HF Qwen3.5 RMSNorm forward scales by (1 + weight); GGUF folds +1.
        return source + np.float32(1)
    if operation == "gdn_v_tiled":
        # Fixture config: 2 K heads x 16, 4 V heads x 16, width 64.
        # Q and K stay unchanged; V's grouped K-head order becomes tiled.
        if source.shape != (128, 64):
            raise AssertionError(f"GDN QKV fixture shape changed: {source.shape}")
        v = source[64:, :].reshape(2, 2, 16, 64)
        return np.concatenate((source[:64, :], v.transpose(1, 0, 2, 3).reshape(64, 64)))
    raise AssertionError(f"unexpected GGUF tensor representation operation: {operation}")


def assert_exact_post_cast_tensor(
    hf_tensor, gguf_tensor, name: str, dtype: str, operation: str = "identity",
) -> None:
    """Reject shape, on-disk dtype, and even single-value conversion errors."""
    import numpy as np

    actual = np.asarray(gguf_tensor)
    source = np.asarray(hf_tensor)
    target_dtype = np.dtype(dtype)
    if actual.dtype != target_dtype:
        raise AssertionError(f"{name}: GGUF dtype {actual.dtype} != {target_dtype}")
    if source.shape != actual.shape:
        raise AssertionError(f"{name}: HF shape {source.shape} != GGUF shape {actual.shape}")
    if not np.isfinite(source).all() or not np.isfinite(actual).all():
        raise AssertionError(f"{name}: nonfinite tensor values")
    expected = transform_hf_values(source, operation).astype(target_dtype)
    if not np.array_equal(actual, expected):
        mismatch_count = int(np.count_nonzero(actual != expected))
        raise AssertionError(f"{name}: {mismatch_count} mismatching GGUF elements after {dtype} cast")


def verify(merged_dir: Path, gguf_path: Path, gguf_python: Path) -> dict:
    # Import the reader from the exact checked-out, SHA-pinned converter tree.
    # The actual file is read via its tensor table; GGUF magic alone is never
    # accepted as numerical conversion evidence.
    sys.path.insert(0, str(gguf_python))
    from gguf import GGUFReader
    from safetensors import safe_open

    expected_keys = {item[0] for item in TENSOR_MAP}
    shards = sorted(merged_dir.glob("*.safetensors"))
    if not shards:
        raise AssertionError("merged HF safetensors checkpoint missing")
    hf = {}
    for shard in shards:
        with safe_open(str(shard), framework="np") as source:
            for key in expected_keys.intersection(source.keys()):
                if key in hf:
                    raise AssertionError(f"duplicate HF checkpoint tensor: {key}")
                hf[key] = source.get_tensor(key)
    missing = sorted(expected_keys.difference(hf))
    if missing:
        raise AssertionError(f"required HF checkpoint tensors missing: {missing}")

    reader = GGUFReader(gguf_path)
    gguf = {item.name: item for item in reader.tensors}
    passed = []
    for hf_key, gguf_key, dtype, operation in TENSOR_MAP:
        item = gguf.get(gguf_key)
        if item is None:
            raise AssertionError(f"required converted GGUF tensor missing: {gguf_key}")
        assert_exact_post_cast_tensor(hf[hf_key], item.data, gguf_key, dtype, operation)
        passed.append({"hf": hf_key, "gguf": gguf_key,
                       "dtype": dtype, "representation": operation,
                       "elements": int(item.n_elements)})

    return {
        "schema": "TRAININGDAY_QWEN35_SYNTHETIC_GGUF_TENSOR_PARITY_V1",
        "scope": "random synthetic 32-layer Qwen3.5 text model; direct tensors only",
        "gguf_sha256": hashlib.sha256(gguf_path.read_bytes()).hexdigest(),
        "assertion": "exact full tensor values after documented representation and F16/F32 casts",
        "matched": len(passed),
        "tensors": passed,
        "official_pretrained_weights_exercised": False,
        "hf_gguf_inference_logit_parity": "NOT_TESTED",
        "koboldcpp_compatibility": "UNKNOWN",
        "gpu_fit": "UNKNOWN",
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: python -m trainingday.qwen35_gguf_tensor_parity_cpu "
                         "MERGED_HF_DIR GGUF_FILE PINNED_LLAMA_CPP_GGUF_PY")
    print(json.dumps(verify(Path(sys.argv[1]), Path(sys.argv[2]),
                            Path(sys.argv[3])), sort_keys=True))


if __name__ == "__main__":
    main()
