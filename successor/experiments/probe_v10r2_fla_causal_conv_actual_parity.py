from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.v10r2_lane_b_runtime import (
    _qwen_fla_causal_conv_update_wrapper,
    _qwen_fla_causal_conv_wrapper,
)


def _metrics(a, b):
    import torch

    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    ref_norm = torch.linalg.vector_norm(af).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    cosine = torch.nn.functional.cosine_similarity(
        af.reshape(1, -1), bf.reshape(1, -1), dim=1
    ).item()
    return {
        "ref_l2": float(ref_norm),
        "diff_l2": float(diff_norm),
        "relative_l2": float(diff_norm / max(ref_norm, 1e-12)),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "cosine": float(cosine),
        "all_finite": bool(
            torch.isfinite(af).all().item() and torch.isfinite(bf).all().item()
        ),
    }


def _reference_full(x, weight, activation):
    import torch.nn.functional as F

    width = weight.shape[1]
    y = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias=None,
        padding=width - 1,
        groups=x.shape[1],
    )[:, :, : x.shape[-1]]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y


def _reference_update(x, cache, weight, activation):
    import torch
    import torch.nn.functional as F

    seq_len = x.shape[-1]
    state_len = cache.shape[-1]
    joined = torch.cat([cache, x], dim=-1).to(weight.dtype)
    cache_out = joined[:, :, -state_len:].clone()
    y = F.conv1d(
        joined,
        weight.unsqueeze(1),
        bias=None,
        padding=0,
        groups=x.shape[1],
    )[:, :, -seq_len:]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y.to(x.dtype), cache_out


def run_probe(seed: int = 20261004) -> dict:
    import torch
    from fla.modules.convolution import causal_conv1d, causal_conv1d_update

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Exact Qwen3.5 text config bound from D:/VERA/models/latest-trained/base:
    # key_dim = 16 * 128 = 2048
    # value_dim = 32 * 128 = 4096
    # conv_dim = 2 * key_dim + value_dim = 8192
    channels = 8192
    width = 4

    full = _qwen_fla_causal_conv_wrapper(causal_conv1d)
    update = _qwen_fla_causal_conv_update_wrapper(causal_conv1d_update)

    thresholds = {
        "forward_relative_l2_max": 0.01,
        "forward_cosine_min": 0.999,
        "gradient_relative_l2_max": 0.02,
        "gradient_cosine_min": 0.999,
        "update_relative_l2_max": 0.01,
        "update_cosine_min": 0.999,
    }

    cases = []
    failures = []
    for seq_len in (17, 64, 128, 512):
        x0 = torch.randn(1, channels, seq_len, device=device, dtype=dtype)
        w0 = torch.randn(channels, width, device=device, dtype=dtype) * 0.02
        grad0 = torch.randn_like(x0)

        xr = x0.clone().requires_grad_(True)
        wr = w0.clone().requires_grad_(True)
        yr = _reference_full(xr, wr, "silu")
        yr.backward(grad0)

        xf = x0.clone().requires_grad_(True)
        wf = w0.clone().requires_grad_(True)
        yf = full(
            x=xf,
            weight=wf,
            bias=None,
            activation="silu",
            seq_idx=None,
        )
        yf.backward(grad0)

        case = {
            "shape": [1, channels, seq_len],
            "width": width,
            "bias": False,
            "forward": _metrics(yr, yf),
            "grad_x": _metrics(xr.grad, xf.grad),
            "grad_weight": _metrics(wr.grad, wf.grad),
        }

        cache0 = torch.randn(
            1, channels, width, device=device, dtype=dtype
        )
        token0 = torch.randn(1, channels, 1, device=device, dtype=dtype)
        ref_y, ref_cache = _reference_update(
            token0, cache0.clone(), w0, "silu"
        )
        fast_cache = cache0.clone()
        fast_y = update(
            token0,
            fast_cache,
            w0,
            None,
            "silu",
        )
        case["update_output"] = _metrics(ref_y, fast_y)
        case["update_cache"] = _metrics(ref_cache, fast_cache)

        for name in ("forward",):
            m = case[name]
            if (
                not m["all_finite"]
                or m["relative_l2"] > thresholds["forward_relative_l2_max"]
                or m["cosine"] < thresholds["forward_cosine_min"]
            ):
                failures.append(f"seq{seq_len}:{name}")

        for name in ("grad_x", "grad_weight"):
            m = case[name]
            if (
                not m["all_finite"]
                or m["relative_l2"] > thresholds["gradient_relative_l2_max"]
                or m["cosine"] < thresholds["gradient_cosine_min"]
            ):
                failures.append(f"seq{seq_len}:{name}")

        for name in ("update_output", "update_cache"):
            m = case[name]
            if (
                not m["all_finite"]
                or m["relative_l2"] > thresholds["update_relative_l2_max"]
                or m["cosine"] < thresholds["update_cosine_min"]
            ):
                failures.append(f"seq{seq_len}:{name}")

        cases.append(case)

    return {
        "schema": "V10R2_FLA_CAUSAL_CONV_ACTUAL_PARITY_V1",
        "status": "PASS" if not failures else "HOLD",
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "qwen35_bound_dimensions": {
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "conv_dim": channels,
            "conv_kernel_width": width,
            "conv_bias": False,
        },
        "thresholds": thresholds,
        "cases": cases,
        "failures": failures,
        "claim_ceiling": "ACTUAL_DIMENSION_MICRO_PARITY_ONLY_NOT_WHOLE_MODEL_OR_TRAINING_QUALIFICATION",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_probe()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
        print("OUTPUT_SHA256=" + hashlib.sha256(args.output.read_bytes()).hexdigest())
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
