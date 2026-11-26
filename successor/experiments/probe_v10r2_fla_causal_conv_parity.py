from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from successor.experiments.v10r2_lane_b_runtime import (
    _qwen_fla_causal_conv_update_wrapper,
    _qwen_fla_causal_conv_wrapper,
)


def _metrics(a, b):
    import torch

    af = a.detach().float()
    bf = b.detach().float()
    diff = (af - bf).abs()
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-7)
    cosine = torch.nn.functional.cosine_similarity(
        af.reshape(1, -1), bf.reshape(1, -1), dim=1
    ).item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
        "cosine": float(cosine),
        "all_finite": bool(
            torch.isfinite(af).all().item() and torch.isfinite(bf).all().item()
        ),
    }


def _reference_full(x, weight, bias, activation):
    import torch.nn.functional as F

    width = weight.shape[1]
    y = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias,
        padding=width - 1,
        groups=x.shape[1],
    )[:, :, : x.shape[-1]]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y


def _reference_update(x, cache, weight, bias, activation):
    import torch.nn.functional as F

    seq_len = x.shape[-1]
    state_len = cache.shape[-1]
    joined = F.pad(x, (state_len, 0))
    joined[:, :, :state_len] = cache
    cache_out = joined[:, :, -state_len:].clone()
    y = F.conv1d(
        joined,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=x.shape[1],
    )[:, :, -seq_len:]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y, cache_out


def run_probe(seed: int = 20261003) -> dict:
    import torch
    from fla.modules.convolution import causal_conv1d, causal_conv1d_update

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    full = _qwen_fla_causal_conv_wrapper(causal_conv1d)
    update = _qwen_fla_causal_conv_update_wrapper(causal_conv1d_update)

    cases = []
    for batch, channels, seq_len, width in (
        (1, 64, 17, 4),
        (2, 128, 64, 4),
        (1, 256, 127, 4),
    ):
        x0 = torch.randn(batch, channels, seq_len, device=device, dtype=dtype)
        w0 = torch.randn(channels, width, device=device, dtype=dtype) * 0.05
        bias0 = torch.randn(channels, device=device, dtype=dtype) * 0.01
        grad0 = torch.randn_like(x0)

        xr = x0.clone().requires_grad_(True)
        wr = w0.clone().requires_grad_(True)
        br = bias0.clone().requires_grad_(True)
        yr = _reference_full(xr, wr, br, "silu")
        yr.backward(grad0)

        xf = x0.clone().requires_grad_(True)
        wf = w0.clone().requires_grad_(True)
        bf = bias0.clone().requires_grad_(True)
        yf = full(
            x=xf,
            weight=wf,
            bias=bf,
            activation="silu",
            seq_idx=None,
        )
        yf.backward(grad0)

        case = {
            "shape": [batch, channels, seq_len],
            "width": width,
            "forward": _metrics(yr, yf),
            "grad_x": _metrics(xr.grad, xf.grad),
            "grad_weight": _metrics(wr.grad, wf.grad),
            "grad_bias": _metrics(br.grad, bf.grad),
        }

        cache0 = torch.randn(
            batch, channels, width, device=device, dtype=dtype
        )
        token0 = torch.randn(batch, channels, 1, device=device, dtype=dtype)
        ref_cache = cache0.clone()
        ref_y, ref_cache_expected = _reference_update(
            token0, ref_cache, w0, bias0, "silu"
        )
        fast_cache = cache0.clone()
        fast_y = update(
            token0,
            fast_cache,
            w0,
            bias0,
            "silu",
        )
        case["update_output"] = _metrics(ref_y, fast_y)
        case["update_cache"] = _metrics(ref_cache_expected, fast_cache)
        cases.append(case)

    thresholds = {
        "forward_max_abs": 0.025,
        "forward_cosine_min": 0.999,
        "gradient_max_abs": 0.05,
        "gradient_cosine_min": 0.995,
        "update_max_abs": 0.025,
        "update_cosine_min": 0.999,
    }
    failures = []
    for idx, case in enumerate(cases):
        for name in ("forward", "update_output", "update_cache"):
            m = case[name]
            if (
                not m["all_finite"]
                or m["max_abs"] > thresholds[
                    "update_max_abs" if name.startswith("update") else "forward_max_abs"
                ]
                or m["cosine"] < thresholds[
                    "update_cosine_min" if name.startswith("update") else "forward_cosine_min"
                ]
            ):
                failures.append(f"case{idx}:{name}")
        for name in ("grad_x", "grad_weight", "grad_bias"):
            m = case[name]
            if (
                not m["all_finite"]
                or m["max_abs"] > thresholds["gradient_max_abs"]
                or m["cosine"] < thresholds["gradient_cosine_min"]
            ):
                failures.append(f"case{idx}:{name}")

    result = {
        "schema": "V10R2_FLA_CAUSAL_CONV_PARITY_V1",
        "status": "PASS" if not failures else "HOLD",
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "thresholds": thresholds,
        "cases": cases,
        "failures": failures,
        "claim_ceiling": "MICRO_PARITY_ONLY_NOT_WHOLE_MODEL_OR_TRAINING_QUALIFICATION",
    }
    return result


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
