from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from successor.experiments.v10r2_lane_b_runtime import apply_qwen35_acceleration


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


def run_probe(seed: int = 20261004) -> dict:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = AutoConfig.from_pretrained(
        r"D:\VERA\models\latest-trained\base"
    ).text_config
    device = torch.device("cuda")
    dtype = torch.bfloat16

    thresholds = {
        "output_relative_l2_max": 0.02,
        "output_cosine_min": 0.995,
        "gradient_relative_l2_max": 0.03,
        "gradient_cosine_min": 0.995,
    }

    cases = []
    failures = []

    for seq_len in (17, 64):
        torch.manual_seed(seed + seq_len)
        ref = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=dtype)
        fast = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=dtype)
        fast.load_state_dict(ref.state_dict(), strict=True)

        # The acceleration helper intentionally discovers Qwen DeltaNet modules
        # under names ending in `.linear_attn`, matching the full-model layout.
        ref_container = torch.nn.Module()
        ref_container.layer = torch.nn.Module()
        ref_container.layer.linear_attn = ref
        fast_container = torch.nn.Module()
        fast_container.layer = torch.nn.Module()
        fast_container.layer.linear_attn = fast

        ref_report = apply_qwen35_acceleration(
            ref_container, backend="fla_triton"
        )
        fast_report = apply_qwen35_acceleration(
            fast_container, backend="fla_triton_full"
        )

        x0 = torch.randn(
            1, seq_len, cfg.hidden_size, device=device, dtype=dtype
        )
        upstream = torch.randn_like(x0)

        xr = x0.clone().requires_grad_(True)
        yr = ref(xr, attention_mask=None)
        yr.backward(upstream)

        xf = x0.clone().requires_grad_(True)
        yf = fast(xf, attention_mask=None)
        yf.backward(upstream)

        selected = {
            "conv1d.weight": (
                ref.conv1d.weight.grad,
                fast.conv1d.weight.grad,
            ),
            "in_proj_qkv.weight": (
                ref.in_proj_qkv.weight.grad,
                fast.in_proj_qkv.weight.grad,
            ),
            "out_proj.weight": (
                ref.out_proj.weight.grad,
                fast.out_proj.weight.grad,
            ),
        }

        case = {
            "seq_len": seq_len,
            "reference_backend": ref_report,
            "candidate_backend": fast_report,
            "output": _metrics(yr, yf),
            "grad_input": _metrics(xr.grad, xf.grad),
            "parameter_gradients": {
                name: _metrics(a, b) for name, (a, b) in selected.items()
            },
        }

        om = case["output"]
        if (
            not om["all_finite"]
            or om["relative_l2"] > thresholds["output_relative_l2_max"]
            or om["cosine"] < thresholds["output_cosine_min"]
        ):
            failures.append(f"seq{seq_len}:output")

        for name, gm in [
            ("grad_input", case["grad_input"]),
            *case["parameter_gradients"].items(),
        ]:
            if (
                not gm["all_finite"]
                or gm["relative_l2"] > thresholds["gradient_relative_l2_max"]
                or gm["cosine"] < thresholds["gradient_cosine_min"]
            ):
                failures.append(f"seq{seq_len}:{name}")

        cases.append(case)

        del ref, fast, xr, xf, yr, yf
        torch.cuda.empty_cache()

    return {
        "schema": "V10R2_QWEN35_MODULE_CONV_BACKEND_PARITY_V1",
        "status": "PASS" if not failures else "HOLD",
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "subject": {
            "module": "Qwen3_5GatedDeltaNet",
            "hidden_size": cfg.hidden_size,
            "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
            "linear_num_key_heads": cfg.linear_num_key_heads,
            "linear_key_head_dim": cfg.linear_key_head_dim,
            "linear_num_value_heads": cfg.linear_num_value_heads,
            "linear_value_head_dim": cfg.linear_value_head_dim,
            "reference_backend": "fla_triton",
            "candidate_backend": "fla_triton_full",
            "only_intended_difference": "causal_conv_backend",
        },
        "thresholds": thresholds,
        "cases": cases,
        "failures": failures,
        "claim_ceiling": "SINGLE_MODULE_PARITY_ONLY_NOT_FULL_MODEL_OR_TRAINING_QUALIFICATION",
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
