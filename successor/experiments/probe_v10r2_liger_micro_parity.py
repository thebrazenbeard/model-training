from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path


def _metrics(a, b) -> dict:
    import torch

    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    ref_l2 = float(torch.linalg.vector_norm(af).item())
    diff_l2 = float(torch.linalg.vector_norm(diff).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            af.reshape(1, -1), bf.reshape(1, -1), dim=1
        ).item()
    )
    return {
        "ref_l2": ref_l2,
        "diff_l2": diff_l2,
        "relative_l2": diff_l2 / max(ref_l2, 1e-12),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "cosine": cosine,
        "all_finite": bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
    }


def _metric_pass(metric: dict, *, relative_l2_max: float, cosine_min: float) -> bool:
    return bool(
        metric["all_finite"]
        and metric["relative_l2"] <= relative_l2_max
        and metric["cosine"] >= cosine_min
    )


def run_probe(liger_path: Path, seed: int = 20261004) -> dict:
    if not liger_path.exists():
        raise RuntimeError(f"liger path missing:{liger_path}")
    sys.path.insert(0, str(liger_path))

    import torch
    import torch.nn.functional as F
    import triton
    from types import SimpleNamespace
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers.monkey_patch import (
        _patch_rms_norm_module,
        _patch_swiglu_module,
    )
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    thresholds = {
        "relative_l2_max": 0.02,
        "cosine_min": 0.995,
        "loss_relative_max": 0.005,
    }
    failures: list[str] = []
    results: dict[str, dict] = {}

    # Qwen3.5 Gemma-style RMSNorm: (1 + weight), BF16 in/out.
    dim = 256
    ref_rms = Qwen3_5RMSNorm(dim).to(device=device, dtype=dtype)
    cand_rms = Qwen3_5RMSNorm(dim).to(device=device, dtype=dtype)
    cand_rms.load_state_dict(ref_rms.state_dict(), strict=True)
    _patch_rms_norm_module(
        cand_rms, offset=1.0, casting_mode="gemma", in_place=False
    )
    x0 = torch.randn(2, 32, dim, device=device, dtype=dtype)
    upstream = torch.randn_like(x0)
    xr = x0.clone().requires_grad_(True)
    xc = x0.clone().requires_grad_(True)
    yr = ref_rms(xr)
    yc = cand_rms(xc)
    yr.backward(upstream)
    yc.backward(upstream)
    rms = {
        "output": _metrics(yr, yc),
        "grad_input": _metrics(xr.grad, xc.grad),
        "grad_weight": _metrics(ref_rms.weight.grad, cand_rms.weight.grad),
    }
    results["rms_norm"] = rms
    for name, metric in rms.items():
        if not _metric_pass(metric, **{
            "relative_l2_max": thresholds["relative_l2_max"],
            "cosine_min": thresholds["cosine_min"],
        }):
            failures.append(f"rms_norm:{name}")

    # Qwen3.5 SwiGLU preserves the gate/up/down Linear modules used by PEFT.
    cfg = SimpleNamespace(hidden_size=256, hidden_act="silu")
    intermediate = 512
    ref_mlp = Qwen3_5MLP(cfg, intermediate).to(device=device, dtype=dtype)
    cand_mlp = Qwen3_5MLP(cfg, intermediate).to(device=device, dtype=dtype)
    cand_mlp.load_state_dict(ref_mlp.state_dict(), strict=True)
    _patch_swiglu_module(cand_mlp, LigerQwen3MoeSwiGLUMLP)
    m0 = torch.randn(2, 32, 256, device=device, dtype=dtype)
    mup = torch.randn_like(m0)
    mr = m0.clone().requires_grad_(True)
    mc = m0.clone().requires_grad_(True)
    myr = ref_mlp(mr)
    myc = cand_mlp(mc)
    myr.backward(mup)
    myc.backward(mup)
    mlp = {
        "output": _metrics(myr, myc),
        "grad_input": _metrics(mr.grad, mc.grad),
        "grad_gate": _metrics(ref_mlp.gate_proj.weight.grad, cand_mlp.gate_proj.weight.grad),
        "grad_up": _metrics(ref_mlp.up_proj.weight.grad, cand_mlp.up_proj.weight.grad),
        "grad_down": _metrics(ref_mlp.down_proj.weight.grad, cand_mlp.down_proj.weight.grad),
    }
    results["swiglu"] = mlp
    for name, metric in mlp.items():
        if not _metric_pass(
            metric,
            relative_l2_max=thresholds["relative_l2_max"],
            cosine_min=thresholds["cosine_min"],
        ):
            failures.append(f"swiglu:{name}")

    # Completion-only masking: pre-shifted labels retain -100 ignore positions.
    tokens, hidden_size, vocab = 64, 256, 4096
    h0 = torch.randn(tokens, hidden_size, device=device, dtype=dtype) * 0.02
    w0 = torch.randn(vocab, hidden_size, device=device, dtype=dtype) * 0.02
    labels = torch.randint(0, vocab, (tokens,), device=device)
    labels[::4] = -100

    hr = h0.clone().requires_grad_(True)
    wr = w0.clone().requires_grad_(True)
    ref_loss = F.cross_entropy(hr @ wr.t(), labels, ignore_index=-100)
    ref_loss.backward()

    hc = h0.clone().requires_grad_(True)
    wc = w0.clone().requires_grad_(True)
    liger_loss = LigerFusedLinearCrossEntropyLoss()(wc, hc, labels)
    liger_loss.backward()

    loss_ref = float(ref_loss.detach().float().item())
    loss_cand = float(liger_loss.detach().float().item())
    loss_relative = abs(loss_cand - loss_ref) / max(abs(loss_ref), 1e-12)
    flce = {
        "loss_reference": loss_ref,
        "loss_candidate": loss_cand,
        "loss_relative_error": loss_relative,
        "grad_input": _metrics(hr.grad, hc.grad),
        "grad_weight": _metrics(wr.grad, wc.grad),
        "masked_label_count": int((labels == -100).sum().item()),
    }
    results["fused_linear_cross_entropy"] = flce
    if loss_relative > thresholds["loss_relative_max"]:
        failures.append("fused_linear_cross_entropy:loss")
    for name in ("grad_input", "grad_weight"):
        if not _metric_pass(
            flce[name],
            relative_l2_max=thresholds["relative_l2_max"],
            cosine_min=thresholds["cosine_min"],
        ):
            failures.append(f"fused_linear_cross_entropy:{name}")

    return {
        "schema": "V10R2_LIGER_QWEN35_MICRO_PARITY_V1",
        "status": "PASS" if not failures else "HOLD",
        "seed": seed,
        "liger_path": str(liger_path),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "liger_version": metadata.version("liger-kernel"),
        "triton_version": triton.__version__,
        "dtype": str(dtype),
        "thresholds": thresholds,
        "results": results,
        "failures": failures,
        "claim_ceiling": (
            "WINDOWS_TRITON_MICRO_PARITY_ONLY_NOT_WHOLE_MODEL_"
            "NOT_TRAINING_QUALIFICATION"
        ),
    }



def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liger-path", type=Path, required=True)
    parser.add_argument("--liger-wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = run_probe(args.liger_path)
    if args.liger_wheel is not None:
        result["liger_wheel_path"] = str(args.liger_wheel)
        result["liger_wheel_sha256"] = hashlib.sha256(
            args.liger_wheel.read_bytes()
        ).hexdigest()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
        print("OUTPUT_SHA256=" + hashlib.sha256(args.output.read_bytes()).hexdigest())
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
