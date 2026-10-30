"""CPU-only MeZO-LoRA *method* pilot, NOT official Qwen/Vera training.

The frozen toy backbone has two trainable low-rank tensors. This experiment
tests correct/incorrect supervision, gradient-free update mechanics, and a
standard gradient baseline with exact train/holdout provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch
from torch import nn
from torch.nn import functional as F


class TinyFrozenBackboneLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(8, 16)
        self.head = nn.Linear(16, 1)
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(16, 2) * 0.05)
        self.lora_b = nn.Parameter(torch.zeros(2, 1))

    def forward(self, features):
        h = torch.tanh(self.features(features))
        return self.head(h) + h @ self.lora_a @ self.lora_b

    def adapter_parameters(self):
        return [self.lora_a, self.lora_b]


def make_data(*, seed: int, count: int):
    if count <= 0 or count > 1024:
        raise ValueError("bounded synthetic data count required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((count, 8), generator=generator, dtype=torch.float32)
    # Original synthetic linear oracle. No V2 or V3 evaluation files accessed.
    linear = x[:, 0] * 1.5 - x[:, 1] * 0.6 + x[:, 2] * 0.4
    labels = torch.where(linear >= 0, 1.0, -1.0).reshape(-1, 1)
    return x, labels


def _digest(parameters):
    return hashlib.sha256(b"".join(
        p.detach().cpu().contiguous().numpy().tobytes() for p in parameters
    )).hexdigest()


def zo_update(parameters, loss_fn, *, epsilon: float, lr: float, seed: int,
              perturbations=None):
    """Two-sided MeZO directional finite difference; no backward() or grad state."""
    if epsilon <= 0 or lr <= 0:
        raise ValueError("positive epsilon and learning rate required")
    params = list(parameters)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = (list(perturbations) if perturbations is not None else [
        torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in params
    ])
    if (len(noise) != len(params)
            or any(z.shape != p.shape or p.device.type != "cpu"
                   or p.grad is not None for p, z in zip(params, noise))):
        raise ValueError("CPU LoRA/no-gradient perturbation invariant violated")
    with torch.no_grad():
        for p, z in zip(params, noise):
            p.add_(z, alpha=epsilon)
        try:
            plus = float(loss_fn().item())
        finally:
            for p, z in zip(params, noise):
                p.add_(z, alpha=-epsilon)
        for p, z in zip(params, noise):
            p.add_(z, alpha=-epsilon)
        try:
            minus = float(loss_fn().item())
        finally:
            for p, z in zip(params, noise):
                p.add_(z, alpha=epsilon)
        if not all(torch.isfinite(torch.tensor(v)) for v in (plus, minus)):
            raise RuntimeError("non-finite MeZO loss")
        directional_derivative = (plus - minus) / (2.0 * epsilon)
        for p, z in zip(params, noise):
            p.add_(z, alpha=-lr * directional_derivative)
    return {"forward_evaluations": 2, "loss_plus": plus,
            "loss_minus": minus,
            "directional_derivative": directional_derivative}


def run_arm(*, method: str, mode: str, steps: int, seed: int):
    if method not in ("mezo", "adamw") or mode not in ("zero", "correct", "corrupt"):
        raise ValueError("unsupported bounded method or control")
    if not 0 <= steps <= 200 or (mode == "zero" and steps != 0):
        raise ValueError("invalid step/control combination")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    model = TinyFrozenBackboneLoRA()
    parameters = model.adapter_parameters()
    initial_sha = _digest(parameters)
    train_x, train_y = make_data(seed=seed + 101, count=48)
    hold_x, hold_y = make_data(seed=seed + 202, count=32)
    assert not ({tuple(row) for row in train_x.tolist()} &
                {tuple(row) for row in hold_x.tolist()})
    data_sha = hashlib.sha256(train_x.numpy().tobytes() +
                              train_y.numpy().tobytes()).hexdigest()
    hold_sha = hashlib.sha256(hold_x.numpy().tobytes() +
                              hold_y.numpy().tobytes()).hexdigest()
    optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0) \
        if method == "adamw" and mode != "zero" else None
    with torch.no_grad():
        baseline_loss = float(F.mse_loss(model(hold_x), hold_y))
        baseline_exact = int(((model(hold_x) >= 0) == (hold_y >= 0)).sum())
    calls = 0
    start = time.perf_counter()
    for step in range(steps):
        idx = torch.arange(step * 16, step * 16 + 16) % len(train_x)
        x, labels = train_x[idx], train_y[idx]
        if mode == "corrupt":
            labels = -labels
        def loss_fn():
            return F.mse_loss(model(x), labels)
        if method == "mezo":
            update = zo_update(parameters, loss_fn, epsilon=0.005,
                               lr=0.02, seed=seed + step * 997)
            calls += update["forward_evaluations"]
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            calls += 1
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start
    final_sha = _digest(parameters)
    if (mode == "zero") != (final_sha == initial_sha):
        raise RuntimeError("zero/update parameter digest invariant violated")
    with torch.no_grad():
        final_loss = float(F.mse_loss(model(hold_x), hold_y))
        exact = int(((model(hold_x) >= 0) == (hold_y >= 0)).sum())
    return {
        "status": "TOY_CPU_LORA_METHOD_ABLATION_NOT_4B",
        "model": "random_frozen_8x16_toy_backbone_r2_lora",
        "method": method, "mode": mode, "seed": seed,
        "optimizer_steps": steps, "train_rows": len(train_x),
        "heldout_rows": len(hold_x),
        "train_sha256": data_sha, "heldout_sha256": hold_sha,
        "initial_sha256": initial_sha, "final_sha256": final_sha,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "forward_evaluations": calls,
        "baseline_heldout_mse": baseline_loss,
        "heldout_mse": final_loss,
        "baseline_heldout_sign_correct": baseline_exact,
        "heldout_sign_correct": exact,
        "optimizer_seconds": elapsed,
        "cuda_allocations": 0, "official_4b_model_trained": False,
        "reasoning_generalization_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mezo", "adamw"), required=True)
    parser.add_argument("--mode", choices=("zero", "correct", "corrupt"),
                        required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=29037)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_arm(method=args.method, mode=args.mode, steps=args.steps,
                     seed=args.seed)
    from pathlib import Path
    path = Path(args.output)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
