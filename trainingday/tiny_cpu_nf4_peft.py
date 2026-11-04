"""Fail-closed, tiny CPU NF4+PEFT smoke test; never a Qwen weight/training test.

This module deliberately constructs only a 64x64 synthetic quantized projection.
All pretrained-model loads, GPU use, and host workload scheduling are out of scope.
"""

from __future__ import annotations

import json
import math
import resource
import time


def run_tiny_cpu_nf4_peft() -> dict:
    """Observe a real CPU NF4 forward/backward and one PEFT AdamW update.

    Fail if any dependency, quantized storage, adapter gradient, or device gate
    is unavailable. A dense FP32 substitute must never count as success.
    """
    start = time.perf_counter()

    import bitsandbytes as bnb
    import peft
    import torch
    import transformers

    if torch.cuda.is_available():
        raise RuntimeError("CUDA is available; this test requires a CPU-only runner")

    torch.manual_seed(20261104)
    width = 64
    base = bnb.nn.Linear4bit(
        width, width, bias=False, quant_type="nf4",
        compute_dtype=torch.float32, compress_statistics=False,
        quant_storage=torch.uint8,
    )
    with torch.no_grad():
        base.weight.data.copy_(torch.randn(width, width) * 0.08)

    # Params4bit.to('cpu') must do real quantization even when source is CPU.
    base = base.to(torch.device("cpu"))
    quant = base.weight
    state = getattr(quant, "quant_state", None)
    if (
        not isinstance(quant, bnb.nn.Params4bit)
        or state is None
        or not bool(getattr(quant, "bnb_quantized", False))
        or quant.dtype != torch.uint8
        or quant.device.type != "cpu"
        or quant.numel() != width * width // 2
        or getattr(state, "quant_type", None) != "nf4"
        or tuple(getattr(state, "shape", ())) != (width, width)
        or bool(getattr(state, "nested", True))
    ):
        raise RuntimeError("not a packed CPU NF4 weight; FP32 is not an acceptable fallback")

    packed_before = quant.detach().clone()
    absmax_before = state.absmax.detach().clone()
    code_before = state.code.detach().clone()

    try:
        bnb.matmul_4bit(torch.zeros(2, width), quant, quant_state=None)
    except ValueError as exc:
        if "quant_state" not in str(exc):
            raise
    else:
        raise RuntimeError("missing NF4 quantization state did not fail closed")

    class SyntheticModel(torch.nn.Module):
        def __init__(self, projection):
            super().__init__()
            self.proj = projection

        def forward(self, inputs):
            return self.proj(inputs)

    configuration = peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["proj"], bias="none",
        task_type=None,
    )
    model = peft.get_peft_model(SyntheticModel(base), configuration)
    model.train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    a = [(name, parameter) for name, parameter in trainable if ".lora_A." in name]
    b = [(name, parameter) for name, parameter in trainable if ".lora_B." in name]
    if len(trainable) != 2 or len(a) != 1 or len(b) != 1:
        raise RuntimeError("PEFT did not install exactly one real LoRA A/B pair")
    with torch.no_grad():
        # PEFT's default B=0 makes A's first gradient zero. Set both nonzero to
        # require a genuine two-sided derivative through a quantized forward.
        torch.nn.init.normal_(b[0][1], std=0.02)

    inputs = torch.randn(4, width, dtype=torch.float32, requires_grad=True)
    base_output = base(inputs).detach()
    if not torch.isfinite(base_output).all() or base_output.abs().sum().item() == 0:
        raise RuntimeError("packed NF4 projection produced no finite base contribution")

    before_adapter = {name: parameter.detach().clone() for name, parameter in trainable}
    output = model(inputs)
    if output.shape != (4, width) or not torch.isfinite(output).all():
        raise RuntimeError("real PEFT-wrapped NF4 forward did not produce finite output")
    if torch.allclose(output.detach(), base_output):
        raise RuntimeError("LoRA contribution was absent from the quantized forward")

    loss = torch.nn.functional.mse_loss(output, torch.zeros_like(output))
    loss.backward()
    for name, parameter in trainable:
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum().item() == 0:
            raise RuntimeError(f"no finite nonzero LoRA gradient through NF4 forward: {name}")
    if inputs.grad is None or not torch.isfinite(inputs.grad).all() or inputs.grad.abs().sum().item() == 0:
        raise RuntimeError("no valid input gradient through actual CPU NF4 forward/backward")

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=0.01,
        foreach=False, fused=False,
    )
    optimizer.step()
    for name, parameter in trainable:
        if torch.equal(parameter.detach(), before_adapter[name]):
            raise RuntimeError(f"AdamW did not update real PEFT LoRA parameter: {name}")
    if (
        not torch.equal(base.weight.detach(), packed_before)
        or not torch.equal(state.absmax.detach(), absmax_before)
        or not torch.equal(state.code.detach(), code_before)
        or base.weight.grad is not None
    ):
        raise RuntimeError("frozen packed NF4 base or quantization state changed during LoRA step")

    elapsed = time.perf_counter() - start
    # Linux ru_maxrss is KiB; the hosted workflow is Linux-only.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if not math.isfinite(elapsed) or elapsed >= 60 or rss >= 2 * 1024**3:
        raise RuntimeError("tiny CPU NF4 rehearsal exceeded explicit time/memory bounds")

    return {
        "status": "TINY_SYNTHETIC_CPU_BNB_NF4_PEFT_STEP_ONLY",
        "device": "cpu",
        "base_quantization": "packed_nf4_uint8",
        "base_packed_bytes": packed_before.numel(),
        "synthetic_projection_shape": [width, width],
        "lora_trainable_tensors": 2,
        "lora_a_b_updated": ["A", "B"],
        "optimizer": "torch.optim.AdamW(foreach=False,fused=False)",
        "optimizer_steps_observed": 1,
        "torch_version": torch.__version__,
        "bitsandbytes_version": bnb.__version__,
        "peft_version": peft.__version__,
        "transformers_version": transformers.__version__,
        "elapsed_seconds": elapsed,
        "max_rss_bytes": rss,
        "official_qwen_weights_loaded": False,
        "official_qwen_forward_or_backward": False,
        "cuda_used": False,
        "physical_training_authorized": False,
        "qwen35_4gb_hardware_fit": "UNKNOWN",
    }


if __name__ == "__main__":
    print(json.dumps(run_tiny_cpu_nf4_peft(), sort_keys=True))
