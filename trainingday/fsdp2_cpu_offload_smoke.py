"""Synthetic FSDP2 CPU-offload hardware probe; never a protected Qwen launcher."""
import hashlib
import json
import tempfile
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy

def digest(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        data = param.detach()
        if hasattr(data, "to_local"):
            data = data.to_local()
        b = data.float().cpu().contiguous().numpy().tobytes()
        h.update(name.encode("utf-8"))
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(0)
    torch.manual_seed(413)
    with tempfile.TemporaryDirectory(prefix="bt2-fsdp2-smoke-") as root:
        dist.init_process_group("nccl", init_method="file://" + root + "/rendezvous", rank=0, world_size=1)
        try:
            model = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 8))
            policy = CPUOffloadPolicy(pin_memory=False)
            fully_shard(model[0], offload_policy=policy)
            fully_shard(model[2], offload_policy=policy)
            fully_shard(model, offload_policy=policy)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
            before = digest(model)
            x = torch.randn(4, 64, device="cuda")
            y = torch.randn(4, 8, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            records = []
            for step in range(1, 4):
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                loss = ((model(x) - y) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                if not grads or any(g is None for g in grads):
                    raise RuntimeError("missing sharded gradients")
                if any(g.device.type != "cpu" for g in grads):
                    raise RuntimeError("FSDP gradients not CPU-resident")
                optimizer.step()
                torch.cuda.synchronize()
                records.append({"step": step, "loss": round(loss.item(), 8),
                                "seconds": round(time.perf_counter()-started, 5)})
            after = digest(model)
            cpu_shards = all(p.device.type == "cpu" for p in model.parameters())
            optimizer_tensors = [v for state in optimizer.state.values()
                                 for v in state.values() if isinstance(v, torch.Tensor)]
            cpu_optimizer = bool(optimizer_tensors) and all(v.device.type == "cpu"
                                                        for v in optimizer_tensors)
            if before == after or not cpu_shards or not cpu_optimizer:
                raise RuntimeError("unchanged weights or shards not CPU-resident")
            print(json.dumps({"status": "SYNTHETIC_FSDP2_CPU_OFFLOAD_STEPS_PASS",
                "model": "random_sequential_64_64_8", "qwen_weights_used": False,
                "training_authorized": False, "steps": records,
                "optimizer": "AdamW", "sharded_parameter_device": "cpu",
                "gradient_device": "cpu", "optimizer_state_device": "cpu",
                "before_sha256": before, "after_sha256": after,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}))
        finally:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()