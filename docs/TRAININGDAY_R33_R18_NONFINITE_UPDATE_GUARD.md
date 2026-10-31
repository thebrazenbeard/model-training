# R33 R18 source guard: nonfinite adapter values

Scope: source-only, deterministic CPU/mock regressions. **No CUDA, optimizer allocation, model weights, or real training was exercised in this wave.**

## Source invariants

- MeZO validates each FP32 LoRA trainable before its paired update, again after the final noise-direction update, and before adapter parameter attestation. A NaN/Inf post-update tensor aborts the step. Its prior CPU snapshot is used for a best-effort restoration; catastrophic device errors can still prevent rollback. No successful-step return or checkpoint occurs along the rejected path.
- QLoRA checks `requires_grad` tensors for finite values before initial digest, immediately following `optimizer.step()` in synthetic fit and actual training, and before the final digest/save. A NaN/Inf post-update tensor raises before `steps += 1`, before FIT, and before `save_pretrained`. The optimizer state is **not** claimed to be recoverable after such a failure; abort and resume only from an independently verified safe checkpoint.
- Each check uses `torch.isfinite(tensor.detach()).all().item()`, including a device synchronization caused by `.item()`. This introduces measurable training overhead; wall-time matching must include it for the method using it.

## Unresolved AdamW8bit numerical comparison

R18 QLoRA `_load_model` instantiates `bitsandbytes.optim.AdamW8bit(..., lr=lr)` without explicitly specifying `weight_decay`. The **installed bitsandbytes version and its runtime default remain unverified in this CPU-only environment**. Changing the source to a guessed value would alter numerical comparability without evidence. This wave **holds** any claim that QLoRA/MeZO have matched optimizer regularization; before a fairness/qualification claim, independently capture `bitsandbytes.__version__`, the live class signature/default and optimizer param-group `weight_decay`, then preregister the intended comparison. No default was changed.

## Gates still open

- Real CUDA fit/update and post-update finite-tensor probe on Lappy.
- Real fresh-process reload and adapter-tensor digest attestation.
- Exact-head independent hostile review.
- Equal-time comparison and protected held-out behavioral qualification.

Source review, deterministic mock passes and Git commit do **not** authorize a production model release.
