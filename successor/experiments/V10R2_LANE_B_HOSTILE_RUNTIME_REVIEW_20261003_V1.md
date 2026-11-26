# V10R2 Lane B Hostile Runtime Review — 2026-10-03

Scope: accelerated local development training/evaluation on Lappy only. This is not final-bank review, independent qualification, merge authority, deployment authority, or production activation.

Evidence base:
- surviving V10 lineage base: `9b0860c15d0c8f9128e07a6033e00d686ed0de82`
- Lane B probe-time runtime binding V1: `e9861abb9d1c9ec80e02b4cb7c08f9474adb642b6996b139e2d768a12843d8de`
- current Lane B runtime binding V2: `8880dc796d4e0646ee0e061c2b83cc743106067b9955410a6ffb41a2d68fc25f`
- successful non-mutating probe receipt: `8e0efe2740e8e97a39985b53aa2360e76ff72968c9bb7224e1e290af790859d5`
- train corpus SHA-256: `a232732a7db1f22c7edabb984fb16a3fe5350022e0a152576d80877eb9430300`
- frozen validation SHA-256: `ccc57ad20e8dfbc826ce49f064e692e0eab3fa396ed602dfba54ad9e252bd6d7`

> **Challenge: the 13–20× DeltaNet microbenchmark speedups may be irrelevant to whole-model training.**
> Confirmed in part. The first whole-model eight-microbatch FLA probe took 297.659 s because variable sequence lengths triggered substantial Triton JIT/specialization cost. Microbench results are therefore evidence of kernel potential, not end-to-end training speed. Lane B is testing fixed 512-token right-padding before calling the path accelerated in practice.

> **Challenge: installing FLA does not mean the stock Transformers Qwen3.5 fast path is valid on this Windows environment.**
> Confirmed. Transformers still warns that its complete fast path is unavailable. Directly wiring FLA causal convolution failed because FLA's helper returned a tuple where Qwen3.5 expected a tensor. Lane B rejected FLA causal convolution and only patches the gated-delta recurrence; Qwen/Torch causal convolution remains authoritative.

> **Challenge: FLA is numerically different from the Torch reference, so a faster kernel could silently alter training.**
> The kernels are not bit-identical. Forward max absolute error was <= 0.0009765625 in the tested shapes. Backward output cosine was 0.9999397 and Q/K/V/g/beta gradient cosines were all > 0.99993. This is strong parity evidence but not identity. Any trained adapter must still be compared on the same frozen held-out diagnostic and public retention-shadow diagnostics.

> **Challenge: fixed-length padding may improve compilation while subtly changing the training objective.**
> Right-padding after the real sequence with attention-mask zeroes and label `-100` should not alter losses on preceding causal tokens, but that is still an inference until measured. Lane B must compare the same row window with and without fixed padding before treating the method as equivalent.

> **Challenge: PyTorch memory counters above 4 GiB prove the run exceeded physical VRAM and therefore the receipt is impossible.**
> Rejected. PyTorch allocator accounting during the probe reported peaks above the card's nominal 4096 MiB, while live `nvidia-smi` resident memory remained around the physical-device ceiling. These counters describe different accounting domains. Lane B will report both and will not translate allocator counters into a claim of physically resident VRAM.

> **Challenge: TorchAO `adamw_torch_8bit` is now qualified because the class resolves.**
> Rejected. Current evidence proves only that Transformers resolves the optimizer to `torchao.optim.adam.AdamW8bit` in the Lane B environment. A real CUDA optimizer step has not yet passed under Lane B. Until that happens, BNB8 remains the only optimizer with real V10R2 step evidence.

> **Challenge: public retention benchmarks can be used to choose the winning development candidate.**
> Rejected by the frozen repository policy. The six pinned shadow benchmarks are diagnostic-only: not final-bank evidence, not promotion-gate evidence, and not candidate-selection authority. Lane B's evaluator binds them for comparable diagnostics only.

> **Challenge: `D:\VERA\models\latest-trained` is the current V10R2 parent because its name says "latest."**
> Rejected by direct artifact lineage. It is the older H07 lineage. Current successful V10R2 adapters are the distinct step1/step4/step12 artifacts under `D:\VERA\models\adapters\v10r2-*`. Lane B does not use the legacy H07 adapter as a V10R2 parent.

> **Challenge: stale `C:\Vera` references under `D:\VERA\models` are harmless because the files are historical.**
> They are provenance debt. Fifteen current text references still name the obsolete root. Lane B does not execute those scripts unchanged and treats `D:\VERA` as authoritative.

Decision ceiling: FLA/Triton gated-delta is mechanically qualified for further local development probing. It is not yet qualified as the preferred end-to-end training path until fixed-shape whole-model timing and at least one real optimizer step are observed. TorchAO8 remains pending a real GPU optimizer step.
