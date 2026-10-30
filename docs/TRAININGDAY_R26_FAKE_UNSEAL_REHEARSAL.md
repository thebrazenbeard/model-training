# TrainingDay R26-01 — fake-only eight-arm evaluation lifecycle rehearsal

**Classification:** `SOURCE/CPU TEST`, `NO_PRIVATE_BANK_ACCESS`, `RELEASE_HOLD`.

The new `tests/test_r26_fake_fullstack_unseal.py` uses the existing public
R18 fixture creator to construct eight *synthetic* MeZO/QLoRA
`frozen`/`zero`/`correct`/`corrupt` arm receipts and adapter byte stand-ins.
It deliberately invokes the actual `trainingday.r18_eval.evaluate()` and
actual `_preunseal_inventory()`; the real `_bank()` is patched out and never
reads a bank, nonce, evaluation prompt, answer, or protected label.

## Negative and fake-positive evidence

| Scenario | Actual source path | Expected protected-bank result |
| --- | --- | --- |
| Eight valid public fake arms, actual unattested source | inventory → real unconditional independent-attester HOLD | zero bank calls, no CUDA, no emitted score |
| Missing arm, duplicate arm, wrong-plan inventory, digest recomputed | actual eight-arm inventory validator | zero bank/attester/load calls |
| Repinned forged optimizer receipt; altered saved adapter bytes | actual receipt contract and adapter hash validator | zero bank calls |
| Fake VRAM-fit error after test-only attester patch | CUDA preflight | zero preload and bank calls |
| Fake base OOM or partial adapter failure | model preload | zero bank calls; no score files |
| Fake model preloaded, then synthetic scorer error | model session → fake unseal → scorer → cleanup | same fake model session closed; no score files |
| Fake bank fails to report observable open timestamp | synthetic fake-bank callback contract | scorer blocked; session closed |
| Fake 64-row blinded aggregate score lifecycle | preflight → preload → fake-bank → fake-score → private/public synthetic files → cleanup | outputs produced in temporary fixture only; automatic promotion false |
| Receipt modified after fake score, before publishing | actual post-scoring rehash | rejected; no final score files |

The test-only patch of `_require_independent_preunseal_attestation` allows
execution of downstream mocked branches **within the unit test only**. The
production evaluator continues to raise at that gate unconditionally.
The fake positive case is an illustration of control flow, **not** a
permission to remove the gate or open sealed bank data.

## Explicit failure of qualification

> **Hostile custody challenge:** the evaluator and raw `_bank()` helper remain
> callable by the same process/OS principal. Monkeypatching a Python gate can
> simulate a custodian but is not an independent custodian. No test here
> proves that protected plaintext is inaccessible to the training principal.

Unresolved: independent evaluator principal and host-enforced bank ACL/token
scope, trustworthy attester key and expiry, externally pinned eight-arm
neural receipts, complete physical Qwen3.5-4B base/adapter preload and
fresh-process tensor byte proof, physical GPU fit/Modern Standby guard,
valid equal-time MeZO/QLoRA training pair, behavioral improvement, and
fresh untouched final evaluation. The original R18 time budget overrun
remains `TIME_BUDGET_OVERRUN_INVALID` and must not be used for comparison.

### Reproduction (public fixtures only)

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python -m unittest -q \
  tests.test_r26_fake_fullstack_unseal tests.test_r18_eval \
  tests.test_r20_attestation_contract
```

`SOURCE_REVIEW = PASS` must not be treated as neural, host, private-bank,
promotion, deployment, or merge authority. The actual bank remains CLOSED.
