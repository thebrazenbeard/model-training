# TRAININGDAY R27-05 — static supply-chain witness (SOURCE/CPU only)

This is an **offline, public-toy, nonauthorizing** verifier. It checks whether a
small supplied byte string matches its own declared SHA-256, whether a direct
`requirements.txt`-style dependency list uses exact **version** pins, whether
source revisions are syntactically full 40-character Git commits, and whether
ZIP / `.tar.gz` metadata contains unsafe archive member paths, links, special
files, duplicates, excessive entry counts, or high expansion ratios. Portable
path checks also reject nested Windows device aliases (`CON`, `NUL.txt`,
`COM1`–`COM9`, `LPT1`–`LPT9`, superscript variants), trailing periods/spaces,
and disguised ZIP directory entries. It **does
not** fetch, decompress into the filesystem, install, load, import or execute
the supplied artifact, and it does not inspect protected model/eval data.
All TAR PAX extended and global headers are refused because their path,
link-target, owner and xattr overrides are outside this small admission scope.

`STATIC_BYTES_MATCH_ONLY` means *internal declaration/bytes consistency*.
Even a self-consistent byte/hash pair can be fabricated by an attacker. The
decision always returns:

- `licenses_authenticated=False`
- `dependency_wheel_hashes_verified=False`
- `reproducible_execution_verified=False`
- `binary_execution_authorized=False`
- `model_release_authorized=False`

Any missing, unsupported, overlarge or malformed input remains HOLD. This
strictly bounded assessor accepts observed public fixture artifacts no larger
than 4 MiB each; it cannot attest actual Qwen3.5-4B base/tensor artifacts.

## Live-source inspection receipt (not a runtime probe)

At exact model-training local source-equivalent commit
`2082feccbbcb876b689c93346bca3154895b7e49`, tree
`505ef315e13824c5e4aa1d4a09c4dcc89b9da6d4` (published as Draft #152
remote head `583a6b41a1be405e76dc26b792a45e6dde5c219d`):

| Source | SHA-256 of source file | SOURCE_INSPECTION finding |
| --- | --- | --- |
| `requirements.txt` | `1092091c9cbaaa2d682b24409f1b21dc24d5225db721d6079280eec00d84aaa6` | Five direct package **versions** pinned with `==`; no verified complete lock including transitive wheel hashes. |
| `.github/workflows/tests.yml` | `f0267cfbd194526803007aa4a00b7fb925e01a9fbda0a31fa2738bc3593548ab` | CI uses mutable `actions/checkout@v4`, `actions/setup-python@v5`; installs pinned `torch==2.14.0+cpu` without `--require-hashes`. |
| `.github/workflows/identity-bootcamp.yml` | `f64d16139b6d1169b0cc79faad2c70e26047ed8e6d90980eab14cac1c8b0ecbd` | Downloaded llama.cpp archive and GGUF are checked against explicit SHA-256 before use, but the GitHub release tarball is extracted by `tar -xzf` without a separate static path/link/special-file admission gate. Model URL resolves `main` but the downstream expected SHA constrains contents. Checkout/setup/cache/upload action tags remain mutable. |
| `.github/workflows/v4-1-training-build.yml` | `69930a34cf0fe67d6396be8e429438205bc722a3daf82f424539bede0819a1b1` | Checkout/setup action versions use immutable SHAs; `datasets`, `transformers`, `jinja2`, `pytest`, `sentence-transformers` versions pinned, but downloaded wheel hash/ABI resolution remains unverified. |

The direct `tar -xzf` observation is a **hardening frontier**, not evidence that
the currently SHA-pinned archive is malicious. A member-admission gate would
be needed before extraction, with verified source checksum and a separately
trusted publisher artifact commitment; this prototype provides safe toy
metadata rules but is **not wired to that production workflow**.

The source version pins are not a reproducible environment lock. The actual
binary distribution for a version, transitive resolver graph, OS/ABI, Python
interpreter, CUDA driver, artifact signer, SPDX/license rights and training
corpus provenance require independent attestation. A declared SPDX string is
not legal authorization or evidence of usage rights.

## Test receipt

Run only the public standard-library suite:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest -q tests.test_r27_supply_chain
```

Negative cases include unpinned/Git/direct dependency references, duplicate
requirements, forged schema using custom equality, source `main`, traversal
locators, extreme manifest integers, SHA mismatch, missing/extra bytes,
ZIP traversal/absolute Windows and POSIX paths, ZIP symlinks/duplicate names,
ZIP expansion bomb and nested archive, ZIP entry-count overflow, ZIP
directory-mode-with-data forgery, Win32 device aliases/trailing-period entries
in both formats, and `.tar.gz` path traversal, symlink and hardlink. An actual CPU source test does not imply
full CI or genuine model training.
Global and per-member TAR PAX claims (including `linkpath=../../escape` and
`SCHILY.xattr.security.selinux`) are explicit negative fixtures.

## Hard boundaries

This module is not a package manager, general archive extractor, dependency
resolver, trusted build system, external publisher certificate checker, or
production admission gate. It is deliberately not imported by the real R18
trainer, quantizer or serving process. No dependency install, model download,
paid compute, private-bank access, credential, GPU training, merge, model
promotion or deployment was performed for this work.
