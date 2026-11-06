# TRAININGDAY Neuro-Sync local artifact byte readback

**Status: source-only / local byte observations / authority HARD HOLD.**

This is a small read-only verifier for the artifact roles named in Neuro-Sync
Draft PRs [#38](https://github.com/thebrazenbeard/neuro-sync/pull/38) and
[#48](https://github.com/thebrazenbeard/neuro-sync/pull/48). The existing #38
inspector parses PEFT safetensors; this module does **not** duplicate that
parser, load model weights, or claim to prove PEFT structure. It checks the
bytes of explicitly named regular files against **untrusted expected SHA-256
and size declarations**.

## Typed roles and completeness

| Role in local verifier | Relation to Neuro-Sync #38/#48 |
| --- | --- |
| `donor_trained_checkpoint` | Full donor checkpoint (PR #37 `donor_trained`); never alias an adapter digest |
| `recipient_base_checkpoint` | Full receiver/base checkpoint (single-file only) |
| `adapter_weights` | Raw local adapter weights bytes; compare to #38 `adapter_sha256` |
| `adapter_config` | Raw local `adapter_config.json` bytes; compare to #38 `config_sha256` |
| `tokenizer` | One named tokenizer file; not proof of a complete tokenizer bundle |

Two exact manifest profiles:

- `FULL_FIVE_ROLE`: all five files required, observed without role/path/SHA
  aliasing. **Still not independently witnessed.**
- `PEFT_ADAPTER_ONLY_PARTIAL_HOLD`: only adapter weights, adapter config,
  tokenizer. It **never invents** a full donor checkpoint or receiver-base
  hash and cannot qualify a complete transfer.

The #48 `target_map_sha256` is a **canonical structural target-map digest**,
not the SHA-256 of a target-map file. Rights records and dataset split hashes
are different evidence classes and are **not** inferred from this verifier.

## Host and filesystem boundary

- On supported POSIX hosts, the root and each path component are opened through
  held directory descriptors with `O_NOFOLLOW`; no `Path.resolve` or
  `commonpath` path-check race is used. All regular files are checked before
  and after streaming against held-file metadata and the final directory entry.
- Final symlinks, symlinked parent paths, root symlinks, `..`, absolute file
  names, UNC/drive syntax, special files, hardlink aliases, duplicate roles,
  unexpected manifest fields and mismatched size/hash cause `ArtifactHold`.
- A final no-follow reopening of the declared allowroot compares held/current
  directory identity and ctime. This rejects the tested case where an attacker
  relocates the original root while the scan uses its held descriptor, then
  substitutes a different directory at the original pathname. It is a
  **point-in-time check**, not an immutable path lease: files or directories
  may change immediately afterward, so any later model use requires its own
  trusted handle-bound custody/readback.
- Per-file declared size is bounded to 32 GiB, sum to 64 GiB, and the scan
  checks a ten-minute monotonic budget. A stalled single filesystem syscall
  or malicious remote filesystem cannot be interrupted by that software
  deadline; the host must apply its own process and storage timeouts.
- On Windows, ordinary Python lacks the needed POSIX dirfd/`O_NOFOLLOW`
  semantics. The module deliberately returns
  `PLATFORM_NO_ATOMIC_PATH_CONTAINMENT` and performs no file hashing rather
  than treating NTFS reparse/junction paths as safe. A separately reviewed
  handle-based Windows implementation is required for Lappy/VN1.
- The allowroot must be selected and constrained by a trusted host. Exposing
  this function directly to arbitrary model callers could turn it into a
  file-existence/hash oracle even though it never writes anything.

## Evidence ceiling

A matching digest proves only that **this local process observed bytes
consistent with a caller-supplied declaration**. It does not authenticate the
producer, source revision, model weight role, rights holder, trained origin,
independent host custody, later persistence, adapter semantics or behavioral
transfer. A caller in the same Python process can fabricate objects or inspect
private module values; the constructor guard prevents accidental fabricated
success, **not a malicious host-attestation forgery**.

No `LocalByteEvidence` object can grant physical launch, state commit, rights
verification, an independent witness or trained-model qualification. Results
are invalidated for future use if files are replaced after observation.

Focused standalone tests use synthetic temporary files only:

```sh
python -B -m unittest -q test_neurosync_artifact_bytes.py
```

When placed at `trainingday/neurosync_artifact_bytes.py` and
`tests/test_neurosync_artifact_bytes.py`:

```sh
python -B -m unittest discover -s tests -p 'test_neurosync_artifact_bytes.py' -q
```
