# Official Qwen3.5-4B metadata byte pins — source-only candidate

This records exact metadata bytes at `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It is not a model
checkpoint, signed attestation, protected-data admission, host lease, or
permission to train. The production `require_official_base` gate remains
**fail-closed** with `TRUSTED_OFFICIAL_METADATA_PIN_ITEMS = ()` until an
independent reviewer admits the source pin set through that separately
controlled change.

| Path | Source at pinned revision | Exact bytes | SHA-256 of actual bytes |
| --- | --- | ---: | --- |
| `config.json` | `/raw/` Git blob | 3,161 | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| `tokenizer_config.json` | `/raw/` Git blob | 16,710 | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `tokenizer.json` | **`/resolve/` LFS object** | **12,807,982** | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `model.safetensors.index.json` | `/raw/` Git blob | 76,196 | `cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d` |
| `chat_template.jinja` | `/raw/` Git blob | 7,756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `merges.txt` | `/raw/` Git blob | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `vocab.json` | `/raw/` Git blob | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

At the pinned revision, `raw/tokenizer.json` serves a 133-byte Git LFS
**pointer**, not the actual tokenizer. Its pointer digest must never be used
as the tokenizer's byte identity. `resolve/tokenizer.json` returned the
12,807,982-byte object with the `x-linked-etag` SHA-256 that matched the
streamed digest independently computed for the content. No weight shard was
downloaded or hashed as part of this source audit.

## Offline check of a separate local metadata snapshot

```bash
python -m trainingday.official_metadata_bytes --snapshot /absolute/path/to/metadata
python -m unittest tests.test_trainingday_official_metadata_bytes -v
```

The `--snapshot` check reads exactly seven named files, hashes them in bounded
chunks, rejects absent, wrong-size, altered, symbolic-link and nonregular
objects, and emits a source-only receipt. It reads **no safetensors shard**,
never downloads anything, and does not change the protected admission gate.
The positive tests use tiny synthetic fixtures solely through an explicitly
untrusted test seam; synthetic passing tests are not a real official snapshot
validation.

### Path races and non-atomicity (source-only limitations)

On POSIX, the checker opens the absolute path component-by-component using
`dir_fd`/`openat`, `O_DIRECTORY` and `O_NOFOLLOW`, then reads each named file
relative to the anchored directory descriptor with `O_NOFOLLOW`. It retains
all opened file descriptors, rechecks each named entry after the last hash,
and rewalks the root path to detect persistent swaps. Tests deliberately
replace ancestor directories with symbolic links, rename the root after the
first anchor, remove an earlier verified file, and replace an earlier file
with identical bytes but a different inode. All must fail closed.

**This does not establish an atomic immutable seven-file snapshot.** Active
writers can still alter or restore files between observations, including
after the final check. The receipt states
`atomic_immutable_snapshot_verified=false` and
`host_path_custody_verified=false` even after all byte hashes match.
Independent immutable-storage custody, host attestation, and owner-authorized
signing must remain separate. On Windows or POSIX without the required secure
directory-descriptor primitives, this source-only checker deliberately returns
`HOLD`; it does not claim portable safe traversal by a path pre-check.

## Still required for physical Qwen training

- Independently verify the **entire** two official weight shards on VN1/VN2
  by actual local byte SHA-256 (official LFS pointer OIDs alone are metadata).
- Establish trusted host identity, storage custody, GPU ownership/lease and
  owner signatures, without making Lappy the permanent weight store.
- Independently admit exact source pins and training/evaluator data rights
  before changing protected gates or running an optimizer.
- Preserve `$0` hosted spend and do not equate a Git/CI PASS with physical
  model readiness.
