# Neuro-Sync ↔ TRAININGDAY — bounded JSONL split-content audit

**Status:** optional offline **LOCAL_CONTENT_INSPECTION**. This tool does
actual read-only parsing and cryptographic hashing of operator-selected files.
It is not independent rights, custody, model, or training verification.
All results are **HOLD**, even when no overlap is detected.

## Input

The operator explicitly supplies two JSONL files inside a narrowly scoped,
existing, local allowed root. Every nonblank line must be one closed-shape
record (all values synthetic here):

    {"id":"sample-1","role":"user","text":"How did the voyage end?"}

or a bounded multi-turn chat:

    {"id":"sample-2","messages":[{"role":"user","content":"What is the status?"},{"role":"assistant","content":"Unknown without evidence."}]}

Accepted roles: system, developer, user, assistant, tool.
Unknown fields (including explicit hidden answer keys), duplicate JSON object
keys, malformed UTF-8/JSON, empty normalized text and oversized inputs stop
the audit. A malicious actor can still conceal an answer in an allowed string.

## Checks

- Streams and SHA-256 hashes actual file bytes (not caller-declared digests);
  refuses changing paths, symlinks and nonregular inputs to the extent supported
  by the local platform. Distinct files with identical digest trigger HOLD.
- Computes **exact normalized text equality** using NFKC, casefold, punctuation
  and whitespace cleanup. Detects overlap even when roles or example IDs differ.
- Detects repeated user prompts with different assistant answers and shared
  multi-turn prefixes with divergent later turns.
- Measures candidate **lexical near duplicates** with fixed 5-character shingle
  Jaccard threshold 0.82; flags for manual review, never automatic deletion.
- Emits only counts, file SHA-256 digests, typed status and constant warnings;
  no raw messages, prompts, answers, identifiers or filenames in the JSON output.
- Refuses excessive files, lines, records, text and comparison fanout rather
  than silently truncating:

| Resource | Current hard bound |
|---|---:|
| Each input file | 8 MiB |
| JSONL line | 32 KiB |
| Records per file | 5,000 |
| Messages per record | 16 |
| Text characters per record | 8,192 |
| Total text chars per file | 2,000,000 |
| Indexed lexical shingles | 500,000 |
| Candidate comparisons | 100,000 |

## Run

After review and installation as
model-training/trainingday/neurosync_dataset_audit.py:

    python -B -m trainingday.neurosync_dataset_audit \
      --allowed-root /path/to/approved-local-datasets \
      --train /path/to/approved-local-datasets/train.jsonl \
      --heldout /path/to/approved-local-datasets/heldout.jsonl

The tool prints a JSON summary. Exit code **1** means a bounded inspection
completed but still holds all training authority. Exit code **2** means
inspection could not safely complete and also holds. It **never** returns a
zero exit code that downstream code might mistake for launch permission.

Offline tests:

    python -B -m unittest discover -s tests -p 'test_neurosync_dataset_audit.py' -v

## Deliberate limitations

> **UNKNOWN:** A result with zero matches does not establish that train and
> heldout are semantically disjoint. Reworded paraphrases, confusable Unicode,
> compressed/encrypted content and training elsewhere can escape this lexical
> audit. Shared prompts or similar ngrams can also create false positives.

> **UNKNOWN:** File-size/inode/mtime/ctime consistency and a local SHA-256 do
> not authenticate a separate evaluator or owner-rights approval. Train and
> heldout are read sequentially, not as an atomic snapshot. A hostile process
> controlling filesystem metadata can still invalidate evidence.

> **PLATFORM:** On POSIX with dir_fd, O_NOFOLLOW and O_DIRECTORY, every
> path component is opened without following symlinks. Windows lacks that
> descriptor-walk guarantee; successful results explicitly include
> PLATFORM_PATH_CONTAINMENT_NOT_ATOMIC_NO_DIRFD_WALK. Use only host-trusted,
> owner-controlled directories on Windows and never expose this tool as a
> privileged file-access broker.

> **HOLD:** No physical optimization, Qwen/PEFT transfer, model identity,
> persistent state commit, or heldout promotion is authorized. Source tests
> are not neural evaluations or proof of learned retention.
