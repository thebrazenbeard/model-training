# W51 offline conversion capacity screen — source-only candidate

**Not installed, not merged, not host-authoritative.** This isolated candidate
accepts caller-provided numerical host observations and rejects inadequate
memory, swap, or staging space for the proposed authentic Qwen3.5-4B + R16
conversion experiment. It never launches or authorizes a process, even if the
numerical screen passes. It cannot verify the observations, owner presence,
model bytes, a HostBroker lease, or a WSL child exit witness.

The conservatively chosen thresholds reproduce the W46 runbook:

- At least 12 GiB **physical Windows host RAM available**, independently of
  any WSL2 guest memory report.
- At least 12 GiB WSL guest available RAM when WSL is involved.
- At least 1 GiB free swap and 25 GiB staging space.

Sample observations are untrusted; the sample schema intentionally rejects
caller-supplied lease/authority flags. Missing fields and duplicate JSON keys
also fail closed. Every CLI invocation exits code `2`, including numerical
success, so no shell pipeline can mistake it for permission to load weights.

Run tests from this directory:

`python3 -B -m unittest discover -s tests -v`

Future owner-qualified execution must independently rehash official
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` bytes, check
the October 30 correct-label R16 adapter, use pinned
`llama.cpp b12098@149f5641b931aaf33697d70f9abc90c3e5300247`, and obtain
a real host-owned exclusive conversion slot, WSL child process custody and
continuous host-wide resource limits. Those effects are deliberately absent.

Source reference: [W46 runbook](https://github.com/thebrazenbeard/chat-communication-bus/issues/404#issuecomment-6646859840).
