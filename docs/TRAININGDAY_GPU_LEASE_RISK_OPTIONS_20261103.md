# TRAININGDAY GPU ownership: two different contracts

**Status: decision needed, physical training remains HOLD.** This is an
evidence-backed risk distinction, not a modification of the currently installed
broker or an owner-approved relaxation of its contract.

## Current Lappy evidence (2026-11-03, read-only host probe)

| Evidence class | Observation |
| --- | --- |
| REAL_SYSTEM_PROBE | GeForce RTX 3050 Laptop GPU, 4096 MiB total, 3778 MiB free at probe, 0% reported utilization. |
| REAL_SYSTEM_PROBE | Windows driver model WDDM, CUDA compute mode `Default`. |
| REAL_SYSTEM_PROBE | The compute-process query listed Blender PID 15108 and Microsoft Edge PID 30772. Their per-process GPU memory was `N/A`. |
| REAL_SYSTEM_PROBE | Authenticated installed HostBroker `STATUS_V1` was `HOLD`: zero authorized launches, no GPU lease, rights decision, or NAS witness. |
| SOURCE | [NVIDIA CUDA on WSL user guide §5.2](https://docs.nvidia.com/cuda/wsl-user-guide/) warns that WSL `nvidia-smi` active-compute-process queries and modifiable compute mode are unsupported. |

The aggregate 0% figure and 3778 MiB of free memory do **not** mean no other
application has a GPU context, cannot allocate memory next, or is governed by
the proposed training broker. A source-controlled file lock or JobObject cannot
prevent unrelated Windows/WDDM clients from using the display GPU.

## Contract A — hardware-exclusive GPU (current strict requirement)

The broker must independently establish that no other GPU client can execute
on the device throughout the lease, including nonparticipating applications.
On this WDDM/WSL2 GeForce route **no qualifying enforcement mechanism has been
observed**. Incomplete process telemetry is not a positive ownership witness.

**Disposition: FAIL/HOLD for positive lease issuance.** Do not infer `PASS`
from idle telemetry, a named mutex, a successfully owned Vera child process,
an untrusted JSON receipt, or a passed hosted Windows unit test.

## Contract B — exclusive TRAININGDAY workload, nonexclusive physical GPU

This is a *potentially useful but weaker, owner-approval-required alternative*:

- The installed Windows service owns an OS-exclusive reservation and strictly
  increasing host/NAS epoch **among all authorized Vera training workers**.
- The service verifies the requester's Windows access token and retained
  process `HANDLE` identity (kernel PID **and creation time**); the client cannot
  choose the executable, training path, or lease token.
- It owns an immutable run plan, protected data/model handles, a Windows Job,
  and kill-on-owner-loss custody. It records terminal completion before release.
- It measures GPU/host availability immediately before dispatch, refuses a
  detected foreign GPU process or inadequate VRAM, and aborts safely on CUDA
  OOM or failed health checks. It does not kill unrelated applications.
- Absence of a detected foreign process still carries **residual competition
  risk** because WDDM/WSL telemetry is incomplete. This risk must be accepted
  explicitly, not hidden under the word “exclusive.”
- For **protected R201 training**, independently authenticated corpus rights,
  protected evaluator custody, NAS Q6 monotonic antirollback witness, official
  base hashes, and low-memory optimizer qualification are **unchanged prerequisites**.
  A separately proposed public-only Tier 1 fit probe is described in
  [TRAININGDAY_TIER1_PUBLIC_FIT_PROPOSAL_20261103.md](TRAININGDAY_TIER1_PUBLIC_FIT_PROPOSAL_20261103.md):
  it is not protected training and has no power to grant Tier 2 authority.

**Disposition: PROPOSED, NOT APPROVED.** An owner may choose to change the
GPU-specific acceptance policy from A to B. Until then, current Contract A
and the installed broker's unconditional `HOLD` remain authoritative.

## Native source identity regression

`JobObjectChildCustody.ReadOwnedKernelIdentityForNativeTestOnly()` queries the
existing retained Windows child process handle for the kernel-reported PID
and process creation FILETIME. Native tests reject a forged PID and a
same-PID/different-creation-time claim and confirm the HANDLE identity stays
stable during a harmless JobObject child's lifetime. This supports **Vera
child-process identity only**. It does not authenticate a caller's token,
reserve CUDA hardware, reconcile foreign processes, or make Contract B live.

**Next protected effect, if B is deliberately selected:** independently
reviewed implementation/installation of a host-owned broker, one owner-approved
maintenance window, a native Windows+WSL2 process/lease crash matrix, witnessed
NAS compare-and-append, an actual GPU memory/OOM qualification with bounded
cleanup, and host readback of the resulting running binary and state. No source
file or hosted CI result substitutes for that installed qualification.
