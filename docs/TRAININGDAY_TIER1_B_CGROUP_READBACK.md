# TrainingDay Tier 1 B: read-only Linux cgroup-v2 custody gate

**Source candidate. The installed Lappy/WSL2 broker remains HOLD.**

This supplements Draft #312's negative process-group/orphan witness. A
Windows JobObject cannot guarantee nested WSL CUDA descendant death, and a
Linux `killpg()` does not contain a process that starts a new session.

`trainingday/linux_cgroup_scope_readback.py` reads Linux kernel `/proc` and
`/sys/fs/cgroup`; it does not create scopes, call `cgroup.kill`, spawn workers,
allocate CUDA, approve rights, or issue a launch grant. A proposed owner-owned
scope must use a dedicated `trainingday-...` leaf. The observer rejects:

- missing, reused or stale retained PID creation ticks;
- a different observed unified cgroup-v2 membership;
- broad, malformed or traversed scope paths;
- symlink substitution within the proposed scope path;
- missing or contradictory `cgroup.events`/`cgroup.procs` readback.

`cgroup.procs` contains direct processes only. `cgroup.events` `populated`
reflects the subtree at one observation. `scope_empty_at_observation` is
**not** retrospective evidence that no worker escaped its scope, proof of
crash containment, or a stable cross-boot custody record. The reader always
returns `physical_training_authorized=false`,
`windows_wsl2_crash_qualified=false`, and
`all_prior_descendants_proven_dead=false`.

## Remaining host-native work (not accomplished here)

1. The Windows HostBroker must independently authenticate and bind its owner
   process to an owner-only, delegated WSL2 cgroup-v2 scope before dispatch.
   Verify cgroup mount/delegation permissions and distinguish the owner's
   kernel handles from caller-controlled pathnames.
2. A broker-independent host-owned recovery service must survive Windows
   broker and `wsl.exe` crashes; its exact authority to stop the owned Linux
   scope must be enrolled and tested, not inferred from source.
3. Inject an owned harmless child that escapes its process group, then an
   owned CUDA test process; crash broker, `wsl.exe`, and Linux supervisor in
   independent scenarios. Kernel-observe owned scope and global process/GPU
   absence; treat PID reuse, missing cgroup state or a survivor as HOLD.
4. Review actual WSL2 cgroup migration, systemd delegation, nested children,
   host reboot, deadline, immutable grant replay epoch, and terminal receipt.
   No `wsl --terminate` on a shared distro, no interference with Blender,
   PreActive, Edge or unrelated processes.

Hosted Linux tests exercise parser and kernel observation only, including
negative cases. They **do not qualify real Windows/WSL2 process custody**.

