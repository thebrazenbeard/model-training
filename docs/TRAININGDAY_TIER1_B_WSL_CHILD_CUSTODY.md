# TRAININGDAY Tier 1 Contract B: nested WSL child custody

**SOURCE/TEST WITNESS ONLY. Physical job remains HOLD.**

Patrick selected bounded, nonexclusive Windows WDDM coexistence for a public
synthetic hardware-fit experiment. That decision **does not** prove cleanup of
Linux CUDA workers when the Windows HostBroker or its `wsl.exe` handle dies.
The broker's Windows JobObject owns Windows child handles; a separate Linux
process group or cgroup owns Linux descendants. They are different kernels.

`trainingday/linux_child_custody_probe.py` and its hosted Linux tests cover
only *harmless* Python sleep processes:

- a dedicated Linux session/group leader has a kernel PID+start-time identity;
- a forged creation-time observation is rejected before group signalling;
- a still-live retained test leader can receive targeted TERM and bounded
  escalation without sending signals to unrelated groups;
- killing the controlling supervisor while its child is in a *different*
  session leaves that Linux child alive. The test harness then cleans up only
  its own child group. That is a **counterexample** to a claim that Windows
  JobObject closure automatically proves nested Linux worker death.

No source fixture, daemon, environment flag or submitted status JSON may
turn this receipt into launch authority. The helper deliberately returns
`all_linux_descendants_proven_dead=false`,
`windows_wsl_broker_crash_qualified=false`, and
`physical_training_authorized=false` even after harmless cleanup succeeds.

## Required live Windows + WSL2 qualification before Tier 1 launch

1. HostBroker owns an immutable bounded plan and retains a Windows process
   handle, including PID and kernel creation time. Its protected process/job
   identity is separate from the Linux worker identity.
2. The Linux side runs inside an owner-controlled *dedicated* process or cgroup
   scope. If cgroup v2 `cgroup.kill` is proposed, verify actual host-delegated
   cgroup permissions and membership; merely writing a pathname in a manifest
   is not proof of containment.
3. Stop/timeout/CUDA OOM paths terminate the **exact owned** Linux scope,
   observe every descendant exit, and journal terminal status before releasing
   the host-local Tier 1 replay epoch. Do not call `wsl --terminate` on a
   shared Ubuntu distribution or kill unrelated processes.
4. Inject a Windows broker crash, a killed `wsl.exe`, killed Linux supervisor,
   stale/reused PID, and a nested child which detaches its process group.
   An unidentified descendant, insufficient WSL `/proc` visibility, ambiguous
   cgroup state or a surviving test CUDA worker means **HOLD**.
5. Verify no Tier 2 R201/evaluator/NAS Q6 mounts or state changes. Record
   observed GPU-memory/OOM behavior without claiming hardware exclusivity or
   a hard WDDM per-process VRAM partition.

Hosted `ubuntu-latest` exercises real Linux process semantics, **not** WSL2,
Windows SCM, a HostBroker service crash, a live CUDA kernel, or privileged
cgroup kill. Those remain independent owner-host release blockers.
