// Native Windows custody regression: a harmless child must run in a real
// JobObject and terminate on owner-handle close. No model or GPU use.
using System;
using Vera.TrainingDay.HostBroker;

internal static class NativeCustodyTests {
    private static void Require(bool condition, string reason) {
        if (!condition) throw new Exception(reason);
    }
    public static int Main() {
        using (var custody = JobObjectChildCustody.LaunchHarmlessProbe()) {
            Require(custody.ProcessId > 0, "child PID absent");
            Require(custody.IsInJob, "child was not bound to native job");
            Require(custody.HasExpectedLimits,
                "kernel Job readback missing job-wide memory limit, CPU hard cap or kill-on-close");
            Require(custody.IsAlive, "child ended before custody close");
            // Keep the kernel PROCESS handle pinned while closing the Job,
            // eliminating a PID-reuse race in the termination assertion.
            Require(custody.CloseJobAndConfirmChildExit(),
                "native child did not exit after closing owning Job handle");
        }
        Console.WriteLine("NATIVE_JOB_CUSTODY_PASS child_assigned=true"
            + " memory_limit_bytes=" + JobObjectChildCustody.ProbeJobMemoryBytes
            + " cpu_hard_cap_rate=" + JobObjectChildCustody.ProbeCpuRate
            + " killed_on_close=true");
        return 0;
    }
}
