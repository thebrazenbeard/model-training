// Native Windows custody regression: a harmless child must run in a real
// JobObject and terminate on owner-handle close. No model or GPU use.
using System;
using Vera.TrainingDay.HostBroker;

internal static class NativeCustodyTests {
    private static void Require(bool condition, string reason) {
        if (!condition) throw new Exception(reason);
    }
    private static void RequireRejected(ulong bytes, uint rate, string why) {
        try {
            using (JobObjectChildCustody.LaunchHarmlessProbeWithTestLimits(bytes, rate)) {}
            throw new Exception("unsafe Job fixture accepted " + why);
        } catch (ArgumentOutOfRangeException) {
            // Deliberate pre-kernel validation rejection.
        }
    }
    public static int Main() {
        RequireRejected(0UL, 2500, "zero memory");
        RequireRejected(ulong.MaxValue, 2500, "memory overflow");
        RequireRejected(5UL*1024UL*1024UL*1024UL+1UL, 2500, "unaligned memory");
        RequireRejected(256UL*1024UL*1024UL, 0, "zero CPU cap");
        RequireRejected(256UL*1024UL*1024UL, uint.MaxValue, "CPU cap overflow");
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
        using (var large = JobObjectChildCustody.LaunchHarmlessProbeWithTestLimits(
                JobObjectChildCustody.LargeProbeJobMemoryBytes, JobObjectChildCustody.ProbeCpuRate)) {
            Require(large.IsInJob && large.IsAlive, "5 GiB fixture not owned/alive");
            Require(large.HasExpectedLimits, "5 GiB native Job memory/CPU readback mismatch");
            Require(large.CloseJobAndConfirmChildExit(), "5 GiB fixture child exit unconfirmed");
        }
        Console.WriteLine("NATIVE_JOB_CUSTODY_PASS child_assigned=true"
            + " memory_limit_bytes=" + JobObjectChildCustody.ProbeJobMemoryBytes
            + " large_job_memory_bytes=" + JobObjectChildCustody.LargeProbeJobMemoryBytes
            + " cpu_hard_cap_rate=" + JobObjectChildCustody.ProbeCpuRate
            + " rejected_invalid_caps=true killed_on_close=true");
        return 0;
    }
}
