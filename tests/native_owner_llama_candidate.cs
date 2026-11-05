// Windows native SOURCE-only regression. No file, socket or process is opened.
using System;
using Vera.TrainingDay.HostBroker;

internal static class NativeOwnerLlamaCandidateTests {
    static void Require(bool condition, string failure) {
        if (!condition) throw new Exception(failure);
    }
    static void MustHold(OwnerLlamaPlanReview report, string reason) {
        Require(!report.CanPhysicalRun && report.PhysicalRunState=="PHYSICAL_RUN_HOLD",
            "a source review emitted launch authority");
        Require(report.HasHold(reason), "missing negative policy hold " + reason);
    }
    static OwnerLlamaObservation Fixture() {
        return new OwnerLlamaObservation {
            Machine="LAPPY",
            OwnerSid="S-1-5-21-100-200-300-1001",
            ApprovedOwnerSid="S-1-5-21-100-200-300-1001",
            OwnerTokenObserved=true,
            PathsProtected=true,
            ExecutablePath=@"C:\Program Files\VERA\llama-server.exe",
            ApprovedExecutablePath=@"C:\Program Files\VERA\llama-server.exe",
            ExecutableSha256=new string('a',64),
            ApprovedExecutableSha256=new string('a',64),
            ModelPath=OwnerLlamaServerLaunchCandidate.ModelPath,
            ModelSha256=OwnerLlamaServerLaunchCandidate.ModelSha256,
            ModelBytes=OwnerLlamaServerLaunchCandidate.ModelBytes,
            FreePhysicalBytes=OwnerLlamaServerLaunchCandidate.JobMemoryCeiling
                + OwnerLlamaServerLaunchCandidate.RequiredHostHeadroom + 1024UL,
            CpuPercent=10, DiskPercent=2, ActiveInferenceProcesses=0,
            AgeSeconds=1, PortAvailable=true,
            RequestedArguments=OwnerLlamaServerLaunchCandidate.ExpectedArguments()
        };
    }
    public static int Main() {
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(null),"NO_OBSERVATION");
        OwnerLlamaObservation o=Fixture();
        OwnerLlamaPlanReview proposed=OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o);
        Require(proposed.SourceInputsMatch,"self-consistent synthetic fixture unexpectedly rejected");
        Require(!proposed.CanPhysicalRun && proposed.PhysicalRunState=="PHYSICAL_RUN_HOLD",
            "synthetic readiness issued real launch authority");
        try {
            OwnerLlamaServerLaunchCandidate.PhysicalRunIsForbidden(proposed);
            throw new Exception("PHYSICAL_RUN was accidentally enabled");
        } catch (InvalidOperationException e) {
            Require(e.Message.StartsWith("PHYSICAL_RUN_HOLD"),"unexpected launch result");
        }
        o=Fixture(); o.Machine="NOT_LAPPY";
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"HOST_MISMATCH");
        o=Fixture(); o.OwnerTokenObserved=false;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"OWNER_IDENTITY_UNVERIFIED");
        o=Fixture(); o.ApprovedOwnerSid="S-1-5-21-100-200-300-9999";
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"OWNER_IDENTITY_UNVERIFIED");
        o=Fixture(); o.ExecutablePath=@"C:\Temp\llama-server.exe";
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"EXECUTABLE_PATH_UNPINNED");
        o=Fixture(); o.ExecutablePath=@"C:\Program Files\VERA\..\llama-server.exe";
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"EXECUTABLE_PATH_UNPINNED");
        o=Fixture(); o.ExecutableSha256=new string('b',64);
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"EXECUTABLE_HASH_MISMATCH");
        o=Fixture(); o.ModelPath=@"D:\VERA\models\gguf\forged.gguf";
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"MODEL_PATH_MISMATCH");
        o=Fixture(); o.ModelSha256=new string('b',64);
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"MODEL_HASH_MISMATCH");
        o=Fixture(); o.ModelBytes=1;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"MODEL_SIZE_MISMATCH");
        o=Fixture(); o.RequestedArguments=new[] { "--host", "0.0.0.0" };
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"ARGUMENTS_CHANGED");
        o=Fixture(); o.RequestedArguments[7]="999"; // caller changes GPU layers
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"ARGUMENTS_CHANGED");
        o=Fixture(); o.FreePhysicalBytes=OwnerLlamaServerLaunchCandidate.JobMemoryCeiling;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"LOW_MEMORY");
        o=Fixture(); o.CpuPercent=95;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"BUSY_CPU");
        o=Fixture(); o.DiskPercent=80;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"BUSY_DISK");
        o=Fixture(); o.ActiveInferenceProcesses=1;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"CONCURRENT_INFERENCE");
        o=Fixture(); o.PortAvailable=false;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"PORT_OCCUPIED");
        o=Fixture(); o.AgeSeconds=99;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"STALE_TELEMETRY");
        o=Fixture(); o.PathsProtected=false;
        MustHold(OwnerLlamaServerLaunchCandidate.ReviewSyntheticObservation(o),"PATH_CUSTODY_UNVERIFIED");
        string[] args=OwnerLlamaServerLaunchCandidate.ExpectedArguments();
        Require(Array.IndexOf(args,"127.0.0.1")>=0 && Array.IndexOf(args,"0.0.0.0")<0,
            "loopback-only binding lost");
        Require(Array.IndexOf(args,"--lora")<0 && Array.IndexOf(args,"--no-webui")>=0
            && Array.IndexOf(args,"--offline")>=0,"unsafe GGUF flags introduced");
        Console.WriteLine("OWNER_LLAMA_SOURCE_HOLD_PASS negative_cases=19 physical_run=false"
            + " model_sha_pinned=true loopback=127.0.0.1 port=11435");
        return 0;
    }
}
