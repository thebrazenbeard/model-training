// SOURCE-ONLY planning/negative-regression surface. No physical launch API.
// It must never be imported by the installed TrainingDay broker or interpreted
// as rights, model compatibility, host identity, or process-lease authority.
using System;
using System.Collections.Generic;
using System.IO;

namespace Vera.TrainingDay.HostBroker {
internal sealed class OwnerLlamaObservation {
    // Synthetic native-test inputs. Production must acquire these itself from
    // protected host policy, Windows tokens/handles, hashes and kernel probes.
    internal string Machine, OwnerSid, ApprovedOwnerSid;
    internal bool OwnerTokenObserved, PathsProtected;
    internal string ExecutablePath, ApprovedExecutablePath;
    internal string ExecutableSha256, ApprovedExecutableSha256;
    internal string ModelPath, ModelSha256;
    internal long ModelBytes;
    internal ulong FreePhysicalBytes;
    internal int CpuPercent, DiskPercent, ActiveInferenceProcesses, AgeSeconds;
    internal bool PortAvailable;
    internal string[] RequestedArguments;
}

internal sealed class OwnerLlamaPlanReview {
    internal readonly string[] Holds;
    // Even matching synthetic inputs cannot authorize a real host operation.
    internal bool CanPhysicalRun { get { return false; } }
    internal string PhysicalRunState { get { return "PHYSICAL_RUN_HOLD"; } }
    internal OwnerLlamaPlanReview(List<string> holds) { Holds=holds.ToArray(); }
    internal bool HasHold(string reason) {
        foreach (string item in Holds) if (item==reason) return true;
        return false;
    }
}

internal static class OwnerLlamaServerLaunchCandidate {
    internal const string ExpectedMachine="LAPPY";
    internal const string ModelPath=@"D:\VERA\models\gguf\model_q5_k_s.gguf";
    internal const string ModelSha256="15150f534dc90ee15c82320ccc014463db7985205a7161228cff89b925fd1216";
    internal const long ModelBytes=3072993056L;
    internal const ulong JobMemoryCeiling=5UL*1024UL*1024UL*1024UL;
    internal const ulong RequiredHostHeadroom=2UL*1024UL*1024UL*1024UL;
    internal const uint CpuHardCapRate=2500;
    internal const int StartupAndConversationTtlSeconds=180;
    internal const int HostPort=11435;

    // Exact ordered argv allowlist. No user-defined executable, flags, LoRA,
    // URL, tokens, dynamic model loading, GPU layers, or non-loopback binding.
    internal static string[] ExpectedArguments() {
        return new[] { "--model", ModelPath, "--host", "127.0.0.1",
            "--port", "11435", "--n-gpu-layers", "0", "--threads", "2",
            "--parallel", "1", "--ctx-size", "2048", "--no-webui", "--offline" };
    }

    static bool ValidDigest(string value) {
        if (String.IsNullOrEmpty(value) || value.Length!=64) return false;
        foreach (char c in value) if (!((c>='0' && c<='9') ||
                (c>='a' && c<='f') || (c>='A' && c<='F'))) return false;
        return true;
    }

    static bool CanonicalLocalPath(string value) {
        if (String.IsNullOrWhiteSpace(value) || value.StartsWith(@"\\", StringComparison.Ordinal))
            return false;
        try { return Path.IsPathRooted(value)
            && String.Equals(Path.GetFullPath(value), value, StringComparison.OrdinalIgnoreCase); }
        catch (ArgumentException) { return false; }
        catch (NotSupportedException) { return false; }
        catch (PathTooLongException) { return false; }
    }

    static bool ExactlyExpectedArguments(string[] args) {
        string[] expected=ExpectedArguments();
        if (args==null || args.Length!=expected.Length) return false;
        for (int i=0; i<expected.Length; i++)
            if (!String.Equals(args[i],expected[i],StringComparison.Ordinal)) return false;
        return true;
    }

    internal static OwnerLlamaPlanReview ReviewSyntheticObservation(OwnerLlamaObservation observed) {
        var holds=new List<string>();
        if (observed==null) { holds.Add("NO_OBSERVATION"); return new OwnerLlamaPlanReview(holds); }
        if (!String.Equals(observed.Machine,ExpectedMachine,StringComparison.OrdinalIgnoreCase))
            holds.Add("HOST_MISMATCH");
        if (!observed.OwnerTokenObserved || String.IsNullOrEmpty(observed.OwnerSid)
                || !observed.OwnerSid.StartsWith("S-1-5-21-",StringComparison.Ordinal)
                || !String.Equals(observed.OwnerSid,observed.ApprovedOwnerSid,StringComparison.Ordinal))
            holds.Add("OWNER_IDENTITY_UNVERIFIED");
        if (!observed.PathsProtected) holds.Add("PATH_CUSTODY_UNVERIFIED");
        if (!CanonicalLocalPath(observed.ExecutablePath)
                || !String.Equals(observed.ExecutablePath,observed.ApprovedExecutablePath,
                    StringComparison.OrdinalIgnoreCase)
                || !String.Equals(Path.GetFileName(observed.ExecutablePath),"llama-server.exe",
                    StringComparison.OrdinalIgnoreCase))
            holds.Add("EXECUTABLE_PATH_UNPINNED");
        if (!ValidDigest(observed.ApprovedExecutableSha256)
                || !ValidDigest(observed.ExecutableSha256)
                || !String.Equals(observed.ExecutableSha256,observed.ApprovedExecutableSha256,
                    StringComparison.OrdinalIgnoreCase))
            holds.Add("EXECUTABLE_HASH_MISMATCH");
        if (!CanonicalLocalPath(observed.ModelPath)
                || !String.Equals(observed.ModelPath,ModelPath,StringComparison.OrdinalIgnoreCase))
            holds.Add("MODEL_PATH_MISMATCH");
        if (observed.ModelBytes!=ModelBytes) holds.Add("MODEL_SIZE_MISMATCH");
        if (!ValidDigest(observed.ModelSha256)
                || !String.Equals(observed.ModelSha256,ModelSha256,StringComparison.OrdinalIgnoreCase))
            holds.Add("MODEL_HASH_MISMATCH");
        if (!ExactlyExpectedArguments(observed.RequestedArguments))
            holds.Add("ARGUMENTS_CHANGED");
        if (observed.FreePhysicalBytes<JobMemoryCeiling+RequiredHostHeadroom)
            holds.Add("LOW_MEMORY");
        if (observed.CpuPercent<0 || observed.CpuPercent>20) holds.Add("BUSY_CPU");
        if (observed.DiskPercent<0 || observed.DiskPercent>10) holds.Add("BUSY_DISK");
        if (observed.ActiveInferenceProcesses!=0) holds.Add("CONCURRENT_INFERENCE");
        if (!observed.PortAvailable) holds.Add("PORT_OCCUPIED");
        if (observed.AgeSeconds<0 || observed.AgeSeconds>15) holds.Add("STALE_TELEMETRY");
        return new OwnerLlamaPlanReview(holds);
    }

    internal static void PhysicalRunIsForbidden(OwnerLlamaPlanReview ignored) {
        // This must remain unconditional even if every synthetic field matches.
        throw new InvalidOperationException("PHYSICAL_RUN_HOLD: owner-host native lease not qualified");
    }
}
}
