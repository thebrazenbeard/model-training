// Crash-conservative host-owned reservation journal for HARMLESS TEST JOBS.
// Never an R18 data-rights grant, GPU exclusivity proof, or launch capability.
// The production broker does NOT instantiate this class: native tests exercise
// the OS file lock and durable replay floor before any future integration.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace Vera.TrainingDay.HostBroker {
internal sealed class HostOwnedProbeReservation : IDisposable {
    const int MaximumJournalBytes = 4 * 1024 * 1024;
    const int MaximumEntries = 20000;
    const string Genesis = "0000000000000000000000000000000000000000000000000000000000000000";
    readonly FileStream journal;
    readonly HashSet<string> consumed = new HashSet<string>(StringComparer.Ordinal);
    string head = Genesis;
    string nonce, request, phase = "GENESIS";
    long epoch;
    bool disposed, poisoned;

    // This accepts a path only for a native platform regression in an isolated
    // test directory. Production must instead open a fixed SYSTEM-owned path,
    // prove its ACL/identity and obtain an independent NAS replay witness.
    internal static HostOwnedProbeReservation OpenForNativeTestOnly(string fullPath) {
        if (String.IsNullOrEmpty(fullPath) || !Path.IsPathRooted(fullPath)
                || Path.GetFullPath(fullPath) != fullPath)
            throw new ArgumentException("native test journal path must be absolute");
        FileStream stream = new FileStream(fullPath, FileMode.OpenOrCreate,
            FileAccess.ReadWrite, FileShare.None, 4096, FileOptions.WriteThrough);
        try { return new HostOwnedProbeReservation(stream); }
        catch { stream.Dispose(); throw; }
    }

    private HostOwnedProbeReservation(FileStream stream) {
        journal = stream;
        RebuildAndVerify();
    }

    static bool Digest(string value) {
        if (value == null || value.Length != 64) return false;
        for (int i = 0; i < value.Length; i++)
            if (!((value[i] >= '0' && value[i] <= '9') ||
                  (value[i] >= 'a' && value[i] <= 'f'))) return false;
        return true;
    }

    static string Sha(string value) {
        using (SHA256 hash = SHA256.Create()) {
            byte[] raw = hash.ComputeHash(Encoding.ASCII.GetBytes(value));
            var output = new StringBuilder(raw.Length * 2);
            foreach (byte b in raw) output.Append(b.ToString("x2"));
            return output.ToString();
        }
    }

    void RebuildAndVerify() {
        if (journal.Length > MaximumJournalBytes)
            throw new InvalidDataException("probe journal exceeds bounded size");
        byte[] bytes = new byte[(int)journal.Length];
        journal.Position = 0;
        int received = 0;
        while (received < bytes.Length) {
            int n = journal.Read(bytes, received, bytes.Length - received);
            if (n <= 0) throw new EndOfStreamException("probe journal truncated");
            received += n;
        }
        for (int i = 0; i < bytes.Length; i++)
            if (bytes[i] != 10 && (bytes[i] < 32 || bytes[i] > 126))
                throw new InvalidDataException("probe journal contains invalid byte");
        if (bytes.Length > 0 && bytes[bytes.Length - 1] != 10)
            throw new InvalidDataException("probe journal incomplete final record");
        string raw = Encoding.ASCII.GetString(bytes);
        string[] lines = raw.Split('\n');
        if (lines.Length > MaximumEntries + 1)
            throw new InvalidDataException("probe journal entry limit exceeded");
        for (int i = 0; i < lines.Length - 1; i++) {
            string[] parts = lines[i].Split('|');
            if (parts.Length != 7 || parts[0] != "1"
                    || !Digest(parts[3]) || !Digest(parts[4])
                    || parts[5] != head || !Digest(parts[6]))
                throw new InvalidDataException("probe journal record invalid");
            long nextEpoch;
            if (!Int64.TryParse(parts[1], System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out nextEpoch)
                    || nextEpoch <= 0 || nextEpoch > Int64.MaxValue - 1)
                throw new InvalidDataException("probe epoch invalid");
            string unsigned = String.Join("|", parts, 0, 6);
            if (Sha(unsigned) != parts[6])
                throw new InvalidDataException("probe journal hash chain invalid");
            switch (parts[2]) {
              case "RESERVED":
                if (!(phase == "GENESIS" || phase == "TERMINAL")
                        || nextEpoch != epoch + 1 || !consumed.Add(parts[3]))
                    throw new InvalidDataException("probe journal epoch/nonce replay");
                epoch = nextEpoch; nonce = parts[3]; request = parts[4];
                break;
              case "DISPATCH_INTENT":
                if (phase != "RESERVED" || nextEpoch != epoch
                        || parts[3] != nonce || parts[4] != request)
                    throw new InvalidDataException("probe journal dispatch transition");
                break;
              case "TERMINAL":
                if (phase != "DISPATCH_INTENT" || nextEpoch != epoch
                        || parts[3] != nonce || parts[4] != request)
                    throw new InvalidDataException("probe journal terminal transition");
                break;
              default:
                throw new InvalidDataException("probe journal state invalid");
            }
            phase = parts[2];
            head = parts[6];
        }
        journal.Position = journal.Length;
    }

    void Append(string transition) {
        if (poisoned) throw new InvalidOperationException("probe journal poisoned; HOLD");
        string unsigned = "1|" + epoch.ToString(System.Globalization.CultureInfo.InvariantCulture)
            + "|" + transition + "|" + nonce + "|" + request + "|" + head;
        string line = unsigned + "|" + Sha(unsigned) + "\n";
        byte[] bytes = Encoding.ASCII.GetBytes(line);
        if (journal.Length + bytes.Length > MaximumJournalBytes)
            throw new InvalidOperationException("probe journal full; fail closed");
        try {
            journal.Write(bytes, 0, bytes.Length);
            journal.Flush(true); // an append is durable before the next effect
        } catch {
            poisoned = true; // never retry through an ambiguously persisted tail
            throw;
        }
        head = Sha(unsigned);
        phase = transition;
    }

    internal long Reserve(string requestSha256, string requestNonceSha256) {
        if (disposed) throw new ObjectDisposedException("HostOwnedProbeReservation");
        if (poisoned) throw new InvalidOperationException("probe journal poisoned; HOLD");
        if (!Digest(requestSha256) || !Digest(requestNonceSha256))
            throw new ArgumentException("digest shape invalid");
        if (!(phase == "GENESIS" || phase == "TERMINAL"))
            throw new InvalidOperationException("previous probe not reconciled; HOLD");
        if (consumed.Contains(requestNonceSha256))
            throw new InvalidOperationException("probe request nonce replay; HOLD");
        if (epoch == Int64.MaxValue - 1)
            throw new InvalidOperationException("probe epoch exhausted");
        epoch++;
        nonce = requestNonceSha256; request = requestSha256;
        consumed.Add(nonce);
        Append("RESERVED");
        return epoch;
    }

    internal void MarkDispatchIntent() {
        if (disposed) throw new ObjectDisposedException("HostOwnedProbeReservation");
        if (poisoned) throw new InvalidOperationException("probe journal poisoned; HOLD");
        if (phase != "RESERVED") throw new InvalidOperationException("probe not reserved");
        Append("DISPATCH_INTENT");
    }

    internal void MarkTerminalAfterOwnedProbeExit(JobObjectChildCustody child) {
        if (disposed) throw new ObjectDisposedException("HostOwnedProbeReservation");
        if (poisoned) throw new InvalidOperationException("probe journal poisoned; HOLD");
        if (phase != "DISPATCH_INTENT" || child == null || !child.IsInJob)
            throw new InvalidOperationException("probe terminal custody missing");
        if (!child.CloseJobAndConfirmChildExit())
            throw new InvalidOperationException("probe child exit unconfirmed; HOLD");
        Append("TERMINAL");
    }

    internal string Phase { get { return phase; } }
    internal long Epoch { get { return epoch; } }
    public void Dispose() {
        if (disposed) return;
        disposed = true;
        journal.Flush(true);
        journal.Dispose();
        // An outstanding RESERVED or DISPATCH_INTENT remains unresolved on disk.
    }
}
}
