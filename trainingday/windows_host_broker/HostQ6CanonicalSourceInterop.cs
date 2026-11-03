// SOURCE-ONLY cross-language HostLaunch/Q6 CAS canonicalization witness.
// No Ed25519, trusted issuer, NAS CAS, clock, GPU, or launch authorization.
// This type is intentionally NOT referenced by the installed broker service.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Web.Script.Serialization;

namespace Vera.TrainingDay.HostBroker {
internal sealed class HostQ6SourceDisposition {
    internal readonly string Status;
    internal readonly string HostLaunchSha256;
    internal readonly string Q6CasSha256;
    // Deliberately no implicit boolean conversion or positive disposition.
    internal bool AuthorizesTraining { get { return false; } }
    internal HostQ6SourceDisposition(string host, string q6) {
        Status = "SOURCE_ONLY_NO_AUTHORITY";
        HostLaunchSha256 = host;
        Q6CasSha256 = q6;
    }
}

internal static class HostQ6CanonicalSourceInterop {
    const string QwenRevision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

    static Dictionary<string, object> Object(object value, params string[] keys) {
        var obj = value as Dictionary<string, object>;
        if (obj == null || obj.Count != keys.Length)
            throw new FormatException("source object shape invalid");
        foreach (string key in keys)
            if (!obj.ContainsKey(key)) throw new FormatException("source field missing: " + key);
        return obj;
    }

    static string Text(Dictionary<string, object> obj, string key) {
        string value = obj[key] as string;
        if (value == null) throw new FormatException("source text invalid: " + key);
        return value;
    }

    static long Number(Dictionary<string, object> obj, string key) {
        if (obj[key] is int) return (int)obj[key];
        if (obj[key] is long) return (long)obj[key];
        throw new FormatException("source integer invalid: " + key);
    }

    static bool Flag(Dictionary<string, object> obj, string key) {
        if (!(obj[key] is bool)) throw new FormatException("source boolean invalid: " + key);
        return (bool)obj[key];
    }

    static bool Hex(string value, int length) {
        if (value.Length != length) return false;
        foreach (char c in value)
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        return true;
    }

    static string QuoteAscii(string value) {
        var sb = new StringBuilder("\"");
        foreach (char c in value) {
            if (c > 127) throw new FormatException("non-ASCII subject unsupported");
            switch (c) {
              case '"': sb.Append("\\\""); break;
              case '\\': sb.Append("\\\\"); break;
              case '\b': sb.Append("\\b"); break;
              case '\f': sb.Append("\\f"); break;
              case '\n': sb.Append("\\n"); break;
              case '\r': sb.Append("\\r"); break;
              case '\t': sb.Append("\\t"); break;
              default:
                if (c < 32) sb.Append("\\u" + ((int)c).ToString("x4", CultureInfo.InvariantCulture));
                else sb.Append(c);
                break;
            }
        }
        return sb.Append('"').ToString();
    }

    // Python: json.dumps(obj, sort_keys=True, separators=(",", ":"),
    // ensure_ascii=True, allow_nan=False). This native fixture has only ASCII
    // keys/strings, booleans, and integral numbers. Unsupported types reject.
    static string Canonical(object value) {
        var dictionary = value as Dictionary<string, object>;
        if (dictionary != null) {
            var keys = new List<string>(dictionary.Keys);
            keys.Sort(StringComparer.Ordinal);
            var fields = new List<string>();
            foreach (string key in keys) fields.Add(QuoteAscii(key) + ":" + Canonical(dictionary[key]));
            return "{" + String.Join(",", fields.ToArray()) + "}";
        }
        var array = value as object[];
        if (array != null) {
            var fields = new List<string>();
            foreach (object member in array) fields.Add(Canonical(member));
            return "[" + String.Join(",", fields.ToArray()) + "]";
        }
        if (value is string) return QuoteAscii((string)value);
        if (value is bool) return (bool)value ? "true" : "false";
        if (value is int) return ((int)value).ToString(CultureInfo.InvariantCulture);
        if (value is long) return ((long)value).ToString(CultureInfo.InvariantCulture);
        if (value == null) return "null";
        throw new FormatException("noncanonical source JSON value");
    }

    static string Sha(string ascii) {
        using (var hash = SHA256.Create()) {
            var bytes = hash.ComputeHash(Encoding.ASCII.GetBytes(ascii));
            var sb = new StringBuilder(64);
            foreach (byte b in bytes) sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
            return sb.ToString();
        }
    }

    // This narrow helper enables an adversarial test to rehash a tampered
    // fixture, proving subject cross-binding is checked beyond checksum shape.
    internal static string SourceDigestForNativeTestOnly(string json, bool host) {
        var obj = new JavaScriptSerializer().DeserializeObject(json) as Dictionary<string, object>;
        if (obj == null || !obj.ContainsKey(host ? "host_launch" : "q6_cas_request"))
            throw new FormatException("missing native source subject");
        return Sha(Canonical(obj[host ? "host_launch" : "q6_cas_request"]) + (host ? "" : "\n"));
    }

    // Native TEST ONLY: caller-controlled fixture is never authority. This
    // checks deterministic source-byte parity and subject cross-binding only.
    internal static HostQ6SourceDisposition AssessFixtureForNativeTestOnly(string json) {
        if (json == null || json.Length > 65536)
            throw new FormatException("source fixture length invalid");
        var serializer = new JavaScriptSerializer();
        serializer.MaxJsonLength = 65536;
        var root = Object(serializer.DeserializeObject(json), "schema", "source",
            "golden_request_digests", "host_launch", "q6_cas_request", "authority");
        if (Text(root, "schema") != "TRAININGDAY_Q6_HOST_SOURCE_VECTOR_V1")
            throw new FormatException("source fixture schema changed");
        var golden = Object(root["golden_request_digests"], "host_launch_sha256", "q6_cas_sha256");
        var host = Object(root["host_launch"], "schema", "source", "rights", "plan", "lease");
        var source = Object(host["source"], "source_commit_sha1", "source_tree_sha1",
            "corpus_manifest_sha256", "corpus_producer_commit_sha1", "training_input_sha256",
            "base_binding_sha256", "base_model_id", "base_revision_sha1");
        var rights = Object(host["rights"], "decision_id", "decision_sha256",
            "covered_manifest_sha256", "declared_disposition", "purported_owner");
        var plan = Object(host["plan"], "run_id", "protocol", "mode", "optimizer_steps",
            "selected_cases", "seed", "learning_rate_ppb", "expected_zero_digest_sha256");
        var lease = Object(host["lease"], "host_instance", "caller_principal", "gpu_lease_id",
            "gpu_lease_epoch", "request_nonce", "issued_at_unix", "expires_at_unix",
            "expected_witness_epoch", "expected_witness_head_sha256");
        var cas = Object(root["q6_cas_request"], "schema", "operation", "nonce", "expected", "new");
        var expected = Object(cas["expected"], "commit_index", "head_sha256");
        var next = Object(cas["new"], "commit_index", "head_sha256");
        var authority = Object(root["authority"], "source_only_unqualified",
            "authorizes_training", "host_gpu_lease_issued", "independent_q6_witness_installed",
            "model_execution_authenticated");

        if (Text(host, "schema") != "TRAININGDAY_HOST_LAUNCH_DECLARATION_V1"
                || Text(source, "base_model_id") != "Qwen/Qwen3.5-4B"
                || Text(source, "base_revision_sha1") != QwenRevision
                || Text(rights, "declared_disposition") != "HOLD_CLAIM"
                || Text(rights, "covered_manifest_sha256") != Text(source, "corpus_manifest_sha256")
                || Text(cas, "schema") != "labyrinth.q6-independent-witness.source-r121.v1"
                || Text(cas, "operation") != "CAS"
                || Number(lease, "expected_witness_epoch") != Number(expected, "commit_index")
                || Text(lease, "expected_witness_head_sha256") != Text(expected, "head_sha256")
                || Number(next, "commit_index") != Number(expected, "commit_index") + 1
                || Text(next, "head_sha256") == Text(expected, "head_sha256")
                || Number(lease, "gpu_lease_epoch") <= 0
                || Number(lease, "issued_at_unix") >= Number(lease, "expires_at_unix")
                || !Hex(Text(lease, "request_nonce"), 64)
                || !Hex(Text(cas, "nonce"), 32)
                || !Hex(Text(expected, "head_sha256"), 64)
                || !Hex(Text(next, "head_sha256"), 64)
                || !Flag(authority, "source_only_unqualified")
                || Flag(authority, "authorizes_training")
                || Flag(authority, "host_gpu_lease_issued")
                || Flag(authority, "independent_q6_witness_installed")
                || Flag(authority, "model_execution_authenticated"))
            throw new FormatException("host/Q6 source subject or authority mismatch");
        // Use the remaining known fields in the canonical subject, not as
        // caller-supplied authorization indicators.
        if (Text(plan, "run_id").Length == 0 || Text(rights, "decision_id").Length == 0)
            throw new FormatException("empty immutable plan or rights declaration");
        string hostSha = Sha(Canonical(host)); // HostLaunch V1 has NO trailing LF.
        string casSha = Sha(Canonical(cas) + "\n"); // Q6 R121 includes one LF.
        if (hostSha != Text(golden, "host_launch_sha256")
                || casSha != Text(golden, "q6_cas_sha256"))
            throw new FormatException("Python/native canonical source digest mismatch");
        return new HostQ6SourceDisposition(hostSha, casSha);
    }
}
}
