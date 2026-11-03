// Windows-native source fixture interoperability, NEVER a positive lease.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;
using Vera.TrainingDay.HostBroker;

internal static class HostQ6SourceInteropNativeTests {
    static int assertions;
    static void Require(bool value, string label) {
        if (!value) throw new Exception("native source regression: " + label);
        assertions++;
    }
    static void Reject(string value, string label) {
        bool rejected = false;
        try { HostQ6CanonicalSourceInterop.AssessFixtureForNativeTestOnly(value); }
        catch (FormatException) { rejected = true; }
        Require(rejected, label);
    }
    static void RejectWire(string value, string label) {
        bool rejected = false;
        try { UniqueJsonMemberGuard.RequireUniqueKeys(value); }
        catch (FormatException) { rejected = true; }
        Require(rejected, label);
    }
    static Dictionary<string, object> Obj(object v) {
        return (Dictionary<string, object>)v;
    }
    static string Rehash(Dictionary<string, object> root) {
        var serializer = new JavaScriptSerializer();
        string raw = serializer.Serialize(root);
        var golden = Obj(root["golden_request_digests"]);
        golden["host_launch_sha256"] = HostQ6CanonicalSourceInterop.SourceDigestForNativeTestOnly(raw, true);
        golden["q6_cas_sha256"] = HostQ6CanonicalSourceInterop.SourceDigestForNativeTestOnly(raw, false);
        return serializer.Serialize(root);
    }
    static Dictionary<string, object> Parse(string source) {
        return Obj(new JavaScriptSerializer().DeserializeObject(source));
    }
    static void Main(string[] argv) {
        if (argv.Length != 1 || !File.Exists(argv[0]))
            throw new ArgumentException("fixture file required");
        string raw = File.ReadAllText(argv[0]);
        var verified = HostQ6CanonicalSourceInterop.AssessFixtureForNativeTestOnly(raw);
        Require(verified.Status == "SOURCE_ONLY_NO_AUTHORITY", "fixture always HOLD");
        Require(!verified.AuthorizesTraining, "never an executable authority");
        Require(verified.HostLaunchSha256 ==
            "be5f84f4711c9bb4eb3b4ff86b9c7e9d4b75972beb69c633bcf98ca982e66f5b",
            "Python HostLaunch digest parity");
        Require(verified.Q6CasSha256 ==
            "23fbdfba43880bd6b43b5e752805d80dd3ecbe20dd7e5871f2d7a6925abd2370",
            "Python Q6 CAS digest parity");
        Require(!HostQ6CanonicalSourceInterop.AssessFixtureForNativeTestOnly(raw).AuthorizesTraining,
            "repeated source fixture never proves anti-replay");

        // Duplicate key semantics must be rejected BEFORE deserialization to
        // Dictionary can collapse conflicting/wire-ambiguous authority fields.
        Reject(raw.Replace("\"authority\": {", "\"authority\": {}, \"authority\": {"),
            "duplicate top-level authority object");
        Reject(raw.Replace("\"authority\": {", "\"auth\\u006frity\": {}, \"authority\": {"),
            "escaped spelling aliases authority object");
        Reject(raw.Replace("\"expected_witness_epoch\": 12,",
            "\"expected_witness_epoch\": 12, \"expected_witness_epoch\": 13,"),
            "duplicate host lease witness epoch");
        Reject(raw.Replace("\"expected\": {\"commit_index\": 12,",
            "\"expected\": {\"commit_index\": 12, \"commit_index\": 12,"),
            "nested duplicate Q6 expected CAS key");
        RejectWire("{\"a\":[{\"k\":1,\"k\":2}]}",
            "duplicate nested in array rejected");
        RejectWire("{\"x\":1,\"\\u0078\":1}",
            "escaped and literal duplicate key rejected");
        UniqueJsonMemberGuard.RequireUniqueKeys("{\"expected\": {\"commit_index\": 12},"
            + "\"new\": {\"commit_index\": 13}}");
        Require(true, "same key in distinct objects remains valid");
        RejectWire("{\"safe\":1,\"safe\":2}", "plain duplicate stand-alone rejected");
        RejectWire("{\"a\":1} trailing", "trailing JSON text rejected");

        // An actual UTF-8 wire bound, not merely a UTF-16 character bound.
        // This ignored root source field is outside both canonical SHA inputs;
        // only the pre-deserialization size guard can reject the oversized JSON.
        const string originalCommit = "752a3e80df9be868926d18c574e4b9739e6f7581";
        Require(raw.Contains(originalCommit), "size falsifier has expected source field");
        string utf8Oversized = raw.Replace(originalCommit, new string('\u00e9', 34000));
        Require(utf8Oversized.Length < 65536
                && Encoding.UTF8.GetByteCount(utf8Oversized) > 65536,
                "non-ASCII source is below char cap but above byte cap");
        Reject(utf8Oversized, "UTF-8 wire above 64 KiB rejected");
        RejectWire("{\"x\":\"" + '\uD800' + "\"}",
            "unpaired UTF-16 high surrogate rejected before decode");
        string emoji = Char.ConvertFromUtf32(0x1F600);
        UniqueJsonMemberGuard.RequireUniqueKeys("{\"x\":\"" + emoji + "\"}");
        Require(true, "valid surrogate pair allowed under byte cap");
        RejectWire("{\"\\uD83D\\uDE00\":1,\"" + emoji + "\":2}",
            "escaped and literal surrogate pair duplicate key rejected");

        Reject(raw.Replace("HOLD_CLAIM", "ALLOW"), "caller rights claim mutation");
        Reject(raw.Replace("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "0000000000000000000000000000000000000000"), "base revision mismatch");
        Reject(raw.Replace("\"authorizes_training\": false", "\"authorizes_training\": true"),
            "caller positive flag is not authority");

        // Rehash both source subjects to prevent a trivial hash-mismatch test
        // from disguising an absent host-to-Q6 epoch/head binding check.
        var epochRoot = Parse(raw);
        var lease = Obj(Obj(epochRoot["host_launch"])["lease"]);
        lease["expected_witness_epoch"] = 11;
        Reject(Rehash(epochRoot), "Q6 epoch splice despite fresh source digests");

        var headRoot = Parse(raw);
        var expected = Obj(Obj(headRoot["q6_cas_request"])["expected"]);
        expected["head_sha256"] = new string('4', 64);
        Reject(Rehash(headRoot), "Q6 expected head splice despite fresh digest");

        var casRoot = Parse(raw);
        var next = Obj(Obj(casRoot["q6_cas_request"])["new"]);
        next["commit_index"] = 14;
        Reject(Rehash(casRoot), "Q6 noncontiguous CAS despite fresh digest");

        var signerRoot = Parse(raw);
        Obj(signerRoot["host_launch"])["caller_public_key"] = "attacker-supplied";
        Reject(Rehash(signerRoot), "extra caller-provided key rejected");

        var privilegeRoot = Parse(raw);
        Obj(privilegeRoot["authority"])["host_gpu_lease_issued"] = true;
        Reject(Rehash(privilegeRoot), "caller-produced lease marker rejected");

        // A coordinated, fully self-consistent caller rewrite can pass
        // STRUCTURAL checks, but still obtains only a literal HOLD. There is
        // no trusted signer, external CAS readback, or durable replay state.
        var selfRoot = Parse(raw);
        var selfLease = Obj(Obj(selfRoot["host_launch"])["lease"]);
        var selfCas = Obj(selfRoot["q6_cas_request"]);
        Obj(selfCas["expected"])["commit_index"] = 13;
        Obj(selfCas["new"])["commit_index"] = 14;
        selfLease["expected_witness_epoch"] = 13;
        Require(!HostQ6CanonicalSourceInterop.AssessFixtureForNativeTestOnly(
                    Rehash(selfRoot)).AuthorizesTraining,
                "self-consistent forged subjects still never authorize");

        Console.WriteLine("NATIVE_HOST_Q6_SOURCE_INTEROP_HOLD_PASS assertions=" + assertions);
    }
}
