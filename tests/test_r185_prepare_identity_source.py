"""Source-only PREPARE_V2 identity-custody checks; Windows execution is separate."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "trainingday/windows_host_broker/PrepareV2PeerIdentity.cs"
INSTALLED_BOOTSTRAP = ROOT / "trainingday/windows_host_broker/TrainingDayHostBroker.cs"


class PrepareV2IdentitySourceTests(unittest.TestCase):
    def source(self):
        return SOURCE.read_text(encoding="utf-8")

    def test_peer_pid_and_image_are_from_windows_not_claimed_request(self):
        source = self.source()
        for required in (
            "GetNamedPipeClientProcessId", "OpenProcess(", "GetProcessTimes(",
            "QueryFullProcessImageNameW", "GetNamedPipeClientProcessId(pipe.SafePipeHandle",
            "WindowsIdentity.GetCurrent(true)", "pipe.RunAsClient(",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertIn("PROCESS_QUERY_LIMITED_INFORMATION", source)
        self.assertIn("SYNCHRONIZE", source)
        self.assertIn("observedPid != confirmedPid", source)
        self.assertIn("creationFileTime != confirmedCreationFileTime", source)

    def test_low_impersonation_and_dead_process_fail_before_observation(self):
        source = self.source()
        self.assertIn("TokenImpersonationLevel.Impersonation", source)
        self.assertIn("TokenImpersonationLevel.Delegation", source)
        self.assertIn("new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator)", source)
        self.assertIn("new UnauthorizedAccessException", source)
        self.assertIn("WaitForSingleObject", source)
        self.assertIn("PROCESS_EXITED_OR_UNOBSERVABLE", source)
        self.assertIn("CloseHandle(processHandle)", source)

    def test_spoofed_claim_replay_or_matching_claim_never_authorizes(self):
        source = self.source()
        for name in ("claimedPid", "claimedSid", "claimedImagePath", "claimedCreationFileTime"):
            self.assertIn(name, source)
        self.assertIn("HOLD_PEER_MISMATCH", source)
        self.assertIn("HOLD_NO_DURABLE_REPLAY_WITNESS", source)
        self.assertIn('public const string State = "HOLD"', source)
        self.assertIn("public bool AuthorizesTraining { get { return false; } }", source)
        self.assertNotRegex(source, re.compile(r'(?i)\b(allow_launch|dispatch_training|gpu_reserve)\s*\('))

    def test_existing_service_is_unchanged_and_deny_only(self):
        bootstrap = INSTALLED_BOOTSTRAP.read_text(encoding="utf-8")
        self.assertNotIn("PrepareV2PeerIdentity", bootstrap)
        self.assertIn('"authorized_launches\\\":0', bootstrap)
        self.assertIn('"state\\\":\\\"HOLD', bootstrap)


if __name__ == "__main__":
    unittest.main()
