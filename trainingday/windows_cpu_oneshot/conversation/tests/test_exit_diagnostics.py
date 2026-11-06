"""Diagnostic reasoning only; never launches or retries the model."""

import unittest

from q4_conversation_eval.exit_diagnostics import ExitObservation, diagnose_exit


def obs(**overrides):
    row = dict(
        elapsed_ms=82, exit_code=0, stdout_bytes=0, stderr_bytes=0,
        process_created=True, owner_stop_reported=False,
    )
    row.update(overrides)
    return ExitObservation(**row)


class ExitDiagnosticTests(unittest.TestCase):
    def test_empty_fast_zero_exit_is_not_success_or_model_evidence(self):
        result = diagnose_exit(obs())
        self.assertEqual(result["classification"], "EARLY_EMPTY_ZERO_EXIT")
        self.assertFalse(result["physical_inference_proved"])
        self.assertIn("unconfirmed", result["evidence_state"].lower())
        self.assertEqual(result["normalized_exit_hex"], "0x00000000")

    def test_ntstatus_signed_and_unsigned_are_equivalent(self):
        signatures = (
            (0xC0000135, "DLL_LOAD_MISSING"),
            (0xC000007B, "INVALID_IMAGE_FORMAT"),
            (0xC0000142, "DLL_INITIALIZATION_FAILED"),
            (0xC000001D, "ILLEGAL_INSTRUCTION"),
            (0xC0000005, "ACCESS_VIOLATION"),
            (0xC0000006, "IN_PAGE_ERROR"),
            (0xC0000017, "VIRTUAL_MEMORY_EXHAUSTED"),
            (0xC000013A, "CONTROL_C_EXIT"),
        )
        for code, expected in signatures:
            with self.subTest(status=f"0x{code:08X}"):
                unsigned = diagnose_exit(obs(exit_code=code))
                signed = diagnose_exit(obs(exit_code=code - 2**32))
                self.assertEqual(unsigned["classification"], expected)
                self.assertEqual(signed["classification"], expected)
                self.assertEqual(unsigned["normalized_exit_hex"], signed["normalized_exit_hex"])
                self.assertFalse(unsigned["physical_inference_proved"])

    def test_broker_stop_and_create_failure_never_laundered(self):
        stopped = diagnose_exit(obs(exit_code=0, owner_stop_reported=True))
        self.assertEqual(stopped["classification"], "OWNER_STOP_REPORTED")
        missing = diagnose_exit(obs(process_created=False, exit_code=None))
        self.assertEqual(missing["classification"], "PROCESS_NOT_CREATED")
        self.assertFalse(missing["physical_inference_proved"])

    def test_zero_exit_with_output_does_not_attest_model(self):
        actual = diagnose_exit(obs(elapsed_ms=60_000, stdout_bytes=128))
        self.assertEqual(actual["classification"], "OUTPUT_PRESENT_UNVERIFIED")
        self.assertFalse(actual["physical_inference_proved"])
        diagnostic = diagnose_exit(obs(elapsed_ms=60_000, stderr_bytes=128))
        self.assertEqual(diagnostic["classification"], "EXIT_ZERO_STDERR_ONLY")

    def test_invalid_record_is_rejected(self):
        for change in (
            {"elapsed_ms": -1},
            {"stdout_bytes": -1},
            {"stderr_bytes": -1},
            {"exit_code": "0"},
            {"exit_code": -2**31 - 1},
            {"exit_code": 2**32},
            {"owner_stop_reported": "no"},
            {"process_created": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnose_exit(obs(**change))

    def test_no_exception_or_raw_response_contents_in_report(self):
        for report in (
            diagnose_exit(obs(exit_code=0xC000001D)),
            diagnose_exit(obs(exit_code=0)),
        ):
            self.assertFalse(any(k in report for k in (
                "stdout", "stderr", "prompt", "model_response", "command_line",
            )))
            self.assertIn("next_read_only_discriminator", report)


if __name__ == "__main__":
    unittest.main()
