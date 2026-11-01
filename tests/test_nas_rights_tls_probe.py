"""CPU-only contract tests for strict TLS transport diagnostics.

These tests neither contact a NAS nor prove a training-authority grant.
"""

import ssl
import unittest
from unittest.mock import MagicMock, patch

from trainingday.nas_rights_tls_probe import (
    NasTlsProbeError,
    parse_tls_origin,
    probe_verified_tls_transport,
)


class NasRightsTlsProbeTests(unittest.TestCase):
    def test_origin_parser_does_not_accept_downgrade_or_embedded_credentials(self):
        for value in (
            "http://nas.example:9443",
            "https://someone:secret@nas.example:9443",
            "https://nas.example:9443/path",
            "https://nas.example:9443/?token=secret",
            "https://nas.example:9443?",
            "https://nas.example:9443/#fragment",
            "https://nas.example:9443#",
            "https://nas.example:99999",
            "https://nas.example:0",
            "https://nas.example:9443\\@evil.test",
            "https://nas.example%2Etest:9443",
            "https://nas.example:9443 ",
            "https://nas.example:9443/../../",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(NasTlsProbeError):
                parse_tls_origin(value)

    def test_normalized_host_and_ip_are_not_authorization(self):
        host = parse_tls_origin("https://TheSimsVault:9443")
        self.assertEqual((host.hostname, host.port), ("thesimsvault", 9443))
        ip = parse_tls_origin("https://192.0.2.1:9443/")
        self.assertEqual((ip.hostname, ip.port), ("192.0.2.1", 9443))

    def test_tls_client_context_enforces_certificate_and_hostname(self):
        real_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.assertEqual(real_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(real_context.check_hostname)
        self.assertIsNone(real_context.keylog_filename)

    def test_mocked_success_is_labeled_transport_only(self):
        context = MagicMock()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        tls = context.wrap_socket.return_value.__enter__.return_value
        tls.version.return_value = "TLSv1.3"
        tls.getpeercert.return_value = b"synthetic DER"  # synthetic, not a trust proof
        with patch("trainingday.nas_rights_tls_probe.ssl.SSLContext", return_value=context), \
             patch("trainingday.nas_rights_tls_probe.socket.create_connection") as connect:
            result = probe_verified_tls_transport("https://nas.example:9443")
        self.assertEqual(result["status"], "TLS_TRANSPORT_ONLY_NOT_NAS_AUTHORITY")
        self.assertFalse(result["signed_nas_grant_verified"])
        self.assertFalse(result["physical_training_authorized"])
        self.assertEqual(connect.call_args.args[0], ("nas.example", 9443))
        self.assertEqual(context.wrap_socket.call_args.kwargs["server_hostname"], "nas.example")
        context.load_default_certs.assert_called_once_with(purpose=ssl.Purpose.SERVER_AUTH)

    def test_failed_validation_stays_denied_with_no_unverified_retry(self):
        context = MagicMock()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        context.wrap_socket.side_effect = ssl.SSLCertVerificationError(
            "certificate verify failed: unable to get local issuer certificate")
        with patch("trainingday.nas_rights_tls_probe.ssl.SSLContext", return_value=context), \
             patch("trainingday.nas_rights_tls_probe.socket.create_connection"):
            result = probe_verified_tls_transport("https://nas.example:9443")
        self.assertEqual(result["status"], "TLS_REJECTED")
        self.assertFalse(result["certificate_hostname_verified"])
        self.assertFalse(result["signed_nas_grant_verified"])


if __name__ == "__main__":
    unittest.main()
