"""Strict NAS TLS *transport* probe; never a TrainingDay grant issuer.

This probe requires a separately chosen HTTPS origin.  A successful TLS
handshake authenticates a certificate against this host's trusted CA store
and the requested hostname; it does not authenticate the NAS training-rights
issuer, assign a GPU lease, or authorize model execution.  It intentionally
has no custom trust-bundle, insecure mode, client key, or CLI switch.

No network call occurs by importing this module or parsing an origin.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import re
import socket
import ssl
from typing import NamedTuple
from urllib.parse import urlsplit


_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


class NasTlsProbeError(ValueError):
    """Malformed or ambiguous NAS TLS diagnostic target."""


class TlsOrigin(NamedTuple):
    hostname: str
    port: int


def parse_tls_origin(value: str) -> TlsOrigin:
    """Accept only an unambiguous HTTPS origin, without embedded secrets.

    DNS names, a single-label host, and IP SAN candidates are accepted.  This
    does *not* assert that a NAS certificate contains any such identity.
    """
    if (type(value) is not str or not 9 <= len(value) <= 512
            or not value.isascii() or any(c.isspace() or ord(c) < 0x20 for c in value)
            or "\\" in value or "%" in value or "@" in value
            or "?" in value or "#" in value):
        raise NasTlsProbeError("invalid or ambiguous NAS TLS origin")
    try:
        parts = urlsplit(value)
        if (parts.scheme != "https" or not parts.netloc or parts.path not in ("", "/")
                or parts.query or parts.fragment or parts.username is not None
                or parts.password is not None):
            raise NasTlsProbeError("NAS TLS probe requires credential-free HTTPS origin")
        hostname = parts.hostname
        port = parts.port if parts.port is not None else 443
    except ValueError as exc:
        raise NasTlsProbeError("malformed NAS TLS hostname or port") from exc
    if (not hostname or not 1 <= port <= 65535 or hostname.endswith(".")
            or "#" in hostname or "?" in hostname):
        raise NasTlsProbeError("invalid NAS TLS hostname or port")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        labels = hostname.split(".")
        if (len(hostname) > 253 or any(_DNS_LABEL.fullmatch(label) is None
                                      for label in labels)):
            raise NasTlsProbeError("invalid DNS hostname")
    return TlsOrigin(hostname.lower(), port)


def _reject(reason: str, *, verify_code: int | None = None,
            verify_message: str | None = None) -> dict:
    return {
        "status": "TLS_REJECTED",
        "failure": reason,
        "verification_code": verify_code,
        "verification_message": verify_message,
        "certificate_hostname_verified": False,
        "signed_nas_grant_verified": False,
        "physical_training_authorized": False,
    }


def probe_verified_tls_transport(origin: str, *, timeout_seconds: float = 3.0) -> dict:
    """Attempt one strict certificate-validated TLS handshake; send no HTTP.

    Call only from an authorized, bounded host/network diagnostic workflow.
    The result is transport evidence *at this moment*, not source-rights,
    trusted NAS issuer enrollment, witnessed lease, or launch authority.
    Failure never retries with certificate or hostname validation disabled.
    """
    target = parse_tls_origin(origin)
    if (type(timeout_seconds) not in (float, int) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 10):
        raise NasTlsProbeError("invalid bounded TLS timeout")

    # Using PROTOCOL_TLS_CLIENT avoids SSLKEYLOGFILE environment-driven key
    # logging that ssl.create_default_context() can enable by default.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs(purpose=ssl.Purpose.SERVER_AUTH)
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        return _reject("TRUST_CONFIGURATION_INCOMPLETE")
    try:
        with socket.create_connection((target.hostname, target.port),
                                      timeout=timeout_seconds) as tcp:
            with context.wrap_socket(tcp, server_hostname=target.hostname) as tls:
                leaf_der = tls.getpeercert(binary_form=True)
                if type(leaf_der) is not bytes or not leaf_der:
                    return _reject("VERIFIED_PEER_CERTIFICATE_MISSING")
                return {
                    "status": "TLS_TRANSPORT_ONLY_NOT_NAS_AUTHORITY",
                    "hostname": target.hostname,
                    "port": target.port,
                    "tls_version": tls.version(),
                    "peer_leaf_sha256": hashlib.sha256(leaf_der).hexdigest(),
                    "certificate_hostname_verified": True,
                    "signed_nas_grant_verified": False,
                    "physical_training_authorized": False,
                }
    except ssl.SSLCertVerificationError as exc:
        return _reject("CERTIFICATE_VALIDATION_FAILED",
                       verify_code=getattr(exc, "verify_code", None),
                       verify_message=getattr(exc, "verify_message", None))
    except (ssl.SSLError, OSError) as exc:
        return _reject(type(exc).__name__)
