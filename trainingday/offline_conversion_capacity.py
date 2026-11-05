"""Source-only, never-authorizing capacity screen for a Qwen GGUF conversion.

This accepts UNTRUSTED observations.  It does not probe a host, open model
weights, reserve resources, obtain a lease, or launch a process.  Even when
numeric capacity passes, independently authenticated host-owned admission is
still required and this module always reports launch_authorized=False.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any


SCHEMA = "TRAININGDAY_OFFLINE_CONVERSION_CAPACITY_V1"
RAM_MIN_MIB = 12 * 1024
SWAP_MIN_MIB = 1024
DISK_MIN_MIB = 25 * 1024
MAX_INPUT_BYTES = 8192
ALLOWED = frozenset({
    "schema", "host_id", "environment", "observed_utc",
    "host_ram_available_mib", "guest_ram_available_mib",
    "swap_available_mib", "staging_free_mib",
})


class CapacityInputError(ValueError):
    """Untrusted sample is malformed; no launch is permitted."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapacityInputError("duplicate JSON key")
        result[key] = value
    return result


def parse_sample(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_INPUT_BYTES:
        raise CapacityInputError("sample bytes missing or over limit")
    try:
        sample = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                CapacityInputError("nonfinite JSON value")))
    except (ValueError, UnicodeError) as exc:
        raise CapacityInputError("invalid JSON sample") from exc
    if type(sample) is not dict or not set(sample).issubset(ALLOWED):
        raise CapacityInputError("unexpected sample fields")
    return sample


def evaluate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Report numeric readiness only; never interpret a caller's lease claim."""
    if type(sample) is not dict or not set(sample).issubset(ALLOWED):
        raise CapacityInputError("invalid sample object")
    if sample.get("schema") != SCHEMA:
        raise CapacityInputError("unknown sample schema")
    host = sample.get("host_id")
    when = sample.get("observed_utc")
    environment = sample.get("environment")
    if (type(host) is not str or not 1 <= len(host) <= 80
            or type(when) is not str or not when.endswith("Z")
            or not 20 <= len(when) <= 40
            or environment not in ("windows", "linux", "windows_wsl2")):
        raise CapacityInputError("invalid host, timestamp or environment")
    try:
        parsed_time = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CapacityInputError("invalid UTC timestamp") from exc
    if parsed_time.tzinfo != timezone.utc:
        raise CapacityInputError("timestamp must be UTC")
    required = ("host_ram_available_mib", "swap_available_mib", "staging_free_mib")
    for field in required:
        value = sample.get(field)
        if type(value) is not int or not 0 <= value <= 1024 * 1024:
            raise CapacityInputError("capacity field missing or malformed: " + field)
    guest = sample.get("guest_ram_available_mib")
    if environment == "windows_wsl2":
        if type(guest) is not int or not 0 <= guest <= 1024 * 1024:
            raise CapacityInputError("WSL requires separate guest memory observation")
    elif guest is not None:
        raise CapacityInputError("guest memory only permitted for WSL")

    failures = []
    if sample["host_ram_available_mib"] < RAM_MIN_MIB:
        failures.append("HOST_RAM_BELOW_12_GIB")
    if environment == "windows_wsl2" and guest < RAM_MIN_MIB:
        failures.append("WSL_GUEST_RAM_BELOW_12_GIB")
    if sample["swap_available_mib"] < SWAP_MIN_MIB:
        failures.append("SWAP_BELOW_1_GIB")
    if sample["staging_free_mib"] < DISK_MIN_MIB:
        failures.append("STAGING_BELOW_25_GIB")
    # Numeric PASS is still NOT a grant. Host identity, telemetry freshness,
    # owner consent, immutable plan, child custody and lease are unverified.
    return {
        "schema": "TRAININGDAY_OFFLINE_CONVERSION_CAPACITY_RESULT_V1",
        "host_id_claim": host,
        "sample_observed_utc_claim": when,
        "evidence_class": "UNTRUSTED_CALLER_SUPPLIED_SAMPLE",
        "numeric_capacity": "HOLD" if failures else "SCREEN_PASS_ONLY",
        "reasons": failures,
        "launch_authorized": False,
        "owner_lease_verified": False,
        "model_bytes_verified": False,
        "physical_inference_or_conversion_performed": False,
    }


def main() -> int:
    try:
        sample = parse_sample(sys.stdin.buffer.read(MAX_INPUT_BYTES + 1))
        report = evaluate_sample(sample)
    except CapacityInputError as exc:
        report = {"schema": "TRAININGDAY_OFFLINE_CONVERSION_CAPACITY_RESULT_V1",
                  "numeric_capacity": "HOLD", "launch_authorized": False,
                  "error": str(exc)}
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    # Deliberate nonzero exit even when the numeric screen passes.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
