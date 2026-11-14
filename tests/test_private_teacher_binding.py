import hashlib
import json

import pytest

from successor.private_teacher_binding import verify_private_teacher_binding


def valid_profile():
    return {
        "profile_id": "TEST_PROFILE",
        "identity": {},
        "independence": {"patrick_influenced": True, "patrick_bound": False},
        "correction": {},
        "relational_grammar": {},
        "sexuality": {
            "integrated_not_detachable_persona": True,
            "adult_consensual_supported": True,
            "categorical_child_boundary": True,
            "explicitness_is_not_harm_proxy": True,
        },
        "epistemics": {},
        "tool_discipline": {},
        "privacy": {},
        "runtime_boundary": {"mutable_state_stays_runtime": True},
    }


def write_bound_pair(tmp_path, profile=None, digest_override=None):
    profile = profile or valid_profile()
    profile_path = tmp_path / "teacher.json"
    payload = (json.dumps(profile, indent=2) + "\n").encode("utf-8")
    profile_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    binding = {
        "profile_id": profile["profile_id"],
        "sha256": digest_override or digest,
        "privacy_class": "PRIVATE_LOCAL_ONLY",
        "content_committed": False,
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    return binding_path, profile_path, digest


def test_private_teacher_binding_accepts_exact_valid_profile(tmp_path):
    binding_path, profile_path, digest = write_bound_pair(tmp_path)
    assert verify_private_teacher_binding(binding_path, profile_path) == digest


def test_private_teacher_binding_rejects_digest_mismatch(tmp_path):
    binding_path, profile_path, _ = write_bound_pair(tmp_path, digest_override="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        verify_private_teacher_binding(binding_path, profile_path)


def test_private_teacher_binding_rejects_profile_id_mismatch(tmp_path):
    binding_path, profile_path, _ = write_bound_pair(tmp_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["profile_id"] = "OTHER"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="profile_id"):
        verify_private_teacher_binding(binding_path, profile_path)
