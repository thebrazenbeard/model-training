from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .teacher_profile import validate_teacher_profile


def verify_private_teacher_binding(binding_path, profile_path) -> str:
    binding_path = Path(binding_path)
    profile_path = Path(profile_path)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    payload = profile_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    if binding.get("privacy_class") != "PRIVATE_LOCAL_ONLY":
        raise ValueError("teacher profile privacy class must be PRIVATE_LOCAL_ONLY")
    if binding.get("content_committed") is not False:
        raise ValueError("private teacher profile content must not be committed")
    if binding.get("sha256") != digest:
        raise ValueError("private teacher profile digest mismatch")

    profile = json.loads(payload.decode("utf-8"))
    if binding.get("profile_id") != profile.get("profile_id"):
        raise ValueError("private teacher profile profile_id mismatch")
    validate_teacher_profile(profile)
    return digest
