"""Fail-closed boundary for physical TRAININGDAY work.

This module is deliberately *not* a grant verifier. Until an independently
authenticated, host-owned launcher can bind caller/process identity, an
exclusive GPU lease, route ownership, data rights and an immutable run plan,
no Python caller has authority to start a GPU process or mutate checkpoints.

No CLI flag, environment variable, source receipt, locally supplied callback,
or self-signed token may turn this HOLD into an authorization.
"""

from __future__ import annotations

from typing import NoReturn


class PhysicalLaunchAuthorityUnavailable(RuntimeError):
    """No host-owned, independently authenticated launch authority is wired."""


def require_physical_launch_authority(*, operation: str) -> NoReturn:
    """Hold all physical R16/R18 invocations before host or GPU effects.

    An eventual implementation needs an authenticated host broker holding
    exclusive lease and process identity; this source-only placeholder never
    grants authority. Add no positive result until physically qualified.
    """
    raise PhysicalLaunchAuthorityUnavailable(
        f"TRAININGDAY physical {operation} HARD HOLD: "
        "host-owned lease/route/data-rights authority not installed")
