from __future__ import annotations

import math
from statistics import NormalDist


_NORMAL = NormalDist()


def cluster_design_effect(cluster_size: int, icc: float) -> float:
    if cluster_size < 1:
        raise ValueError("cluster_size must be >= 1")
    if not 0 <= icc < 1:
        raise ValueError("icc must be in [0, 1)")
    return 1.0 + (cluster_size - 1) * icc


def effective_n(n: int, *, cluster_size: int, icc: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    return n / cluster_design_effect(cluster_size, icc)


def normal_paired_power(*, n: float, delta: float, discordance: float, alpha: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")
    if abs(delta) > discordance:
        raise ValueError("absolute delta cannot exceed discordance")
    zcrit = _NORMAL.inv_cdf(1 - alpha / 2)
    mu = abs(delta) * math.sqrt(n / discordance)
    upper = 1 - _NORMAL.cdf(zcrit - mu)
    lower = _NORMAL.cdf(-zcrit - mu)
    return max(0.0, min(1.0, upper + lower))


def required_n(*, delta: float, discordance: float, alpha: float, power: float) -> int:
    if delta == 0:
        raise ValueError("delta must be nonzero")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")
    if abs(delta) > discordance:
        raise ValueError("absolute delta cannot exceed discordance")
    zcrit = _NORMAL.inv_cdf(1 - alpha / 2)
    zpower = _NORMAL.inv_cdf(power)
    return math.ceil(discordance * ((zcrit + zpower) / abs(delta)) ** 2)


def planning_matrix(
    *,
    nominal_n: int,
    delta: float,
    alpha: float,
    discordances: tuple[float, ...],
    iccs: tuple[float, ...],
    cluster_size: int,
) -> list[dict]:
    rows = []
    for discordance in discordances:
        for icc in iccs:
            neff = effective_n(nominal_n, cluster_size=cluster_size, icc=icc)
            rows.append(
                {
                    "nominal_n": nominal_n,
                    "cluster_size": cluster_size,
                    "icc": icc,
                    "design_effect": cluster_design_effect(cluster_size, icc),
                    "effective_n": neff,
                    "delta": delta,
                    "discordance": discordance,
                    "alpha": alpha,
                    "approx_power": normal_paired_power(
                        n=neff,
                        delta=delta,
                        discordance=discordance,
                        alpha=alpha,
                    ),
                }
            )
    return rows
