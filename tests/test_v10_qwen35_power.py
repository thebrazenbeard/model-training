from successor.experiments.power_v10_qwen35 import (
    cluster_design_effect,
    effective_n,
    normal_paired_power,
    required_n,
)


def test_cluster_design_effect():
    assert cluster_design_effect(10, 0.0) == 1.0
    assert cluster_design_effect(10, 0.1) == 1.9


def test_effective_n_decreases_with_clustering():
    assert effective_n(500, cluster_size=10, icc=0.1) < 500


def test_required_n_increases_with_discordance():
    assert required_n(delta=0.05, discordance=0.3, alpha=0.01, power=0.8) > required_n(
        delta=0.05, discordance=0.1, alpha=0.01, power=0.8
    )


def test_10k_overall_has_more_power_than_500_dimension():
    overall = normal_paired_power(n=10000, delta=0.05, discordance=0.2, alpha=0.01)
    dimension = normal_paired_power(n=500, delta=0.05, discordance=0.2, alpha=0.01)
    assert overall > 0.99
    assert dimension < 0.8
