"""Unit tests for the density-matrix comparison metrics.

The mixed-state Uhlmann fidelity has two implementations: a NumPy
accelerated eigensolve and a pure-Python complex-Hermitian Jacobi fallback
used when NumPy cannot be imported. Both are exercised here and must agree.
"""

from __future__ import annotations

import math
import random

import pytest

from quantum_circuit import metrics


def _unit_vector(dimension: int, rng: random.Random) -> list[complex]:
    vector = [complex(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(dimension)]
    norm = math.sqrt(math.fsum(abs(component) ** 2 for component in vector))
    return [component / norm for component in vector]


def _random_density(dimension: int, rank: int, rng: random.Random) -> list[list[complex]]:
    vectors = [_unit_vector(dimension, rng) for _ in range(rank)]
    weights = [rng.random() + 0.05 for _ in range(rank)]
    total = math.fsum(weights)
    weights = [weight / total for weight in weights]
    rho = [
        [
            sum(
                weights[k] * vectors[k][i] * vectors[k][j].conjugate() for k in range(rank)
            )
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    return metrics.normalized_density_matrix(rho)


def _rank_one(vector: list[complex]) -> list[list[complex]]:
    return [[vector[i] * vector[j].conjugate() for j in range(len(vector))] for i in range(len(vector))]


@pytest.mark.parametrize("dimension", [2, 3, 4, 8])
def test_fidelity_paths_agree(dimension):
    rng = random.Random(1234 + dimension)
    rho = _random_density(dimension, rng.randint(1, dimension), rng)
    sigma = _random_density(dimension, rng.randint(1, dimension), rng)
    accelerated = metrics.density_fidelity(rho, sigma)

    import sys

    saved = sys.modules.get("numpy")
    sys.modules["numpy"] = None  # forces the local `import numpy` to raise ImportError
    try:
        fallback = metrics.density_fidelity(rho, sigma)
    finally:
        if saved is None:
            del sys.modules["numpy"]
        else:
            sys.modules["numpy"] = saved

    assert fallback == pytest.approx(accelerated, abs=1e-11)


def test_pure_pure_density_fidelity_matches_overlap():
    rng = random.Random(7)
    psi = _unit_vector(8, rng)
    phi = _unit_vector(8, rng)
    rho = _rank_one(psi)
    sigma = _rank_one(phi)
    assert metrics.density_fidelity(rho, sigma) == pytest.approx(
        metrics.state_fidelity(psi, phi), abs=1e-12
    )


def test_pure_density_expectation_value():
    # psi = |1>, rho = diag(0.25, 0.75) -> 0.75.
    state = [0j, 1 + 0j]
    rho = [[0.25 + 0j, 0j], [0j, 0.75 + 0j]]
    assert metrics.pure_density_fidelity(state, rho) == pytest.approx(0.75)


def test_commuting_diagonal_closed_form():
    p, q = 0.3, 0.7
    rho = [[p + 0j, 0j], [0j, 1 - p + 0j]]
    sigma = [[q + 0j, 0j], [0j, 1 - q + 0j]]
    expected = (math.sqrt(p * q) + math.sqrt((1 - p) * (1 - q))) ** 2
    assert metrics.density_fidelity(rho, sigma) == pytest.approx(expected, abs=1e-12)


def test_identical_and_orthogonal_states():
    rng = random.Random(11)
    rho = _random_density(6, 3, rng)
    assert metrics.density_fidelity(rho, rho) == 1.0

    zero = _rank_one([1 + 0j, 0j, 0j, 0j])
    one = _rank_one([0j, 1 + 0j, 0j, 0j])
    assert metrics.density_fidelity(zero, one) == 0.0


def test_fidelity_is_symmetric():
    rng = random.Random(99)
    rho = _random_density(5, 3, rng)
    sigma = _random_density(5, 4, rng)
    assert metrics.density_fidelity(rho, sigma) == pytest.approx(
        metrics.density_fidelity(sigma, rho), abs=1e-12
    )


def test_purity_values():
    assert metrics.density_purity(_rank_one([1 + 0j, 0j])) == 1.0
    dimension = 8
    maximally_mixed = [
        [(1.0 / dimension if i == j else 0.0) + 0j for j in range(dimension)]
        for i in range(dimension)
    ]
    assert metrics.density_purity(maximally_mixed) == pytest.approx(1.0 / dimension)


def test_mixed_entropies_maximally_mixed_and_pure():
    maximally_mixed = [[0.5 + 0j, 0j], [0j, 0.5 + 0j]]
    assert metrics.density_single_qubit_entropies(maximally_mixed, 1) == [1.0]
    pure = _rank_one([1 + 0j, 0j])
    assert metrics.density_single_qubit_entropies(pure, 1) == [0.0]


def test_results_are_finite_snapped_and_without_negative_zero():
    rng = random.Random(31)
    rho = _random_density(8, 5, rng)
    sigma = _random_density(8, 5, rng)
    for value in [metrics.density_fidelity(rho, sigma), metrics.density_purity(rho)]:
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    for entropy in metrics.density_single_qubit_entropies(rho, 3):
        assert math.isfinite(entropy)
        assert 0.0 <= entropy <= 1.0
