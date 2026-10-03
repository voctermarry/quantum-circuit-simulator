"""Shared simulation core behind the ``simulate`` and ``probabilities`` surfaces.

This module converges everything the command line and the Python DSL agree
on: state evolution (state vector or density matrix), measurement-probability
aggregation, deterministic shot sampling, and assembly of the canonical
result objects both surfaces emit. It is independent of
:mod:`quantum_circuit.cli`: it never reads files, standard input or
environment variables and never writes to stdout or stderr.

Callers supply an already validated :class:`~quantum_circuit.openqasm.Program`,
a shots/seed pair and an already validated noise model (or ``None``). The
result is the exact payload dictionary the corresponding surface reports;
the only failure raised here is :class:`SimulationError`, a domain error
each surface maps onto its own error convention (the command line emits a
``simulation_error`` line with exit code 3, the DSL raises ``ValueError``).
"""

from __future__ import annotations

import math

from .noise import MAX_NOISE_QUBITS, simulate_density_matrix
from .openqasm import Program
from .simulator import (
    measurement_probabilities,
    sample_counts,
    sample_counts_from_probabilities,
    simulate_state_vector,
)

# Probabilities are reported in full; entries this close to 0 are omitted
# and entries this close to 1 are reported as exactly 1. Reported values
# sum to within 1e-12 of 1.
PROBABILITY_ZERO_TOLERANCE = 1e-15
PROBABILITY_ONE_TOLERANCE = 1e-15


class SimulationError(Exception):
    """A validated circuit cannot be simulated within the core's limits."""


def snap_probability(value: float) -> float | None:
    """Snap one probability for deterministic output.

    Returns ``None`` when the entry is omitted (magnitude at most
    ``1e-15``), ``1.0`` when it is within ``1e-15`` of 1, and otherwise a
    finite ``float`` clamped to ``(0, 1)`` so simulation round-off can
    never produce an out-of-range value.
    """
    if abs(value) <= PROBABILITY_ZERO_TOLERANCE:
        return None
    if abs(1.0 - value) <= PROBABILITY_ONE_TOLERANCE:
        return 1.0
    return min(1.0, max(0.0, float(value)))


def _check_noise_qubit_limit(program: Program) -> None:
    if program.num_qubits > MAX_NOISE_QUBITS:
        raise SimulationError(
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"got {program.num_qubits}"
        )


def basis_probabilities(program: Program, noise_model: dict[str, float] | None) -> list[float]:
    """Evolve *program* and return its basis-state probability vector.

    Without a noise model this is the squared magnitude of the final state
    vector (register sizes are already capped at the 20-qubit state-vector
    limit by the parser and the DSL). With a noise model the circuit must
    fit the density-matrix limit and the result is the diagonal of the
    final density matrix.
    """
    if noise_model is None:
        state = simulate_state_vector(program)
        return [abs(amplitude) ** 2 for amplitude in state]
    _check_noise_qubit_limit(program)
    return simulate_density_matrix(program, noise_model)


def probability_payload(
    program: Program,
    basis_probabilities: list[float],
    noise_model: dict[str, float] | None,
) -> dict[str, object]:
    """Build the deterministic ``probabilities`` success payload.

    The aggregated classical distribution is renormalized by its total
    first, mirroring the normalization the sampling path applies to basis
    probabilities, so simulation round-off (notably density-matrix drift)
    cannot move the reported sum away from 1.
    """
    aggregated = measurement_probabilities(program, basis_probabilities)
    total = math.fsum(aggregated.values())
    probabilities: dict[str, float] = {}
    for key, value in aggregated.items():
        normalized = value / total if total else value
        snapped = snap_probability(normalized)
        if snapped is not None:
            probabilities[key] = snapped

    if noise_model is None:
        return {
            "schema_version": 1,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "probabilities": probabilities,
        }
    return {
        "schema_version": 2,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "noise_model": noise_model,
        "probabilities": probabilities,
    }


def probabilities_result(
    program: Program, noise_model: dict[str, float] | None
) -> dict[str, object]:
    """Evolve *program* and build the canonical ``probabilities`` payload."""
    return probability_payload(program, basis_probabilities(program, noise_model), noise_model)


def sample_result(
    program: Program,
    noise_model: dict[str, float] | None,
    shots: int,
    seed: int,
) -> dict[str, object]:
    """Evolve *program*, sample it and build the canonical ``simulate`` payload.

    Sampling is deterministic for a given (*shots*, *seed*): each call draws
    from a fresh RNG seeded by *seed*, so equal arguments give equal counts.
    """
    if noise_model is None:
        state = simulate_state_vector(program)
        counts = sample_counts(program, state, shots, seed)
        return {
            "schema_version": 1,
            "shots": shots,
            "seed": seed,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "counts": counts,
        }
    _check_noise_qubit_limit(program)
    probabilities = simulate_density_matrix(program, noise_model)
    counts = sample_counts_from_probabilities(program, probabilities, shots, seed)
    return {
        "schema_version": 2,
        "shots": shots,
        "seed": seed,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "noise_model": noise_model,
        "counts": counts,
    }


__all__ = [
    "PROBABILITY_ONE_TOLERANCE",
    "PROBABILITY_ZERO_TOLERANCE",
    "SimulationError",
    "basis_probabilities",
    "probabilities_result",
    "probability_payload",
    "sample_result",
    "snap_probability",
]
