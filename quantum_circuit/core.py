"""CLI-independent simulation application core.

Both the command line (``simulate``, ``probabilities`` and
``verify-samples``) and the Python DSL (:class:`~quantum_circuit.circuit.Circuit`)
prepare and validate their own inputs, then delegate here for the actual
work: state evolution, measurement probability aggregation, deterministic
sampling and canonical success-payload assembly.

This module never imports :mod:`quantum_circuit.cli` and performs no file,
standard input, environment or standard stream access. It receives only a
validated :class:`~quantum_circuit.openqasm.Program` together with already
validated sampling parameters and a canonical noise model (as produced by
:func:`quantum_circuit.noise.validate_noise_model` /
:func:`quantum_circuit.noise.parse_noise_model`); failures intrinsic to a
simulation are reported as :class:`SimulationError`.
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
    """The prepared circuit cannot be simulated (``simulation_error``).

    The only current case is a noisy circuit beyond the density-matrix
    qubit limit; command-line callers map it to exit code 3.
    """


def noisy_qubit_limit_error(num_qubits: int) -> SimulationError:
    return SimulationError(
        f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
        f"got {num_qubits}"
    )


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


def evolve_basis_probabilities(
    program: Program, noise_model: dict[str, float] | None = None
) -> list[float]:
    """Evolve *program* and return per-basis-state measurement probabilities.

    Without a noise model the final state vector is evolved (the parser
    caps registers at the 20-qubit limit) and amplitudes are squared; with
    one the density matrix is evolved (at most :data:`~quantum_circuit.noise.MAX_NOISE_QUBITS`
    qubits) and its diagonal returned. Raises :class:`SimulationError` when
    a noisy circuit exceeds the density-matrix limit.
    """
    if noise_model is None:
        state = simulate_state_vector(program)
        return [abs(amplitude) ** 2 for amplitude in state]
    if program.num_qubits > MAX_NOISE_QUBITS:
        raise noisy_qubit_limit_error(program.num_qubits)
    return simulate_density_matrix(program, noise_model)


def probability_payload(
    program: Program,
    basis_probabilities: list[float],
    noise_model: dict[str, float] | None = None,
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


def run_probabilities(
    program: Program, noise_model: dict[str, float] | None = None
) -> dict[str, object]:
    """Evolve *program* and assemble its ``probabilities`` payload."""
    basis_probabilities = evolve_basis_probabilities(program, noise_model)
    return probability_payload(program, basis_probabilities, noise_model)


def run_simulation(
    program: Program,
    shots: int,
    seed: int,
    noise_model: dict[str, float] | None = None,
) -> dict[str, object]:
    """Evolve a prepared circuit and sample it (the ``simulate`` payload).

    Sampling draws from a fresh RNG seeded by *seed*, so equal arguments
    produce equal results and no random state is shared between calls.
    Raises :class:`SimulationError` when a noisy circuit exceeds the
    density-matrix qubit limit.
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
    if program.num_qubits > MAX_NOISE_QUBITS:
        raise noisy_qubit_limit_error(program.num_qubits)
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
