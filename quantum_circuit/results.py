"""Deterministic result payloads shared by the command line and the Python DSL.

Both the CLI (``simulate``/``probabilities``) and :class:`Circuit`
(``sample``/``probabilities``) build their success objects here, with a fixed
key order, so the JSON the CLI writes and the dicts the DSL returns contain
exactly the same fields and values.
"""

from __future__ import annotations

import math

from .openqasm import Program
from .simulator import measurement_probabilities, sample_counts_from_probabilities

# Probabilities are reported in full; entries this close to 0 are omitted
# and entries this close to 1 are reported as exactly 1. Reported values
# sum to within 1e-12 of 1.
PROBABILITY_ZERO_TOLERANCE = 1e-15
PROBABILITY_ONE_TOLERANCE = 1e-15


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


def simulation_payload(
    program: Program,
    basis_probabilities: list[float],
    shots: int,
    seed: int,
    noise_model: dict[str, float] | None,
) -> dict[str, object]:
    """Build the ``simulate``/``sample`` success payload from basis weights.

    *basis_probabilities* are the per-basis-state outcome probabilities:
    squared state-vector amplitudes on the noiseless path, the density
    matrix diagonal on the noisy path. Sampling draws a fresh RNG seeded
    from *seed*, so repeated calls with the same arguments return equal
    counts and never share random state.
    """
    counts = sample_counts_from_probabilities(program, basis_probabilities, shots, seed)
    if noise_model is None:
        return {
            "schema_version": 1,
            "shots": shots,
            "seed": seed,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "counts": counts,
        }
    return {
        "schema_version": 2,
        "shots": shots,
        "seed": seed,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "noise_model": noise_model,
        "counts": counts,
    }
