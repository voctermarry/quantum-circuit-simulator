"""CLI-independent simulation application core.

Both the command line and the Python DSL
(:class:`~quantum_circuit.circuit.Circuit`) prepare and validate their own
inputs, then delegate here for the actual work: state evolution, measurement
probability aggregation, deterministic sampling, observable expectations,
state and unitary comparison, batch/reconcile scheduling and canonical
success-payload assembly.

This module never imports the command modules
(:mod:`quantum_circuit.cli`, :mod:`quantum_circuit.commands` or
:mod:`quantum_circuit.cli_io`) and performs no file, standard input,
environment or standard stream access. It receives only validated inputs --
a parsed :class:`~quantum_circuit.openqasm.Program`, already validated
sampling parameters, canonical noise models (as produced by
:func:`quantum_circuit.noise.parse_noise_model`) and validated JSON document
content (as produced by :mod:`quantum_circuit.documents`); failures
intrinsic to a simulation are reported as :class:`SimulationError`.
"""

from __future__ import annotations

import math

from .documents import BATCH_SCHEMA_VERSION, json_equal
from .equivalence import EQUIVALENCE_TOLERANCE, measurement_layout, unitary_distance
from .metrics import (
    density_fidelity,
    density_purity,
    density_single_qubit_entropies,
    normalized_density_matrix,
    pure_density_fidelity,
    single_qubit_entropies,
    state_fidelity,
)
from .noise import MAX_NOISE_QUBITS, evolve_density_matrix, simulate_density_matrix
from .observables import (
    density_matrix_expectation,
    snap_expectation,
    state_vector_expectation,
)
from .openqasm import Program
from .simulator import (
    measurement_probabilities,
    sample_counts,
    sample_counts_from_probabilities,
    simulate_state_vector,
    unitary_matrix,
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


def compute_simulation(
    program: Program,
    noise_model: dict[str, float] | None,
    shots: int,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Run one prepared simulation task, capturing limit failures.

    Returns ``(result, None)`` on success or ``(None, error_payload)`` when
    the circuit is rejected, so batch-style callers can embed the task
    error and command-line callers can emit it unchanged.
    """
    try:
        return run_simulation(program, shots, seed, noise_model), None
    except SimulationError as exc:
        return None, {"error": "simulation_error", "message": str(exc)}


def run_verification(
    program: Program,
    counts: dict[str, int],
    noise_model: dict[str, float] | None,
    tolerance: float,
) -> dict[str, object]:
    """Compare observed *counts* against the exact distribution.

    *counts* is a validated samples-document counts mapping (see
    :func:`quantum_circuit.documents.parse_samples_document`). Expected
    probabilities follow the exact ``probabilities`` convention
    (aggregation, renormalization, near-zero/near-one snapping); the
    distance is the total variation distance over the union of both
    distributions' keys, with outcomes absent from either side contributing
    probability zero. Raises :class:`SimulationError` when a noisy circuit
    exceeds the density-matrix qubit limit.
    """
    basis_probabilities = evolve_basis_probabilities(program, noise_model)
    expected = probability_payload(program, basis_probabilities, noise_model)["probabilities"]
    assert isinstance(expected, dict)

    shots = sum(counts.values())
    observed: dict[str, float] = {}
    for key in sorted(counts):
        snapped = snap_probability(counts[key] / shots)
        if snapped is not None:
            observed[key] = snapped

    keys = set(expected) | set(observed)
    distance = 0.5 * math.fsum(
        abs(expected.get(key, 0.0) - observed.get(key, 0.0)) for key in keys
    )
    accepted = distance <= tolerance

    return {
        "schema_version": 1,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "shots": shots,
        "noise_model": noise_model,
        "tolerance": tolerance,
        "total_variation_distance": distance,
        "accepted": accepted,
        "expected_probabilities": expected,
        "observed_probabilities": observed,
    }


def run_expectation(
    program: Program,
    observables: list[tuple[str, object]],
    noise_model: dict[str, float] | None,
) -> dict[str, object]:
    """Evaluate Pauli-product *observables* on the final pre-measurement state.

    *observables* is the validated ``(id, operators)`` list produced by
    :func:`quantum_circuit.observables.parse_observables`. Raises
    :class:`SimulationError` when a noisy circuit exceeds the
    density-matrix qubit limit.
    """
    if noise_model is None:
        # The parser already caps the register at the 20-qubit
        # state-vector limit.
        state = simulate_state_vector(program)
        results = [
            {
                "id": observable_id,
                "expectation": snap_expectation(state_vector_expectation(state, operators)),
            }
            for observable_id, operators in observables
        ]
        return {
            "schema_version": 1,
            "num_qubits": program.num_qubits,
            "noise_model": None,
            "results": results,
        }
    if program.num_qubits > MAX_NOISE_QUBITS:
        raise noisy_qubit_limit_error(program.num_qubits)
    rho = normalized_density_matrix(evolve_density_matrix(program, noise_model))
    results = [
        {
            "id": observable_id,
            "expectation": snap_expectation(density_matrix_expectation(rho, operators)),
        }
        for observable_id, operators in observables
    ]
    return {
        "schema_version": 1,
        "num_qubits": program.num_qubits,
        "noise_model": noise_model,
        "results": results,
    }


def run_equivalence(left: Program, right: Program) -> dict[str, object]:
    """Compare two noiseless circuits and assemble the ``equivalent`` payload.

    Both programs must already satisfy the equivalence qubit cap (enforced
    by the caller while loading, so a rejected side never reaches here).
    """
    left_clbits, left_measurements = measurement_layout(left)
    right_clbits, right_measurements = measurement_layout(right)

    if left.num_qubits != right.num_qubits:
        return {
            "schema_version": 1,
            "equivalent": False,
            "reason": "qubit_count_mismatch",
            "left_num_qubits": left.num_qubits,
            "right_num_qubits": right.num_qubits,
            "distance": None,
            "tolerance": EQUIVALENCE_TOLERANCE,
        }

    distance = unitary_distance(unitary_matrix(left), unitary_matrix(right))

    same_transform = distance <= EQUIVALENCE_TOLERANCE
    same_layout = left_clbits == right_clbits and left_measurements == right_measurements

    if not same_transform:
        reason = "unitary_distance"
        equivalent = False
    elif not same_layout:
        reason = "measurement_layout_mismatch"
        equivalent = False
    else:
        reason = "equivalent"
        equivalent = True

    return {
        "schema_version": 1,
        "equivalent": equivalent,
        "reason": reason,
        "left_num_qubits": left.num_qubits,
        "right_num_qubits": right.num_qubits,
        "distance": distance,
        "tolerance": EQUIVALENCE_TOLERANCE,
    }


def run_state_metrics(
    left: Program,
    right: Program,
    left_model: dict[str, float] | None,
    right_model: dict[str, float] | None,
) -> dict[str, object]:
    """Compare the (optionally noisy) final states of two circuits.

    A noisy side evolves a full density matrix, a model-less side keeps the
    state-vector path; the caller enforces the density-matrix qubit limit
    per side before calling, so a rejected side never reaches here.
    """
    left_state: list[complex] | None = None
    right_state: list[complex] | None = None
    left_rho: list[list[complex]] | None = None
    right_rho: list[list[complex]] | None = None
    if left_model is None:
        left_state = simulate_state_vector(left)
    else:
        left_rho = normalized_density_matrix(evolve_density_matrix(left, left_model))
    if right_model is None:
        right_state = simulate_state_vector(right)
    else:
        right_rho = normalized_density_matrix(evolve_density_matrix(right, right_model))

    same_qubits = left.num_qubits == right.num_qubits
    reason = "compared" if same_qubits else "qubit_count_mismatch"

    if not same_qubits:
        fidelity: float | None = None
    elif left_rho is None and right_rho is None:
        fidelity = state_fidelity(left_state, right_state)
    elif left_rho is not None and right_rho is not None:
        fidelity = density_fidelity(left_rho, right_rho)
    elif left_rho is not None:
        fidelity = pure_density_fidelity(right_state, left_rho)
    else:
        fidelity = pure_density_fidelity(left_state, right_rho)

    left_purity = 1.0 if left_rho is None else density_purity(left_rho)
    right_purity = 1.0 if right_rho is None else density_purity(right_rho)
    left_entropy = (
        single_qubit_entropies(left_state, left.num_qubits)
        if left_rho is None
        else density_single_qubit_entropies(left_rho, left.num_qubits)
    )
    right_entropy = (
        single_qubit_entropies(right_state, right.num_qubits)
        if right_rho is None
        else density_single_qubit_entropies(right_rho, right.num_qubits)
    )

    if left_model is None and right_model is None:
        # No noise model on either side: the schema_version 1 payload.
        return {
            "schema_version": 1,
            "left_num_qubits": left.num_qubits,
            "right_num_qubits": right.num_qubits,
            "reason": reason,
            "fidelity": fidelity,
            "left_single_qubit_entropy": left_entropy,
            "right_single_qubit_entropy": right_entropy,
        }
    return {
        "schema_version": 2,
        "left_num_qubits": left.num_qubits,
        "right_num_qubits": right.num_qubits,
        "reason": reason,
        "left_noise_model": left_model,
        "right_noise_model": right_model,
        "fidelity": fidelity,
        "left_purity": left_purity,
        "right_purity": right_purity,
        "left_single_qubit_entropy": left_entropy,
        "right_single_qubit_entropy": right_entropy,
    }


# Prepared batch/reconcile task: the manifest job together with the circuit
# and noise model produced by input preparation, or the preparation error
# payload to embed instead of a result.
PreparedTask = tuple[
    dict[str, object],
    Program | None,
    dict[str, float] | None,
    dict[str, object] | None,
]


def run_batch(prepared: list[PreparedTask]) -> dict[str, object]:
    """Evolve every prepared task in manifest order and assemble the summary.

    Tasks whose preparation failed keep their embedded error; the rest are
    sampled with the exact ``simulate`` semantics. Task isolation is
    preserved: one task's failure never affects another's result.
    """
    results: list[dict[str, object]] = []
    succeeded = 0
    for job, program, noise_model, error in prepared:
        output: dict[str, object] | None = None
        if error is None:
            assert program is not None
            output, error = compute_simulation(
                program,
                noise_model,
                job.get("shots", 1024),
                job.get("seed", 0),
            )
        if error is None:
            succeeded += 1
            results.append({"id": job["id"], "status": "succeeded", "output": output})
        else:
            results.append({"id": job["id"], "status": "failed", "error": error})

    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "job_count": len(prepared),
        "succeeded": succeeded,
        "failed": len(prepared) - succeeded,
        "results": results,
    }


def run_reconciliation(
    prepared: list[PreparedTask],
    baseline_results: list[dict[str, object]],
) -> dict[str, object]:
    """Re-run every prepared task and compare against the baseline entries.

    Each task is re-run with the exact batch-simulate semantics; the report
    preserves the complete baseline and current task result on any
    mismatch.
    """
    entries: list[dict[str, object]] = []
    matched = 0
    for (job, program, noise_model, error), reference in zip(prepared, baseline_results):
        output: dict[str, object] | None = None
        if error is None:
            assert program is not None
            output, error = compute_simulation(
                program,
                noise_model,
                job.get("shots", 1024),
                job.get("seed", 0),
            )
        current: dict[str, object] = (
            {"id": job["id"], "status": "succeeded", "output": output}
            if error is None
            else {"id": job["id"], "status": "failed", "error": error}
        )
        entry: dict[str, object] = {"id": job["id"]}
        if current["status"] != reference["status"]:
            entry["consistent"] = False
            entry["reason"] = "status_mismatch"
            entry["expected"] = reference
            entry["actual"] = current
            entries.append(entry)
            continue

        if current["status"] == "succeeded":
            equal = json_equal(reference["output"], output)
            reason = "output_mismatch"
        else:
            equal = json_equal(reference["error"], error)
            reason = "error_mismatch"

        if equal:
            entry["consistent"] = True
            entry["reason"] = "identical"
            matched += 1
        else:
            entry["consistent"] = False
            entry["reason"] = reason
            # Preserve the complete baseline and current task result.
            entry["expected"] = reference
            entry["actual"] = current
        entries.append(entry)

    mismatched = len(prepared) - matched
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "job_count": len(prepared),
        "matched": matched,
        "mismatched": mismatched,
        "consistent": mismatched == 0,
        "results": entries,
    }
