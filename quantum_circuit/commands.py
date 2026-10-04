"""Pure command logic: payload assembly, task execution and orchestration.

Every function here takes already-read and validated inputs (parsed
programs, canonical noise models, parsed JSON documents) and returns the
ready success payload — or raises :class:`~quantum_circuit.errors.CommandFailure`
carrying the stderr error name/message/positions and exit code. Nothing in
this module reads files, standard input or environment, writes streams, or
imports the command entry point, so the complete result-producing logic of
each subcommand can be driven (and tested) without a terminal.

The batch and reconcile commands additionally separate preparation from
execution: their runners take a prebuilt ``prepared`` list whose entries
are ``(program, noise_model, embedded_error)`` triples, where
*embedded_error* is the exact error object embedded for a failed task.
"""

from __future__ import annotations

import json
import math

from .core import (
    SimulationError,
    evolve_basis_probabilities,
    probability_payload,
    run_expectation,
    run_probabilities,
    run_simulation,
    snap_probability,
)
from .documents import (
    BATCH_SCHEMA_VERSION,
    DocumentError,
    json_equal,
    load_json,
    parse_samples_document,
    validate_baseline,
    validate_batch_manifest,
    check_finite_numbers,
)
from .equivalence import (
    EQUIVALENCE_TOLERANCE,
    measurement_layout,
    unitary_distance,
)
from .errors import CommandFailure
from .metrics import (
    density_fidelity,
    density_purity,
    density_single_qubit_entropies,
    normalized_density_matrix,
    pure_density_fidelity,
    single_qubit_entropies,
    state_fidelity,
)
from .noise import MAX_NOISE_QUBITS, evolve_density_matrix
from .observables import (
    ObservableError,
    parse_observables,
)
from .optimizer import optimize as optimize_program
from .simulator import simulate_state_vector, unitary_matrix

PreparedTask = tuple[object, dict[str, float] | None, dict[str, object] | None]


# ----------------------------------------------------------- shared mappers


def read_observables(text: str, num_qubits: int):
    """Parse the observables document, mapping :class:`ObservableError`."""
    try:
        return parse_observables(text, num_qubits)
    except ObservableError as exc:
        raise CommandFailure("observable_error", str(exc), 2) from None


def samples_counts(text: str, num_clbits: int) -> dict[str, int]:
    """Validate the samples document, mapping its error name/exit code."""
    try:
        return parse_samples_document(text, num_clbits)
    except DocumentError as exc:
        raise CommandFailure("sample_input_error", str(exc), 2) from None


def manifest_jobs(text: str, error_name: str) -> list[dict[str, object]]:
    """Parse a batch manifest for *error_name* (batch or reconcile)."""
    try:
        data = load_json(text, "batch manifest")
        return validate_batch_manifest(data)
    except json.JSONDecodeError as exc:
        raise CommandFailure(error_name, f"batch manifest is not valid JSON: {exc}", 2) from None
    except DocumentError as exc:
        raise CommandFailure(error_name, str(exc), 2) from None


def baseline_results(text: str, jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Parse and validate a reconcile baseline against the manifest jobs."""
    try:
        data = load_json(text, "baseline")
        check_finite_numbers(data, "baseline")
        return validate_baseline(data, jobs)
    except json.JSONDecodeError as exc:
        raise CommandFailure("reconcile_input_error", f"baseline is not valid JSON: {exc}", 2) from None
    except DocumentError as exc:
        raise CommandFailure("reconcile_input_error", str(exc), 2) from None


def stdin_conflict(values: list[str], error_name: str, message: str) -> CommandFailure | None:
    """Return the standard-input conflict failure when more than one is '-'."""
    if sum(value == "-" for value in values) > 1:
        return CommandFailure(error_name, message, 2)
    return None


# ------------------------------------------------------------- single tasks


def simulate_task(
    program,
    noise_model: dict[str, float] | None,
    shots: int,
    seed: int,
) -> dict[str, object]:
    """Sample one prepared circuit (the ``simulate`` success payload)."""
    try:
        return run_simulation(program, shots, seed, noise_model)
    except SimulationError as exc:
        raise CommandFailure("simulation_error", str(exc), 3) from None


def probabilities_task(program, noise_model: dict[str, float] | None) -> dict[str, object]:
    """Exact distribution payload (``probabilities``)."""
    try:
        return run_probabilities(program, noise_model)
    except SimulationError as exc:
        raise CommandFailure("simulation_error", str(exc), 3) from None


def verify_samples_task(
    program,
    noise_model: dict[str, float] | None,
    counts: dict[str, int],
    tolerance: float,
) -> tuple[dict[str, object], int]:
    """Assemble the verify-samples report; exit code 3 when rejected."""
    try:
        basis_probabilities = evolve_basis_probabilities(program, noise_model)
    except SimulationError as exc:
        raise CommandFailure("simulation_error", str(exc), 3) from None

    # Expected probabilities follow the exact ``probabilities`` convention
    # (aggregation, renormalization, near-zero/near-one snapping).
    expected = probability_payload(program, basis_probabilities, noise_model)["probabilities"]
    assert isinstance(expected, dict)

    shots = sum(counts.values())
    observed: dict[str, float] = {}
    for key in sorted(counts):
        snapped = snap_probability(counts[key] / shots)
        if snapped is not None:
            observed[key] = snapped

    # Total variation distance over the union of both distributions' keys;
    # outcomes absent from either side contribute probability zero.
    keys = set(expected) | set(observed)
    distance = 0.5 * math.fsum(
        abs(expected.get(key, 0.0) - observed.get(key, 0.0)) for key in keys
    )
    accepted = distance <= tolerance

    result: dict[str, object] = {
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
    return result, 0 if accepted else 3


def expectation_task(
    program,
    noise_model: dict[str, float] | None,
    observables: list[tuple[str, list[tuple[int, str]]]],
) -> dict[str, object]:
    """Evaluate every observable against the final (pre-measurement) state."""
    try:
        return run_expectation(program, noise_model, observables)
    except SimulationError as exc:
        raise CommandFailure("simulation_error", str(exc), 3) from None


def equivalent_task(left, right) -> dict[str, object]:
    """Compare two prepared, size-checked noiseless circuits."""
    left_qubits = left.num_qubits
    right_qubits = right.num_qubits

    left_clbits, left_measurements = measurement_layout(left)
    right_clbits, right_measurements = measurement_layout(right)

    if left_qubits != right_qubits:
        return {
            "schema_version": 1,
            "equivalent": False,
            "reason": "qubit_count_mismatch",
            "left_num_qubits": left_qubits,
            "right_num_qubits": right_qubits,
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
        "left_num_qubits": left_qubits,
        "right_num_qubits": right_qubits,
        "distance": distance,
        "tolerance": EQUIVALENCE_TOLERANCE,
    }


def optimize_task(program) -> dict[str, object]:
    """Simplify one prepared circuit into its canonical payload."""
    original_gate_count = sum(1 for op in program.operations if op.kind != "measure")
    gates, changed, qasm = optimize_program(program)
    return {
        "schema_version": 1,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "original_gate_count": original_gate_count,
        "optimized_gate_count": len(gates),
        "changed": changed,
        "qasm": qasm,
    }


def state_metrics_task(left, right, left_model, right_model) -> dict[str, object]:
    """Compare the final states of two prepared (optionally noisy) circuits."""
    # A noisy side evolves a full density matrix (at most MAX_NOISE_QUBITS);
    # a model-less side keeps the 20-qubit state-vector path.
    if left_model is not None and left.num_qubits > MAX_NOISE_QUBITS:
        raise CommandFailure(
            "simulation_error",
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"left input has {left.num_qubits}",
            3,
            side="left",
        ) from None
    if right_model is not None and right.num_qubits > MAX_NOISE_QUBITS:
        raise CommandFailure(
            "simulation_error",
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"right input has {right.num_qubits}",
            3,
            side="right",
        ) from None

    noisy = left_model is not None or right_model is not None

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

    if not noisy:
        # No noise model on either side: keep the schema_version 1 payload
        # byte-for-byte identical to the historical command.
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


# ------------------------------------------------------------- batch runners


def _task_output(
    program,
    noise_model: dict[str, float] | None,
    shots: int,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Sample one task, converting only the simulation failure to payload."""
    try:
        return run_simulation(program, shots, seed, noise_model), None
    except SimulationError as exc:
        return None, CommandFailure("simulation_error", str(exc), 3).to_payload()


def run_batch(
    jobs: list[dict[str, object]],
    prepared: list[PreparedTask],
) -> tuple[dict[str, object], int]:
    """Execute prepared tasks in manifest order (failures embedded)."""
    results: list[dict[str, object]] = []
    succeeded = 0
    for job, (program, noise_model, prepare_error) in zip(jobs, prepared):
        output: dict[str, object] | None = None
        error = prepare_error
        if error is None:
            output, error = _task_output(
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

    summary: dict[str, object] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "job_count": len(jobs),
        "succeeded": succeeded,
        "failed": len(jobs) - succeeded,
        "results": results,
    }
    return summary, 0 if succeeded == len(jobs) else 3


def run_reconcile(
    jobs: list[dict[str, object]],
    baseline: list[dict[str, object]],
    prepared: list[PreparedTask],
) -> tuple[dict[str, object], int]:
    """Re-run prepared tasks and diff them against the validated baseline."""
    entries: list[dict[str, object]] = []
    matched = 0
    for job, (program, noise_model, prepare_error), reference in zip(jobs, prepared, baseline):
        output: dict[str, object] | None = None
        error = prepare_error
        if error is None:
            output, error = _task_output(
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

    mismatched = len(jobs) - matched
    report: dict[str, object] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "job_count": len(jobs),
        "matched": matched,
        "mismatched": mismatched,
        "consistent": mismatched == 0,
        "results": entries,
    }
    return report, 0 if mismatched == 0 else 3
