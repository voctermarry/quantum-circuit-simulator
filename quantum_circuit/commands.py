"""Command handlers: orchestrate preparation, computation and delivery.

Each handler corresponds to one subcommand and follows the same pipeline:
validate argument combinations, prepare inputs through
:mod:`quantum_circuit.cli_io` (reading, decoding, parsing, document
validation), guard the optional ``--output`` export against overwriting an
input, run the pure computation in :mod:`quantum_circuit.core` (or the
relevant domain module), then deliver the result. Handlers own the error
mapping: every failure leaves this module as an emitted single-line JSON
error on stderr plus the command's documented exit code.
"""

from __future__ import annotations

from . import core
from .cli_io import (
    RunContext,
    base_dir_for,
    emit_error,
    emit_error_payload,
    error_payload,
    finish,
    guard_output_conflict,
    prepare_circuit,
    prepare_noise_model,
    prepare_simulation,
    read_utf8_input,
    register_manifest_references,
)
from .core import SimulationError
from .documents import (
    BatchManifestError,
    ReconcileInputError,
    SampleInputError,
    parse_baseline_document,
    parse_batch_manifest,
    parse_samples_document,
)
from .equivalence import MAX_EQUIVALENCE_QUBITS
from .estimation import estimate as estimate_program
from .noise import MAX_NOISE_QUBITS
from .observables import ObservableError, parse_observables
from .openqasm import Program
from .optimizer import optimize as optimize_program


def _emit_and_return(payload: dict[str, object], exit_code: int) -> int:
    emit_error_payload(payload)
    return exit_code


def _guard(ctx: RunContext | None) -> int | None:
    """Emit and return exit code 2 when the export would overwrite an input."""
    if ctx is None:
        return None
    payload = guard_output_conflict(ctx)
    if payload is None:
        return None
    return _emit_and_return(payload, 2)


# ------------------------------------------------------------------ simulate


def simulate(
    source_arg: str,
    shots: int,
    seed: int,
    noise_model_arg: str | None,
    ctx: RunContext | None = None,
) -> int:
    program, noise_model, error, exit_code = prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    if (code := _guard(ctx)) is not None:
        return code
    assert program is not None
    result, error = core.compute_simulation(program, noise_model, shots, seed)
    if error is not None:
        return _emit_and_return(error, 3)
    assert result is not None
    return finish(ctx, result, 0)


# --------------------------------------------------------------- probabilities


def probabilities(
    source_arg: str, noise_model_arg: str | None, ctx: RunContext | None = None
) -> int:
    program, noise_model, error, exit_code = prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert program is not None

    # Refuse an input-overwriting export before any state is evolved.
    if (code := _guard(ctx)) is not None:
        return code

    try:
        result = core.run_probabilities(program, noise_model)
    except SimulationError as exc:
        emit_error("simulation_error", str(exc))
        return 3
    return finish(ctx, result, 0)


# --------------------------------------------------------------- verify-samples


def verify_samples(
    source_arg: str,
    samples_arg: str,
    noise_model_arg: str | None,
    tolerance: float,
    ctx: RunContext | None = None,
) -> int:
    # Three inputs may request standard input; at most one may be '-'. The
    # conflict is detected before any input is read.
    stdin_args = [source_arg, samples_arg]
    if noise_model_arg is not None:
        stdin_args.append(noise_model_arg)
    if sum(value == "-" for value in stdin_args) > 1:
        emit_error(
            "verification_error",
            "at most one of the circuit source, samples document and noise model "
            "can be read from standard input",
        )
        return 2

    # Circuit and noise model use the exact simulate/probabilities path
    # (reading, UTF-8, parsing, validation, model normalization).
    program, noise_model, error, exit_code = prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert program is not None

    samples_text, error, exit_code = read_utf8_input(
        samples_arg, "samples", side="samples", ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert samples_text is not None

    # The samples document is fully validated against the parsed classical
    # register width before any state is evolved.
    try:
        counts = parse_samples_document(samples_text, program.num_clbits)
    except SampleInputError as exc:
        emit_error("sample_input_error", str(exc))
        return 2

    # Refuse an input-overwriting export before any state is evolved.
    if (code := _guard(ctx)) is not None:
        return code

    try:
        result = core.run_verification(program, counts, noise_model, tolerance)
    except SimulationError as exc:
        emit_error("simulation_error", str(exc))
        return 3
    return finish(ctx, result, 0 if result["accepted"] else 3)


# --------------------------------------------------------------- expectation


def expectation(
    source_arg: str,
    observables_arg: str,
    noise_model_arg: str | None,
    ctx: RunContext | None = None,
) -> int:
    # Three inputs may request standard input; at most one may be '-'. The
    # conflict is detected before any input is read.
    stdin_args = [source_arg, observables_arg]
    if noise_model_arg is not None:
        stdin_args.append(noise_model_arg)
    if sum(value == "-" for value in stdin_args) > 1:
        emit_error(
            "expectation_error",
            "at most one of the circuit source, observables document and noise model "
            "can be read from standard input",
        )
        return 2

    # Circuit and noise model use the exact simulate/probabilities path
    # (reading, UTF-8, parsing, validation, model normalization).
    program, noise_model, error, exit_code = prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert program is not None

    observables_text, error, exit_code = read_utf8_input(
        observables_arg, "observables", side="observables", ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert observables_text is not None

    # The observables document is fully validated against the parsed
    # register size before any state is evolved.
    try:
        observables = parse_observables(observables_text, program.num_qubits)
    except ObservableError as exc:
        emit_error("observable_error", str(exc))
        return 2

    # Refuse an input-overwriting export before any state is evolved.
    if (code := _guard(ctx)) is not None:
        return code

    try:
        result = core.run_expectation(program, observables, noise_model)
    except SimulationError as exc:
        emit_error("simulation_error", str(exc))
        return 3
    return finish(ctx, result, 0)


# ------------------------------------------------------------- batch-simulate


def batch_simulate(manifest_arg: str, ctx: RunContext | None = None) -> int:
    manifest_text, error, exit_code = read_utf8_input(manifest_arg, "manifest", ctx=ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert manifest_text is not None
    base_dir = base_dir_for(manifest_arg)

    try:
        jobs = parse_batch_manifest(manifest_text)
    except BatchManifestError as exc:
        emit_error("batch_input_error", str(exc))
        return 2

    # Register every path the validated manifest names, then guard against
    # an input-overwriting export before any task file is opened or any
    # simulation runs. References are registered from the manifest itself,
    # so a missing/unreadable task is still covered.
    if ctx is not None:
        register_manifest_references(ctx, jobs, base_dir)
        if (code := _guard(ctx)) is not None:
            return code

    # Read, decode, parse and validate every task first (task failures are
    # embedded), then evolve each prepared circuit in manifest order.
    prepared: list[core.PreparedTask] = []
    for job in jobs:
        program, noise_model, task_error, _exit_code = prepare_simulation(
            job["source"],
            job.get("noise_model"),
            base_dir,
            ctx,
        )
        prepared.append((job, program, noise_model, task_error))

    summary = core.run_batch(prepared)
    return finish(ctx, summary, 0 if summary["failed"] == 0 else 3)


# --------------------------------------------------------------- reconcile


def reconcile(manifest_arg: str, baseline_arg: str, ctx: RunContext | None = None) -> int:
    if manifest_arg == "-" and baseline_arg == "-":
        # Standard input cannot serve both inputs; do not read it.
        emit_error(
            "reconciliation_error",
            "manifest and baseline cannot both be read from standard input",
        )
        return 2

    # Read and decode both inputs fully (manifest first, then baseline) and
    # validate both before any task file is touched.
    manifest_text, error, exit_code = read_utf8_input(
        manifest_arg, "manifest", side="manifest", ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert manifest_text is not None
    base_dir = base_dir_for(manifest_arg)

    baseline_text, error, exit_code = read_utf8_input(
        baseline_arg, "baseline", side="baseline", ctx=ctx
    )
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert baseline_text is not None

    try:
        jobs = parse_batch_manifest(manifest_text)
    except BatchManifestError as exc:
        emit_error("reconcile_input_error", str(exc))
        return 2

    try:
        baseline_results = parse_baseline_document(baseline_text, jobs)
    except ReconcileInputError as exc:
        emit_error("reconcile_input_error", str(exc))
        return 2

    # Register every path the validated manifest names (plus the manifest
    # and baseline already read), then guard against an input-overwriting
    # export before any task file is opened or any task is re-run.
    if ctx is not None:
        register_manifest_references(ctx, jobs, base_dir)
        if (code := _guard(ctx)) is not None:
            return code

    # Prepare (read, parse, validate) every task first, then re-run them in
    # manifest order with the exact batch-simulate semantics.
    prepared: list[core.PreparedTask] = []
    for job in jobs:
        program, noise_model, task_error, _exit_code = prepare_simulation(
            job["source"],
            job.get("noise_model"),
            base_dir,
            ctx,
        )
        prepared.append((job, program, noise_model, task_error))

    report = core.run_reconciliation(prepared, baseline_results)
    return finish(ctx, report, 0 if report["consistent"] else 3)


# ------------------------------------------------------------------ optimize


def optimize(source_arg: str, ctx: RunContext | None = None) -> int:
    program, error, exit_code = prepare_circuit(source_arg, ctx=ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert program is not None
    if (code := _guard(ctx)) is not None:
        return code

    original_gate_count = sum(1 for op in program.operations if op.kind != "measure")
    gates, changed, qasm = optimize_program(program)
    result: dict[str, object] = {
        "schema_version": 1,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "original_gate_count": original_gate_count,
        "optimized_gate_count": len(gates),
        "changed": changed,
        "qasm": qasm,
    }
    return finish(ctx, result, 0)


# ------------------------------------------------------------------ estimate


def estimate(source_arg: str, mode: str, ctx: RunContext | None = None) -> int:
    program, error, exit_code = prepare_circuit(source_arg, ctx=ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert program is not None
    if (code := _guard(ctx)) is not None:
        return code

    return finish(ctx, estimate_program(program, mode), 0)


# ------------------------------------------------------------------ equivalent


def _prepare_comparison_program(
    side: str, source_arg: str, ctx: RunContext | None
) -> tuple[Program | None, dict[str, object] | None, int]:
    """Load one side of an equivalence comparison, enforcing the size cap."""
    program, error, code = prepare_circuit(source_arg, side=side, ctx=ctx)
    if error is not None:
        return None, error, code
    assert program is not None

    if program.num_qubits > MAX_EQUIVALENCE_QUBITS:
        return (
            None,
            error_payload(
                "simulation_error",
                f"equivalence comparison supports at most {MAX_EQUIVALENCE_QUBITS} qubits, "
                f"{side} input has {program.num_qubits}",
                side=side,
            ),
            3,
        )
    return program, None, 0


def equivalent(left_arg: str, right_arg: str, ctx: RunContext | None = None) -> int:
    if left_arg == "-" and right_arg == "-":
        # Standard input cannot serve both sides; do not read it.
        emit_error(
            "comparison_error",
            "left and right inputs cannot both be read from standard input",
        )
        return 2

    # Fully validate each side (read, parse, semantics, size) in LEFT, RIGHT
    # order, reporting only the first failure.
    left, error, exit_code = _prepare_comparison_program("left", left_arg, ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert left is not None
    right, error, exit_code = _prepare_comparison_program("right", right_arg, ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert right is not None

    if (code := _guard(ctx)) is not None:
        return code

    return finish(ctx, core.run_equivalence(left, right), 0)


# --------------------------------------------------------------- state-metrics


def state_metrics(
    left_arg: str,
    right_arg: str,
    left_noise_arg: str | None,
    right_noise_arg: str | None,
    ctx: RunContext | None = None,
) -> int:
    # The two sources and the two noise models are four stdin-capable
    # inputs; at most one may be '-'. Detect the conflict without touching
    # standard input.
    stdin_count = sum(
        value == "-"
        for value in (left_arg, left_noise_arg, right_arg, right_noise_arg)
    )
    if stdin_count > 1:
        if left_arg == "-" and right_arg == "-":
            # Keep the historical two-source message byte-for-byte.
            message = "left and right inputs cannot both be read from standard input"
        else:
            message = (
                "at most one of the left source, right source, left noise model and "
                "right noise model can be read from standard input"
            )
        emit_error("metrics_error", message)
        return 2

    # Fully validate in left-source, left-model, right-source, right-model
    # order, reporting only the first failure (each tagged with its side).
    left, error, exit_code = prepare_circuit(left_arg, side="left", ctx=ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert left is not None
    left_model = None
    if left_noise_arg is not None:
        left_model, error, exit_code = prepare_noise_model(
            left_noise_arg, side="left", ctx=ctx
        )
        if error is not None:
            return _emit_and_return(error, exit_code)

    right, error, exit_code = prepare_circuit(right_arg, side="right", ctx=ctx)
    if error is not None:
        return _emit_and_return(error, exit_code)
    assert right is not None
    right_model = None
    if right_noise_arg is not None:
        right_model, error, exit_code = prepare_noise_model(
            right_noise_arg, side="right", ctx=ctx
        )
        if error is not None:
            return _emit_and_return(error, exit_code)

    if (code := _guard(ctx)) is not None:
        return code

    # A noisy side evolves a full density matrix (at most MAX_NOISE_QUBITS);
    # a model-less side keeps the 20-qubit state-vector path.
    if left_model is not None and left.num_qubits > MAX_NOISE_QUBITS:
        emit_error(
            "simulation_error",
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"left input has {left.num_qubits}",
            side="left",
        )
        return 3
    if right_model is not None and right.num_qubits > MAX_NOISE_QUBITS:
        emit_error(
            "simulation_error",
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"right input has {right.num_qubits}",
            side="right",
        )
        return 3

    return finish(ctx, core.run_state_metrics(left, right, left_model, right_model), 0)
