"""Command line entry point for quantum-circuit-simulator.

This module is intentionally thin: it owns argument parsing and the
ordering of the three layers, nothing else.

* :mod:`quantum_circuit.preparation` reads files/stdin, decodes UTF-8 and
  parses/validates every input, reporting failures as
  :class:`~quantum_circuit.errors.CommandFailure` data.
* :mod:`quantum_circuit.commands` (with the domain modules
  :mod:`quantum_circuit.core`, :mod:`quantum_circuit.observables`, ...)
  performs the pure computation and assembles success payloads without
  touching files or streams.
* :mod:`quantum_circuit.delivery` writes stdout/stderr and the atomic
  ``--output`` export.
"""

from __future__ import annotations

import argparse
import math
import sys

from . import __version__
from . import commands, preparation
from .delivery import (
    RunContext,
    emit_failure,
    finish,
    output_conflict,
    register_manifest_references,
)
from .documents import json_equal
from .errors import CommandFailure
from .estimation import estimate as estimate_payload
from .simulator import simulate_state_vector  # backwards-compatible CLI name

# Backwards-compatible private alias (exercised by the CLI tests).
_json_equal = json_equal


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _tolerance(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a finite number between 0 and 1, got {value!r}") from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"must be a finite number between 0 and 1, got {value!r}")
    return parsed


def _guard(ctx: RunContext | None) -> int | None:
    """Refuse an input-overwriting export before computation runs."""
    if ctx is None:
        return None
    conflict = output_conflict(ctx)
    if conflict is not None:
        emit_failure(conflict)
        return conflict.exit_code
    return None


def _fail(failure: CommandFailure) -> int:
    emit_failure(failure)
    return failure.exit_code


# --------------------------------------------------------------- simulate etc.


def _simulate(
    source_arg: str,
    shots: int,
    seed: int,
    noise_model_arg: str | None,
    ctx: RunContext | None = None,
) -> int:
    register = ctx.register_input if ctx is not None else None
    program, noise_model, failure = preparation.prepare_simulation(
        source_arg, noise_model_arg, register=register
    )
    if failure is not None:
        return _fail(failure)
    if (code := _guard(ctx)) is not None:
        return code
    try:
        result = commands.simulate_task(program, noise_model, shots, seed)
    except CommandFailure as exc:
        return _fail(exc)
    return finish(ctx, result, 0)


def _probabilities(
    source_arg: str, noise_model_arg: str | None, ctx: RunContext | None = None
) -> int:
    register = ctx.register_input if ctx is not None else None
    program, noise_model, failure = preparation.prepare_simulation(
        source_arg, noise_model_arg, register=register
    )
    if failure is not None:
        return _fail(failure)
    if (code := _guard(ctx)) is not None:
        return code
    try:
        result = commands.probabilities_task(program, noise_model)
    except CommandFailure as exc:
        return _fail(exc)
    return finish(ctx, result, 0)


# --------------------------------------------------------------- verify-samples


def _verify_samples(
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
    conflict = commands.stdin_conflict(
        stdin_args,
        "verification_error",
        "at most one of the circuit source, samples document and noise model "
        "can be read from standard input",
    )
    if conflict is not None:
        return _fail(conflict)

    register = ctx.register_input if ctx is not None else None
    # Circuit and noise model use the exact simulate/probabilities path
    # (reading, UTF-8, parsing, validation, model normalization).
    program, noise_model, failure = preparation.prepare_simulation(
        source_arg, noise_model_arg, register=register
    )
    if failure is not None:
        return _fail(failure)

    samples_text, failure = preparation.read_document(
        "samples", "samples", samples_arg, register
    )
    if failure is not None:
        return _fail(failure)

    # The samples document is fully validated against the parsed classical
    # register width before any state is evolved.
    try:
        counts = commands.samples_counts(samples_text, program.num_clbits)
    except CommandFailure as exc:
        return _fail(exc)

    if (code := _guard(ctx)) is not None:
        return code

    try:
        result, exit_code = commands.verify_samples_task(
            program, noise_model, counts, tolerance
        )
    except CommandFailure as exc:
        return _fail(exc)
    return finish(ctx, result, exit_code)


# --------------------------------------------------------------- expectation


def _expectation(
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
    conflict = commands.stdin_conflict(
        stdin_args,
        "expectation_error",
        "at most one of the circuit source, observables document and noise model "
        "can be read from standard input",
    )
    if conflict is not None:
        return _fail(conflict)

    register = ctx.register_input if ctx is not None else None
    # Circuit and noise model use the exact simulate/probabilities path
    # (reading, UTF-8, parsing, validation, model normalization).
    program, noise_model, failure = preparation.prepare_simulation(
        source_arg, noise_model_arg, register=register
    )
    if failure is not None:
        return _fail(failure)

    observables_text, failure = preparation.read_document(
        "observables", "observables", observables_arg, register
    )
    if failure is not None:
        return _fail(failure)

    # The observables document is fully validated against the parsed
    # register size before any state is evolved.
    try:
        observables = commands.read_observables(observables_text, program.num_qubits)
    except CommandFailure as exc:
        return _fail(exc)

    if (code := _guard(ctx)) is not None:
        return code

    try:
        result = commands.expectation_task(program, noise_model, observables)
    except CommandFailure as exc:
        return _fail(exc)
    return finish(ctx, result, 0)


# ------------------------------------------------------------- batch-simulate


def _prepare_jobs(jobs, base_dir, register):
    """Read/parse/validate every manifest task, keeping failures as data."""
    prepared = []
    for job in jobs:
        program, noise_model, failure = preparation.prepare_simulation(
            job["source"], job.get("noise_model"), base_dir, register
        )
        embedded = None if failure is None else failure.to_payload()
        prepared.append((program, noise_model, embedded))
    return prepared


def _batch_simulate(manifest_arg: str, ctx: RunContext | None = None) -> int:
    register = ctx.register_input if ctx is not None else None
    manifest_text, _base_dir, failure = preparation.read_batch_manifest(
        manifest_arg, register
    )
    if failure is not None:
        return _fail(failure)

    try:
        jobs = commands.manifest_jobs(manifest_text, "batch_input_error")
    except CommandFailure as exc:
        return _fail(exc)

    # Register every path the validated manifest names, then guard against
    # an input-overwriting export before any task file is opened or any
    # simulation runs. References are registered from the manifest itself,
    # so a missing/unreadable task is still covered.
    if ctx is not None:
        register_manifest_references(ctx, jobs, _base_dir)
        if (code := _guard(ctx)) is not None:
            return code

    # Read, decode, parse and validate every task first (task failures are
    # embedded), then evolve each prepared circuit in manifest order.
    prepared = _prepare_jobs(jobs, _base_dir, register)
    summary, exit_code = commands.run_batch(jobs, prepared)
    return finish(ctx, summary, exit_code)


# --------------------------------------------------------------- reconcile


def _reconcile(manifest_arg: str, baseline_arg: str, ctx: RunContext | None = None) -> int:
    conflict = commands.stdin_conflict(
        [manifest_arg, baseline_arg],
        "reconciliation_error",
        "manifest and baseline cannot both be read from standard input",
    )
    if conflict is not None:
        return _fail(conflict)

    register = ctx.register_input if ctx is not None else None
    # Read and decode both inputs fully (manifest first, then baseline) and
    # validate both before any task file is touched.
    raw_manifest, base_dir, failure = preparation.read_manifest_or_baseline(
        "manifest", manifest_arg, register
    )
    if failure is not None:
        return _fail(failure)
    raw_baseline, _base_dir, failure = preparation.read_manifest_or_baseline(
        "baseline", baseline_arg, register
    )
    if failure is not None:
        return _fail(failure)

    try:
        jobs = commands.manifest_jobs(raw_manifest.decode("utf-8"), "reconcile_input_error")
    except CommandFailure as exc:
        return _fail(exc)
    try:
        baseline_results = commands.baseline_results(raw_baseline.decode("utf-8"), jobs)
    except CommandFailure as exc:
        return _fail(exc)

    # Register every path the validated manifest names (plus the manifest
    # and baseline already read), then guard against an input-overwriting
    # export before any task file is opened or any task is re-run.
    if ctx is not None:
        register_manifest_references(ctx, jobs, base_dir)
        if (code := _guard(ctx)) is not None:
            return code

    # Prepare (read, parse, validate) every task first, then re-run them in
    # manifest order with the exact batch-simulate semantics.
    prepared = _prepare_jobs(jobs, base_dir, register)
    report, exit_code = commands.run_reconcile(jobs, baseline_results, prepared)
    return finish(ctx, report, exit_code)


# ------------------------------------------------------------- optimize etc.


def _optimize(source_arg: str, ctx: RunContext | None = None) -> int:
    register = ctx.register_input if ctx is not None else None
    source, failure = preparation.read_source(source_arg, register)
    if failure is not None:
        return _fail(failure)
    program, failure = preparation.parse_program(source)
    if failure is not None:
        return _fail(failure)
    if (code := _guard(ctx)) is not None:
        return code
    return finish(ctx, commands.optimize_task(program), 0)


def _estimate(source_arg: str, mode: str, ctx: RunContext | None = None) -> int:
    register = ctx.register_input if ctx is not None else None
    source, failure = preparation.read_source(source_arg, register)
    if failure is not None:
        return _fail(failure)
    program, failure = preparation.parse_program(source)
    if failure is not None:
        return _fail(failure)
    if (code := _guard(ctx)) is not None:
        return code
    return finish(ctx, estimate_payload(program, mode), 0)


# --------------------------------------------------------------- equivalent


def _equivalent(left_arg: str, right_arg: str, ctx: RunContext | None = None) -> int:
    conflict = commands.stdin_conflict(
        [left_arg, right_arg],
        "comparison_error",
        "left and right inputs cannot both be read from standard input",
    )
    if conflict is not None:
        return _fail(conflict)

    register = ctx.register_input if ctx is not None else None
    # Fully validate each side (read, parse, semantics, size) in LEFT, RIGHT
    # order, reporting only the first failure.
    left, failure = preparation.load_comparison_program("left", left_arg, register)
    if failure is not None:
        return _fail(failure)
    right, failure = preparation.load_comparison_program("right", right_arg, register)
    if failure is not None:
        return _fail(failure)

    if (code := _guard(ctx)) is not None:
        return code
    return finish(ctx, commands.equivalent_task(left, right), 0)


# ------------------------------------------------------------- state-metrics


def _state_metrics(
    left_arg: str,
    right_arg: str,
    left_noise_arg: str | None,
    right_noise_arg: str | None,
    ctx: RunContext | None = None,
) -> int:
    # The two sources and the two noise models are four stdin-capable
    # inputs; at most one may be '-'. Detect the conflict without touching
    # standard input.
    stdin_values = [left_arg, right_arg]
    if left_noise_arg is not None:
        stdin_values.append(left_noise_arg)
    if right_noise_arg is not None:
        stdin_values.append(right_noise_arg)
    if sum(value == "-" for value in stdin_values) > 1:
        if left_arg == "-" and right_arg == "-":
            # Keep the historical two-source message byte-for-byte.
            message = "left and right inputs cannot both be read from standard input"
        else:
            message = (
                "at most one of the left source, right source, left noise model and "
                "right noise model can be read from standard input"
            )
        emit_failure(CommandFailure("metrics_error", message, 2))
        return 2

    register = ctx.register_input if ctx is not None else None
    # Fully validate in left-source, left-model, right-source, right-model
    # order, reporting only the first failure (each tagged with its side).
    left, failure = preparation.load_program("left", left_arg, register)
    if failure is not None:
        return _fail(failure)
    left_model = None
    if left_noise_arg is not None:
        left_model, failure = preparation.load_metrics_noise_model(
            "left", left_noise_arg, register
        )
        if failure is not None:
            return _fail(failure)

    right, failure = preparation.load_program("right", right_arg, register)
    if failure is not None:
        return _fail(failure)
    right_model = None
    if right_noise_arg is not None:
        right_model, failure = preparation.load_metrics_noise_model(
            "right", right_noise_arg, register
        )
        if failure is not None:
            return _fail(failure)

    if (code := _guard(ctx)) is not None:
        return code

    try:
        result = commands.state_metrics_task(left, right, left_model, right_model)
    except CommandFailure as exc:
        return _fail(exc)
    return finish(ctx, result, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quantum-circuit-simulator",
        description="Quantum circuit construction, simulation and verification",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")

    simulate_parser = sub.add_parser("simulate", help="simulate an OpenQASM 2.0 subset circuit")
    simulate_parser.add_argument("source", help="path to the OpenQASM source file, or '-' for stdin")
    simulate_parser.add_argument("--shots", type=_positive_int, default=1024, help="number of samples (default: 1024)")
    simulate_parser.add_argument("--seed", type=int, default=0, help="signed integer RNG seed (default: 0)")
    simulate_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for density-matrix simulation ('-' reads stdin)",
    )
    simulate_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    probabilities_parser = sub.add_parser(
        "probabilities",
        help="report exact final-state measurement probabilities without sampling",
    )
    probabilities_parser.add_argument(
        "source", help="path to the OpenQASM source file, or '-' for stdin"
    )
    probabilities_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for density-matrix probabilities ('-' reads stdin)",
    )
    probabilities_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    expectation_parser = sub.add_parser(
        "expectation",
        help="report Pauli-product observable expectation values of the final pre-measurement state",
    )
    expectation_parser.add_argument(
        "source", help="path to the OpenQASM source file, or '-' for stdin"
    )
    expectation_parser.add_argument(
        "observables",
        help="path to the UTF-8 JSON observables document, or '-' for stdin",
    )
    expectation_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for density-matrix evolution ('-' reads stdin)",
    )
    expectation_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    verify_parser = sub.add_parser(
        "verify-samples",
        help="verify external measurement counts against the exact classical distribution",
    )
    verify_parser.add_argument(
        "source", help="path to the OpenQASM source file, or '-' for stdin"
    )
    verify_parser.add_argument(
        "samples",
        help="path to the UTF-8 JSON samples document, or '-' for stdin",
    )
    verify_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for density-matrix probabilities ('-' reads stdin)",
    )
    verify_parser.add_argument(
        "--tolerance",
        metavar="D",
        type=_tolerance,
        default=0.05,
        help="maximum accepted total variation distance, a finite number in [0, 1] (default: 0.05)",
    )
    verify_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    equivalent_parser = sub.add_parser(
        "equivalent",
        help="check whether two noiseless OpenQASM circuits implement the same transformation",
    )
    equivalent_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    equivalent_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")
    equivalent_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    optimize_parser = sub.add_parser(
        "optimize",
        help="simplify an OpenQASM circuit into a deterministic canonical form",
    )
    optimize_parser.add_argument("source", help="path to the OpenQASM source file, or '-' for stdin")
    optimize_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    estimate_parser = sub.add_parser(
        "estimate",
        help="estimate circuit size and feasibility without evolving it",
    )
    estimate_parser.add_argument("source", help="path to the OpenQASM source file, or '-' for stdin")
    estimate_parser.add_argument(
        "--mode",
        choices=("state-vector", "density-matrix", "unitary"),
        default="state-vector",
        help="simulation mode to size (default: state-vector)",
    )
    estimate_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    metrics_parser = sub.add_parser(
        "state-metrics",
        help="compare the (optionally noisy) final states of two circuits and measure per-qubit entropy",
    )
    metrics_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    metrics_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")
    metrics_parser.add_argument(
        "--left-noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for the left circuit ('-' reads stdin)",
    )
    metrics_parser.add_argument(
        "--right-noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for the right circuit ('-' reads stdin)",
    )
    metrics_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    batch_parser = sub.add_parser(
        "batch-simulate",
        help="run a manifest of independent simulations in list order",
    )
    batch_parser.add_argument(
        "manifest",
        help="path to the UTF-8 JSON batch manifest, or '-' for stdin",
    )
    batch_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    reconcile_parser = sub.add_parser(
        "reconcile",
        help="re-run a batch manifest and compare against a baseline result file",
    )
    reconcile_parser.add_argument(
        "manifest",
        help="path to the UTF-8 JSON batch manifest, or '-' for stdin",
    )
    reconcile_parser.add_argument(
        "baseline",
        help="path to the UTF-8 JSON batch-simulate result file, or '-' for stdin",
    )
    reconcile_parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    ctx = RunContext(args.output) if getattr(args, "output", None) is not None else None

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed, args.noise_model, ctx)

    if args.command == "probabilities":
        return _probabilities(args.source, args.noise_model, ctx)

    if args.command == "expectation":
        return _expectation(args.source, args.observables, args.noise_model, ctx)

    if args.command == "verify-samples":
        return _verify_samples(args.source, args.samples, args.noise_model, args.tolerance, ctx)

    if args.command == "batch-simulate":
        return _batch_simulate(args.manifest, ctx)

    if args.command == "reconcile":
        return _reconcile(args.manifest, args.baseline, ctx)

    if args.command == "equivalent":
        return _equivalent(args.left, args.right, ctx)

    if args.command == "optimize":
        return _optimize(args.source, ctx)

    if args.command == "estimate":
        return _estimate(args.source, args.mode, ctx)

    if args.command == "state-metrics":
        return _state_metrics(
            args.left,
            args.right,
            args.left_noise_model,
            args.right_noise_model,
            ctx,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
