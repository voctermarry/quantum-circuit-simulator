"""Command line entry point for quantum-circuit-simulator.

This module only parses arguments and dispatches to the command handlers
in :mod:`quantum_circuit.commands`. Input preparation and result delivery
live in :mod:`quantum_circuit.cli_io`, JSON document validation in
:mod:`quantum_circuit.documents`, and all computation in
:mod:`quantum_circuit.core` and the domain modules, none of which depend
on this module.
"""

from __future__ import annotations

import argparse
import math
import sys

from . import __version__, commands
from .cli_io import RunContext


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


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="write the result JSON line to PATH instead of stdout ('-' is not allowed)",
    )


def _build_parser() -> argparse.ArgumentParser:
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
    _add_output_argument(simulate_parser)

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
    _add_output_argument(probabilities_parser)

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
    _add_output_argument(expectation_parser)

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
    _add_output_argument(verify_parser)

    equivalent_parser = sub.add_parser(
        "equivalent",
        help="check whether two noiseless OpenQASM circuits implement the same transformation",
    )
    equivalent_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    equivalent_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")
    _add_output_argument(equivalent_parser)

    optimize_parser = sub.add_parser(
        "optimize",
        help="simplify an OpenQASM circuit into a deterministic canonical form",
    )
    optimize_parser.add_argument("source", help="path to the OpenQASM source file, or '-' for stdin")
    _add_output_argument(optimize_parser)

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
    _add_output_argument(estimate_parser)

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
    _add_output_argument(metrics_parser)

    batch_parser = sub.add_parser(
        "batch-simulate",
        help="run a manifest of independent simulations in list order",
    )
    batch_parser.add_argument(
        "manifest",
        help="path to the UTF-8 JSON batch manifest, or '-' for stdin",
    )
    _add_output_argument(batch_parser)

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
    _add_output_argument(reconcile_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    ctx = RunContext(args.output) if getattr(args, "output", None) is not None else None

    if args.command == "simulate":
        return commands.simulate(args.source, args.shots, args.seed, args.noise_model, ctx)

    if args.command == "probabilities":
        return commands.probabilities(args.source, args.noise_model, ctx)

    if args.command == "expectation":
        return commands.expectation(args.source, args.observables, args.noise_model, ctx)

    if args.command == "verify-samples":
        return commands.verify_samples(args.source, args.samples, args.noise_model, args.tolerance, ctx)

    if args.command == "batch-simulate":
        return commands.batch_simulate(args.manifest, ctx)

    if args.command == "reconcile":
        return commands.reconcile(args.manifest, args.baseline, ctx)

    if args.command == "equivalent":
        return commands.equivalent(args.left, args.right, ctx)

    if args.command == "optimize":
        return commands.optimize(args.source, ctx)

    if args.command == "estimate":
        return commands.estimate(args.source, args.mode, ctx)

    if args.command == "state-metrics":
        return commands.state_metrics(
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
