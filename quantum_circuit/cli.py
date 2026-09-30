"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .openqasm import ParseError, ValidationError, parse
from .simulator import sample_counts, simulate_state_vector


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _emit_error(error: str, message: str, line: int | None = None, column: int | None = None) -> None:
    payload: dict[str, object] = {"error": error, "message": message}
    if line is not None:
        payload["line"] = line
        payload["column"] = column
    sys.stderr.write(json.dumps(payload) + "\n")


def _simulate(source_arg: str, shots: int, seed: int) -> int:
    try:
        if source_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(source_arg, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        _emit_error("io_error", f"cannot read source {source_arg!r}: {exc.strerror or exc}")
        return 1

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error("io_error", f"source {source_arg!r} is not valid UTF-8: {exc}")
        return 1

    try:
        program = parse(source)
    except ParseError as exc:
        _emit_error("parse_error", exc.message, exc.line, exc.column)
        return 2
    except ValidationError as exc:
        _emit_error("validation_error", exc.message, exc.line, exc.column)
        return 2

    state = simulate_state_vector(program)
    counts = sample_counts(program, state, shots, seed)

    result = {
        "schema_version": 1,
        "shots": shots,
        "seed": seed,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "counts": counts,
    }
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


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

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
