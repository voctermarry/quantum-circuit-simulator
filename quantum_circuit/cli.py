"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import re
import sys

from . import __version__
from .qasm import ParseError, ValidationError, parse
from .simulator import run_state_vector, sample_counts

_UNSIGNED_INT = re.compile(r"[0-9]+\Z")
_SIGNED_INT = re.compile(r"[+-]?[0-9]+\Z")


def _strict_int(value: str, pattern: re.Pattern[str]) -> int | None:
    if not pattern.match(value):
        return None
    return int(value)


def _positive_int(value: str) -> int:
    number = _strict_int(value, _UNSIGNED_INT)
    if number is None:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"shots must be a positive integer, got {number}")
    return number


def _signed_int(value: str) -> int:
    number = _strict_int(value, _SIGNED_INT)
    if number is None:
        raise argparse.ArgumentTypeError(f"expected an integer seed, got {value!r}")
    return number


def _error_payload(error: str, line: int | None = None, column: int | None = None) -> str:
    payload: dict[str, object] = {"error": error}
    if line is not None:
        payload["line"] = line
    if column is not None:
        payload["column"] = column
    return json.dumps(payload, ensure_ascii=False)


def _read_source(source: str) -> str | None:
    try:
        if source == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(source, "rb") as handle:
                data = handle.read()
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _run_simulate(args: argparse.Namespace) -> int:
    text = _read_source(args.source)
    if text is None:
        print(_error_payload("io_error"), file=sys.stderr)
        return 1

    try:
        program = parse(text)
    except ParseError as error:
        print(_error_payload("parse_error", error.line, error.column), file=sys.stderr)
        return 2
    except ValidationError as error:
        print(_error_payload("validation_error", error.line, error.column), file=sys.stderr)
        return 2

    state = run_state_vector(program)
    counts = sample_counts(program, state, args.shots, args.seed)

    result = {
        "schema_version": 1,
        "shots": args.shots,
        "seed": args.seed,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "counts": counts,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantum-circuit-simulator", description="Quantum circuit construction, simulation and verification")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")

    simulate = sub.add_parser("simulate", help="simulate an OpenQASM 2.0 circuit")
    simulate.add_argument("source", help="UTF-8 OpenQASM source file, or - for standard input")
    simulate.add_argument("--shots", type=_positive_int, default=1024, help="number of shots (positive integer, default 1024)")
    simulate.add_argument("--seed", type=_signed_int, default=0, help="sampling seed (signed integer, default 0)")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "simulate":
        return _run_simulate(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
