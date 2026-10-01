"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .equivalence import (
    MAX_EQUIVALENCE_QUBITS,
    TOLERANCE,
    circuit_unitary,
    measurement_layout,
    unitary_distance,
)
from .noise import (
    MAX_NOISE_QUBITS,
    NoiseModelError,
    parse_noise_model,
    simulate_density_matrix,
)
from .openqasm import ParseError, ValidationError, parse
from .simulator import sample_counts, sample_counts_from_probabilities, simulate_state_vector


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _emit_error(
    error: str,
    message: str,
    line: int | None = None,
    column: int | None = None,
    input_side: str | None = None,
) -> None:
    payload: dict[str, object] = {"error": error, "message": message}
    if input_side is not None:
        payload["input"] = input_side
    if line is not None:
        payload["line"] = line
        payload["column"] = column
    sys.stderr.write(json.dumps(payload) + "\n")


def _simulate(source_arg: str, shots: int, seed: int, noise_model_arg: str | None) -> int:
    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        _emit_error(
            "noise_model_error",
            "circuit source and noise model cannot both be read from standard input",
        )
        return 2

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

    noise_model: dict[str, float] | None = None
    if noise_model_arg is not None:
        try:
            if noise_model_arg == "-":
                raw_model = sys.stdin.buffer.read()
            else:
                with open(noise_model_arg, "rb") as handle:
                    raw_model = handle.read()
        except OSError as exc:
            _emit_error("io_error", f"cannot read noise model {noise_model_arg!r}: {exc.strerror or exc}")
            return 1
        try:
            model_text = raw_model.decode("utf-8")
        except UnicodeDecodeError as exc:
            _emit_error("noise_model_error", f"noise model {noise_model_arg!r} is not valid UTF-8: {exc}")
            return 2
        try:
            noise_model = parse_noise_model(model_text)
        except NoiseModelError as exc:
            _emit_error("noise_model_error", str(exc))
            return 2

    try:
        program = parse(source)
    except ParseError as exc:
        _emit_error("parse_error", exc.message, exc.line, exc.column)
        return 2
    except ValidationError as exc:
        _emit_error("validation_error", exc.message, exc.line, exc.column)
        return 2

    if noise_model is None:
        state = simulate_state_vector(program)
        counts = sample_counts(program, state, shots, seed)
        result: dict[str, object] = {
            "schema_version": 1,
            "shots": shots,
            "seed": seed,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "counts": counts,
        }
    else:
        if program.num_qubits > MAX_NOISE_QUBITS:
            _emit_error(
                "simulation_error",
                f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
                f"got {program.num_qubits}",
            )
            return 3
        probabilities = simulate_density_matrix(program, noise_model)
        counts = sample_counts_from_probabilities(program, probabilities, shots, seed)
        result = {
            "schema_version": 2,
            "shots": shots,
            "seed": seed,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "noise_model": noise_model,
            "counts": counts,
        }
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def _equivalent(left_arg: str, right_arg: str) -> int:
    if left_arg == "-" and right_arg == "-":
        _emit_error(
            "comparison_error",
            "left and right inputs cannot both be read from standard input",
        )
        return 2

    programs = []
    for side, source_arg in (("left", left_arg), ("right", right_arg)):
        try:
            if source_arg == "-":
                data = sys.stdin.buffer.read()
            else:
                with open(source_arg, "rb") as handle:
                    data = handle.read()
        except OSError as exc:
            _emit_error("io_error", f"cannot read {side} source {source_arg!r}: {exc.strerror or exc}")
            return 1

        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            _emit_error("io_error", f"{side} source {source_arg!r} is not valid UTF-8: {exc}")
            return 1

        try:
            programs.append(parse(source))
        except ParseError as exc:
            _emit_error("parse_error", exc.message, exc.line, exc.column, input_side=side)
            return 2
        except ValidationError as exc:
            _emit_error("validation_error", exc.message, exc.line, exc.column, input_side=side)
            return 2

    left, right = programs
    for side, program in (("left", left), ("right", right)):
        if program.num_qubits > MAX_EQUIVALENCE_QUBITS:
            _emit_error(
                "simulation_error",
                f"equivalence comparison supports at most {MAX_EQUIVALENCE_QUBITS} qubits, "
                f"{side} input has {program.num_qubits}",
            )
            return 3

    result: dict[str, object] = {"schema_version": 1}
    if left.num_qubits != right.num_qubits:
        result["equivalent"] = False
        result["reason"] = "qubit_count_mismatch"
        distance: float | None = None
    else:
        distance = unitary_distance(circuit_unitary(left), circuit_unitary(right))
        if distance > TOLERANCE:
            result["equivalent"] = False
            result["reason"] = "unitary_distance"
        elif measurement_layout(left) != measurement_layout(right):
            result["equivalent"] = False
            result["reason"] = "measurement_layout_mismatch"
        else:
            result["equivalent"] = True
            result["reason"] = "equivalent"
    result["left_num_qubits"] = left.num_qubits
    result["right_num_qubits"] = right.num_qubits
    result["distance"] = distance
    result["tolerance"] = TOLERANCE
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
    simulate_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for density-matrix simulation ('-' reads stdin)",
    )

    equivalent_parser = sub.add_parser(
        "equivalent", help="check whether two noiseless OpenQASM circuits are equivalent"
    )
    equivalent_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    equivalent_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed, args.noise_model)

    if args.command == "equivalent":
        return _equivalent(args.left, args.right)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
