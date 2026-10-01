"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .equivalence import (
    EQUIVALENCE_TOLERANCE,
    MAX_EQUIVALENCE_QUBITS,
    measurement_layout,
    unitary_distance,
)
from .metrics import single_qubit_entropies, state_fidelity
from .noise import (
    MAX_NOISE_QUBITS,
    NoiseModelError,
    parse_noise_model,
    simulate_density_matrix,
)
from .openqasm import ParseError, ValidationError, parse
from .optimizer import optimize as optimize_program
from .simulator import sample_counts, sample_counts_from_probabilities, simulate_state_vector, unitary_matrix


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
    side: str | None = None,
) -> None:
    payload: dict[str, object] = {"error": error}
    if side is not None:
        payload["input"] = side
    payload["message"] = message
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


def _read_source(source_arg: str) -> str | None:
    """Read a UTF-8 circuit source from a path or ``-`` (stdin).

    Returns the decoded source, or ``None`` after emitting a single-line
    ``io_error`` on stderr (the corresponding exit code is always 1).
    """
    try:
        if source_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(source_arg, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        _emit_error("io_error", f"cannot read source {source_arg!r}: {exc.strerror or exc}")
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error("io_error", f"source {source_arg!r} is not valid UTF-8: {exc}")
        return None


def _optimize(source_arg: str) -> int:
    source = _read_source(source_arg)
    if source is None:
        return 1

    try:
        program = parse(source)
    except ParseError as exc:
        _emit_error("parse_error", exc.message, exc.line, exc.column)
        return 2
    except ValidationError as exc:
        _emit_error("validation_error", exc.message, exc.line, exc.column)
        return 2

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
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def _load_side_program(side: str, source_arg: str):
    """Read and parse one side of a two-circuit comparison.

    Returns the parsed :class:`Program`, or an exit code (1 or 2) after
    emitting the appropriate single-line error on stderr. *side* is
    ``"left"`` or ``"right"`` and is reported as ``input`` on every failure.
    """
    try:
        if source_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(source_arg, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        _emit_error(
            "io_error",
            f"cannot read source {source_arg!r}: {exc.strerror or exc}",
            side=side,
        )
        return None, 1

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error("io_error", f"source {source_arg!r} is not valid UTF-8: {exc}", side=side)
        return None, 1

    try:
        program = parse(source)
    except ParseError as exc:
        _emit_error("parse_error", exc.message, exc.line, exc.column, side=side)
        return None, 2
    except ValidationError as exc:
        _emit_error("validation_error", exc.message, exc.line, exc.column, side=side)
        return None, 2

    return program, None


def _load_comparison_program(side: str, source_arg: str):
    """Read, parse and size-check one side of an equivalence comparison.

    Returns the parsed :class:`Program`, or an exit code (1, 2 or 3) after
    emitting the appropriate single-line error on stderr. *side* is ``"left"``
    or ``"right"`` and is reported as ``input`` on every failure.
    """
    program, error_code = _load_side_program(side, source_arg)
    if program is None:
        return None, error_code

    if program.num_qubits > MAX_EQUIVALENCE_QUBITS:
        _emit_error(
            "simulation_error",
            f"equivalence comparison supports at most {MAX_EQUIVALENCE_QUBITS} qubits, "
            f"{side} input has {program.num_qubits}",
            side=side,
        )
        return None, 3

    return program, None


def _equivalent(left_arg: str, right_arg: str) -> int:
    if left_arg == "-" and right_arg == "-":
        # Standard input cannot serve both sides; do not read it.
        _emit_error(
            "comparison_error",
            "left and right inputs cannot both be read from standard input",
        )
        return 2

    # Fully validate each side (read, parse, semantics, size) in LEFT, RIGHT
    # order, reporting only the first failure.
    left, error_code = _load_comparison_program("left", left_arg)
    if left is None:
        return error_code
    right, error_code = _load_comparison_program("right", right_arg)
    if right is None:
        return error_code

    left_qubits = left.num_qubits
    right_qubits = right.num_qubits

    left_clbits, left_measurements = measurement_layout(left)
    right_clbits, right_measurements = measurement_layout(right)

    if left_qubits != right_qubits:
        result: dict[str, object] = {
            "schema_version": 1,
            "equivalent": False,
            "reason": "qubit_count_mismatch",
            "left_num_qubits": left_qubits,
            "right_num_qubits": right_qubits,
            "distance": None,
            "tolerance": EQUIVALENCE_TOLERANCE,
        }
        sys.stdout.write(json.dumps(result) + "\n")
        return 0

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

    result = {
        "schema_version": 1,
        "equivalent": equivalent,
        "reason": reason,
        "left_num_qubits": left_qubits,
        "right_num_qubits": right_qubits,
        "distance": distance,
        "tolerance": EQUIVALENCE_TOLERANCE,
    }
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


def _state_metrics(left_arg: str, right_arg: str) -> int:
    if left_arg == "-" and right_arg == "-":
        # Standard input cannot serve both sides; do not read it.
        _emit_error(
            "metrics_error",
            "left and right inputs cannot both be read from standard input",
        )
        return 2

    # Fully validate each side (read, parse, semantics) in LEFT, RIGHT
    # order, reporting only the first failure.
    left, error_code = _load_side_program("left", left_arg)
    if left is None:
        return error_code
    right, error_code = _load_side_program("right", right_arg)
    if right is None:
        return error_code

    # Terminal measurements are ignored by the state-vector evolution, so
    # the measurement layouts of the two sides need not agree.
    left_state = simulate_state_vector(left)
    right_state = simulate_state_vector(right)

    if left.num_qubits == right.num_qubits:
        reason = "compared"
        fidelity: float | None = state_fidelity(left_state, right_state)
    else:
        reason = "qubit_count_mismatch"
        fidelity = None

    result: dict[str, object] = {
        "schema_version": 1,
        "left_num_qubits": left.num_qubits,
        "right_num_qubits": right.num_qubits,
        "reason": reason,
        "fidelity": fidelity,
        "left_single_qubit_entropy": single_qubit_entropies(left_state, left.num_qubits),
        "right_single_qubit_entropy": single_qubit_entropies(right_state, right.num_qubits),
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
    simulate_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="optional JSON noise model for density-matrix simulation ('-' reads stdin)",
    )

    equivalent_parser = sub.add_parser(
        "equivalent",
        help="check whether two noiseless OpenQASM circuits implement the same transformation",
    )
    equivalent_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    equivalent_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")

    optimize_parser = sub.add_parser(
        "optimize",
        help="simplify an OpenQASM circuit into a deterministic canonical form",
    )
    optimize_parser.add_argument("source", help="path to the OpenQASM source file, or '-' for stdin")

    metrics_parser = sub.add_parser(
        "state-metrics",
        help="compare the noiseless final states of two circuits evolved from the all-zero state",
    )
    metrics_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    metrics_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed, args.noise_model)

    if args.command == "equivalent":
        return _equivalent(args.left, args.right)

    if args.command == "optimize":
        return _optimize(args.source)

    if args.command == "state-metrics":
        return _state_metrics(args.left, args.right)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
