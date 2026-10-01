"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .noise import NoiseModelError, parse_noise_model
from .openqasm import ParseError, ValidationError, parse
from .simulator import MAX_NOISE_QUBITS, sample_counts, sample_counts_noisy, simulate_density_matrix, simulate_state_vector


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


def _read_input(path_arg: str, label: str) -> tuple[bytes | None, int | None]:
    """Read raw bytes from *path_arg* (``-`` means stdin)."""
    try:
        if path_arg == "-":
            return sys.stdin.buffer.read(), None
        with open(path_arg, "rb") as handle:
            return handle.read(), None
    except OSError as exc:
        _emit_error("io_error", f"cannot read {label} {path_arg!r}: {exc.strerror or exc}")
        return None, 1


def _simulate(source_arg: str, shots: int, seed: int, noise_model_arg: str | None) -> int:
    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        _emit_error("noise_model_error", "circuit source and noise model cannot both be read from stdin")
        return 2

    data, rc = _read_input(source_arg, "source")
    if rc is not None:
        return rc
    assert data is not None

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

    if noise_model_arg is not None:
        model_data, rc = _read_input(noise_model_arg, "noise model")
        if rc is not None:
            return rc
        assert model_data is not None
        try:
            model_text = model_data.decode("utf-8")
        except UnicodeDecodeError as exc:
            _emit_error("noise_model_error", f"noise model {noise_model_arg!r} is not valid UTF-8: {exc}")
            return 2
        try:
            noise = parse_noise_model(model_text)
        except NoiseModelError as exc:
            _emit_error("noise_model_error", str(exc))
            return 2

        if program.num_qubits > MAX_NOISE_QUBITS:
            _emit_error(
                "simulation_error",
                f"noisy simulation supports at most {MAX_NOISE_QUBITS} qubits, got {program.num_qubits}",
            )
            return 3

        rho = simulate_density_matrix(program, noise)
        counts = sample_counts_noisy(program, rho, shots, seed)

        result = {
            "schema_version": 2,
            "shots": shots,
            "seed": seed,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "noise_model": noise,
            "counts": counts,
        }
        sys.stdout.write(json.dumps(result) + "\n")
        return 0

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
    simulate_parser.add_argument(
        "--noise-model",
        metavar="PATH",
        default=None,
        help="path to a JSON noise model enabling density-matrix simulation, or '-' for stdin",
    )

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed, args.noise_model)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
