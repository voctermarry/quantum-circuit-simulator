"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .equivalence import (
    EQUIVALENCE_TOLERANCE,
    MAX_EQUIVALENCE_QUBITS,
    measurement_layout,
    unitary_distance,
)
from .estimation import estimate as estimate_program
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


def _error_payload(
    error: str,
    message: str,
    line: int | None = None,
    column: int | None = None,
    side: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"error": error}
    if side is not None:
        payload["input"] = side
    payload["message"] = message
    if line is not None:
        payload["line"] = line
        payload["column"] = column
    return payload


def _emit_error(
    error: str,
    message: str,
    line: int | None = None,
    column: int | None = None,
    side: str | None = None,
) -> None:
    _emit_error_payload(_error_payload(error, message, line, column, side))


def _emit_error_payload(payload: dict[str, object]) -> None:
    sys.stderr.write(json.dumps(payload) + "\n")


def _run_simulation(
    source_arg: str,
    shots: int,
    seed: int,
    noise_model_arg: str | None,
    base_dir: str | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, int]:
    """Run one simulation with the exact semantics of ``simulate``.

    Returns ``(output, error, exit_code)``: on success *error* is ``None``
    and *output* is the payload printed by ``simulate``; on failure *output*
    is ``None`` and *error* is the payload ``simulate`` would write to
    stderr (including line/column where applicable).

    When *base_dir* is given (the batch case), relative paths are opened
    relative to it while error messages keep quoting *source_arg* and
    *noise_model_arg* verbatim, so messages match a standalone ``simulate``
    invocation with the same path strings.
    """

    def open_path(path: str) -> str:
        if base_dir is not None and not os.path.isabs(path):
            return os.path.normpath(os.path.join(base_dir, path))
        return path

    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        return (
            None,
            _error_payload(
                "noise_model_error",
                "circuit source and noise model cannot both be read from standard input",
            ),
            2,
        )

    try:
        if source_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(open_path(source_arg), "rb") as handle:
                data = handle.read()
    except OSError as exc:
        return None, _error_payload("io_error", f"cannot read source {source_arg!r}: {exc.strerror or exc}"), 1

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, _error_payload("io_error", f"source {source_arg!r} is not valid UTF-8: {exc}"), 1

    noise_model: dict[str, float] | None = None
    if noise_model_arg is not None:
        try:
            if noise_model_arg == "-":
                raw_model = sys.stdin.buffer.read()
            else:
                with open(open_path(noise_model_arg), "rb") as handle:
                    raw_model = handle.read()
        except OSError as exc:
            return (
                None,
                _error_payload(
                    "io_error", f"cannot read noise model {noise_model_arg!r}: {exc.strerror or exc}"
                ),
                1,
            )
        try:
            model_text = raw_model.decode("utf-8")
        except UnicodeDecodeError as exc:
            return (
                None,
                _error_payload(
                    "noise_model_error",
                    f"noise model {noise_model_arg!r} is not valid UTF-8: {exc}",
                ),
                2,
            )
        try:
            noise_model = parse_noise_model(model_text)
        except NoiseModelError as exc:
            return None, _error_payload("noise_model_error", str(exc)), 2

    try:
        program = parse(source)
    except ParseError as exc:
        return None, _error_payload("parse_error", exc.message, exc.line, exc.column), 2
    except ValidationError as exc:
        return None, _error_payload("validation_error", exc.message, exc.line, exc.column), 2

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
            return (
                None,
                _error_payload(
                    "simulation_error",
                    f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
                    f"got {program.num_qubits}",
                ),
                3,
            )
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
    return result, None, 0


def _simulate(source_arg: str, shots: int, seed: int, noise_model_arg: str | None) -> int:
    result, error, exit_code = _run_simulation(source_arg, shots, seed, noise_model_arg)
    if error is not None:
        _emit_error_payload(error)
        return exit_code
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


# ------------------------------------------------------------- batch-simulate

BATCH_SCHEMA_VERSION = 1
MAX_BATCH_JOBS = 100
_BATCH_ROOT_KEYS = ("schema_version", "jobs")
_BATCH_JOB_KEYS = ("id", "source", "shots", "seed", "noise_model")


class _BatchManifestError(Exception):
    """The batch manifest is structurally invalid (``batch_input_error``)."""


def _manifest_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _BatchManifestError(f"duplicate key {key!r} in batch manifest")
        result[key] = value
    return result


def _reject_manifest_constant(value: str) -> None:
    raise _BatchManifestError(f"batch manifest contains non-JSON constant {value!r}")


def _is_json_int(value: object) -> bool:
    # bool is a subclass of int but is not a JSON integer here.
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_batch_manifest(data: object) -> list[dict[str, object]]:
    """Validate the parsed manifest, returning the raw jobs in list order.

    Every structural, type and range rule is checked here, before any task
    file is read. Raises :class:`_BatchManifestError` on the first problem.
    """
    if not isinstance(data, dict):
        raise _BatchManifestError("batch manifest must be a JSON object")

    for key in data:
        if key not in _BATCH_ROOT_KEYS:
            raise _BatchManifestError(f"unknown batch manifest key {key!r}")

    if "schema_version" not in data:
        raise _BatchManifestError("batch manifest is missing required key 'schema_version'")
    schema_version = data["schema_version"]
    if not _is_json_int(schema_version) or schema_version != BATCH_SCHEMA_VERSION:
        raise _BatchManifestError(
            f"batch manifest schema_version must be {BATCH_SCHEMA_VERSION}, got {schema_version!r}"
        )

    if "jobs" not in data:
        raise _BatchManifestError("batch manifest is missing required key 'jobs'")
    jobs = data["jobs"]
    if not isinstance(jobs, list):
        raise _BatchManifestError("batch manifest 'jobs' must be an array")
    if not 1 <= len(jobs) <= MAX_BATCH_JOBS:
        raise _BatchManifestError(
            f"batch manifest 'jobs' must contain between 1 and {MAX_BATCH_JOBS} jobs, got {len(jobs)}"
        )

    seen_ids: set[str] = set()
    for index, job in enumerate(jobs):
        where = f"job at jobs[{index}]"
        if not isinstance(job, dict):
            raise _BatchManifestError(f"{where} must be a JSON object")
        for key in job:
            if key not in _BATCH_JOB_KEYS:
                raise _BatchManifestError(f"{where} has unknown key {key!r}")

        if "id" not in job:
            raise _BatchManifestError(f"{where} is missing required key 'id'")
        job_id = job["id"]
        if not isinstance(job_id, str) or not job_id:
            raise _BatchManifestError(f"{where} 'id' must be a non-empty string")
        if job_id in seen_ids:
            raise _BatchManifestError(f"duplicate job id {job_id!r}")
        seen_ids.add(job_id)

        if "source" not in job:
            raise _BatchManifestError(f"{where} is missing required key 'source'")
        source = job["source"]
        if not isinstance(source, str):
            raise _BatchManifestError(f"{where} 'source' must be a string")
        if source == "-":
            raise _BatchManifestError(f"{where} 'source' must not be '-'; stdin is reserved for the manifest")

        if "shots" in job:
            shots = job["shots"]
            if not _is_json_int(shots) or shots <= 0:
                raise _BatchManifestError(f"{where} 'shots' must be a positive integer")

        if "seed" in job and not _is_json_int(job["seed"]):
            raise _BatchManifestError(f"{where} 'seed' must be an integer")

        if "noise_model" in job:
            noise_model = job["noise_model"]
            if not isinstance(noise_model, str):
                raise _BatchManifestError(f"{where} 'noise_model' must be a string path")
            if noise_model == "-":
                raise _BatchManifestError(
                    f"{where} 'noise_model' must not be '-'; stdin is reserved for the manifest"
                )

    return jobs


def _batch_simulate(manifest_arg: str) -> int:
    try:
        if manifest_arg == "-":
            raw_manifest = sys.stdin.buffer.read()
            base_dir = os.getcwd()
        else:
            with open(manifest_arg, "rb") as handle:
                raw_manifest = handle.read()
            base_dir = os.path.dirname(os.path.abspath(manifest_arg))
    except OSError as exc:
        _emit_error("io_error", f"cannot read manifest {manifest_arg!r}: {exc.strerror or exc}")
        return 1

    try:
        manifest_text = raw_manifest.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error("io_error", f"manifest {manifest_arg!r} is not valid UTF-8: {exc}")
        return 1

    try:
        data = json.loads(
            manifest_text,
            object_pairs_hook=_manifest_pairs,
            parse_constant=_reject_manifest_constant,
        )
        jobs = _validate_batch_manifest(data)
    except (json.JSONDecodeError, _BatchManifestError) as exc:
        # Syntax problems are reported with the same shape as every other
        # manifest rejection: a single batch_input_error line.
        if isinstance(exc, json.JSONDecodeError):
            message = f"batch manifest is not valid JSON: {exc}"
        else:
            message = str(exc)
        _emit_error("batch_input_error", message)
        return 2

    results: list[dict[str, object]] = []
    succeeded = 0
    for job in jobs:
        output, error, _exit_code = _run_simulation(
            job["source"],
            job.get("shots", 1024),
            job.get("seed", 0),
            job.get("noise_model"),
            base_dir=base_dir,
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
    sys.stdout.write(json.dumps(summary) + "\n")
    return 0 if succeeded == len(jobs) else 3


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


def _estimate(source_arg: str, mode: str) -> int:
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

    sys.stdout.write(json.dumps(estimate_program(program, mode)) + "\n")
    return 0


def _load_program(side: str, source_arg: str):
    """Read, parse and validate one side of a two-circuit comparison.

    Returns the parsed :class:`Program`, or an exit code (1 or 2) after
    emitting the appropriate single-line error on stderr. *side* is
    ``"left"`` or ``"right"`` and is reported as ``input`` on every
    failure.
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
    """Load one side of an equivalence comparison, enforcing the size cap.

    Like :func:`_load_program`, but additionally rejects circuits larger
    than ``MAX_EQUIVALENCE_QUBITS`` with exit code 3.
    """
    program, error_code = _load_program(side, source_arg)
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
    left, error_code = _load_program("left", left_arg)
    if left is None:
        return error_code
    right, error_code = _load_program("right", right_arg)
    if right is None:
        return error_code

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

    metrics_parser = sub.add_parser(
        "state-metrics",
        help="compare the noiseless final states of two circuits and measure per-qubit entanglement",
    )
    metrics_parser.add_argument("left", help="path to the left OpenQASM source file, or '-' for stdin")
    metrics_parser.add_argument("right", help="path to the right OpenQASM source file, or '-' for stdin")

    batch_parser = sub.add_parser(
        "batch-simulate",
        help="run a manifest of independent simulations in list order",
    )
    batch_parser.add_argument(
        "manifest",
        help="path to the UTF-8 JSON batch manifest, or '-' for stdin",
    )

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed, args.noise_model)

    if args.command == "batch-simulate":
        return _batch_simulate(args.manifest)

    if args.command == "equivalent":
        return _equivalent(args.left, args.right)

    if args.command == "optimize":
        return _optimize(args.source)

    if args.command == "estimate":
        return _estimate(args.source, args.mode)

    if args.command == "state-metrics":
        return _state_metrics(args.left, args.right)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
