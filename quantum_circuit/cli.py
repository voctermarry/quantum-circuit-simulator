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
    sys.stderr.write(json.dumps(_error_payload(error, message, line, column, side)) + "\n")


def _read_utf8_file(
    path: str, label: str, read_error: str, decode_error: str, decode_code: int = 1
):
    """Read and UTF-8 decode a file.

    *label* describes the file kind in messages (``"source"`` or
    ``"noise model"``); *read_error* and *decode_error* name the errors used
    for unreadable files and invalid UTF-8 respectively. Unreadable files
    always use exit code 1; *decode_code* sets the code for invalid UTF-8
    (noise models use 2). Returns ``(text, None, 0)`` or
    ``(None, payload, code)``.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        return None, _error_payload(read_error, f"cannot read {label} {path!r}: {exc.strerror or exc}"), 1
    try:
        return data.decode("utf-8"), None, 0
    except UnicodeDecodeError as exc:
        return None, _error_payload(decode_error, f"{label} {path!r} is not valid UTF-8: {exc}"), decode_code


def _run_simulation(source_arg: str, shots: int, seed: int, noise_model_arg: str | None):
    """Run one simulation exactly like the ``simulate`` subcommand.

    Returns ``(result, None, 0)`` on success or ``(None, payload, code)`` on
    failure, where *payload* is the same single-line error object the
    subcommand writes to stderr. Paths of ``-`` read standard input, so
    callers that cannot represent stdin must reject them up front.
    """
    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        return (
            None,
            _error_payload(
                "noise_model_error",
                "circuit source and noise model cannot both be read from standard input",
            ),
            2,
        )

    if source_arg == "-":
        try:
            source = sys.stdin.buffer.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, _error_payload("io_error", f"source '-' is not valid UTF-8: {exc}"), 1
    else:
        source, payload, payload_code = _read_utf8_file(
            source_arg, "source", "io_error", "io_error"
        )
        if payload is not None:
            return None, payload, payload_code

    noise_model: dict[str, float] | None = None
    if noise_model_arg is not None:
        if noise_model_arg == "-":
            raw_model = sys.stdin.buffer.read()
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
        else:
            model_text, payload, payload_code = _read_utf8_file(
                noise_model_arg, "noise model", "io_error", "noise_model_error", decode_code=2
            )
            if payload is not None:
                return None, payload, payload_code
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
    result, payload, code = _run_simulation(source_arg, shots, seed, noise_model_arg)
    if result is None:
        sys.stderr.write(json.dumps(payload) + "\n")
        return code
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


# -------------------------------------------------------------- batch-simulate

BATCH_SCHEMA_VERSION = 1
MAX_BATCH_JOBS = 100
_BATCH_ROOT_KEYS = frozenset(("schema_version", "jobs"))
_BATCH_JOB_KEYS = frozenset(("id", "source", "shots", "seed", "noise_model"))
_BATCH_DEFAULT_SHOTS = 1024
_BATCH_DEFAULT_SEED = 0


class _BatchValidationError(Exception):
    """The batch manifest is structurally invalid (``batch_input_error``)."""


def _batch_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _BatchValidationError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_batch_constant(value: str) -> None:
    # json accepts NaN/Infinity by default; they are not valid JSON numbers.
    raise _BatchValidationError(f"non-JSON constant {value!r}")


def _validate_batch_manifest(data: object) -> list[dict[str, object]]:
    """Validate a decoded batch manifest, returning normalized job dicts.

    Raises :class:`_BatchValidationError` on the first problem found.
    """
    if not isinstance(data, dict):
        raise _BatchValidationError("batch manifest root must be a JSON object")
    for key in data:
        if key not in _BATCH_ROOT_KEYS:
            raise _BatchValidationError(f"unknown manifest key {key!r}")
    if "schema_version" not in data:
        raise _BatchValidationError("missing required key 'schema_version'")
    version = data["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != BATCH_SCHEMA_VERSION:
        raise _BatchValidationError(f"schema_version must be {BATCH_SCHEMA_VERSION}")
    if "jobs" not in data:
        raise _BatchValidationError("missing required key 'jobs'")

    jobs = data["jobs"]
    if not isinstance(jobs, list):
        raise _BatchValidationError("'jobs' must be an array")
    if not 1 <= len(jobs) <= MAX_BATCH_JOBS:
        raise _BatchValidationError(f"'jobs' must contain between 1 and {MAX_BATCH_JOBS} jobs")

    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, job in enumerate(jobs):
        location = f"jobs[{index}]"
        if not isinstance(job, dict):
            raise _BatchValidationError(f"{location} must be a JSON object")
        for key in job:
            if key not in _BATCH_JOB_KEYS:
                raise _BatchValidationError(f"{location}: unknown key {key!r}")

        if "id" not in job:
            raise _BatchValidationError(f"{location}: missing required key 'id'")
        job_id = job["id"]
        if not isinstance(job_id, str):
            raise _BatchValidationError(f"{location}: 'id' must be a string")
        if not job_id:
            raise _BatchValidationError(f"{location}: 'id' must be a non-empty string")
        if job_id in seen_ids:
            raise _BatchValidationError(f"{location}: duplicate job id {job_id!r}")
        seen_ids.add(job_id)

        if "source" not in job:
            raise _BatchValidationError(f"{location}: missing required key 'source'")
        source = job["source"]
        if not isinstance(source, str):
            raise _BatchValidationError(f"{location}: 'source' must be a string")
        if source == "-":
            raise _BatchValidationError(f"{location}: 'source' cannot be '-' (standard input)")

        shots = job.get("shots", _BATCH_DEFAULT_SHOTS)
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise _BatchValidationError(f"{location}: 'shots' must be a positive integer")

        seed = job.get("seed", _BATCH_DEFAULT_SEED)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise _BatchValidationError(f"{location}: 'seed' must be an integer")

        noise_model: str | None = None
        if "noise_model" in job:
            noise_model = job["noise_model"]
            if not isinstance(noise_model, str):
                raise _BatchValidationError(f"{location}: 'noise_model' must be a string")
            if noise_model == "-":
                raise _BatchValidationError(f"{location}: 'noise_model' cannot be '-' (standard input)")

        normalized.append(
            {
                "id": job_id,
                "source": source,
                "shots": shots,
                "seed": seed,
                "noise_model": noise_model,
            }
        )
    return normalized


def _load_batch_manifest(manifest_arg: str):
    """Read and fully validate the batch manifest before any task is touched.

    Returns ``(jobs, base_dir, None, 0)`` or ``(None, None, payload, code)``.
    Unreadable or non-UTF-8 manifests are ``io_error`` (1); malformed JSON or
    invalid structure are ``batch_input_error`` (2).
    """
    try:
        if manifest_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(manifest_arg, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        return None, None, _error_payload(
            "io_error", f"cannot read batch manifest {manifest_arg!r}: {exc.strerror or exc}"
        ), 1

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, None, _error_payload(
            "io_error", f"batch manifest {manifest_arg!r} is not valid UTF-8: {exc}"
        ), 1

    try:
        parsed = json.loads(
            text, object_pairs_hook=_batch_object_pairs, parse_constant=_reject_batch_constant
        )
        jobs = _validate_batch_manifest(parsed)
    except (json.JSONDecodeError, _BatchValidationError) as exc:
        return None, None, _error_payload("batch_input_error", str(exc)), 2

    if manifest_arg == "-":
        base_dir = os.getcwd()
    else:
        base_dir = os.path.dirname(os.path.abspath(manifest_arg))
    return jobs, base_dir, None, 0


def _resolve_job_path(base_dir: str, path: str) -> str:
    """Resolve a task path against the manifest directory (or cwd for stdin)."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _batch_simulate(manifest_arg: str) -> int:
    jobs, base_dir, payload, code = _load_batch_manifest(manifest_arg)
    if jobs is None:
        sys.stderr.write(json.dumps(payload) + "\n")
        return code

    results: list[dict[str, object]] = []
    succeeded = 0
    failed = 0
    for job in jobs:
        source_path = _resolve_job_path(base_dir, job["source"])
        noise_arg = job["noise_model"]
        noise_path = _resolve_job_path(base_dir, noise_arg) if noise_arg is not None else None
        output, error_payload, _error_code = _run_simulation(
            source_path, job["shots"], job["seed"], noise_path
        )
        if output is not None:
            entry: dict[str, object] = {
                "id": job["id"],
                "status": "succeeded",
                "output": output,
            }
            succeeded += 1
        else:
            entry = {"id": job["id"], "status": "failed", "error": error_payload}
            failed += 1
        results.append(entry)

    summary: dict[str, object] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "job_count": len(jobs),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    return 3 if failed else 0


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
        help="run multiple independent simulations described by a JSON manifest",
    )
    batch_parser.add_argument("manifest", help="path to the UTF-8 JSON manifest, or '-' for stdin")

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

    if args.command == "estimate":
        return _estimate(args.source, args.mode)

    if args.command == "state-metrics":
        return _state_metrics(args.left, args.right)

    if args.command == "batch-simulate":
        return _batch_simulate(args.manifest)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
