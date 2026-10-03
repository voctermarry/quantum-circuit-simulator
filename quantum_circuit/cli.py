"""Command line entry point for quantum-circuit-simulator."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import tempfile

from . import __version__
from .equivalence import (
    EQUIVALENCE_TOLERANCE,
    MAX_EQUIVALENCE_QUBITS,
    measurement_layout,
    unitary_distance,
)
from .estimation import estimate as estimate_program
from .metrics import (
    density_fidelity,
    density_purity,
    density_single_qubit_entropies,
    normalized_density_matrix,
    pure_density_fidelity,
    single_qubit_entropies,
    state_fidelity,
)
from .noise import (
    MAX_NOISE_QUBITS,
    NoiseModelError,
    evolve_density_matrix,
    parse_noise_model,
    simulate_density_matrix,
)
from .observables import (
    ObservableError,
    density_matrix_expectation,
    parse_observables,
    snap_expectation,
    state_vector_expectation,
)
from .openqasm import ParseError, Program, ValidationError, parse
from .optimizer import optimize as optimize_program
from .simulator import (
    measurement_probabilities,
    sample_counts,
    sample_counts_from_probabilities,
    simulate_state_vector,
    unitary_matrix,
)


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


# ------------------------------------------------------------- result export


class _RunContext:
    """Track the optional ``--output PATH`` target and every input file read.

    The target is absolutized once (anchored at the invocation working
    directory). Every non-stdin input the command reads is registered using
    the same path resolution used to open it, so an export that would
    overwrite a reproducibility input can be refused before any result is
    computed.
    """

    def __init__(self, output: str | None):
        self.output_arg = output
        self.target: str | None = None if output is None else os.path.abspath(output)
        self.inputs: set[str] = set()

    def register_input(self, path: str, base_dir: str | None = None) -> None:
        if self.target is None or path == "-":
            return
        if base_dir is not None and not os.path.isabs(path):
            resolved = os.path.join(base_dir, path)
        else:
            resolved = path
        self.inputs.add(os.path.abspath(resolved))


def _guard_output_conflict(ctx: _RunContext) -> int | None:
    """Refuse to export over an input file (``output_error``, exit code 2).

    Invoked only after the command's own validation has succeeded but
    before any result-producing computation runs, so no simulation or
    re-run task executes and no file is modified.
    """
    if ctx.target is not None and ctx.target in ctx.inputs:
        _emit_error(
            "output_error",
            f"output path {ctx.output_arg!r} is the same file as an input read by this command",
        )
        return 2
    return None


def _register_manifest_references(
    ctx: _RunContext,
    jobs: list[dict[str, object]],
    base_dir: str | None,
) -> None:
    """Register every circuit/noise path a *validated* manifest names.

    Conflict protection is based on the manifest's declared references,
    not only on files that opened successfully, so a failed task cannot
    export a report over a path a later re-run would treat as input. The
    manifest is structurally valid at this point (paths are non-``-``
    strings); a missing or unreadable file is still covered.
    """
    if ctx.target is None:
        return
    for job in jobs:
        source = str(job["source"])
        ctx.register_input(source, base_dir)
        noise_model = job.get("noise_model")
        if noise_model is not None:
            ctx.register_input(str(noise_model), base_dir)


def _write_output_atomic(target: str, data: bytes) -> str | None:
    """Write *data* to *target* without ever leaving a partial result.

    Returns ``None`` on success, otherwise a human-readable error message.
    The data lands in a temporary file in the target's parent directory and
    is moved into place atomically, so a failure leaves any pre-existing
    target byte-for-byte unchanged and removes the temporary file.
    """
    parent = os.path.dirname(target) or "."
    if not os.path.isdir(parent):
        return f"cannot write output {target!r}: parent directory does not exist"

    # The target may be a new file or an existing regular file. Symlinks are
    # followed (a link to a regular file is a valid target; os.replace swaps
    # the link itself without touching its referent), while a directory --
    # including a symlink to one -- and other non-regular types are refused.
    try:
        existing = os.stat(target)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        return f"cannot write output {target!r}: {exc.strerror or exc}"
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        kind = "a directory" if stat.S_ISDIR(existing.st_mode) else "not a regular file"
        return f"cannot write output {target!r}: target is {kind}"

    tmp_name: str | None = None
    try:
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".qcs-output-", suffix=".tmp", dir=parent)
        except OSError as exc:
            return f"cannot write output {target!r}: {exc.strerror or exc}"
        try:
            # A new file gets the usual umask-derived mode; overwriting an
            # existing regular file keeps its permissions (as open("w") would).
            current_umask = os.umask(0)
            os.umask(current_umask)
            mode = (
                stat.S_IMODE(existing.st_mode)
                if existing is not None
                else 0o666 & ~current_umask
            )
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            return f"cannot write output {target!r}: {exc.strerror or exc}"
        try:
            os.replace(tmp_name, target)
        except OSError as exc:
            return f"cannot write output {target!r}: {exc.strerror or exc}"
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return None


def _finish(ctx: _RunContext | None, result: dict[str, object], exit_code: int) -> int:
    """Emit a ready result: stdout as before, or the exported file.

    With no ``--output`` the stdout/stderr/exit behavior is exactly the
    historical one. With an export, stdout stays empty and the original
    exit code (including the batch/reconcile code 3) is preserved whenever
    the file lands; an export failure is an ``output_error`` with code 1.
    """
    if ctx is None or ctx.target is None:
        sys.stdout.write(json.dumps(result) + "\n")
        return exit_code
    if ctx.output_arg == "-":
        _emit_error("output_error", "output path must not be '-'")
        return 1
    message = _write_output_atomic(ctx.target, (json.dumps(result) + "\n").encode("utf-8"))
    if message is not None:
        _emit_error("output_error", message)
        return 1
    return exit_code


def _prepare_simulation(
    source_arg: str,
    noise_model_arg: str | None,
    base_dir: str | None = None,
    ctx: _RunContext | None = None,
) -> tuple[Program | None, dict[str, float] | None, dict[str, object] | None, int]:
    """Read, decode, parse and validate one simulation input.

    Returns ``(program, noise_model, error, exit_code)``: on success *error*
    is ``None`` and *program* is the parsed circuit (with *noise_model*
    ``None`` on the state-vector path); on failure *program* is ``None`` and
    *error* is the payload the calling command writes to stderr (including
    line/column where applicable).

    Reading (files, UTF-8, ``-`` for stdin, relative paths) and error
    semantics match ``simulate`` for both ``simulate`` and
    ``probabilities``. No qubit-limit or simulation error is raised here;
    callers apply their own limits.

    When *base_dir* is given (the batch case), relative paths are opened
    relative to it while error messages keep quoting *source_arg* and
    *noise_model_arg* verbatim, so messages match a standalone invocation
    with the same path strings.
    """

    def open_path(path: str) -> str:
        if base_dir is not None and not os.path.isabs(path):
            return os.path.normpath(os.path.join(base_dir, path))
        return path

    def register(path: str) -> None:
        if ctx is not None:
            ctx.register_input(path, base_dir)

    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        return (
            None,
            None,
            _error_payload(
                "noise_model_error",
                "circuit source and noise model cannot both be read from standard input",
            ),
            2,
        )

    if source_arg != "-":
        source_resolved = open_path(source_arg)
    try:
        if source_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(source_resolved, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        return None, None, _error_payload(
            "io_error", f"cannot read source {source_arg!r}: {exc.strerror or exc}"
        ), 1
    if source_arg != "-":
        register(source_arg)

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, None, _error_payload(
            "io_error", f"source {source_arg!r} is not valid UTF-8: {exc}"
        ), 1

    noise_model: dict[str, float] | None = None
    if noise_model_arg is not None:
        if noise_model_arg != "-":
            noise_resolved = open_path(noise_model_arg)
        try:
            if noise_model_arg == "-":
                raw_model = sys.stdin.buffer.read()
            else:
                with open(noise_resolved, "rb") as handle:
                    raw_model = handle.read()
        except OSError as exc:
            return (
                None,
                None,
                _error_payload(
                    "io_error", f"cannot read noise model {noise_model_arg!r}: {exc.strerror or exc}"
                ),
                1,
            )
        if noise_model_arg != "-":
            register(noise_model_arg)
        try:
            model_text = raw_model.decode("utf-8")
        except UnicodeDecodeError as exc:
            return (
                None,
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
            return None, None, _error_payload("noise_model_error", str(exc)), 2

    try:
        program = parse(source)
    except ParseError as exc:
        return None, None, _error_payload("parse_error", exc.message, exc.line, exc.column), 2
    except ValidationError as exc:
        return None, None, _error_payload("validation_error", exc.message, exc.line, exc.column), 2

    return program, noise_model, None, 0


def _compute_simulation(
    program: Program,
    noise_model: dict[str, float] | None,
    shots: int,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None, int]:
    """Evolve a prepared circuit and sample it (the ``simulate`` payload).

    Companion to :func:`_prepare_simulation`; reading and parsing happen
    there, size checks and evolution here, so an output conflict can be
    refused between the two without running a simulation.
    """
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


def _simulate(
    source_arg: str,
    shots: int,
    seed: int,
    noise_model_arg: str | None,
    ctx: _RunContext | None = None,
) -> int:
    program, noise_model, error, exit_code = _prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        _emit_error_payload(error)
        return exit_code
    if ctx is not None and (code := _guard_output_conflict(ctx)) is not None:
        return code
    assert program is not None
    result, error, exit_code = _compute_simulation(program, noise_model, shots, seed)
    if error is not None:
        _emit_error_payload(error)
        return exit_code
    assert result is not None
    return _finish(ctx, result, 0)


# --------------------------------------------------------------- probabilities

# Probabilities are reported in full; entries this close to 0 are omitted
# and entries this close to 1 are reported as exactly 1. Reported values
# sum to within 1e-12 of 1.
_PROBABILITY_ZERO_TOLERANCE = 1e-15
_PROBABILITY_ONE_TOLERANCE = 1e-15


def _snap_probability(value: float) -> float | None:
    """Snap one probability for deterministic output.

    Returns ``None`` when the entry is omitted (magnitude at most
    ``1e-15``), ``1.0`` when it is within ``1e-15`` of 1, and otherwise a
    finite ``float`` clamped to ``(0, 1)`` so simulation round-off can
    never produce an out-of-range value.
    """
    if abs(value) <= _PROBABILITY_ZERO_TOLERANCE:
        return None
    if abs(1.0 - value) <= _PROBABILITY_ONE_TOLERANCE:
        return 1.0
    return min(1.0, max(0.0, float(value)))


def _probability_payload(
    program: Program,
    basis_probabilities: list[float],
    noise_model: dict[str, float] | None,
) -> dict[str, object]:
    """Build the deterministic ``probabilities`` success payload.

    The aggregated classical distribution is renormalized by its total
    first, mirroring the normalization the sampling path applies to basis
    probabilities, so simulation round-off (notably density-matrix drift)
    cannot move the reported sum away from 1.
    """
    aggregated = measurement_probabilities(program, basis_probabilities)
    total = math.fsum(aggregated.values())
    probabilities: dict[str, float] = {}
    for key, value in aggregated.items():
        normalized = value / total if total else value
        snapped = _snap_probability(normalized)
        if snapped is not None:
            probabilities[key] = snapped

    if noise_model is None:
        return {
            "schema_version": 1,
            "num_qubits": program.num_qubits,
            "num_clbits": program.num_clbits,
            "probabilities": probabilities,
        }
    return {
        "schema_version": 2,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "noise_model": noise_model,
        "probabilities": probabilities,
    }


def _probabilities(
    source_arg: str, noise_model_arg: str | None, ctx: _RunContext | None = None
) -> int:
    program, noise_model, error, exit_code = _prepare_simulation(source_arg, noise_model_arg, ctx=ctx)
    if error is not None:
        _emit_error_payload(error)
        return exit_code
    assert program is not None

    # Refuse an input-overwriting export before any state is evolved.
    if ctx is not None and (code := _guard_output_conflict(ctx)) is not None:
        return code

    if noise_model is None:
        # Register sizes are already capped at 20 by the parser, matching
        # the state-vector limit enforced for ``simulate``.
        state = simulate_state_vector(program)
        basis_probabilities = [abs(amplitude) ** 2 for amplitude in state]
    else:
        if program.num_qubits > MAX_NOISE_QUBITS:
            _emit_error(
                "simulation_error",
                f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
                f"got {program.num_qubits}",
            )
            return 3
        basis_probabilities = simulate_density_matrix(program, noise_model)

    result = _probability_payload(program, basis_probabilities, noise_model)
    return _finish(ctx, result, 0)


# -------------------------------------------------------------- verify-samples

VERIFY_SCHEMA_VERSION = 1
_SAMPLES_ROOT_KEYS = ("schema_version", "counts")
DEFAULT_VERIFY_TOLERANCE = 0.05


class _SampleInputError(Exception):
    """The samples document is structurally invalid (``sample_input_error``)."""


def _reject_samples_constant(value: str) -> None:
    raise _SampleInputError(f"samples document contains non-JSON constant {value!r}")


def _samples_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _SampleInputError(f"duplicate key {key!r} in samples document")
        result[key] = value
    return result


def _tolerance_value(value: str) -> float:
    """Parse ``--tolerance``: a finite JSON-style number in ``[0, 1]``."""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a finite number between 0 and 1, got {value!r}"
        ) from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number between 0 and 1, got {value!r}"
        )
    return parsed


def _validate_samples_document(
    data: object, num_clbits: int
) -> tuple[dict[str, int], int]:
    """Validate the parsed samples document against the classical register.

    Returns the raw counts and their total (the shot count). Raises
    :class:`_SampleInputError` on any structural, type, range or key-width
    problem. Validation is purely structural: it runs before any state is
    evolved.
    """
    if not isinstance(data, dict):
        raise _SampleInputError("samples document must be a JSON object")
    for key in data:
        if key not in _SAMPLES_ROOT_KEYS:
            raise _SampleInputError(f"unknown samples document key {key!r}")

    if "schema_version" not in data:
        raise _SampleInputError("samples document is missing required key 'schema_version'")
    schema_version = data["schema_version"]
    if not _is_json_int(schema_version) or schema_version != VERIFY_SCHEMA_VERSION:
        raise _SampleInputError(
            f"samples document schema_version must be {VERIFY_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )

    if "counts" not in data:
        raise _SampleInputError("samples document is missing required key 'counts'")
    counts = data["counts"]
    if not isinstance(counts, dict) or not counts:
        raise _SampleInputError("samples document 'counts' must be a non-empty JSON object")

    total = 0
    for key, value in counts.items():
        if not isinstance(key, str) or len(key) != num_clbits or any(bit not in "01" for bit in key):
            raise _SampleInputError(
                f"samples document 'counts' key {key!r} is not a "
                f"{num_clbits}-bit measurement string"
            )
        if not _is_json_int(value) or value <= 0:
            raise _SampleInputError(
                "samples document 'counts' values must be non-boolean positive integers"
            )
        total += value
    return counts, total


def _read_samples_document(
    samples_arg: str, ctx: _RunContext | None
) -> tuple[str | None, int | None]:
    """Read the UTF-8 samples JSON document from a path or ``-`` (stdin).

    Returns ``(text, None)`` on success or ``(None, exit_code)`` after a
    single ``io_error`` line tagged with ``input`` ``"samples"``.
    """
    try:
        if samples_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(samples_arg, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        _emit_error(
            "io_error",
            f"cannot read samples {samples_arg!r}: {exc.strerror or exc}",
            side="samples",
        )
        return None, 1

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error(
            "io_error",
            f"samples {samples_arg!r} is not valid UTF-8: {exc}",
            side="samples",
        )
        return None, 1

    if ctx is not None and samples_arg != "-":
        ctx.register_input(samples_arg)
    return text, None


def _snap_unit_value(value: float) -> float:
    """Snap one ``[0, 1]`` value for deterministic output.

    Mirrors the probability conventions: magnitude at most ``1e-15`` is
    reported as ``0.0`` (never negative zero), within ``1e-15`` of 1 as
    ``1.0``, otherwise a finite float clamped to ``[0, 1]``.
    """
    if abs(value) <= _PROBABILITY_ZERO_TOLERANCE:
        return 0.0
    if abs(1.0 - value) <= _PROBABILITY_ONE_TOLERANCE:
        return 1.0
    return min(1.0, max(0.0, float(value)))


def _verify_samples(
    source_arg: str,
    samples_arg: str,
    noise_model_arg: str | None,
    tolerance: float,
    ctx: _RunContext | None = None,
) -> int:
    # Three inputs may request standard input; at most one may be '-'. The
    # conflict is detected before any input is read.
    stdin_args = [source_arg, samples_arg]
    if noise_model_arg is not None:
        stdin_args.append(noise_model_arg)
    if sum(value == "-" for value in stdin_args) > 1:
        _emit_error(
            "verification_error",
            "at most one of the circuit source, samples document and noise model "
            "can be read from standard input",
        )
        return 2

    # Circuit and noise model use the exact simulate/probabilities path
    # (reading, UTF-8, parsing, validation, model normalization, ordering).
    program, noise_model, error, exit_code = _prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        _emit_error_payload(error)
        return exit_code
    assert program is not None

    samples_text, read_code = _read_samples_document(samples_arg, ctx)
    if samples_text is None:
        return read_code if read_code is not None else 1

    # The samples document is fully validated against the parsed classical
    # register width before any state is evolved.
    try:
        data = json.loads(
            samples_text,
            object_pairs_hook=_samples_pairs,
            parse_constant=_reject_samples_constant,
        )
        counts, shots = _validate_samples_document(data, program.num_clbits)
    except (json.JSONDecodeError, _SampleInputError) as exc:
        message = (
            f"samples document is not valid JSON: {exc}"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        _emit_error("sample_input_error", message)
        return 2

    # Refuse an input-overwriting export before any state is evolved.
    if ctx is not None and (code := _guard_output_conflict(ctx)) is not None:
        return code

    if noise_model is None:
        # The parser already caps the register at the 20-qubit state-vector
        # limit.
        state = simulate_state_vector(program)
        basis_probabilities = [abs(amplitude) ** 2 for amplitude in state]
    else:
        if program.num_qubits > MAX_NOISE_QUBITS:
            _emit_error(
                "simulation_error",
                f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
                f"got {program.num_qubits}",
            )
            return 3
        basis_probabilities = simulate_density_matrix(program, noise_model)

    # The theoretical distribution is built by the exact probabilities path
    # (measurement aggregation, renormalization, near-zero omission and
    # near-one snapping).
    expected_probabilities: dict[str, float] = _probability_payload(
        program, basis_probabilities, noise_model
    )["probabilities"]

    # Counts normalize to empirical probabilities; outcomes absent from the
    # counts are treated as zero on the union below.
    observed_probabilities = {
        key: _snap_unit_value(counts[key] / shots) for key in sorted(counts)
    }

    all_keys = sorted(set(expected_probabilities) | set(observed_probabilities))
    distance = 0.5 * math.fsum(
        abs(expected_probabilities.get(key, 0.0) - observed_probabilities.get(key, 0.0))
        for key in all_keys
    )
    distance = _snap_unit_value(distance)

    result: dict[str, object] = {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "shots": shots,
        "noise_model": noise_model,
        "tolerance": tolerance,
        "total_variation_distance": distance,
        "accepted": distance <= tolerance,
        "expected_probabilities": expected_probabilities,
        "observed_probabilities": observed_probabilities,
    }
    # A report is produced either way; only the exit code distinguishes them.
    return _finish(ctx, result, 0 if distance <= tolerance else 3)


# --------------------------------------------------------------- expectation


def _read_observables_document(
    observables_arg: str, ctx: _RunContext | None
) -> tuple[str | None, int | None]:
    """Read the UTF-8 observables JSON document from a path or ``-``.

    Returns ``(text, None)`` on success or ``(None, exit_code)`` after a
    single ``io_error`` line tagged with ``input`` ``"observables"``.
    """
    try:
        if observables_arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(observables_arg, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        _emit_error(
            "io_error",
            f"cannot read observables {observables_arg!r}: {exc.strerror or exc}",
            side="observables",
        )
        return None, 1

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error(
            "io_error",
            f"observables {observables_arg!r} is not valid UTF-8: {exc}",
            side="observables",
        )
        return None, 1

    if ctx is not None and observables_arg != "-":
        ctx.register_input(observables_arg)
    return text, None


def _expectation(
    source_arg: str,
    observables_arg: str,
    noise_model_arg: str | None,
    ctx: _RunContext | None = None,
) -> int:
    # Three inputs may request standard input; at most one may be '-'. The
    # conflict is detected before any input is read.
    stdin_args = [source_arg, observables_arg]
    if noise_model_arg is not None:
        stdin_args.append(noise_model_arg)
    if sum(value == "-" for value in stdin_args) > 1:
        _emit_error(
            "expectation_error",
            "at most one of the circuit source, observables document and noise model "
            "can be read from standard input",
        )
        return 2

    # Circuit and noise model use the exact simulate/probabilities path
    # (reading, UTF-8, parsing, validation, model normalization).
    program, noise_model, error, exit_code = _prepare_simulation(
        source_arg, noise_model_arg, ctx=ctx
    )
    if error is not None:
        _emit_error_payload(error)
        return exit_code
    assert program is not None

    observables_text, read_code = _read_observables_document(observables_arg, ctx)
    if observables_text is None:
        return read_code if read_code is not None else 1

    # The observables document is fully validated against the parsed
    # register size before any state is evolved.
    try:
        observables = parse_observables(observables_text, program.num_qubits)
    except ObservableError as exc:
        _emit_error("observable_error", str(exc))
        return 2

    # Refuse an input-overwriting export before any state is evolved.
    if ctx is not None and (code := _guard_output_conflict(ctx)) is not None:
        return code

    if noise_model is None:
        # The parser already caps the register at the 20-qubit
        # state-vector limit.
        state = simulate_state_vector(program)
        results = [
            {
                "id": observable_id,
                "expectation": snap_expectation(state_vector_expectation(state, operators)),
            }
            for observable_id, operators in observables
        ]
        result: dict[str, object] = {
            "schema_version": 1,
            "num_qubits": program.num_qubits,
            "noise_model": None,
            "results": results,
        }
    else:
        if program.num_qubits > MAX_NOISE_QUBITS:
            _emit_error(
                "simulation_error",
                f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
                f"got {program.num_qubits}",
            )
            return 3
        rho = normalized_density_matrix(evolve_density_matrix(program, noise_model))
        results = [
            {
                "id": observable_id,
                "expectation": snap_expectation(density_matrix_expectation(rho, operators)),
            }
            for observable_id, operators in observables
        ]
        result = {
            "schema_version": 1,
            "num_qubits": program.num_qubits,
            "noise_model": noise_model,
            "results": results,
        }
    return _finish(ctx, result, 0)


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


def _batch_simulate(manifest_arg: str, ctx: _RunContext | None = None) -> int:
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
    if ctx is not None and manifest_arg != "-":
        ctx.register_input(manifest_arg)

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

    # Register every path the validated manifest names, then guard against
    # an input-overwriting export before any task file is opened or any
    # simulation runs. References are registered from the manifest itself,
    # so a missing/unreadable task is still covered.
    if ctx is not None:
        _register_manifest_references(ctx, jobs, base_dir)
        if (code := _guard_output_conflict(ctx)) is not None:
            return code

    # Read, decode, parse and validate every task first (task failures are
    # embedded), then evolve each prepared circuit in manifest order.
    prepared: list[
        tuple[dict[str, object], Program | None, dict[str, float] | None, dict[str, object] | None]
    ] = []
    for job in jobs:
        program, noise_model, error, _exit_code = _prepare_simulation(
            job["source"],
            job.get("noise_model"),
            base_dir,
            ctx,
        )
        prepared.append((job, program, noise_model, error))

    results: list[dict[str, object]] = []
    succeeded = 0
    for job, program, noise_model, error in prepared:
        output: dict[str, object] | None = None
        if error is None:
            assert program is not None
            output, error, _exit_code = _compute_simulation(
                program,
                noise_model,
                job.get("shots", 1024),
                job.get("seed", 0),
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
    return _finish(ctx, summary, 0 if succeeded == len(jobs) else 3)


# --------------------------------------------------------------- reconcile

_BASELINE_ROOT_KEYS = ("schema_version", "job_count", "succeeded", "failed", "results")
_BASELINE_RESULT_KEYS = ("id", "status", "output", "error")
_OUTPUT_V1_KEYS = ("schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts")
_OUTPUT_V2_KEYS = (
    "schema_version",
    "shots",
    "seed",
    "num_qubits",
    "num_clbits",
    "noise_model",
    "counts",
)
_NOISE_CHANNELS = ("amplitude_damping", "phase_damping", "bit_flip", "depolarizing")
_ERROR_KEYS = ("error", "message", "line", "column")
_RECONCILE_ERROR_NAMES = (
    "io_error",
    "parse_error",
    "validation_error",
    "noise_model_error",
    "simulation_error",
)
# Task errors that carry a source position; every other batch-simulate task
# error is exactly ``{"error", "message"}``.
_POSITIONED_ERROR_NAMES = ("parse_error", "validation_error")
_RECONCILE_REASONS = ("identical", "status_mismatch", "output_mismatch", "error_mismatch")


class _ReconcileInputError(Exception):
    """One of the two reconcile inputs is invalid (``reconcile_input_error``)."""


def _reject_baseline_constant(value: str) -> None:
    raise _ReconcileInputError(f"baseline contains non-JSON constant {value!r}")


def _pairs_no_dupes(pairs: list[tuple[str, object]], label: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _ReconcileInputError(f"duplicate key {key!r} in {label}")
        result[key] = value
    return result


def _require_int(value: object, description: str) -> int:
    if not _is_json_int(value):
        raise _ReconcileInputError(f"{description} must be an integer")
    return value


def _check_finite_numbers(value: object, description: str) -> None:
    """Reject non-finite floats anywhere in an already-parsed JSON payload.

    Integers, strings, booleans and ``None`` are unproblematic; floats are
    only reachable through explicit decimal literals (constants such as
    ``NaN`` are refused at parse time) but must still be finite.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _ReconcileInputError(f"{description} is not a finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_numbers(item, f"{description}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_numbers(item, f"{description}[{index}]")


def _validate_baseline_output(output: object, where: str) -> dict[str, object]:
    if not isinstance(output, dict):
        raise _ReconcileInputError(f"{where} must be a JSON object")
    for key in output:
        if key not in _OUTPUT_V2_KEYS:
            raise _ReconcileInputError(f"{where} has unknown key {key!r}")

    for required in _OUTPUT_V1_KEYS:
        if required not in output:
            raise _ReconcileInputError(f"{where} is missing required key {required!r}")

    schema_version = _require_int(output["schema_version"], f"{where} 'schema_version'")
    shots = _require_int(output["shots"], f"{where} 'shots'")
    seed = _require_int(output["seed"], f"{where} 'seed'")
    num_qubits = _require_int(output["num_qubits"], f"{where} 'num_qubits'")
    num_clbits = _require_int(output["num_clbits"], f"{where} 'num_clbits'")

    if schema_version not in (1, 2):
        raise _ReconcileInputError(f"{where} 'schema_version' must be 1 or 2")
    if shots <= 0:
        raise _ReconcileInputError(f"{where} 'shots' must be a positive integer")
    if not 1 <= num_qubits <= 20:
        raise _ReconcileInputError(f"{where} 'num_qubits' must be between 1 and 20")
    if not 1 <= num_clbits <= 20:
        raise _ReconcileInputError(f"{where} 'num_clbits' must be between 1 and 20")

    counts = output["counts"]
    if not isinstance(counts, dict) or not counts:
        raise _ReconcileInputError(f"{where} 'counts' must be a non-empty JSON object")
    total = 0
    for key, value in counts.items():
        if not isinstance(key, str) or not key:
            raise _ReconcileInputError(f"{where} 'counts' keys must be non-empty strings")
        if not _is_json_int(value) or value <= 0:
            raise _ReconcileInputError(f"{where} 'counts' values must be positive integers")
        if len(key) != num_clbits or any(bit not in "01" for bit in key):
            raise _ReconcileInputError(
                f"{where} 'counts' key {key!r} is not a {num_clbits}-bit measurement string"
            )
        total += value
    if total != shots:
        raise _ReconcileInputError(f"{where} 'counts' must sum to shots ({shots}), got {total}")

    if schema_version == 2:
        if num_qubits > MAX_NOISE_QUBITS:
            raise _ReconcileInputError(
                f"{where} noisy output must have at most {MAX_NOISE_QUBITS} qubits"
            )
        if "noise_model" not in output:
            raise _ReconcileInputError(f"{where} is missing required key 'noise_model'")
        noise_model = output["noise_model"]
        if not isinstance(noise_model, dict) or not noise_model:
            raise _ReconcileInputError(f"{where} 'noise_model' must be a non-empty JSON object")
        for channel, probability in noise_model.items():
            if channel not in _NOISE_CHANNELS:
                raise _ReconcileInputError(f"{where} has unknown noise channel {channel!r}")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise _ReconcileInputError(f"{where} noise channel {channel!r} must be a number")
            if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
                raise _ReconcileInputError(
                    f"{where} noise channel {channel!r} probability must be between 0 and 1"
                )
    elif "noise_model" in output:
        raise _ReconcileInputError(f"{where} schema_version 1 output must not contain 'noise_model'")

    return output


def _validate_baseline_error(error: object, where: str) -> dict[str, object]:
    if not isinstance(error, dict):
        raise _ReconcileInputError(f"{where} must be a JSON object")
    for key in error:
        if key not in _ERROR_KEYS:
            raise _ReconcileInputError(f"{where} has unknown key {key!r}")

    name = error.get("error")
    if not isinstance(name, str) or name not in _RECONCILE_ERROR_NAMES:
        raise _ReconcileInputError(f"{where} 'error' must name a batch-simulate task error")
    if "message" not in error or not isinstance(error["message"], str):
        raise _ReconcileInputError(f"{where} is missing a string 'message'")

    if name in _POSITIONED_ERROR_NAMES:
        for position in ("line", "column"):
            if position not in error:
                raise _ReconcileInputError(f"{where} is missing required key {position!r}")
            value = error[position]
            if not _is_json_int(value) or value < 1:
                raise _ReconcileInputError(f"{where} {position!r} must be a positive integer")
    elif "line" in error or "column" in error:
        raise _ReconcileInputError(f"{where} {name!r} errors must not carry a source position")

    return error


def _validate_baseline(data: object, jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Validate the batch-result *data* against the manifest *jobs*.

    Self-consistency and correspondence (count, ids, order) are checked
    here, before any task is re-run. Raises :class:`_ReconcileInputError`.
    """
    if not isinstance(data, dict):
        raise _ReconcileInputError("baseline must be a JSON object")
    for key in data:
        if key not in _BASELINE_ROOT_KEYS:
            raise _ReconcileInputError(f"unknown baseline key {key!r}")
    if any(key not in data for key in _BASELINE_ROOT_KEYS):
        raise _ReconcileInputError("baseline is missing one or more required root keys")

    schema_version = _require_int(data["schema_version"], "baseline 'schema_version'")
    if schema_version != BATCH_SCHEMA_VERSION:
        raise _ReconcileInputError(
            f"baseline schema_version must be {BATCH_SCHEMA_VERSION}, got {schema_version!r}"
        )

    job_count = _require_int(data["job_count"], "baseline 'job_count'")
    succeeded = _require_int(data["succeeded"], "baseline 'succeeded'")
    failed = _require_int(data["failed"], "baseline 'failed'")
    if job_count < 0 or succeeded < 0 or failed < 0:
        raise _ReconcileInputError("baseline counters must be non-negative")
    if succeeded + failed != job_count:
        raise _ReconcileInputError("baseline 'succeeded' + 'failed' must equal 'job_count'")
    if job_count != len(jobs):
        raise _ReconcileInputError(
            f"baseline job_count {job_count} does not match manifest job count {len(jobs)}"
        )

    results = data["results"]
    if not isinstance(results, list):
        raise _ReconcileInputError("baseline 'results' must be an array")
    if len(results) != job_count:
        raise _ReconcileInputError("baseline 'results' length must equal 'job_count'")

    seen: set[str] = set()
    actual_succeeded = 0
    for index, entry in enumerate(results):
        where = f"baseline result[{index}]"
        if not isinstance(entry, dict):
            raise _ReconcileInputError(f"{where} must be a JSON object")
        for key in entry:
            if key not in _BASELINE_RESULT_KEYS:
                raise _ReconcileInputError(f"{where} has unknown key {key!r}")

        if "id" not in entry:
            raise _ReconcileInputError(f"{where} is missing required key 'id'")
        job_id = entry["id"]
        if not isinstance(job_id, str) or not job_id:
            raise _ReconcileInputError(f"{where} 'id' must be a non-empty string")
        if job_id in seen:
            raise _ReconcileInputError(f"duplicate baseline result id {job_id!r}")
        seen.add(job_id)
        if job_id != jobs[index]["id"]:
            raise _ReconcileInputError(
                f"{where} id {job_id!r} does not match manifest id {jobs[index]['id']!r}"
            )

        if "status" not in entry:
            raise _ReconcileInputError(f"{where} is missing required key 'status'")
        status = entry["status"]
        if status == "succeeded":
            if "output" not in entry:
                raise _ReconcileInputError(f"{where} succeeded result is missing 'output'")
            if "error" in entry:
                raise _ReconcileInputError(f"{where} succeeded result must not contain 'error'")
            _validate_baseline_output(entry["output"], f"{where} output")
            actual_succeeded += 1
        elif status == "failed":
            if "error" not in entry:
                raise _ReconcileInputError(f"{where} failed result is missing 'error'")
            if "output" in entry:
                raise _ReconcileInputError(f"{where} failed result must not contain 'output'")
            _validate_baseline_error(entry["error"], f"{where} error")
        else:
            raise _ReconcileInputError(f"{where} 'status' must be 'succeeded' or 'failed'")

        required_keys = (
            {"id", "status", "output"} if status == "succeeded" else {"id", "status", "error"}
        )
        if set(entry) != required_keys:
            raise _ReconcileInputError(f"{where} has unexpected keys for status {status!r}")

    if actual_succeeded != succeeded or len(results) - actual_succeeded != failed:
        raise _ReconcileInputError("baseline counters do not match result statuses")

    return results


def _read_reconcile_input(
    side: str, arg: str, ctx: _RunContext | None = None
) -> tuple[bytes, str | None, int | None]:
    """Read one reconcile input, returning ``(raw_bytes, base_dir, error_code)``.

    *side* is ``"manifest"`` or ``"baseline"`` and is reported on every
    failure via the ``input`` field. On failure the code is 1 for an
    unreadable file or invalid UTF-8, after emitting an ``io_error``.
    """
    try:
        if arg == "-":
            raw = sys.stdin.buffer.read()
            base_dir = os.getcwd()
        else:
            with open(arg, "rb") as handle:
                raw = handle.read()
            base_dir = os.path.dirname(os.path.abspath(arg))
    except OSError as exc:
        _emit_error(
            "io_error",
            f"cannot read {side} {arg!r}: {exc.strerror or exc}",
            side=side,
        )
        return b"", None, 1

    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error("io_error", f"{side} {arg!r} is not valid UTF-8: {exc}", side=side)
        return b"", None, 1

    if ctx is not None and arg != "-":
        ctx.register_input(arg)

    return raw, base_dir, None


def _json_equal(expected: object, actual: object) -> bool:
    """Compare parsed JSON ignoring object key order and whitespace.

    Array order and scalar values (including int-vs-float distinctions and
    numeric precision) are significant.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(_json_equal(expected[key], actual[key]) for key in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _json_equal(left, right) for left, right in zip(expected, actual)
        )
    return expected == actual and type(expected) is type(actual)


def _reconcile(manifest_arg: str, baseline_arg: str, ctx: _RunContext | None = None) -> int:
    if manifest_arg == "-" and baseline_arg == "-":
        # Standard input cannot serve both inputs; do not read it.
        _emit_error(
            "reconciliation_error",
            "manifest and baseline cannot both be read from standard input",
        )
        return 2

    # Read and decode both inputs fully (manifest first, then baseline) and
    # validate both before any task file is touched.
    raw_manifest, base_dir, code = _read_reconcile_input("manifest", manifest_arg, ctx)
    if code is not None:
        return code
    raw_baseline, _base_dir, code = _read_reconcile_input("baseline", baseline_arg, ctx)
    if code is not None:
        return code

    try:
        manifest_data = json.loads(
            raw_manifest.decode("utf-8"),
            object_pairs_hook=_manifest_pairs,
            parse_constant=_reject_manifest_constant,
        )
        jobs = _validate_batch_manifest(manifest_data)
    except (json.JSONDecodeError, _BatchManifestError) as exc:
        message = (
            f"batch manifest is not valid JSON: {exc}"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        _emit_error("reconcile_input_error", message)
        return 2

    try:
        baseline_data = json.loads(
            raw_baseline.decode("utf-8"),
            object_pairs_hook=lambda pairs: _pairs_no_dupes(pairs, "baseline"),
            parse_constant=_reject_baseline_constant,
        )
        _check_finite_numbers(baseline_data, "baseline")
        baseline_results = _validate_baseline(baseline_data, jobs)
    except (json.JSONDecodeError, _ReconcileInputError) as exc:
        message = (
            f"baseline is not valid JSON: {exc}"
            if isinstance(exc, json.JSONDecodeError)
            else str(exc)
        )
        _emit_error("reconcile_input_error", message)
        return 2

    # Register every path the validated manifest names (plus the manifest
    # and baseline already read), then guard against an input-overwriting
    # export before any task file is opened or any task is re-run.
    if ctx is not None:
        _register_manifest_references(ctx, jobs, base_dir)
        if (code := _guard_output_conflict(ctx)) is not None:
            return code

    # Prepare (read, parse, validate) every task first, then re-run them in
    # manifest order with the exact batch-simulate semantics.
    prepared: list[
        tuple[dict[str, object], Program | None, dict[str, float] | None, dict[str, object] | None]
    ] = []
    for job in jobs:
        program, noise_model, error, _exit_code = _prepare_simulation(
            job["source"],
            job.get("noise_model"),
            base_dir,
            ctx,
        )
        prepared.append((job, program, noise_model, error))

    # Re-run every task in manifest order with the exact batch-simulate
    # semantics (reading, parsing, noise, qubit limits, sampling, errors).
    entries: list[dict[str, object]] = []
    matched = 0
    for (job, program, noise_model, error), reference in zip(prepared, baseline_results):
        output: dict[str, object] | None = None
        if error is None:
            assert program is not None
            output, error, _exit_code = _compute_simulation(
                program,
                noise_model,
                job.get("shots", 1024),
                job.get("seed", 0),
            )
        current: dict[str, object] = (
            {"id": job["id"], "status": "succeeded", "output": output}
            if error is None
            else {"id": job["id"], "status": "failed", "error": error}
        )
        entry: dict[str, object] = {"id": job["id"]}
        if current["status"] != reference["status"]:
            entry["consistent"] = False
            entry["reason"] = "status_mismatch"
            entry["expected"] = reference
            entry["actual"] = current
            entries.append(entry)
            continue

        if current["status"] == "succeeded":
            equal = _json_equal(reference["output"], output)
            reason = "output_mismatch"
        else:
            equal = _json_equal(reference["error"], error)
            reason = "error_mismatch"

        if equal:
            entry["consistent"] = True
            entry["reason"] = "identical"
            matched += 1
        else:
            entry["consistent"] = False
            entry["reason"] = reason
            # Preserve the complete baseline and current task result.
            entry["expected"] = reference
            entry["actual"] = current
        entries.append(entry)

    mismatched = len(jobs) - matched
    report: dict[str, object] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "job_count": len(jobs),
        "matched": matched,
        "mismatched": mismatched,
        "consistent": mismatched == 0,
        "results": entries,
    }
    return _finish(ctx, report, 0 if mismatched == 0 else 3)


def _read_source(source_arg: str, ctx: _RunContext | None = None) -> str | None:
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
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error("io_error", f"source {source_arg!r} is not valid UTF-8: {exc}")
        return None

    if ctx is not None and source_arg != "-":
        ctx.register_input(source_arg)
    return source


def _parse_or_emit(source: str) -> tuple[Program | None, int | None]:
    """Parse/validate one circuit source, emitting the standard error line.

    Returns ``(program, None)`` on success or ``(None, exit_code)`` after a
    ``parse_error``/``validation_error`` line has been written.
    """
    try:
        program = parse(source)
    except ParseError as exc:
        _emit_error("parse_error", exc.message, exc.line, exc.column)
        return None, 2
    except ValidationError as exc:
        _emit_error("validation_error", exc.message, exc.line, exc.column)
        return None, 2
    return program, None


def _optimize(source_arg: str, ctx: _RunContext | None = None) -> int:
    source = _read_source(source_arg, ctx)
    if source is None:
        return 1

    program, code = _parse_or_emit(source)
    if program is None:
        return code if code is not None else 2
    if ctx is not None and (guard := _guard_output_conflict(ctx)) is not None:
        return guard

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
    return _finish(ctx, result, 0)


def _estimate(source_arg: str, mode: str, ctx: _RunContext | None = None) -> int:
    source = _read_source(source_arg, ctx)
    if source is None:
        return 1

    program, code = _parse_or_emit(source)
    if program is None:
        return code if code is not None else 2
    if ctx is not None and (guard := _guard_output_conflict(ctx)) is not None:
        return guard

    return _finish(ctx, estimate_program(program, mode), 0)


def _load_program(side: str, source_arg: str, ctx: _RunContext | None = None):
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

    if ctx is not None and source_arg != "-":
        ctx.register_input(source_arg)
    return program, None


def _load_comparison_program(
    side: str, source_arg: str, ctx: _RunContext | None = None
):
    """Load one side of an equivalence comparison, enforcing the size cap.

    Like :func:`_load_program`, but additionally rejects circuits larger
    than ``MAX_EQUIVALENCE_QUBITS`` with exit code 3.
    """
    program, error_code = _load_program(side, source_arg, ctx)
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


def _equivalent(left_arg: str, right_arg: str, ctx: _RunContext | None = None) -> int:
    if left_arg == "-" and right_arg == "-":
        # Standard input cannot serve both sides; do not read it.
        _emit_error(
            "comparison_error",
            "left and right inputs cannot both be read from standard input",
        )
        return 2

    # Fully validate each side (read, parse, semantics, size) in LEFT, RIGHT
    # order, reporting only the first failure.
    left, error_code = _load_comparison_program("left", left_arg, ctx)
    if left is None:
        return error_code
    right, error_code = _load_comparison_program("right", right_arg, ctx)
    if right is None:
        return error_code

    if ctx is not None and (guard := _guard_output_conflict(ctx)) is not None:
        return guard

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
        return _finish(ctx, result, 0)

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
    return _finish(ctx, result, 0)


def _load_metrics_noise_model(
    side: str, model_arg: str, ctx: _RunContext | None = None
) -> tuple[dict[str, float] | None, int | None]:
    """Read and validate one side's noise model for ``state-metrics``.

    Returns ``(model, None)`` on success or ``(None, exit_code)`` after a
    single side-tagged error line. Mirrors the ``simulate`` model handling:
    an unreadable file is an ``io_error`` (code 1), invalid UTF-8 or
    non-compliant content is a ``noise_model_error`` (code 2); *side* is
    ``"left"`` or ``"right"`` and reported as ``input``.
    """
    try:
        if model_arg == "-":
            raw_model = sys.stdin.buffer.read()
        else:
            with open(model_arg, "rb") as handle:
                raw_model = handle.read()
    except OSError as exc:
        _emit_error(
            "io_error",
            f"cannot read noise model {model_arg!r}: {exc.strerror or exc}",
            side=side,
        )
        return None, 1

    if ctx is not None and model_arg != "-":
        ctx.register_input(model_arg)

    try:
        model_text = raw_model.decode("utf-8")
    except UnicodeDecodeError as exc:
        _emit_error(
            "noise_model_error",
            f"noise model {model_arg!r} is not valid UTF-8: {exc}",
            side=side,
        )
        return None, 2
    try:
        return parse_noise_model(model_text), None
    except NoiseModelError as exc:
        _emit_error("noise_model_error", str(exc), side=side)
        return None, 2


def _state_metrics(
    left_arg: str,
    right_arg: str,
    left_noise_arg: str | None,
    right_noise_arg: str | None,
    ctx: _RunContext | None = None,
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
        _emit_error("metrics_error", message)
        return 2

    # Fully validate in left-source, left-model, right-source, right-model
    # order, reporting only the first failure (each tagged with its side).
    left, error_code = _load_program("left", left_arg, ctx)
    if left is None:
        return error_code
    left_model: dict[str, float] | None = None
    if left_noise_arg is not None:
        left_model, error_code = _load_metrics_noise_model("left", left_noise_arg, ctx)
        if left_model is None:
            return error_code

    right, error_code = _load_program("right", right_arg, ctx)
    if right is None:
        return error_code
    right_model: dict[str, float] | None = None
    if right_noise_arg is not None:
        right_model, error_code = _load_metrics_noise_model("right", right_noise_arg, ctx)
        if right_model is None:
            return error_code

    if ctx is not None and (guard := _guard_output_conflict(ctx)) is not None:
        return guard

    # A noisy side evolves a full density matrix (at most MAX_NOISE_QUBITS);
    # a model-less side keeps the 20-qubit state-vector path.
    if left_model is not None and left.num_qubits > MAX_NOISE_QUBITS:
        _emit_error(
            "simulation_error",
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"left input has {left.num_qubits}",
            side="left",
        )
        return 3
    if right_model is not None and right.num_qubits > MAX_NOISE_QUBITS:
        _emit_error(
            "simulation_error",
            f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
            f"right input has {right.num_qubits}",
            side="right",
        )
        return 3

    noisy = left_model is not None or right_model is not None

    left_state: list[complex] | None = None
    right_state: list[complex] | None = None
    left_rho: list[list[complex]] | None = None
    right_rho: list[list[complex]] | None = None
    if left_model is None:
        left_state = simulate_state_vector(left)
    else:
        left_rho = normalized_density_matrix(evolve_density_matrix(left, left_model))
    if right_model is None:
        right_state = simulate_state_vector(right)
    else:
        right_rho = normalized_density_matrix(evolve_density_matrix(right, right_model))

    same_qubits = left.num_qubits == right.num_qubits
    reason = "compared" if same_qubits else "qubit_count_mismatch"

    if not same_qubits:
        fidelity: float | None = None
    elif left_rho is None and right_rho is None:
        fidelity = state_fidelity(left_state, right_state)
    elif left_rho is not None and right_rho is not None:
        fidelity = density_fidelity(left_rho, right_rho)
    elif left_rho is not None:
        fidelity = pure_density_fidelity(right_state, left_rho)
    else:
        fidelity = pure_density_fidelity(left_state, right_rho)

    left_purity = 1.0 if left_rho is None else density_purity(left_rho)
    right_purity = 1.0 if right_rho is None else density_purity(right_rho)
    left_entropy = (
        single_qubit_entropies(left_state, left.num_qubits)
        if left_rho is None
        else density_single_qubit_entropies(left_rho, left.num_qubits)
    )
    right_entropy = (
        single_qubit_entropies(right_state, right.num_qubits)
        if right_rho is None
        else density_single_qubit_entropies(right_rho, right.num_qubits)
    )

    if not noisy:
        # No noise model on either side: keep the schema_version 1 payload
        # byte-for-byte identical to the historical command.
        result: dict[str, object] = {
            "schema_version": 1,
            "left_num_qubits": left.num_qubits,
            "right_num_qubits": right.num_qubits,
            "reason": reason,
            "fidelity": fidelity,
            "left_single_qubit_entropy": left_entropy,
            "right_single_qubit_entropy": right_entropy,
        }
    else:
        result = {
            "schema_version": 2,
            "left_num_qubits": left.num_qubits,
            "right_num_qubits": right.num_qubits,
            "reason": reason,
            "left_noise_model": left_model,
            "right_noise_model": right_model,
            "fidelity": fidelity,
            "left_purity": left_purity,
            "right_purity": right_purity,
            "left_single_qubit_entropy": left_entropy,
            "right_single_qubit_entropy": right_entropy,
        }
    return _finish(ctx, result, 0)


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

    verify_parser = sub.add_parser(
        "verify-samples",
        help="compare external measurement counts against exact theoretical probabilities",
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
        type=_tolerance_value,
        default=DEFAULT_VERIFY_TOLERANCE,
        help="maximum accepted total variation distance in [0, 1] (default: 0.05)",
    )
    verify_parser.add_argument(
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

    ctx = _RunContext(args.output) if getattr(args, "output", None) is not None else None

    if args.command == "simulate":
        return _simulate(args.source, args.shots, args.seed, args.noise_model, ctx)

    if args.command == "probabilities":
        return _probabilities(args.source, args.noise_model, ctx)

    if args.command == "verify-samples":
        return _verify_samples(
            args.source,
            args.samples,
            args.noise_model,
            args.tolerance,
            ctx,
        )

    if args.command == "expectation":
        return _expectation(args.source, args.observables, args.noise_model, ctx)

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
