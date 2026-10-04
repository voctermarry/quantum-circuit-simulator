"""Input preparation for the command layer.

This is the only layer that touches the file system and standard input.
Each helper reads and decodes one input, then parses and validates it,
returning either the prepared value or a
:class:`~quantum_circuit.errors.CommandFailure` whose name, message,
positions, ``input`` side and exit code match the historical commands
exactly. Helpers never write standard streams, so the validation order
(read errors, UTF-8 errors, parse errors, semantic errors) is expressed as
plain data flow and remains unit-testable.

Paths inside batch manifests are resolved relative to *base_dir* while
error messages keep quoting the user-supplied path verbatim, exactly as a
standalone invocation would.
"""

from __future__ import annotations

import os
import sys

from .errors import CommandFailure
from .equivalence import MAX_EQUIVALENCE_QUBITS
from .noise import NoiseModelError, parse_noise_model
from .openqasm import ParseError, Program, ValidationError, parse


def _resolve(path: str, base_dir: str | None) -> str:
    if base_dir is not None and not os.path.isabs(path):
        return os.path.normpath(os.path.join(base_dir, path))
    return path


def _read_bytes(path: str, base_dir: str | None) -> bytes:
    """Read raw bytes from *path* (resolved against *base_dir*) or stdin."""
    if path == "-":
        return sys.stdin.buffer.read()
    with open(_resolve(path, base_dir), "rb") as handle:
        return handle.read()


def _read_text(
    path: str,
    base_dir: str | None,
    error_name: str,
    noun: str,
    *,
    utf8_error: str | None = None,
    side: str | None = None,
) -> tuple[str | None, CommandFailure | None]:
    """Read and UTF-8 decode one document.

    *noun* is how the input is quoted in messages (e.g. ``"source"``,
    ``"samples"``, ``"noise model"``). An unreadable file is *error_name*
    (typically ``io_error``); an invalid UTF-8 document is *utf8_error*
    when given (noise models use ``noise_model_error``), otherwise the
    same *error_name*.
    """
    try:
        data = _read_bytes(path, base_dir)
    except OSError as exc:
        return None, CommandFailure(
            error_name, f"cannot read {noun} {path!r}: {exc.strerror or exc}", 1, side=side
        )
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, CommandFailure(
            utf8_error or error_name,
            f"{noun} {path!r} is not valid UTF-8: {exc}",
            2 if utf8_error is not None else 1,
            side=side,
        )


def prepare_simulation(
    source_arg: str,
    noise_model_arg: str | None,
    base_dir: str | None = None,
    register=None,
) -> tuple[Program | None, dict[str, float] | None, CommandFailure | None]:
    """Read, decode, parse and validate one simulation input.

    Returns ``(program, noise_model, failure)``: on success *failure* is
    ``None`` (with *noise_model* ``None`` on the state-vector path); on
    failure *program* is ``None``. Reading (files, UTF-8, ``-`` for stdin,
    relative paths) and error semantics match ``simulate`` for both
    ``simulate`` and ``probabilities``. No qubit-limit or simulation error
    is raised here; callers apply their own limits.

    When *base_dir* is given (the batch case), relative paths are opened
    relative to it while error messages keep quoting *source_arg* and
    *noise_model_arg* verbatim.
    """
    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        return (
            None,
            None,
            CommandFailure(
                "noise_model_error",
                "circuit source and noise model cannot both be read from standard input",
                2,
            ),
        )

    try:
        data = _read_bytes(source_arg, base_dir)
    except OSError as exc:
        return None, None, CommandFailure(
            "io_error", f"cannot read source {source_arg!r}: {exc.strerror or exc}", 1
        )
    if register is not None:
        register(source_arg, base_dir)

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, None, CommandFailure(
            "io_error", f"source {source_arg!r} is not valid UTF-8: {exc}", 1
        )

    noise_model: dict[str, float] | None = None
    if noise_model_arg is not None:
        # An unreadable model is io_error; invalid UTF-8 or bad content is
        # noise_model_error (the historical two-tier behavior).
        try:
            raw_model = _read_bytes(noise_model_arg, base_dir)
        except OSError as exc:
            return (
                None,
                None,
                CommandFailure(
                    "io_error",
                    f"cannot read noise model {noise_model_arg!r}: {exc.strerror or exc}",
                    1,
                ),
            )
        if register is not None:
            register(noise_model_arg, base_dir)
        try:
            model_text = raw_model.decode("utf-8")
        except UnicodeDecodeError as exc:
            return (
                None,
                None,
                CommandFailure(
                    "noise_model_error",
                    f"noise model {noise_model_arg!r} is not valid UTF-8: {exc}",
                    2,
                ),
            )
        try:
            noise_model = parse_noise_model(model_text)
        except NoiseModelError as exc:
            return None, None, CommandFailure("noise_model_error", str(exc), 2)

    try:
        program = parse(source)
    except ParseError as exc:
        return None, None, CommandFailure(
            "parse_error", exc.message, 2, line=exc.line, column=exc.column
        )
    except ValidationError as exc:
        return None, None, CommandFailure(
            "validation_error", exc.message, 2, line=exc.line, column=exc.column
        )

    return program, noise_model, None


def read_document(
    side: str, noun: str, arg: str, register=None
) -> tuple[str | None, CommandFailure | None]:
    """Read a UTF-8 auxiliary JSON document (samples/observables) or stdin.

    *side* is reported verbatim in the ``input`` field on every failure.
    """
    text, failure = _read_text(arg, None, "io_error", noun, side=side)
    if failure is None and register is not None:
        register(arg, None)
    return text, failure


def read_source(
    source_arg: str, register=None
) -> tuple[str | None, CommandFailure | None]:
    """Read a UTF-8 circuit source from a path or ``-`` (stdin)."""
    text, failure = _read_text(source_arg, None, "io_error", "source")
    if failure is None and register is not None:
        register(source_arg, None)
    return text, failure


def parse_program(source: str) -> tuple[Program | None, CommandFailure | None]:
    """Parse/validate one circuit source into a program or a failure."""
    try:
        program = parse(source)
    except ParseError as exc:
        return None, CommandFailure(
            "parse_error", exc.message, 2, line=exc.line, column=exc.column
        )
    except ValidationError as exc:
        return None, CommandFailure(
            "validation_error", exc.message, 2, line=exc.line, column=exc.column
        )
    return program, None


def load_program(
    side: str, source_arg: str, register=None
) -> tuple[Program | None, CommandFailure | None]:
    """Read, parse and validate one side of a two-circuit comparison.

    *side* (``"left"``/``"right"``) is reported as ``input`` on every
    failure. Returns the parsed :class:`Program` or a failure.
    """
    source, failure = _read_text(source_arg, None, "io_error", "source", side=side)
    if failure is not None:
        return None, failure
    program, failure = parse_program(source)
    if program is None:
        # Tag the parse/validation failure with the comparison side.
        assert failure is not None
        return None, CommandFailure(
            failure.error,
            failure.message,
            failure.exit_code,
            side=side,
            line=failure.line,
            column=failure.column,
        )
    if register is not None:
        register(source_arg, None)
    return program, None


def load_comparison_program(
    side: str, source_arg: str, register=None
) -> tuple[Program | None, CommandFailure | None]:
    """Load one equivalence side, additionally enforcing the 8-qubit cap."""
    program, failure = load_program(side, source_arg, register)
    if program is None:
        return None, failure
    if program.num_qubits > MAX_EQUIVALENCE_QUBITS:
        return None, CommandFailure(
            "simulation_error",
            f"equivalence comparison supports at most {MAX_EQUIVALENCE_QUBITS} qubits, "
            f"{side} input has {program.num_qubits}",
            3,
            side=side,
        )
    return program, None


def load_metrics_noise_model(
    side: str, model_arg: str, register=None
) -> tuple[dict[str, float] | None, CommandFailure | None]:
    """Read and validate one side's ``state-metrics`` noise model.

    Mirrors the ``simulate`` model handling but tags every failure with
    *side* (``"left"``/``"right"``).
    """
    try:
        raw_model = _read_bytes(model_arg, None)
    except OSError as exc:
        return None, CommandFailure(
            "io_error",
            f"cannot read noise model {model_arg!r}: {exc.strerror or exc}",
            1,
            side=side,
        )
    if register is not None:
        register(model_arg, None)
    try:
        model_text = raw_model.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, CommandFailure(
            "noise_model_error",
            f"noise model {model_arg!r} is not valid UTF-8: {exc}",
            2,
            side=side,
        )
    try:
        return parse_noise_model(model_text), None
    except NoiseModelError as exc:
        return None, CommandFailure("noise_model_error", str(exc), 2, side=side)


def read_manifest_or_baseline(
    side: str, arg: str, register=None
) -> tuple[bytes | None, str | None, CommandFailure | None]:
    """Read one reconcile-style input, returning raw bytes and base dir.

    *side* is ``"manifest"`` or ``"baseline"``. Standard input resolves
    *base_dir* to the current working directory; a path resolves it to
    the file's own directory. Bytes are UTF-8 checked but returned raw so
    the caller owns JSON error naming.
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
        return None, None, CommandFailure(
            "io_error", f"cannot read {side} {arg!r}: {exc.strerror or exc}", 1, side=side
        )
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, None, CommandFailure(
            "io_error", f"{side} {arg!r} is not valid UTF-8: {exc}", 1, side=side
        )
    if register is not None and arg != "-":
        register(arg, None)
    return raw, base_dir, None


def read_batch_manifest(
    manifest_arg: str, register=None
) -> tuple[str | None, str | None, CommandFailure | None]:
    """Read and UTF-8 decode a batch-simulate manifest (path or stdin)."""
    if manifest_arg == "-":
        try:
            raw_manifest = sys.stdin.buffer.read()
        except OSError as exc:
            return None, None, CommandFailure(
                "io_error", f"cannot read manifest {manifest_arg!r}: {exc.strerror or exc}", 1
            )
        base_dir = os.getcwd()
    else:
        try:
            with open(manifest_arg, "rb") as handle:
                raw_manifest = handle.read()
        except OSError as exc:
            return None, None, CommandFailure(
                "io_error", f"cannot read manifest {manifest_arg!r}: {exc.strerror or exc}", 1
            )
        base_dir = os.path.dirname(os.path.abspath(manifest_arg))
    if register is not None:
        register(manifest_arg, None)
    try:
        return raw_manifest.decode("utf-8"), base_dir, None
    except UnicodeDecodeError as exc:
        return None, None, CommandFailure(
            "io_error", f"manifest {manifest_arg!r} is not valid UTF-8: {exc}", 1
        )
