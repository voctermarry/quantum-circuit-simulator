"""Input preparation and result delivery for the command layer.

This module is the only part of the command layer that touches the file
system, standard input or the standard streams. It reads and decodes
command inputs (files or ``-`` for stdin), parses and validates circuit
sources and noise models into the values the pure computation core
(:mod:`quantum_circuit.core`) consumes, and delivers finished results to
stdout or an atomically exported ``--output`` file.

Failures are reported to the caller as ``(error_payload, exit_code)``
pairs; the command handlers in :mod:`quantum_circuit.commands` emit them
through :func:`emit_error_payload`, keeping error mapping in one place.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile

from .noise import NoiseModelError, parse_noise_model
from .openqasm import ParseError, Program, ValidationError, parse


def error_payload(
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


def emit_error(
    error: str,
    message: str,
    line: int | None = None,
    column: int | None = None,
    side: str | None = None,
) -> None:
    emit_error_payload(error_payload(error, message, line, column, side))


def emit_error_payload(payload: dict[str, object]) -> None:
    sys.stderr.write(json.dumps(payload) + "\n")


# ------------------------------------------------------------- result export


class RunContext:
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


def guard_output_conflict(ctx: RunContext) -> dict[str, object] | None:
    """Refuse to export over an input file (``output_error``, exit code 2).

    Invoked only after the command's own validation has succeeded but
    before any result-producing computation runs, so no simulation or
    re-run task executes and no file is modified. Returns the error payload
    to emit, or ``None`` when the export is safe.
    """
    if ctx.target is not None and ctx.target in ctx.inputs:
        return error_payload(
            "output_error",
            f"output path {ctx.output_arg!r} is the same file as an input read by this command",
        )
    return None


def register_manifest_references(
    ctx: RunContext,
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


def write_output_atomic(target: str, data: bytes) -> str | None:
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


def finish(ctx: RunContext | None, result: dict[str, object], exit_code: int) -> int:
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
        emit_error("output_error", "output path must not be '-'")
        return 1
    message = write_output_atomic(ctx.target, (json.dumps(result) + "\n").encode("utf-8"))
    if message is not None:
        emit_error("output_error", message)
        return 1
    return exit_code


# ---------------------------------------------------------- input preparation


def base_dir_for(arg: str) -> str:
    """The directory relative input paths of a document argument resolve from."""
    return os.getcwd() if arg == "-" else os.path.dirname(os.path.abspath(arg))


def read_utf8_input(
    arg: str,
    label: str,
    *,
    side: str | None = None,
    utf8_error: str = "io_error",
    utf8_code: int = 1,
    ctx: RunContext | None = None,
    base_dir: str | None = None,
) -> tuple[str | None, dict[str, object] | None, int]:
    """Read and UTF-8 decode one text input from a path or ``-`` (stdin).

    Returns ``(text, None, 0)`` on success or ``(None, error_payload,
    exit_code)`` on failure. *label* is the input's name in error messages
    (``"source"``, ``"noise model"``, ``"samples"``, ...); *side* tags the
    payload with an ``input`` field for multi-document commands. Relative
    paths open against *base_dir* when given, while messages keep quoting
    *arg* verbatim.
    """
    if base_dir is not None and arg != "-" and not os.path.isabs(arg):
        open_path = os.path.normpath(os.path.join(base_dir, arg))
    else:
        open_path = arg
    try:
        if arg == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(open_path, "rb") as handle:
                data = handle.read()
    except OSError as exc:
        return (
            None,
            error_payload(
                "io_error", f"cannot read {label} {arg!r}: {exc.strerror or exc}", side=side
            ),
            1,
        )
    if ctx is not None:
        ctx.register_input(arg, base_dir)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return (
            None,
            error_payload(
                utf8_error, f"{label} {arg!r} is not valid UTF-8: {exc}", side=side
            ),
            utf8_code,
        )
    return text, None, 0


def parse_program(
    source: str, *, side: str | None = None
) -> tuple[Program | None, dict[str, object] | None, int]:
    """Parse and validate one circuit source.

    Returns ``(program, None, 0)`` on success or ``(None, error_payload,
    2)`` carrying the standard ``parse_error``/``validation_error`` shape.
    """
    try:
        return parse(source), None, 0
    except ParseError as exc:
        return None, error_payload("parse_error", exc.message, exc.line, exc.column, side=side), 2
    except ValidationError as exc:
        return (
            None,
            error_payload("validation_error", exc.message, exc.line, exc.column, side=side),
            2,
        )


def prepare_circuit(
    source_arg: str,
    *,
    side: str | None = None,
    ctx: RunContext | None = None,
) -> tuple[Program | None, dict[str, object] | None, int]:
    """Read, decode, parse and validate one circuit source argument."""
    source, error, code = read_utf8_input(source_arg, "source", side=side, ctx=ctx)
    if error is not None:
        return None, error, code
    assert source is not None
    return parse_program(source, side=side)


def prepare_noise_model(
    model_arg: str,
    *,
    side: str | None = None,
    ctx: RunContext | None = None,
    base_dir: str | None = None,
) -> tuple[dict[str, float] | None, dict[str, object] | None, int]:
    """Read, decode, parse and validate one noise model argument.

    An unreadable file is an ``io_error`` (code 1); invalid UTF-8 or
    non-compliant content is a ``noise_model_error`` (code 2).
    """
    model_text, error, code = read_utf8_input(
        model_arg,
        "noise model",
        side=side,
        utf8_error="noise_model_error",
        utf8_code=2,
        ctx=ctx,
        base_dir=base_dir,
    )
    if error is not None:
        return None, error, code
    assert model_text is not None
    try:
        return parse_noise_model(model_text), None, 0
    except NoiseModelError as exc:
        return None, error_payload("noise_model_error", str(exc), side=side), 2


def prepare_simulation(
    source_arg: str,
    noise_model_arg: str | None,
    base_dir: str | None = None,
    ctx: RunContext | None = None,
) -> tuple[Program | None, dict[str, float] | None, dict[str, object] | None, int]:
    """Read, decode, parse and validate one simulation input.

    Returns ``(program, noise_model, error, exit_code)``: on success *error*
    is ``None`` and *program* is the parsed circuit (with *noise_model*
    ``None`` on the state-vector path); on failure *program* is ``None`` and
    *error* is the payload the calling command writes to stderr (including
    line/column where applicable).

    Reading (files, UTF-8, ``-`` for stdin, relative paths) and error
    semantics match ``simulate`` for every command that evolves a circuit.
    No qubit-limit or simulation error is raised here; callers apply their
    own limits.

    When *base_dir* is given (the batch case), relative paths are opened
    relative to it while error messages keep quoting *source_arg* and
    *noise_model_arg* verbatim, so messages match a standalone invocation
    with the same path strings.
    """
    if noise_model_arg is not None and source_arg == "-" and noise_model_arg == "-":
        return (
            None,
            None,
            error_payload(
                "noise_model_error",
                "circuit source and noise model cannot both be read from standard input",
            ),
            2,
        )

    source, error, code = read_utf8_input(source_arg, "source", ctx=ctx, base_dir=base_dir)
    if error is not None:
        return None, None, error, code
    assert source is not None

    noise_model: dict[str, float] | None = None
    if noise_model_arg is not None:
        noise_model, error, code = prepare_noise_model(
            noise_model_arg, ctx=ctx, base_dir=base_dir
        )
        if error is not None:
            return None, None, error, code

    program, error, code = parse_program(source)
    if error is not None:
        return None, None, error, code
    assert program is not None
    return program, noise_model, None, 0
