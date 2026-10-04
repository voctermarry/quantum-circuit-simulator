"""Result delivery: standard streams and atomic file export.

This is the only layer that writes standard output/error or result files.
Preparation and pure computation hand it either a ready success payload
(plus the command's exit code) or a
:class:`~quantum_circuit.errors.CommandFailure`; :class:`RunContext`
additionally implements the ``--output`` conflict protection, refusing to
export over any registered input before the command computes or re-runs
anything.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile

from .errors import CommandFailure


def _file_identity(path: str) -> tuple[int, int] | None:
    """Return the ``(device, inode)`` identity of *path*, or ``None``.

    :func:`os.stat` follows symbolic links (and Windows directory
    junctions), so every alias of the same underlying file -- links, hard
    links, junctions, ``.``/``..`` spellings -- yields the same identity.
    A path that does not exist or cannot be stat'ed has no identity.
    """
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_dev, info.st_ino)


class RunContext:
    """Track the optional ``--output PATH`` target and every input file read.

    The target is absolutized once (anchored at the invocation working
    directory). Every non-stdin input the command reads is registered using
    the same path resolution used to open it, in two forms:

    * a lexically normalized spelling (``normcase`` of the absolute path),
      so ``.``/``..`` spellings always collide and case-only differences
      collide exactly on case-insensitive file systems -- this also covers
      manifest-declared paths that do not exist (yet); and
    * the actual file identity when the file exists, so any alias of the
      same file (symbolic links, hard links, junctions) is recognized.

    An export that would overwrite a reproducibility input can therefore
    be refused before any result is computed, while two genuinely
    different files -- however similar their names or contents -- never
    collide.
    """

    def __init__(self, output: str | None):
        self.output_arg = output
        self.target: str | None = None if output is None else os.path.abspath(output)
        self.inputs: set[str] = set()
        self.identities: set[tuple[int, int]] = set()

    def register_input(self, path: str, base_dir: str | None = None) -> None:
        if self.target is None or path == "-":
            return
        if base_dir is not None and not os.path.isabs(path):
            resolved = os.path.join(base_dir, path)
        else:
            resolved = path
        absolute = os.path.abspath(resolved)
        self.inputs.add(os.path.normcase(absolute))
        identity = _file_identity(absolute)
        if identity is not None:
            self.identities.add(identity)


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


def output_conflict(ctx: RunContext) -> CommandFailure | None:
    """Refuse to export over an input file (``output_error``, exit code 2).

    Invoked only after the command's own validation has succeeded but
    before any result-producing computation runs, so no simulation or
    re-run task executes and no file is modified. The target conflicts
    when its normalized spelling matches a registered one (covering
    declared-but-missing manifest references and case-only respellings on
    case-insensitive file systems) or when it exists and its file identity
    matches an existing input's (covering links, hard links and junction
    aliases).
    """
    if ctx.target is None:
        return None
    conflict = os.path.normcase(ctx.target) in ctx.inputs
    if not conflict:
        identity = _file_identity(ctx.target)
        conflict = identity is not None and identity in ctx.identities
    if conflict:
        return CommandFailure(
            "output_error",
            f"output path {ctx.output_arg!r} is the same file as an input read by this command",
            2,
        )
    return None


def emit_failure(failure: CommandFailure) -> None:
    """Write one failure as the canonical single-line stderr JSON."""
    sys.stderr.write(json.dumps(failure.to_payload()) + "\n")


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
        emit_failure(CommandFailure("output_error", "output path must not be '-'", 1))
        return 1
    message = write_output_atomic(ctx.target, (json.dumps(result) + "\n").encode("utf-8"))
    if message is not None:
        emit_failure(CommandFailure("output_error", message, 1))
        return 1
    return exit_code
