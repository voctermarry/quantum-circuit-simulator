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


class RunContext:
    """Track the optional ``--output PATH`` target and every input file read.

    The target is absolutized once (anchored at the invocation working
    directory). Every non-stdin input the command reads is registered using
    the same path resolution used to open it. Conflicts are decided by real
    file identity (:func:`os.path.samefile` -- which follows symlinks,
    Windows junctions and hard links and folds case on case-insensitive
    volumes), not by comparing path strings, so relative ``.``/``..``
    aliases, different absolute spellings, links or case-only aliases cannot
    be used to export over a reproducibility input.

    A manifest-declared task path that currently does not exist (or is not
    readable) cannot be compared by identity; such an entry is registered
    with the same lexical path the preparation layer would open and
    compared lexically against the target when one side is absent.
    """

    def __init__(self, output: str | None):
        self.output_arg = output
        self.target: str | None = None if output is None else os.path.abspath(output)
        # Absolute, lexically normalized spellings of every protected input.
        self.inputs: set[str] = set()

    def register_input(self, path: str, base_dir: str | None = None) -> None:
        """Register *path* as an input the command actually read."""
        if self.target is None or path == "-":
            return
        self.inputs.add(_absolute_input(path, base_dir))

    def register_declared(self, path: str, base_dir: str | None = None) -> None:
        """Register a path a validated manifest declares as a task input.

        The file need not exist or be readable. When it happens to exist
        (possibly through a symlinked manifest directory), conflict
        detection still compares it by identity; a genuinely absent
        declaration falls back to lexical comparison.
        """
        if self.target is None or path == "-":
            return
        self.inputs.add(_absolute_input(path, base_dir))

    def is_conflict(self, input_path: str) -> bool:
        """Return whether *input_path* names the same file as the target."""
        target = self.target
        assert target is not None
        # Whenever both sides exist, real file identity settles it:
        # os.path.samefile follows symlinks and Windows junctions, joins
        # hard links through (st_dev, st_ino), and reports equal identity
        # for case-only aliases on volumes that fold case.
        if os.path.exists(target) and os.path.exists(input_path):
            try:
                return os.path.samefile(target, input_path)
            except OSError:
                pass
        # At least one side does not exist (the normal case for a declared
        # but missing task) or identity could not be computed: fall back to
        # the lexical view of both absolute spellings, folding letter case
        # only on volumes that actually do so.
        return _lexically_same_file(target, input_path)


def _absolute_input(path: str, base_dir: str | None) -> str:
    """Absolutize an input path exactly the way the command opens it."""
    if base_dir is not None and not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.abspath(path)


# Case-folding probe results, cached per probed existing directory.
_case_folding_dirs: dict[str, bool] = {}


def _probe_case_folding(directory: str) -> bool:
    """Return whether *directory* (an existing dir) folds letter case.

    Detected read-only whenever the directory has at least one entry whose
    name carries a letter case: the entry and its case-swapped spelling are
    compared; equal identity means the volume folds case (distinct entries
    can only coexist on a case-sensitive volume). An empty or unlistable
    directory falls back to a create-and-remove probe, and failure to create
    one (a read-only directory) to the platform convention.
    """
    cached = _case_folding_dirs.get(directory)
    if cached is not None:
        return cached

    folding = _case_folding_from_entries(directory)
    if folding is None:
        folding = _case_folding_from_probe(directory)
    _case_folding_dirs[directory] = folding
    return folding


def _case_folding_from_entries(directory: str) -> bool | None:
    """Read-only case-folding test using the directory's own entries."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for name in entries:
        swapped = name.swapcase()
        if swapped == name:
            continue
        direct = os.path.join(directory, name)
        folded = os.path.join(directory, swapped)
        try:
            return os.path.samefile(direct, folded)
        except OSError:
            # The swapped spelling does not resolve (case-sensitive volume)
            # or neither entry can be stated; move on to the next entry.
            continue
    return None


def _case_folding_from_probe(directory: str) -> bool:
    """Create-and-remove probe for directories without cased entries."""
    probe: str | None = None
    try:
        try:
            fd, probe = tempfile.mkstemp(prefix=".qcs-caseprobe-", dir=directory)
        except OSError:
            return os.name == "nt" or sys.platform == "darwin"
        os.close(fd)
        name = os.path.basename(probe)
        return os.path.exists(os.path.join(directory, name.swapcase()))
    finally:
        if probe is not None:
            try:
                os.unlink(probe)
            except OSError:
                pass


def _path_case_folding(path: str) -> bool:
    """Return whether the volume hosting *path* folds letter case.

    The file itself need not exist; the nearest existing ancestor directory
    is probed, since case sensitivity is a directory/volume property.
    """
    directory = os.path.dirname(os.path.normpath(path)) or "."
    while not os.path.isdir(directory):
        parent = os.path.dirname(directory)
        if parent == directory:
            # No existing ancestor (e.g. an unmapped Windows drive): fall
            # back to the platform convention.
            return os.name == "nt" or sys.platform == "darwin"
        directory = parent
    return _probe_case_folding(directory)


def _lexically_same_file(left: str, right: str) -> bool:
    """Compare two absolute spellings without relying on either file existing.

    ``.``/``..`` segments are normalized away first. Letter case is folded
    only when both paths live on case-folding volumes; on a case-sensitive
    volume two case-only spellings are genuinely different files.
    """
    left = os.path.normpath(left)
    right = os.path.normpath(right)
    if left == right:
        return True
    if _path_case_folding(left) and _path_case_folding(right):
        return left.lower() == right.lower()
    return False


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
    strings); a missing or unreadable file is still covered, compared
    lexically against the output target using the manifest directory base.
    Comparison itself is decided per entry later: an existing file by
    identity, an absent declaration lexically.
    """
    if ctx.target is None:
        return
    for job in jobs:
        source = str(job["source"])
        ctx.register_declared(source, base_dir)
        noise_model = job.get("noise_model")
        if noise_model is not None:
            ctx.register_declared(str(noise_model), base_dir)


def output_conflict(ctx: RunContext) -> CommandFailure | None:
    """Refuse to export over an input file (``output_error``, exit code 2).

    Existing inputs and target are compared by real file identity, so
    relative ``.``/``..`` aliases, different absolute spellings, symbolic
    links, Windows junctions or hard links -- and case-only aliases on
    case-folding volumes -- all conflict; two genuinely different files
    never do. A manifest-declared input that is currently missing or
    unreadable is compared by lexical normalization instead.

    Invoked only after the command's own validation has succeeded but
    before any result-producing computation runs, so no simulation or
    re-run task executes and no file is modified.
    """
    if ctx.target is None:
        return None
    for input_path in ctx.inputs:
        if ctx.is_conflict(input_path):
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
