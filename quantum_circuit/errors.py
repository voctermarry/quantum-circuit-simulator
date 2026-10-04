"""Structured command failures shared by input preparation and pure logic.

The command layer speaks about failures as data rather than exceptions:
:class:`CommandFailure` carries exactly the fields that become the
single-line stderr JSON document (``error``/``message`` and, for a few
inputs, ``input``/``line``/``column``) together with the process exit code.

This module performs no file system, standard input or standard stream
access, so preparation and pure computation can build failures without
depending on the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandFailure(Exception):
    """A failed command step as a serializable payload plus exit code.

    Subclassing :class:`Exception` lets pure helpers raise it through
    call stacks while preparation code normally constructs and returns it.
    """

    error: str
    message: str
    exit_code: int
    side: str | None = None
    line: int | None = None
    column: int | None = None

    def to_payload(self) -> dict[str, object]:
        """Return the canonical stderr JSON object (ordered fields)."""
        payload: dict[str, object] = {"error": self.error}
        if self.side is not None:
            payload["input"] = self.side
        payload["message"] = self.message
        if self.line is not None:
            payload["line"] = self.line
            payload["column"] = self.column
        return payload
