"""Unitary-level equivalence of noiseless OpenQASM circuits.

Two circuits are the same quantum transformation when the unitary defined by
their gates (measurements excluded) matches up to a global phase. The
distance metric is

    d = min over |lambda| = 1 of ||U - lambda * V||_F / sqrt(2 * 2**n)

which, with ``t = tr(U† V)`` and ``N = 2**n``, evaluates to
``sqrt(1 - |t| / N)`` because ||U||_F² = ||V||_F² = N and the optimal phase
aligns ``lambda * t`` with the positive real axis. Full equivalence also
requires the classical register width and the qubit-to-clbit measurement
mapping to agree.
"""

from __future__ import annotations

import math

from .openqasm import Program
from .simulator import apply_gates

# Full unitary matrices hold 4**n amplitudes; the comparison entry point is
# capped well below the state-vector register limit.
MAX_EQUIVALENCE_QUBITS = 8

# Fixed tolerance on the normalized Frobenius distance.
TOLERANCE = 1e-10


def circuit_unitary(program: Program) -> list[list[complex]]:
    """Return the unitary of *program*'s gates as a list of columns.

    Column ``k`` is the image of basis state ``k``; entry ``[k][j]`` is the
    matrix element ``U[j][k]``. Measurement operations are excluded.
    """
    n = program.num_qubits
    size = 1 << n
    columns: list[list[complex]] = []
    for basis in range(size):
        state = [0j] * size
        state[basis] = 1 + 0j
        apply_gates(program, state)
        columns.append(state)
    return columns


def unitary_distance(left: list[list[complex]], right: list[list[complex]]) -> float:
    """Normalized global-phase-aware Frobenius distance between two unitaries.

    Both operands are column lists of the same dimension, as returned by
    :func:`circuit_unitary`.
    """
    size = len(left)
    inner = 0j  # tr(U† V) = sum of conj(U[j][k]) * V[j][k]
    for left_col, right_col in zip(left, right):
        for u, v in zip(left_col, right_col):
            inner += u.conjugate() * v
    return math.sqrt(max(0.0, 1.0 - abs(inner) / size))


def measurement_layout(program: Program) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return (number of clbits, sorted (qubit, clbit) measurement pairs)."""
    pairs = sorted(
        (op.targets[0], op.targets[1]) for op in program.operations if op.kind == "measure"
    )
    return program.num_clbits, tuple(pairs)
