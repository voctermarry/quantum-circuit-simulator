"""Equivalence checking for two noiseless OpenQASM circuits.

Two circuits represent the same quantum transformation when their
pre-measurement unitaries agree up to a global phase and their measurement
layout (classical register width plus qubit-to-clbit mapping) is identical.
Register names do not matter.

The unitary distance allowing a global phase is

    d = min_{|lambda|=1} ||U - lambda V||_F / sqrt(2 * 2**n)
      = sqrt(1 - |tr(U^dagger V)| / 2**n)
"""

from __future__ import annotations

import math

from .openqasm import Program

# A dense 2**n x 2**n unitary has 4**n entries; 8 qubits is the comparison cap.
MAX_EQUIVALENCE_QUBITS = 8
EQUIVALENCE_TOLERANCE = 1e-10


def measurement_layout(program: Program) -> tuple[int, frozenset[tuple[int, int]]]:
    """Return (classical register width, set of (qubit, clbit) measurements).

    Register names are discarded: only the width and the qubit-to-clbit
    mapping participate in the comparison.
    """
    pairs = frozenset(
        (op.targets[0], op.targets[1])
        for op in program.operations
        if op.kind == "measure"
    )
    return program.num_clbits, pairs


def unitary_distance(u: list[list[complex]], v: list[list[complex]]) -> float:
    """Global-phase-invariant normalized Frobenius distance of two unitaries.

    ``u`` and ``v`` must be square matrices of the same dimension. Returns
    ``min_{|lambda|=1} ||U - lambda V||_F / sqrt(2 * dim)``, a value in
    ``[0, 1]`` that is ``0`` exactly when ``U`` and ``V`` differ only by a
    global phase.
    """
    dim = len(u)
    # t = tr(U^dagger V) = sum_{i,j} conj(U[j][i]) * V[j][i].
    trace = 0j
    for i in range(dim):
        for j in range(dim):
            trace += u[j][i].conjugate() * v[j][i]

    normalized = abs(trace) / dim
    # Round-off can push |t|/dim a hair above 1; clamp to the valid interval.
    aligned = min(1.0, max(0.0, normalized))
    return math.sqrt(max(0.0, 1.0 - aligned))
