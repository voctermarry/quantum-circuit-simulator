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

    The minimizer is ``lambda = t / |t|`` with ``t = tr(U^dagger V)``. The
    residual norm is evaluated directly (with compensated summation) rather
    than as ``1 - |t|/dim``: that latter form cancels to a few ulps of
    round-off for nearly identical matrices, which a square root then
    amplifies by orders of magnitude.
    """
    dim = len(u)

    # t = tr(U^dagger V) = sum_{i,j} conj(U[j][i]) * V[j][i], summed with
    # compensated real/imaginary accumulators.
    products = [
        u[j][i].conjugate() * v[j][i]
        for i in range(dim)
        for j in range(dim)
    ]
    trace = complex(math.fsum(z.real for z in products), math.fsum(z.imag for z in products))
    magnitude = math.hypot(trace.real, trace.imag)
    if magnitude == 0.0:
        # No phase aligns anything: U and V are fully orthogonal.
        return 1.0

    # The minimizer of ||U - lambda V|| is lambda = conj(t)/|t|.
    lam = trace.conjugate() / magnitude
    residual_sq = math.fsum(
        abs(u[j][i] - lam * v[j][i]) ** 2
        for i in range(dim)
        for j in range(dim)
    )
    normalized = residual_sq / (2 * dim)
    # Round-off can leave a hair outside [0, 1]; clamp to the valid interval.
    return math.sqrt(min(1.0, max(0.0, normalized)))
