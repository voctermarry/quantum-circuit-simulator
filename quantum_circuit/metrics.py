"""Noiseless pure-state comparison metrics for two circuits.

Both circuits are evolved from the all-zero initial state (terminal
measurements are ignored, matching :func:`simulate_state_vector`). The
metrics are the squared overlap of the two final states and, for each
side, the per-qubit von Neumann entropy of the reduced density matrix,
which quantifies the entanglement of that qubit with the rest of the
system.
"""

from __future__ import annotations

import math

# Values within this distance of 0 or 1 are reported as exactly 0 or 1.
SNAP_TOLERANCE = 1e-15


def _snap01(value: float) -> float:
    """Clamp *value* to [0, 1] and snap to the endpoints within tolerance.

    Guarantees a finite, non-negative-zero result: values no larger than
    ``SNAP_TOLERANCE`` in magnitude become ``0.0`` and values within
    ``SNAP_TOLERANCE`` of 1 become ``1.0``.
    """
    value = min(1.0, max(0.0, value))
    if abs(value) <= SNAP_TOLERANCE:
        return 0.0
    if abs(1.0 - value) <= SNAP_TOLERANCE:
        return 1.0
    return value


def state_fidelity(left: list[complex], right: list[complex]) -> float:
    """Return ``|<left|right>|^2`` of the two normalized state vectors.

    Both states must have the same dimension. The states are renormalized
    explicitly so round-off in the simulation cannot push the result out
    of [0, 1].
    """
    norm_left = math.fsum(abs(amplitude) ** 2 for amplitude in left)
    norm_right = math.fsum(abs(amplitude) ** 2 for amplitude in right)
    products = [a.conjugate() * b for a, b in zip(left, right)]
    inner = complex(
        math.fsum(z.real for z in products),
        math.fsum(z.imag for z in products),
    )
    fidelity = (inner.real**2 + inner.imag**2) / (norm_left * norm_right)
    return _snap01(fidelity)


def single_qubit_entropies(state: list[complex], num_qubits: int) -> list[float]:
    """Return the per-qubit reduced-state von Neumann entropies (base 2).

    Entry ``i`` is ``S(rho_i) = -sum lambda log2(lambda)`` where ``rho_i``
    is the reduced density matrix of qubit ``i`` (qubit ``i`` lives in bit
    ``i`` of the basis index) and zero eigenvalues do not contribute. A
    product state gives all zeros; each qubit of a Bell pair gives 1.
    """
    entropies: list[float] = []
    for qubit in range(num_qubits):
        bit = 1 << qubit
        diag0_terms: list[float] = []
        diag1_terms: list[float] = []
        off_re_terms: list[float] = []
        off_im_terms: list[float] = []
        for index, amplitude in enumerate(state):
            if index & bit:
                diag1_terms.append(amplitude.real**2 + amplitude.imag**2)
            else:
                diag0_terms.append(amplitude.real**2 + amplitude.imag**2)
                product = amplitude * state[index | bit].conjugate()
                off_re_terms.append(product.real)
                off_im_terms.append(product.imag)

        rho00 = math.fsum(diag0_terms)
        rho11 = math.fsum(diag1_terms)
        off_magnitude_sq = math.fsum(off_re_terms) ** 2 + math.fsum(off_im_terms) ** 2

        # Eigenvalues of the 2x2 Hermitian reduced density matrix.
        half_trace = (rho00 + rho11) / 2
        half_diff = (rho00 - rho11) / 2
        spread = math.sqrt(half_diff**2 + off_magnitude_sq)

        entropy = 0.0
        for eigenvalue in (half_trace - spread, half_trace + spread):
            # Round-off can leave a hair outside [0, 1]; zero (or slightly
            # negative) eigenvalues do not contribute.
            if eigenvalue > 0.0:
                entropy -= eigenvalue * math.log2(eigenvalue)
        entropies.append(_snap01(entropy))
    return entropies
