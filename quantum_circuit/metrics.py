"""Final-state comparison metrics for two circuits.

Both circuits are evolved from the all-zero initial state (terminal
measurements are ignored, matching :func:`simulate_state_vector`). Without
noise models the metrics are the squared overlap of the two final pure
states; with a noise model a side is evolved as a full density matrix and
the comparison uses the squared Uhlmann fidelity (which reduces to the
squared overlap for two pure states). For each side the per-qubit von
Neumann entropy of the reduced density matrix quantifies the entanglement
of that qubit with the rest of the system, and the purity ``Tr(rho**2)``
quantifies how mixed the noisy evolution left the state.
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


def _qubit_entropy(rho00: float, rho11: float, off_magnitude_sq: float) -> float:
    """Von Neumann entropy (base 2) of a 2x2 reduced density matrix.

    *rho00*/*rho11* are the diagonal entries and *off_magnitude_sq* the
    squared magnitude of the off-diagonal entry. Zero (or, through
    round-off, slightly negative) eigenvalues do not contribute.
    """
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
    return _snap01(entropy)


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
        entropies.append(_qubit_entropy(rho00, rho11, off_magnitude_sq))
    return entropies


def state_purity(state: list[complex]) -> float:
    """Return ``Tr(rho**2)`` for the pure state *state* (its squared norm).

    A normalized state gives exactly 1 after endpoint snapping.
    """
    norm = math.fsum(abs(amplitude) ** 2 for amplitude in state)
    return _snap01(norm * norm)


def density_purity(rho: list[list[complex]]) -> float:
    """Return ``Tr(rho**2)`` of a density matrix.

    For a Hermitian matrix ``Tr(rho**2)`` is the sum of the squared
    magnitudes of all entries, accumulated in a fixed row-major order so
    repeated runs agree bit for bit.
    """
    total = math.fsum(abs(entry) ** 2 for row in rho for entry in row)
    return _snap01(total)


def density_single_qubit_entropies(rho: list[list[complex]], num_qubits: int) -> list[float]:
    """Per-qubit reduced-state von Neumann entropies of a density matrix.

    Same definition as :func:`single_qubit_entropies`, but the reduced
    matrices are obtained by partially tracing *rho* instead of
    reconstructing them from a state vector.
    """
    entropies: list[float] = []
    for qubit in range(num_qubits):
        bit = 1 << qubit
        diag0_terms: list[float] = []
        diag1_terms: list[float] = []
        off_re_terms: list[float] = []
        off_im_terms: list[float] = []
        for index, row in enumerate(rho):
            diagonal = row[index]
            if index & bit:
                diag1_terms.append(diagonal.real)
            else:
                diag0_terms.append(diagonal.real)
                product = row[index | bit]
                off_re_terms.append(product.real)
                off_im_terms.append(product.imag)

        rho00 = math.fsum(diag0_terms)
        rho11 = math.fsum(diag1_terms)
        off_magnitude_sq = math.fsum(off_re_terms) ** 2 + math.fsum(off_im_terms) ** 2
        entropies.append(_qubit_entropy(rho00, rho11, off_magnitude_sq))
    return entropies


def pure_mixed_fidelity_squared(state: list[complex], rho: list[list[complex]]) -> float:
    """Return the squared Uhlmann fidelity of a pure state and a mixed state.

    For ``rho_pure = |psi><psi|`` this is ``<psi|rho|psi>``. The state is
    renormalized explicitly, mirroring :func:`state_fidelity`, and the
    accumulation order is fixed so repeated runs agree bit for bit.
    """
    norm = math.fsum(abs(amplitude) ** 2 for amplitude in state)
    re_terms: list[float] = []
    im_terms: list[float] = []
    for i, row in enumerate(rho):
        left = state[i].conjugate()
        for j, entry in enumerate(row):
            product = left * entry * state[j]
            re_terms.append(product.real)
            im_terms.append(product.imag)
    value = complex(math.fsum(re_terms), math.fsum(im_terms)) / norm
    # The exact value is real and non-negative; any imaginary part is
    # round-off from the evolution.
    return _snap01(value.real)


def uhlmann_fidelity_squared(left: list[list[complex]], right: list[list[complex]]) -> float:
    """Return the squared Uhlmann fidelity of two density matrices.

    Computes ``(Tr sqrt(sqrt(left) right sqrt(left)))**2`` via Hermitian
    eigendecompositions (NumPy, imported lazily so the pure-state paths
    stay dependency-free). For two pure states this equals the squared
    overlap computed by :func:`state_fidelity`.
    """
    # Cap BLAS threading before NumPy loads its backend: the multithreaded
    # LAPACK eigensolver path can be drastically slower than the
    # single-threaded one for the matrix sizes involved here. ``setdefault``
    # keeps any user configuration, and if NumPy is already imported the
    # computation is unaffected (only its threading stays as configured).
    import os

    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    import numpy as np

    left_matrix = np.asarray(left, dtype=complex)
    right_matrix = np.asarray(right, dtype=complex)

    # Square root of the (Hermitian, positive semidefinite) left matrix.
    left_hermitian = (left_matrix + left_matrix.conj().T) / 2
    values, vectors = np.linalg.eigh(left_hermitian)
    root = (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T

    product = root @ right_matrix @ root
    product = (product + product.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(product)
    fidelity = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))) ** 2)
    return _snap01(fidelity)
