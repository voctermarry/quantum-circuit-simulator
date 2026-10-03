"""Final-state comparison metrics for two circuits.

Both circuits evolve from the all-zero initial state (terminal measurements
are ignored, matching :func:`simulate_state_vector` and the density-matrix
path). When neither side carries a noise model both final states are pure
state vectors and the metrics are the squared overlap
``|<left|right>|^2`` plus, for each side, the per-qubit von Neumann entropy
of the reduced density matrix.

With a noise model on either side, that side evolves as a full density
matrix. The comparison metrics then generalize to:

- the squared Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2``,
  which equals ``|<psi|phi>|^2`` when both states are pure and
  ``<psi|rho|psi>`` when exactly one is mixed;
- the purity ``Tr(rho**2)`` of each density matrix (1 for a pure state);
- the same per-qubit binary von Neumann entropy, computed from each side's
  (possibly mixed) reduced density matrix.
"""

from __future__ import annotations

import math

# Values within this distance of 0 or 1 are reported as exactly 0 or 1.
SNAP_TOLERANCE = 1e-15

# Eigenvalues of the numerically rebuilt product ``sqrt(rho) sigma
# sqrt(rho)`` below this absolute floor are treated as exact zeros. Round-off
# leaves phantom eigenvalues of order 1e-17 (NumPy) to 1e-16 (the pure-Python
# Jacobi path) in null subspaces; kept as-is their square roots would inject
# amplified noise into the fidelity root sum.
_EIGENVALUE_NOISE_FLOOR = 1e-13


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


def normalized_density_matrix(rho: list[list[complex]]) -> list[list[complex]]:
    """Return *rho* divided by its trace (a fresh, exactly unit-trace copy).

    Every configured noise channel is trace preserving in exact arithmetic,
    but repeated finite-precision evolution can drift the trace by a few
    ulps, just as state-vector evolution drifts the norm. All density-matrix
    metrics assume a normalized state, so callers normalize once, mirroring
    the explicit renormalization :func:`state_fidelity` applies to vectors.
    """
    size = len(rho)
    trace = math.fsum(rho[i][i].real for i in range(size))
    if trace == 0.0:
        return [row[:] for row in rho]
    return [[entry / trace for entry in row] for row in rho]


def _entropy_from_qubit_block(
    rho00: float, rho11: float, off_re: float, off_im: float
) -> float:
    """Binary entropy of the 2x2 Hermitian block ``[[rho00, z], [z*, rho11]]``.

    ``z`` is supplied as real/imaginary parts. The two eigenvalues of the
    block are computed directly; zero (or slightly negative) eigenvalues do
    not contribute.
    """
    off_magnitude_sq = off_re**2 + off_im**2
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

        entropies.append(
            _entropy_from_qubit_block(
                math.fsum(diag0_terms),
                math.fsum(diag1_terms),
                math.fsum(off_re_terms),
                math.fsum(off_im_terms),
            )
        )
    return entropies


def density_purity(rho: list[list[complex]]) -> float:
    """Return ``Tr(rho**2)`` of a normalized density matrix.

    For a Hermitian matrix this is the squared Frobenius norm
    ``sum |rho_ij|**2``; a pure state gives 1 and the maximally mixed state
    of ``n`` qubits gives ``2**-n``.
    """
    total = math.fsum(abs(entry) ** 2 for row in rho for entry in row)
    return _snap01(total)


def density_single_qubit_entropies(
    rho: list[list[complex]], num_qubits: int
) -> list[float]:
    """Per-qubit reduced-state binary entropies for a (possibly mixed) state.

    Same convention as :func:`single_qubit_entropies`, but the reductions
    are taken from the full density matrix, so the result also reflects
    classical/mixed-state entropy rather than only entanglement.
    """
    entropies: list[float] = []
    size = len(rho)
    for qubit in range(num_qubits):
        bit = 1 << qubit
        diag0_terms: list[float] = []
        diag1_terms: list[float] = []
        off_re_terms: list[float] = []
        off_im_terms: list[float] = []
        for index in range(size):
            if index & bit:
                diag1_terms.append(rho[index][index].real)
            else:
                diag0_terms.append(rho[index][index].real)
                entry = rho[index][index | bit]
                off_re_terms.append(entry.real)
                off_im_terms.append(entry.imag)

        entropies.append(
            _entropy_from_qubit_block(
                math.fsum(diag0_terms),
                math.fsum(diag1_terms),
                math.fsum(off_re_terms),
                math.fsum(off_im_terms),
            )
        )
    return entropies


def pure_density_fidelity(state: list[complex], rho: list[list[complex]]) -> float:
    """Squared Uhlmann fidelity of a pure state with a density matrix.

    Returns ``<psi|rho|psi>`` for the normalized state and unit-trace
    density matrix. Both orders are covered (the expectation is symmetric
    in which side is pure), and it equals ``|<psi|phi>|^2`` when ``rho`` is
    itself a rank-one pure state.
    """
    norm = math.fsum(abs(amplitude) ** 2 for amplitude in state)

    # v = rho @ psi, accumulated per component with fsum for determinism.
    applied: list[complex] = []
    products: list[complex] = []
    for i, row in enumerate(rho):
        terms = [entry * state[j] for j, entry in enumerate(row)]
        component = complex(
            math.fsum(z.real for z in terms),
            math.fsum(z.imag for z in terms),
        )
        applied.append(component)
        products.append(state[i].conjugate() * component)
    expectation = math.fsum(z.real for z in products)
    return _snap01(expectation / norm)


# ------------------------------------------------------------- mixed fidelity


def _hermitian_eigendecomposition(
    matrix: list[list[complex]],
) -> tuple[list[float], list[list[complex]]]:
    """Eigendecompose a complex Hermitian matrix via cyclic Jacobi sweeps.

    Pure-Python reference path used only when NumPy is unavailable. Each
    pivot is first made real by a diagonal phase rotation on its plane,
    then cleared by a real Jacobi rotation ``G = D J`` with
    ``A' = G^H A G``; the same rotations accumulate into ``V`` (starting
    from the identity), so the columns of ``V`` are the eigenvectors and
    ``A = V diag(eigenvalues) V^H``. Sweeps continue until every
    off-diagonal entry is negligible relative to the spectral scale.

    Correct for any size, but the cubic Python cost makes it practical only
    for the small matrices the no-NumPy case realistically compares.
    """
    n = len(matrix)
    work = [row[:] for row in matrix]
    vectors = [[1.0 + 0j if i == j else 0j for j in range(n)] for i in range(n)]

    # Hermitize against round-off: mirror each entry's conjugate.
    for i in range(n):
        work[i][i] = complex(work[i][i].real, 0.0)
        for j in range(i + 1, n):
            value = (work[i][j] + work[j][i].conjugate()) / 2
            work[i][j] = value
            work[j][i] = value.conjugate()

    tolerance = 1e-15
    for _sweep in range(300):
        largest = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                pivot = work[p][q]
                magnitude = abs(pivot)
                if magnitude <= 0.0:
                    continue
                largest = max(largest, magnitude)

                a = work[p][p].real
                d = work[q][q].real
                # G = D J on the (p, q) plane: D = diag(1, e**-i phi)
                # removes the pivot's phase and J = [[c, -s], [s, c]] is a
                # real rotation with tan(2 theta) = 2m / (a - d); the small
                # root is t = -sign(tau)/(|tau| + sqrt(1 + tau**2)).
                tau = (d - a) / (2.0 * magnitude)
                if tau >= 0.0:
                    tangent = -1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    tangent = 1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                cs = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sn = tangent * cs
                phase = pivot.conjugate() / magnitude  # e**(-i phi)
                phase_adjoint = phase.conjugate()  # e**(+i phi)

                # Step 1: B = A G (mix columns p, q).
                col_p = [work[r][p] for r in range(n)]
                col_q = [work[r][q] for r in range(n)]
                for r in range(n):
                    work[r][p] = cs * col_p[r] + sn * phase * col_q[r]
                    work[r][q] = -sn * col_p[r] + cs * phase * col_q[r]

                # Step 2: A' = G^H B (mix rows p, q).
                row_p = work[p][:]
                row_q = work[q][:]
                for r in range(n):
                    work[p][r] = cs * row_p[r] + sn * phase_adjoint * row_q[r]
                    work[q][r] = -sn * row_p[r] + cs * phase_adjoint * row_q[r]

                # Accumulate V' = V G (same column mix as step 1).
                vec_p = [vectors[r][p] for r in range(n)]
                vec_q = [vectors[r][q] for r in range(n)]
                for r in range(n):
                    vectors[r][p] = cs * vec_p[r] + sn * phase * vec_q[r]
                    vectors[r][q] = -sn * vec_p[r] + cs * phase * vec_q[r]

        scale = max(1.0, max(abs(work[i][i]) for i in range(n)))
        if largest <= tolerance * scale:
            break

    eigenvalues = [work[i][i].real for i in range(n)]
    return eigenvalues, vectors


def _hermitian_eigenvalues_numpy(rho: list[list[complex]]) -> tuple[list[float], object]:
    """Eigenvalues and eigenvectors of *rho* via NumPy (optional accelerator).

    Dense eigensolves are cubic; NumPy is used opportunistically when it can
    be imported and is never a declared dependency. Columns of the returned
    matrix are the eigenvectors, matching :func:`_hermitian_eigendecomposition`.
    """
    import numpy as np

    matrix = np.asarray(rho, dtype=np.complex128)
    matrix = (matrix + matrix.conj().T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return [float(value) for value in eigenvalues.real], eigenvectors


def density_fidelity(
    rho: list[list[complex]], sigma: list[list[complex]]
) -> float:
    """Squared Uhlmann fidelity of two normalized density matrices.

    Computes ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2`` from the
    eigenvalues ``mu`` of ``sqrt(rho) sigma sqrt(rho)``:

        F = (sum_k sqrt(max(mu_k, 0))) ** 2

    The two states share dimension. With ``rho = V diag(lambda) V^H`` the
    scaled product is ``B = R V^H sigma V R`` for ``R = diag(sqrt(lambda))``,
    so only one eigendecomposition of each input matrix is needed. The
    product is built only on the subspace of rho's eigenvalues above the
    round-off floor, so phantom null-subspace eigenvalues cannot reach the
    square roots; any residual value below the floor is treated as zero.
    """
    try:
        eigenvalues, vectors = _hermitian_eigenvalues_numpy(rho)
    except ImportError:
        eigenvalues, vectors = _hermitian_eigendecomposition(rho)

    active = [i for i, value in enumerate(eigenvalues) if value > _EIGENVALUE_NOISE_FLOOR]
    if not active:
        return 0.0
    roots = [math.sqrt(eigenvalues[i]) for i in active]

    if not isinstance(vectors, list):
        # NumPy path: columns of V scaled by sqrt(lambda) give M = V R and
        # B = M^H sigma M directly, restricted to the active columns.
        import numpy as np

        sigma_matrix = np.asarray(sigma, dtype=np.complex128)
        basis = vectors[:, np.asarray(active)] * np.asarray(roots, dtype=np.float64)
        product = basis.conj().T @ sigma_matrix @ basis
        product = (product + product.conj().T) * 0.5
        mu = [float(value) for value in np.linalg.eigvalsh(product).real]
    else:
        # B_ij = r_i r_j (V^H sigma V)_ij with eigenvectors in V's columns.
        size = len(rho)

        def transformed(i: int, j: int) -> complex:
            total = 0j
            for k in range(size):
                vki = vectors[k][i].conjugate()
                if vki == 0j:
                    continue
                row = sigma[k]
                inner = 0j
                for l in range(size):
                    v_lj = vectors[l][j]
                    if v_lj != 0j:
                        inner += row[l] * v_lj
                total += vki * inner
            return total

        block = [
            [roots[a] * transformed(active[a], active[b]) * roots[b] for b in range(len(active))]
            for a in range(len(active))
        ]
        mu = _hermitian_eigendecomposition(block)[0]

    root_terms = [
        math.sqrt(value)
        for value in mu
        if value > _EIGENVALUE_NOISE_FLOOR
    ]
    root_sum = math.fsum(root_terms)
    return _snap01(root_sum * root_sum)
