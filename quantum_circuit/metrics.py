"""Noiseless final-state comparison metrics for two circuits.

Both circuits are evolved from the all-zero initial state with the existing
state-vector semantics (terminal measurements are ignored). The metrics are

- ``fidelity``: the squared magnitude of the normalized final-state inner
  product, ``|<left|right>|**2``, in ``[0, 1]``;
- ``single_qubit_entropies``: for each qubit (ascending index) the base-2
  von Neumann entropy of its reduced density matrix, in ``[0, 1]`` — ``0``
  for a qubit factored out of the rest, ``1`` for a maximally entangled one
  (each qubit of a Bell pair).

Every reported value is clamped to its domain and snapped to exactly ``0``
or ``1`` when within ``1e-15`` of that endpoint, so results never contain
NaN, infinities or negative zero, and repeated runs on the same input are
byte-identical.
"""

from __future__ import annotations

import math

# Values within this distance of a domain endpoint are reported as the
# endpoint exactly.
_SNAP_TOLERANCE = 1e-15


def _snap_to_domain(value: float) -> float:
    """Clamp *value* to ``[0, 1]`` and snap near-endpoints to ``0``/``1``.

    The clamp also turns a negative zero into a positive one.
    """
    value = min(1.0, max(0.0, value))
    if value <= _SNAP_TOLERANCE:
        return 0.0
    if 1.0 - value <= _SNAP_TOLERANCE:
        return 1.0
    return value


def state_fidelity(left_state: list[complex], right_state: list[complex]) -> float:
    """Squared magnitude of the inner product of two normalized states.

    ``left_state`` and ``right_state`` must have the same length. The sum is
    evaluated with compensated accumulators so nearly identical states do
    not lose the result to cancellation.
    """
    products = [a.conjugate() * b for a, b in zip(left_state, right_state)]
    inner = complex(math.fsum(z.real for z in products), math.fsum(z.imag for z in products))
    return _snap_to_domain(inner.real * inner.real + inner.imag * inner.imag)


def single_qubit_entropies(state: list[complex], num_qubits: int) -> list[float]:
    """Base-2 von Neumann entropy of each qubit's reduced density matrix.

    Qubit ``i`` lives in bit ``i`` of the basis-state index (q[0] is the
    least significant bit), matching :func:`simulate_state_vector`. Zero
    eigenvalues contribute nothing to the entropy.
    """
    dim = 1 << num_qubits
    entropies: list[float] = []
    for qubit in range(num_qubits):
        bit = 1 << qubit
        step = bit << 1
        # Reduced density matrix [[p0, off], [conj(off), p1]] for this qubit.
        p0_terms: list[float] = []
        p1_terms: list[float] = []
        re_terms: list[float] = []
        im_terms: list[float] = []
        for base in range(0, dim, step):
            for low in range(base, base + bit):
                amp0 = state[low]
                amp1 = state[low + bit]
                p0_terms.append(abs(amp0) ** 2)
                p1_terms.append(abs(amp1) ** 2)
                product = amp0 * amp1.conjugate()
                re_terms.append(product.real)
                im_terms.append(product.imag)
        p0 = math.fsum(p0_terms)
        p1 = math.fsum(p1_terms)
        off = complex(math.fsum(re_terms), math.fsum(im_terms))

        # Eigenvalues of the 2x2 block: (trace ± sqrt((p0-p1)^2 + 4|off|^2)) / 2.
        trace = p0 + p1
        discriminant = (p0 - p1) ** 2 + 4.0 * abs(off) ** 2
        root = math.sqrt(min(trace * trace, max(0.0, discriminant)))
        eigenvalues = ((trace + root) / 2.0, (trace - root) / 2.0)

        entropy = math.fsum(
            -lam * math.log2(lam) for lam in eigenvalues if lam > 0.0
        )
        entropies.append(_snap_to_domain(entropy))
    return entropies
