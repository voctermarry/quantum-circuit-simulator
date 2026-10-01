"""State-vector and density-matrix simulation with deterministic shot sampling."""

from __future__ import annotations

import cmath
import math
import random

from .noise import channel_kraus
from .openqasm import Operation, Program

_SQRT1_2 = 2.0**-0.5

# A density matrix has 4**n entries; noisy simulation is capped well below
# the state-vector register limit to keep it tractable.
MAX_NOISE_QUBITS = 10


def _hadamard(state: list[complex], qubit: int, num_qubits: int) -> None:
    bit = 1 << qubit
    step = bit << 1
    for base in range(0, 1 << num_qubits, step):
        for low in range(base, base + bit):
            high = low | bit
            a = state[low]
            b = state[high]
            state[low] = (a + b) * _SQRT1_2
            state[high] = (a - b) * _SQRT1_2


def _pauli_x(state: list[complex], qubit: int, num_qubits: int) -> None:
    bit = 1 << qubit
    step = bit << 1
    for base in range(0, 1 << num_qubits, step):
        for low in range(base, base + bit):
            high = low | bit
            state[low], state[high] = state[high], state[low]


def _controlled_x(state: list[complex], control: int, target: int, num_qubits: int) -> None:
    cbit = 1 << control
    tbit = 1 << target
    for index in range(1 << num_qubits):
        if (index & cbit) and not (index & tbit):
            other = index | tbit
            if index < other:
                state[index], state[other] = state[other], state[index]


def _rx(state: list[complex], qubit: int, num_qubits: int, theta: float) -> None:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    bit = 1 << qubit
    step = bit << 1
    for base in range(0, 1 << num_qubits, step):
        for low in range(base, base + bit):
            high = low | bit
            a = state[low]
            b = state[high]
            state[low] = c * a - 1j * s * b
            state[high] = -1j * s * a + c * b


def _ry(state: list[complex], qubit: int, num_qubits: int, theta: float) -> None:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    bit = 1 << qubit
    step = bit << 1
    for base in range(0, 1 << num_qubits, step):
        for low in range(base, base + bit):
            high = low | bit
            a = state[low]
            b = state[high]
            state[low] = c * a - s * b
            state[high] = s * a + c * b


def _rz(state: list[complex], qubit: int, num_qubits: int, theta: float) -> None:
    lo_factor = cmath.exp(-0.5j * theta)
    hi_factor = cmath.exp(0.5j * theta)
    bit = 1 << qubit
    for index in range(1 << num_qubits):
        if index & bit:
            state[index] = hi_factor * state[index]
        else:
            state[index] = lo_factor * state[index]


def simulate_state_vector(program: Program) -> list[complex]:
    """Return the final state vector after applying all quantum gates.

    Basis-state integer ``k`` stores qubit ``i`` in bit ``i`` (q[0] is the
    least significant bit). Measurements do not collapse the state: the spec
    guarantees every measurement comes after every gate and each qubit is
    measured at most once, so non-destructive sampling of the final state is
    equivalent.
    """
    n = program.num_qubits
    state = [0j] * (1 << n)
    state[0] = 1 + 0j

    for op in program.operations:
        if op.kind == "x":
            _pauli_x(state, op.targets[0], n)
        elif op.kind == "h":
            _hadamard(state, op.targets[0], n)
        elif op.kind == "cx":
            _controlled_x(state, op.targets[0], op.targets[1], n)
        elif op.kind == "rx":
            _rx(state, op.targets[0], n, op.params[0])
        elif op.kind == "ry":
            _ry(state, op.targets[0], n, op.params[0])
        elif op.kind == "rz":
            _rz(state, op.targets[0], n, op.params[0])
        # "measure" operations affect only the sampled classical outcomes.
    return state


def sample_counts(program: Program, state: list[complex], shots: int, seed: int) -> dict[str, int]:
    """Draw *shots* samples from *state* deterministically from *seed*.

    Returns a mapping from fixed-width classical bit strings (highest clbit
    index first) to occurrence counts. Only observed outcomes are present.
    """
    probabilities = [abs(amplitude) ** 2 for amplitude in state]
    return _sample_from_probabilities(program, probabilities, shots, seed)


def _sample_from_probabilities(
    program: Program, probabilities: list[float], shots: int, seed: int
) -> dict[str, int]:
    """Draw *shots* samples from basis-state *probabilities* deterministically."""
    measurements = [(op.targets[0], op.targets[1]) for op in program.operations if op.kind == "measure"]

    total = sum(probabilities)
    cumulative = []
    running = 0.0
    for probability in probabilities:
        running += probability / total
        cumulative.append(running)
    if cumulative:
        cumulative[-1] = 1.0

    rng = random.Random(seed)
    width = program.num_clbits
    counts: dict[str, int] = {}
    for _ in range(shots):
        point = rng.random()
        # Binary search for the first cumulative value >= point.
        lo, hi = 0, len(cumulative)
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < point:
                lo = mid + 1
            else:
                hi = mid
        basis = lo

        classical = 0
        for qubit, clbit in measurements:
            if (basis >> qubit) & 1:
                classical |= 1 << clbit
        key = format(classical, f"0{width}b") if width else ""
        counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items()))


# ----------------------------------------------------------- density matrices


def _gate_matrix(op: Operation) -> tuple[tuple[complex, ...], ...]:
    """Return the 2x2 unitary of a single-qubit gate operation."""
    kind = op.kind
    if kind == "x":
        return ((0, 1), (1, 0))
    if kind == "h":
        return ((_SQRT1_2, _SQRT1_2), (_SQRT1_2, -_SQRT1_2))
    theta = op.params[0]
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if kind == "rx":
        return ((c, -1j * s), (-1j * s, c))
    if kind == "ry":
        return ((c, -s), (s, c))
    if kind == "rz":
        return ((cmath.exp(-0.5j * theta), 0), (0, cmath.exp(0.5j * theta)))
    raise ValueError(f"not a single-qubit gate: {kind!r}")


# cx with qubits ordered (control, target): pattern bit 0 is the control,
# bit 1 the target, so only |10> and |11> (patterns 1 and 3) are swapped.
_CX_MATRIX = (
    (1, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 1, 0),
    (0, 1, 0, 0),
)


def _superoperator(ops: list[tuple[tuple[complex, ...], ...]], dim: int) -> list[tuple[int, int, int, int, complex]]:
    """Compile Kraus *ops* into sparse superoperator entries.

    Each entry ``(i, j, p, q, coef)`` contributes ``coef * rho[p][q]`` to
    ``rho'[i][j]`` within the extracted block, i.e. ``coef`` is
    ``sum_K K[i][p] * conj(K[j][q])``. Zero coefficients are dropped.
    """
    entries = []
    for i in range(dim):
        for j in range(dim):
            for p in range(dim):
                for q in range(dim):
                    coef = 0j
                    for matrix in ops:
                        coef += matrix[i][p] * matrix[j][q].conjugate()
                    if coef:
                        entries.append((i, j, p, q, coef))
    return entries


def _apply_superoperator(
    rho: list[list[complex]],
    num_qubits: int,
    qubits: tuple[int, ...],
    entries: list[tuple[int, int, int, int, complex]],
) -> None:
    """Apply a sparse superoperator acting on *qubits* to *rho* in place."""
    size = 1 << num_qubits
    dim = 1 << len(qubits)
    masks = []
    for pattern in range(dim):
        mask = 0
        for k, qubit in enumerate(qubits):
            if (pattern >> k) & 1:
                mask |= 1 << qubit
        masks.append(mask)
    involved = 0
    for qubit in qubits:
        involved |= 1 << qubit
    rest = [index for index in range(size) if not index & involved]

    for rest_row in rest:
        rows = [rest_row | masks[p] for p in range(dim)]
        rho_rows = [rho[row] for row in rows]
        for rest_col in rest:
            cols = [rest_col | masks[q] for q in range(dim)]
            block = [[rho_rows[p][cols[q]] for q in range(dim)] for p in range(dim)]
            new = [[0j] * dim for _ in range(dim)]
            for i, j, p, q, coef in entries:
                new[i][j] += coef * block[p][q]
            for p in range(dim):
                row = rho_rows[p]
                for q in range(dim):
                    row[cols[q]] = new[p][q]


def simulate_density_matrix(program: Program, noise: dict[str, float]) -> list[list[complex]]:
    """Return the final density matrix with *noise* applied after every gate.

    The state starts as ``|0...0><0...0|``. After each quantum gate, every
    configured channel is applied to the qubits the gate acts on: for ``cx``
    the two qubits are processed in ascending index order, and for each qubit
    the channels run in the fixed order amplitude_damping, phase_damping,
    bit_flip, depolarizing. The evolution is fully deterministic and does not
    consume any randomness.
    """
    n = program.num_qubits
    size = 1 << n
    rho = [[0j] * size for _ in range(size)]
    rho[0][0] = 1 + 0j

    channels = [(name, _superoperator(channel_kraus(name, p), 2)) for name, p in noise.items()]
    cx_super = _superoperator([_CX_MATRIX], 4)

    for op in program.operations:
        if op.kind == "measure":
            continue
        if op.kind == "cx":
            _apply_superoperator(rho, n, op.targets, cx_super)
            noise_qubits = sorted(op.targets)
        else:
            _apply_superoperator(rho, n, op.targets, _superoperator([_gate_matrix(op)], 2))
            noise_qubits = op.targets
        for qubit in noise_qubits:
            for _name, entries in channels:
                _apply_superoperator(rho, n, (qubit,), entries)
    return rho


def sample_counts_noisy(
    program: Program, rho: list[list[complex]], shots: int, seed: int
) -> dict[str, int]:
    """Draw *shots* samples from the diagonal of density matrix *rho*."""
    probabilities = [rho[index][index].real for index in range(1 << program.num_qubits)]
    return _sample_from_probabilities(program, probabilities, shots, seed)
