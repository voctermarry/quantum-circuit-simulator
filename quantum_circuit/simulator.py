"""State-vector simulation and deterministic shot sampling."""

from __future__ import annotations

import cmath
import math
import random

from .openqasm import Operation, Program

_SQRT1_2 = 2.0**-0.5


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


def _pauli_y(state: list[complex], qubit: int, num_qubits: int) -> None:
    # Y = [[0, -i], [i, 0]]: |0> -> i|1>, |1> -> -i|0>.
    bit = 1 << qubit
    step = bit << 1
    for base in range(0, 1 << num_qubits, step):
        for low in range(base, base + bit):
            high = low | bit
            a = state[low]
            b = state[high]
            state[low] = -1j * b
            state[high] = 1j * a


def _diagonal_phase(state: list[complex], qubit: int, num_qubits: int, factor: complex) -> None:
    """Apply ``diag(1, factor)`` on *qubit* (z/s/sdg/t/tdg share this form)."""
    bit = 1 << qubit
    for index in range(1 << num_qubits):
        if index & bit:
            state[index] = factor * state[index]


def _swap(state: list[complex], first: int, second: int, num_qubits: int) -> None:
    abit = 1 << first
    bbit = 1 << second
    for index in range(1 << num_qubits):
        # Each unordered pair is visited once: the state with first-bit 0
        # and second-bit 1 is exchanged with first-bit 1, second-bit 0.
        if not (index & abit) and (index & bbit):
            other = (index | abit) & ~bbit
            state[index], state[other] = state[other], state[index]


def _controlled_x(state: list[complex], control: int, target: int, num_qubits: int) -> None:
    cbit = 1 << control
    tbit = 1 << target
    for index in range(1 << num_qubits):
        if (index & cbit) and not (index & tbit):
            other = index | tbit
            if index < other:
                state[index], state[other] = state[other], state[index]


def _controlled_single(
    state: list[complex],
    control: int,
    target: int,
    num_qubits: int,
    matrix: tuple[complex, complex, complex, complex],
) -> None:
    """Apply a single-qubit matrix to *target* where the *control* bit is 1."""
    g00, g01, g10, g11 = matrix
    cbit = 1 << control
    tbit = 1 << target
    step = tbit << 1
    for base in range(0, 1 << num_qubits, step):
        for low in range(base, base + tbit):
            if not (low & cbit):
                continue
            high = low | tbit
            a = state[low]
            b = state[high]
            state[low] = g00 * a + g01 * b
            state[high] = g10 * a + g11 * b


def _rotation_matrix(kind: str, theta: float) -> tuple[complex, complex, complex, complex]:
    """The single-qubit matrix of an rx/ry/rz rotation by *theta* radians."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if kind == "rx":
        return (c, -1j * s, -1j * s, c)
    if kind == "ry":
        return (c, -s, s, c)
    # rz
    return (cmath.exp(-0.5j * theta), 0j, 0j, cmath.exp(0.5j * theta))


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


def _apply_gate(state: list[complex], op: Operation, num_qubits: int) -> None:
    """Apply one (non-measurement) *op* to *state* in place."""
    if op.kind == "x":
        _pauli_x(state, op.targets[0], num_qubits)
    elif op.kind == "y":
        _pauli_y(state, op.targets[0], num_qubits)
    elif op.kind == "z":
        _diagonal_phase(state, op.targets[0], num_qubits, -1 + 0j)
    elif op.kind == "s":
        _diagonal_phase(state, op.targets[0], num_qubits, 1j)
    elif op.kind == "sdg":
        _diagonal_phase(state, op.targets[0], num_qubits, -1j)
    elif op.kind == "t":
        _diagonal_phase(state, op.targets[0], num_qubits, cmath.exp(0.25j * math.pi))
    elif op.kind == "tdg":
        _diagonal_phase(state, op.targets[0], num_qubits, cmath.exp(-0.25j * math.pi))
    elif op.kind == "h":
        _hadamard(state, op.targets[0], num_qubits)
    elif op.kind == "swap":
        _swap(state, op.targets[0], op.targets[1], num_qubits)
    elif op.kind == "cx":
        _controlled_x(state, op.targets[0], op.targets[1], num_qubits)
    elif op.kind == "cz":
        _controlled_single(
            state, op.targets[0], op.targets[1], num_qubits, (1 + 0j, 0j, 0j, -1 + 0j)
        )
    elif op.kind == "rx":
        _rx(state, op.targets[0], num_qubits, op.params[0])
    elif op.kind == "ry":
        _ry(state, op.targets[0], num_qubits, op.params[0])
    elif op.kind == "rz":
        _rz(state, op.targets[0], num_qubits, op.params[0])
    elif op.kind in ("crx", "cry", "crz"):
        matrix = _rotation_matrix(op.kind[1:], op.params[0])
        _controlled_single(state, op.targets[0], op.targets[1], num_qubits, matrix)


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
        if op.kind == "measure":
            # Measurements affect only the sampled classical outcomes.
            continue
        _apply_gate(state, op, n)
    return state


def unitary_matrix(program: Program) -> list[list[complex]]:
    """Return the full unitary U induced by the gates (pre-measurement).

    The matrix acts on basis vectors with qubit ``i`` in bit ``i`` (q[0] is
    the least significant bit), matching :func:`simulate_state_vector`.
    Measurement operations are ignored. Columns are evolved independently by
    applying each gate in source order, so for a circuit with gates G1..Gk the
    result is U = Gk @ ... @ G1.
    """
    n = program.num_qubits
    dim = 1 << n
    gate_ops = [op for op in program.operations if op.kind != "measure"]

    columns: list[list[complex]] = []
    for start in range(dim):
        state = [0j] * dim
        state[start] = 1 + 0j
        for op in gate_ops:
            _apply_gate(state, op, n)
        columns.append(state)

    # Transpose columns-of-U into rows.
    return [[columns[col][row] for col in range(dim)] for row in range(dim)]


def measurement_probabilities(program: Program, basis_probabilities: list[float]) -> dict[str, float]:
    """Aggregate basis-state probabilities into classical-outcome probabilities.

    Uses the same qubit-to-clbit mapping as :func:`sample_counts` (each
    measured qubit contributes its bit to the classical index; unwritten
    clbits stay 0) and the same fixed-width, lexicographically ordered key
    convention. With no measurements the only possible outcome is the
    all-zero string, carrying total probability 1.
    """
    width = program.num_clbits
    measurements = [(op.targets[0], op.targets[1]) for op in program.operations if op.kind == "measure"]

    if not measurements:
        return {format(0, f"0{width}b") if width else "": 1.0}

    aggregated: dict[int, float] = {}
    for basis, probability in enumerate(basis_probabilities):
        classical = 0
        for qubit, clbit in measurements:
            if (basis >> qubit) & 1:
                classical |= 1 << clbit
        aggregated[classical] = aggregated.get(classical, 0.0) + probability

    return {
        format(classical, f"0{width}b") if width else "": probability
        for classical, probability in sorted(aggregated.items())
    }


def sample_counts(program: Program, state: list[complex], shots: int, seed: int) -> dict[str, int]:
    """Draw *shots* samples from *state* deterministically from *seed*.

    Returns a mapping from fixed-width classical bit strings (highest clbit
    index first) to occurrence counts. Only observed outcomes are present.
    """
    probabilities = [abs(amplitude) ** 2 for amplitude in state]
    return sample_counts_from_probabilities(program, probabilities, shots, seed)


def sample_counts_from_probabilities(
    program: Program, probabilities: list[float], shots: int, seed: int
) -> dict[str, int]:
    """Draw *shots* samples from explicit basis-state probabilities.

    Used by the density-matrix path, which supplies the diagonal of the
    final density matrix; sampling semantics match the state-vector path.
    """
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
