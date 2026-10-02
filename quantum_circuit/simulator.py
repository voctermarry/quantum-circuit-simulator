"""State-vector simulation and deterministic shot sampling."""

from __future__ import annotations

import cmath
import math
import random

from .openqasm import Operation, Program

_SQRT1_2 = 2.0**-0.5

# Exact-probability output drops entries within this tolerance of 0 and
# reports entries within it of 1 as exactly 1.
PROBABILITY_TOLERANCE = 1e-15


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


def _apply_gate(state: list[complex], op: Operation, num_qubits: int) -> None:
    """Apply one (non-measurement) *op* to *state* in place."""
    if op.kind == "x":
        _pauli_x(state, op.targets[0], num_qubits)
    elif op.kind == "h":
        _hadamard(state, op.targets[0], num_qubits)
    elif op.kind == "cx":
        _controlled_x(state, op.targets[0], op.targets[1], num_qubits)
    elif op.kind == "rx":
        _rx(state, op.targets[0], num_qubits, op.params[0])
    elif op.kind == "ry":
        _ry(state, op.targets[0], num_qubits, op.params[0])
    elif op.kind == "rz":
        _rz(state, op.targets[0], num_qubits, op.params[0])


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


def exact_measurement_probabilities(
    program: Program, probabilities: list[float]
) -> dict[str, object]:
    """Aggregate basis-state probabilities into exact classical outcomes.

    Keys are fixed-width classical bit strings (highest clbit index first),
    matching :func:`sample_counts_from_probabilities`, and are sorted
    lexicographically; classical bits never written by a measurement stay 0.
    Entries whose magnitude is at most :data:`PROBABILITY_TOLERANCE` are
    omitted and entries within the same tolerance of 1 are reported as the
    integer 1; every other value is a finite float in (0, 1). When the
    circuit has no measurements the only entry is the all-zero string with
    probability 1.
    """
    measurements = [(op.targets[0], op.targets[1]) for op in program.operations if op.kind == "measure"]
    width = program.num_clbits
    zero_key = format(0, f"0{width}b") if width else ""
    if not measurements:
        return {zero_key: 1}

    aggregated: dict[str, float] = {}
    for basis, probability in enumerate(probabilities):
        if probability == 0.0:
            continue
        classical = 0
        for qubit, clbit in measurements:
            if (basis >> qubit) & 1:
                classical |= 1 << clbit
        key = format(classical, f"0{width}b") if width else ""
        aggregated[key] = aggregated.get(key, 0.0) + probability

    result: dict[str, object] = {}
    for key in sorted(aggregated):
        value = aggregated[key]
        if abs(value) <= PROBABILITY_TOLERANCE:
            continue
        if abs(value - 1.0) <= PROBABILITY_TOLERANCE:
            result[key] = 1
        else:
            result[key] = value
    return result
