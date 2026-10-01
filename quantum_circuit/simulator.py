"""State-vector simulation and deterministic shot sampling."""

from __future__ import annotations

import cmath
import math
import random

from .openqasm import Program

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


def apply_gates(program: Program, state: list[complex]) -> None:
    """Apply every quantum gate of *program* to *state* in place.

    Measurements are ignored: the spec guarantees every measurement comes
    after every gate and each qubit is measured at most once, so they affect
    only the sampled classical outcomes, not the state.
    """
    n = program.num_qubits
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
    apply_gates(program, state)
    return state


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
