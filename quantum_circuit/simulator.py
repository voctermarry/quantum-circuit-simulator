"""State-vector simulation and deterministic measurement sampling."""

from __future__ import annotations

import math
import random

from .qasm import Program

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _apply_x(state: list[complex], qubit: int) -> None:
    bit = 1 << qubit
    for i in range(len(state)):
        if not i & bit:
            state[i], state[i | bit] = state[i | bit], state[i]


def _apply_h(state: list[complex], qubit: int) -> None:
    step = 1 << qubit
    for block in range(0, len(state), step << 1):
        for offset in range(step):
            i0 = block + offset
            i1 = i0 + step
            a = state[i0]
            b = state[i1]
            state[i0] = (a + b) * _INV_SQRT2
            state[i1] = (a - b) * _INV_SQRT2


def _apply_cx(state: list[complex], control: int, target: int) -> None:
    cbit = 1 << control
    tbit = 1 << target
    for i in range(len(state)):
        if (i & cbit) and not (i & tbit):
            j = i | tbit
            state[i], state[j] = state[j], state[i]


def run_state_vector(program: Program) -> list[complex]:
    """Apply all gates in source order to the |0...0> state."""
    state = [0j] * (1 << program.num_qubits)
    state[0] = 1.0 + 0j
    for gate in program.gates:
        if gate.name == "x":
            _apply_x(state, gate.qubits[0])
        elif gate.name == "h":
            _apply_h(state, gate.qubits[0])
        else:  # cx
            _apply_cx(state, gate.qubits[0], gate.qubits[1])
    return state


def _probability_one(state: list[complex], qubit: int) -> float:
    bit = 1 << qubit
    total = 0.0
    for i, amplitude in enumerate(state):
        if i & bit:
            total += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    return total


def _project(state: list[complex], qubit: int, outcome: int) -> None:
    bit = 1 << qubit
    norm_sq = 0.0
    for i, amplitude in enumerate(state):
        if bool(i & bit) != bool(outcome):
            state[i] = 0j
        else:
            norm_sq += amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    factor = 1.0 / math.sqrt(norm_sq)
    for i in range(len(state)):
        state[i] *= factor


def sample_counts(
    program: Program,
    state: list[complex],
    shots: int,
    seed: int,
) -> dict[str, int]:
    """Run measurements in source order with a seeded PRNG.

    Each shot uses an independent copy of the final state vector and
    collapses measured qubits one by one.  Unmeasured classical bits
    stay zero.  Repeated calls with the same arguments draw the PRNG in
    exactly the same order and return identical counts.
    """
    rng = random.Random(seed)
    counts: dict[str, int] = {}

    for _ in range(shots):
        shot_state = state.copy()
        values: dict[int, int] = {}
        for measurement in program.measurements:
            p_one = _probability_one(shot_state, measurement.qubit)
            outcome = 1 if rng.random() < p_one else 0
            _project(shot_state, measurement.qubit, outcome)
            values[measurement.clbit] = outcome

        # Highest classical index first, fixed width, unmeasured bits zero.
        key = "".join(
            "1" if values.get(index, 0) == 1 else "0"
            for index in range(program.num_clbits - 1, -1, -1)
        )
        counts[key] = counts.get(key, 0) + 1

    return {key: counts[key] for key in sorted(counts)}
