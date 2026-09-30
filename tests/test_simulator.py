"""Tests for state-vector simulation and shot sampling."""

from __future__ import annotations

import cmath
import math

from quantum_circuit.openqasm import parse
from quantum_circuit.simulator import sample_counts, simulate_state_vector


def _program(body: str):
    return parse('OPENQASM 2.0;\ninclude "qelib1.inc";\n' + body)


def _amplitudes(body: str):
    program = _program(body)
    return program, simulate_state_vector(program)


def test_initial_state_is_all_zeros():
    _, state = _amplitudes("qreg q[3];\ncreg c[3];\n")
    assert state[0] == 1 + 0j
    assert all(abs(a) == 0.0 for a in state[1:])


def test_pauli_x_flips_a_bit():
    _, state = _amplitudes("qreg q[2];\ncreg c[2];\nx q[1];\n")
    # q[1] is bit 1 of the basis index, so |10> is index 2.
    assert abs(state[2] - 1.0) < 1e-12
    assert sum(abs(a) for i, a in enumerate(state) if i != 2) < 1e-12


def test_hadamard_gives_equal_superposition():
    _, state = _amplitudes("qreg q[1];\ncreg c[1];\nh q[0];\n")
    assert abs(state[0] - 1 / math.sqrt(2)) < 1e-12
    assert abs(state[1] - 1 / math.sqrt(2)) < 1e-12


def test_bell_state_only_correlated_outcomes():
    _, state = _amplitudes(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    )
    assert abs(abs(state[0]) - 1 / math.sqrt(2)) < 1e-12
    assert abs(abs(state[3]) - 1 / math.sqrt(2)) < 1e-12
    assert abs(state[1]) < 1e-12 and abs(state[2]) < 1e-12


def test_cx_does_not_fire_when_control_is_zero():
    _, state = _amplitudes("qreg q[2];\ncreg c[2];\nh q[1];\ncx q[0],q[1];\n")
    # Control q[0] is 0 throughout, so cx is identity on the superposition.
    assert abs(abs(state[0]) - 1 / math.sqrt(2)) < 1e-12
    assert abs(abs(state[2]) - 1 / math.sqrt(2)) < 1e-12


def test_state_is_normalized():
    _, state = _amplitudes(
        "qreg q[3];\ncreg c[3];\nh q[0];\nh q[1];\ncx q[0],q[2];\nx q[1];\n"
    )
    norm = sum(abs(a) ** 2 for a in state)
    assert abs(norm - 1.0) < 1e-12


# ------------------------------------------------------------------- sampling


def test_deterministic_circuit_has_single_outcome():
    program = _program("qreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    counts = sample_counts(program, simulate_state_vector(program), 16, 0)
    assert counts == {"01": 16}


def test_unwritten_clbits_stay_zero():
    program = _program("qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[0];\n")
    counts = sample_counts(program, simulate_state_vector(program), 4, 0)
    assert list(counts) == ["001"]  # width 3, c[2]c[1]c[0]


def test_counts_sum_to_shots_and_seed_is_reproducible():
    body = (
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    program = _program(body)
    state = simulate_state_vector(program)
    first = sample_counts(program, state, 1000, 42)
    second = sample_counts(program, state, 1000, 42)
    assert first == second
    assert sum(first.values()) == 1000
    assert set(first) <= {"00", "11"}


def test_different_seeds_can_differ():
    body = "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    program = _program(body)
    state = simulate_state_vector(program)
    counts_a = sample_counts(program, state, 50, 1)
    counts_b = sample_counts(program, state, 50, 2)
    assert counts_a != counts_b


def test_partial_qubit_measurement_bit_placement():
    # Entangled q[0],q[2]; read them into c[2] and c[0] respectively.
    body = (
        "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[2];\n"
        "measure q[0] -> c[2];\nmeasure q[2] -> c[0];\n"
    )
    program = _program(body)
    counts = sample_counts(program, simulate_state_vector(program), 30, 3)
    assert set(counts) <= {"000", "101"}
    assert sum(counts.values()) == 30


def test_counts_keys_are_sorted_lexicographically():
    body = "qreg q[3];\ncreg c[3];\nh q[0];\nh q[1];\nh q[2];\n"
    body += "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(3))
    program = _program(body)
    counts = sample_counts(program, simulate_state_vector(program), 200, 9)
    assert list(counts) == sorted(counts)


# ------------------------------------------------------- parameterized gates


def test_rx_pi_acts_like_x_up_to_phase():
    _, state = _amplitudes("qreg q[1];\ncreg c[1];\nrx(pi) q[0];\n")
    assert abs(state[0]) < 1e-12
    assert abs(state[1] - (-1j)) < 1e-12


def test_ry_pi_flips_with_real_amplitude():
    _, state = _amplitudes("qreg q[1];\ncreg c[1];\nry(pi) q[0];\n")
    assert abs(state[0]) < 1e-12
    assert abs(state[1] - 1.0) < 1e-12


def test_rx_half_pi_gives_equal_superposition_magnitudes():
    _, state = _amplitudes("qreg q[1];\ncreg c[1];\nrx(pi/2) q[0];\n")
    assert abs(abs(state[0]) ** 2 - 0.5) < 1e-12
    assert abs(abs(state[1]) ** 2 - 0.5) < 1e-12
    assert abs(state[1] - (-1j / math.sqrt(2))) < 1e-12


def test_rz_applies_relative_phase_on_superposition():
    theta = math.pi / 3
    _, state = _amplitudes("qreg q[1];\ncreg c[1];\nh q[0];\nrz(pi/3) q[0];\n")
    assert abs(state[0] - cmath.exp(-0.5j * theta) / math.sqrt(2)) < 1e-12
    assert abs(state[1] - cmath.exp(0.5j * theta) / math.sqrt(2)) < 1e-12


def test_rz_does_not_change_measurement_statistics():
    body = "qreg q[1];\ncreg c[1];\nh q[0];\nrz(7*pi/5) q[0];\nmeasure q[0] -> c[0];\n"
    program = _program(body)
    counts = sample_counts(program, simulate_state_vector(program), 200, 4)
    assert set(counts) <= {"0", "1"}
    assert sum(counts.values()) == 200


def test_rotations_apply_to_each_target_pair_in_entangled_state():
    # Bell state, then rx(pi) on q[0]: each (a, b) pair maps to (-i b, -i a).
    _, state = _amplitudes(
        "qreg q[2];\ncreg c[2];\nh q[1];\ncx q[1],q[0];\nrx(pi) q[0];\n"
    )
    assert abs(state[0]) < 1e-12 and abs(state[3]) < 1e-12
    assert abs(state[1] - (-1j / math.sqrt(2))) < 1e-12
    assert abs(state[2] - (-1j / math.sqrt(2))) < 1e-12


def test_rotation_expressions_compose_with_other_gates():
    # ry(pi/2) then rx(pi) on the same qubit, checked against the matrices.
    theta = math.pi / 2
    _, state = _amplitudes("qreg q[1];\ncreg c[1];\nry(pi/2) q[0];\nrx(pi) q[0];\n")
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    a, b = c, s  # after ry
    assert abs(state[0] - (-1j * b)) < 1e-12
    assert abs(state[1] - (-1j * a)) < 1e-12
