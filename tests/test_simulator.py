"""Tests for state-vector simulation and deterministic sampling."""

from __future__ import annotations

import math

from quantum_circuit.qasm import parse
from quantum_circuit.simulator import run_state_vector, sample_counts

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def simulate(text: str, shots: int = 1024, seed: int = 0) -> dict[str, int]:
    program = parse(text)
    state = run_state_vector(program)
    return sample_counts(program, state, shots, seed)


def test_x_gate_flips_the_qubit():
    counts = simulate(HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    assert counts == {"1": 1024}


def test_identity_circuit_measures_zeros():
    counts = simulate(HEADER + "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    assert counts == {"00": 1024}


def test_bell_state_splits_evenly_for_even_shots():
    text = (
        HEADER
        + "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    counts = simulate(text, shots=1000, seed=7)
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 1000
    assert 400 < counts["00"] < 600
    assert 400 < counts["11"] < 600


def test_keys_have_fixed_width_and_high_index_first():
    # Flip only qubit 1, measured into classical bit 1: key must be "10".
    text = (
        HEADER
        + "qreg q[2];\ncreg c[2];\nx q[1];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    counts = simulate(text)
    assert list(counts) == ["10"]


def test_unmeasured_classical_bits_remain_zero():
    # Only c[0] is written; c[1] stays 0 so the key is "01".
    text = HEADER + "qreg q[1];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\n"
    counts = simulate(text)
    assert counts == {"01": 1024}


def test_partial_measurement_collapses_entangled_state():
    # Bell state; measuring qubit 0 first always forces qubit 1 to agree.
    text = (
        HEADER
        + "qreg q[2];\ncreg c[1];\nh q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\n"
    )
    counts = simulate(text, shots=200, seed=3)
    assert set(counts) <= {"0", "1"}
    assert sum(counts.values()) == 200


def test_same_seed_gives_byte_identical_counts():
    text = (
        HEADER
        + "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0], q[1];\nx q[2];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    first = simulate(text, shots=500, seed=42)
    second = simulate(text, shots=500, seed=42)
    assert first == second


def test_different_seeds_may_differ():
    text = (
        HEADER
        + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    )
    counts = [simulate(text, shots=40, seed=seed) for seed in range(20)]
    assert any(counts[0] != other for other in counts[1:])


def test_negative_seed_is_accepted():
    text = HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    assert simulate(text, shots=50, seed=-12345) == simulate(text, shots=50, seed=-12345)


def test_state_vector_hadamard_superposition():
    program = parse(HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\n")
    state = run_state_vector(program)
    assert len(state) == 2
    assert math.isclose(state[0].real, 1 / math.sqrt(2), abs_tol=1e-12)
    assert math.isclose(state[1].real, 1 / math.sqrt(2), abs_tol=1e-12)


def test_state_vector_bell_state():
    program = parse(
        HEADER + "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n"
    )
    state = run_state_vector(program)
    assert math.isclose(abs(state[0]), 1 / math.sqrt(2), abs_tol=1e-12)
    assert state[1] == 0j
    assert state[2] == 0j
    assert math.isclose(abs(state[3]), 1 / math.sqrt(2), abs_tol=1e-12)


def test_counts_total_equals_shots():
    text = (
        HEADER
        + "qreg q[2];\ncreg c[2];\nh q[0];\nh q[1];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    for shots in (1, 7, 999):
        counts = simulate(text, shots=shots, seed=11)
        assert sum(counts.values()) == shots


def test_counts_keys_are_sorted_lexicographically():
    text = (
        HEADER
        + "qreg q[3];\ncreg c[3];\nh q[0];\nx q[2];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    counts = simulate(text, shots=200, seed=5)
    keys = list(counts)
    assert keys == sorted(keys)
