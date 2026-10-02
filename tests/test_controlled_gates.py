"""Tests for the controlled gates cz, crx, cry and crz."""

from __future__ import annotations

import json
import math

import pytest

from quantum_circuit import cli
from quantum_circuit.equivalence import unitary_distance
from quantum_circuit.estimation import estimate
from quantum_circuit.noise import simulate_density_matrix
from quantum_circuit.openqasm import ParseError, ValidationError, parse
from quantum_circuit.optimizer import normalize_controlled_angle, optimize
from quantum_circuit.simulator import simulate_state_vector, unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


def _parse(body: str, qreg: str = "q[3]", creg: str = "c[3]"):
    return parse(HEADER + f"qreg {qreg};\ncreg {creg};\n{body}")


# ------------------------------------------------------------------- parsing


def test_new_gates_parse_with_positions():
    program = _parse(
        "cz q[0],q[1];\ncrx(pi/2) q[0],q[1];\ncry(-0.5) q[2],q[0];\ncrz(2*pi) q[1],q[2];\n"
    )
    kinds = [op.kind for op in program.operations]
    assert kinds == ["cz", "crx", "cry", "crz"]
    assert program.operations[0].targets == (0, 1)
    assert program.operations[1].params == (math.pi / 2,)
    assert program.operations[2].targets == (2, 0)
    assert program.operations[3].params == (2 * math.pi,)


@pytest.mark.parametrize("gate", ["cz", "crx", "cry", "crz"])
def test_same_control_and_target_is_validation_error(gate):
    text = f"{gate} q[1],q[1];\n" if gate == "cz" else f"{gate}(0.3) q[1],q[1];\n"
    with pytest.raises(ValidationError) as info:
        _parse(text)
    assert info.value.line >= 1 and info.value.column >= 1
    assert "control and target" in info.value.message


@pytest.mark.parametrize(
    "text",
    [
        "crx q[0],q[1];\n",  # missing parameter list
        "crx(0.1,0.2) q[0],q[1];\n",  # extra parameter
        "crx(0.1) q[0];\n",  # missing second operand
        "crx(0.1) q[0],q[1],q[2];\n",  # extra operand
        "cry() q[0],q[1];\n",  # empty parameter list
        "cz q[0];\n",  # missing target
    ],
)
def test_operand_structure_errors_are_parse_errors(text):
    with pytest.raises(ParseError) as info:
        _parse(text)
    assert info.value.line >= 1 and info.value.column >= 1


# ----------------------------------------------------------------- semantics


def test_cz_flips_phase_only_on_11():
    program = _parse("x q[0];\nx q[1];\ncz q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    state = simulate_state_vector(program)
    assert state[3] == -1 and all(state[i] == 0 for i in (0, 1, 2))


def test_cz_is_a_no_op_when_control_is_zero():
    program = _parse("cz q[0],q[1];\ncrz(pi) q[0],q[1];\ncrx(pi) q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    state = simulate_state_vector(program)
    assert abs(state[0] - 1) < 1e-12


def test_crx_applies_rx_when_control_is_one():
    program = _parse("x q[0];\ncrx(pi) q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    state = simulate_state_vector(program)
    assert abs(state[3] - complex(0, -1)) < 1e-12


def test_cry_applies_ry_when_control_is_one():
    program = _parse("x q[1];\ncry(pi/2) q[1],q[0];\n", qreg="q[2]", creg="c[2]")
    state = simulate_state_vector(program)
    half = 2.0**-0.5
    assert abs(state[2] - half) < 1e-12 and abs(state[3] - half) < 1e-12


def test_crz_applies_rz_when_control_is_one():
    program = _parse("x q[0];\ncrz(pi) q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    state = simulate_state_vector(program)
    assert abs(state[1] - complex(0, -1)) < 1e-12


def test_unitary_of_cz_is_diagonal_phase():
    program = _parse("cz q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    u = unitary_matrix(program)
    for i in range(4):
        for j in range(4):
            expected = (-1 if i == 3 else 1) if i == j else 0
            assert u[i][j] == expected


def test_density_matrix_matches_state_vector_without_noise():
    body = "h q[0];\ncry(0.7) q[0],q[1];\ncz q[1],q[0];\ncrx(0.3) q[1],q[0];\ncrz(0.9) q[0],q[1];\n"
    program = _parse(body, qreg="q[2]", creg="c[2]")
    state = simulate_state_vector(program)
    expected = [abs(a) ** 2 for a in state]
    diag = simulate_density_matrix(program, {"bit_flip": 0.0})
    assert all(abs(a - b) < 1e-12 for a, b in zip(expected, diag))


def test_density_matrix_noisy_evolution_stays_normalized():
    program = _parse(
        "h q[0];\ncrx(0.9) q[0],q[1];\ncz q[0],q[1];\ncry(0.4) q[1],q[0];\n",
        qreg="q[2]",
        creg="c[2]",
    )
    noise = {"amplitude_damping": 0.1, "phase_damping": 0.2, "bit_flip": 0.05, "depolarizing": 0.03}
    diag = simulate_density_matrix(program, noise)
    assert abs(sum(diag) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in diag)


# ----------------------------------------------------------------- estimate


def test_estimate_schema_2_with_ten_gate_counts():
    program = _parse("cz q[0],q[1];\ncrx(0.1) q[0],q[1];\ncry(0.2) q[1],q[2];\ncrz(0.3) q[2],q[0];\n")
    data = estimate(program, "state-vector")
    assert data["schema_version"] == 2
    assert list(data["gate_counts"]) == ["x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz"]
    assert data["gate_counts"] == {
        "x": 0, "h": 0, "cx": 0, "cz": 1, "rx": 0, "ry": 0, "rz": 0,
        "crx": 1, "cry": 1, "crz": 1,
    }
    assert data["gate_count"] == 4
    assert data["circuit_depth"] == 4


def test_estimate_old_gates_keep_schema_1():
    program = _parse("h q[0];\ncx q[0],q[1];\nrx(0.5) q[0];\n")
    data = estimate(program, "state-vector")
    assert data["schema_version"] == 1
    assert list(data["gate_counts"]) == ["x", "h", "cx", "rx", "ry", "rz"]


# ----------------------------------------------------------------- optimize


def test_normalize_controlled_angle_interval():
    assert normalize_controlled_angle(0.7) == 0.7
    assert normalize_controlled_angle(2 * math.pi) == 2 * math.pi
    assert normalize_controlled_angle(-2 * math.pi) == 2 * math.pi
    assert normalize_controlled_angle(3 * math.pi) == -math.pi
    assert normalize_controlled_angle(4 * math.pi) == 0.0
    assert normalize_controlled_angle(6 * math.pi) == 2 * math.pi


def test_cz_pairs_cancel_across_disjoint_gates():
    _, changed, qasm = optimize(_parse("cz q[0],q[1];\nh q[2];\ncz q[0],q[1];\n"))
    assert changed is True
    assert "cz" not in qasm and "h q[2];" in qasm


def test_cz_swapped_bits_do_not_cancel():
    _, changed, qasm = optimize(_parse("cz q[0],q[1];\ncz q[1],q[0];\n", qreg="q[2]", creg="c[2]"))
    assert qasm.count("cz") == 2


@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
def test_controlled_rotations_merge(gate):
    _, changed, qasm = optimize(_parse(f"{gate}(0.3) q[0],q[1];\nx q[2];\n{gate}(0.4) q[0],q[1];\n"))
    assert changed is True
    assert f"{gate}(0.7) q[0],q[1];" in qasm


def test_controlled_rotation_two_pi_is_kept():
    _, _, qasm = optimize(_parse("crz(2*pi) q[0],q[1];\n", qreg="q[2]", creg="c[2]"))
    assert f"crz({2 * math.pi!r}) q[0],q[1];" in qasm


def test_controlled_rotation_four_pi_is_deleted():
    _, changed, qasm = optimize(_parse("crx(4*pi) q[0],q[1];\n", qreg="q[2]", creg="c[2]"))
    assert changed is True
    assert "crx" not in qasm


def test_controlled_rotation_merge_normalizes_to_interval():
    _, _, qasm = optimize(_parse("cry(2*pi) q[0],q[1];\ncry(2*pi) q[0],q[1];\n", qreg="q[2]", creg="c[2]"))
    assert "cry" not in qasm  # 4*pi total is the identity


def test_optimized_new_gates_reparse_and_stay_equivalent():
    body = (
        "h q[0];\ncrx(0.9) q[0],q[1];\ncrx(0.2) q[0],q[1];\n"
        "cz q[0],q[1];\ncz q[0],q[1];\ncry(-0.3) q[1],q[0];\ncry(0.3) q[1],q[0];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    program = _parse(body, qreg="q[2]", creg="c[2]")
    _, changed, qasm = optimize(program)
    assert changed is True
    optimized = parse(qasm)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(optimized)) <= 1e-10
    assert optimize(optimized)[2] == qasm  # idempotent


# -------------------------------------------------------------------- CLI


def test_cli_simulate_and_probabilities_with_new_gates(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncrx(0.9) q[0],q[1];\ncz q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["simulate", path, "--shots", "100", "--seed", "5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert sum(data["counts"].values()) == 100

    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-12


def test_cli_estimate_schema_2_field_order(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\ncz q[0],q[1];\n")
    assert cli.main(["estimate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data) == [
        "schema_version", "mode", "num_qubits", "num_clbits", "gate_count",
        "measurement_count", "gate_counts", "circuit_depth", "entry_count",
        "complex_payload_bytes", "supported", "qubit_limit",
    ]
    assert data["schema_version"] == 2
    assert list(data["gate_counts"]) == ["x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz"]


def test_cli_same_qubit_error_has_position(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\ncrz(0.1) q[1],q[1];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1
