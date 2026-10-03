"""Tests for the gates y, z, s, sdg, t, tdg and swap."""

from __future__ import annotations

import cmath
import json
import math

import pytest

from quantum_circuit import cli
from quantum_circuit.circuit import Circuit
from quantum_circuit.equivalence import unitary_distance
from quantum_circuit.estimation import estimate
from quantum_circuit.noise import simulate_density_matrix
from quantum_circuit.openqasm import ParseError, ValidationError, parse
from quantum_circuit.optimizer import optimize
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


def test_new_gates_parse():
    program = _parse(
        "y q[0];\nz q[1];\ns q[2];\nsdg q[0];\nt q[1];\ntdg q[2];\nswap q[0],q[2];\n"
    )
    kinds = [op.kind for op in program.operations]
    assert kinds == ["y", "z", "s", "sdg", "t", "tdg", "swap"]
    assert program.operations[-1].targets == (0, 2)
    assert all(op.params == () for op in program.operations)


def test_swap_same_operand_is_validation_error_with_position():
    with pytest.raises(ValidationError) as info:
        _parse("swap q[1],q[1];\n")
    assert info.value.line == 5
    assert info.value.column > 1


@pytest.mark.parametrize(
    "text",
    [
        "swap q[0];\n",  # missing second operand
        "swap q[0],q[1],q[2];\n",  # extra operand
        "swap(0.5) q[0],q[1];\n",  # unexpected parameter list
        "y q[0],q[1];\n",  # extra operand on a single-qubit gate
        "t(0.5) q[0];\n",  # unexpected parameter on a fixed gate
    ],
)
def test_operand_structure_errors_are_parse_errors(text):
    with pytest.raises(ParseError) as info:
        _parse(text)
    assert info.value.line >= 1 and info.value.column >= 1


def test_cli_swap_same_operand_is_validation_error(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\nswap q[1],q[1];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "validation_error"
    assert data["line"] == 5
    assert data["column"] > 1


# ----------------------------------------------------------------- semantics


def test_pauli_y_maps_basis_states():
    # y|0> = i|1>
    state = simulate_state_vector(_parse("y q[0];\n", qreg="q[1]", creg="c[1]"))
    assert state == pytest.approx([0j, 1j])
    # y|1> = -i|0>
    state = simulate_state_vector(_parse("x q[0];\ny q[0];\n", qreg="q[1]", creg="c[1]"))
    assert state == pytest.approx([-1j, 0j])


def test_z_s_t_phases():
    # Each phase gate leaves |0> unchanged and multiplies |1> by its phase.
    for gate, phase in [
        ("z", -1 + 0j),
        ("s", 1j),
        ("sdg", -1j),
        ("t", cmath.exp(0.25j * math.pi)),
        ("tdg", cmath.exp(-0.25j * math.pi)),
    ]:
        state = simulate_state_vector(
            _parse(f"x q[0];\n{gate} q[0];\n", qreg="q[1]", creg="c[1]")
        )
        assert state == pytest.approx([0j, phase])


def test_swap_exchanges_qubits():
    # |01> (q[0] = 1) becomes |10> (q[1] = 1).
    state = simulate_state_vector(
        _parse("x q[0];\nswap q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    )
    assert state == pytest.approx([0j, 0j, 1 + 0j, 0j])
    # swap is symmetric in its operands.
    mirror = simulate_state_vector(
        _parse("x q[0];\nswap q[1],q[0];\n", qreg="q[2]", creg="c[2]")
    )
    assert mirror == pytest.approx(state)


def test_unitary_matrix_matches_state_vector():
    program = _parse(
        "h q[0];\nt q[0];\nswap q[0],q[1];\nsdg q[1];\ny q[2];\nz q[0];\n"
    )
    unitary = unitary_matrix(program)
    state = simulate_state_vector(program)
    assert [row[0] for row in unitary] == pytest.approx(state)


def test_density_matrix_diagonal_matches_state_vector():
    body = "h q[0];\ns q[1];\nswap q[0],q[1];\nt q[0];\n"
    program = _parse(body, qreg="q[2]", creg="c[2]")
    expected = [abs(a) ** 2 for a in simulate_state_vector(program)]
    noisy = simulate_density_matrix(program, {"bit_flip": 0.0})
    assert noisy == pytest.approx(expected)


def test_noisy_swap_applies_channels_to_both_qubits():
    # bit_flip with p = 1 fires after every gate on each qubit the gate
    # touched: x q[0] -> flip q0 (back to |00>); swap |00> -> |00>, then
    # flip q0 (index 1), then flip q1 (index 3): both swap qubits noised.
    program = _parse("x q[0];\nswap q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    diagonal = simulate_density_matrix(program, {"bit_flip": 1.0})
    assert diagonal == pytest.approx([0.0, 0.0, 0.0, 1.0])


# ------------------------------------------------------------------ optimize


@pytest.mark.parametrize("gate", ["y", "z", "h", "s", "t"])
def test_inverse_and_self_pairs_cancel(gate):
    inverse = {"s": "sdg", "t": "tdg"}.get(gate, gate)
    program = _parse(f"{gate} q[0];\n{inverse} q[0];\n", qreg="q[1]", creg="c[1]")
    gates, changed, qasm = optimize(program)
    assert changed
    assert gates == ()
    assert f"{gate} q[0];" not in qasm


def test_swap_pair_cancels_across_disjoint_gates():
    program = _parse(
        "swap q[0],q[2];\nh q[1];\nswap q[2],q[0];\n",
    )
    gates, changed, qasm = optimize(program)
    assert changed
    assert [g.kind for g in gates] == ["h"]


def test_optimize_output_reparses_and_stays_equivalent(write_qasm, capsys):
    body = (
        "qreg q[2];\ncreg c[2];\ny q[0];\ns q[1];\nswap q[0],q[1];\n"
        "sdg q[1];\nt q[0];\ntdg q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    path = write_qasm(body)
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 0
    first = json.loads(captured.out)
    optimized_path = write_qasm("", "optimized.qasm")
    with open(optimized_path, "w", encoding="utf-8") as handle:
        handle.write(first["qasm"])
    # The canonical text re-parses and is a fixed point of optimize.
    rc = cli.main(["optimize", optimized_path])
    second = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert second["changed"] is False
    assert second["qasm"] == first["qasm"]
    # The optimized circuit is equivalent to the original.
    rc = cli.main(["equivalent", path, optimized_path])
    result = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert result["equivalent"] is True


# ------------------------------------------------------------------ estimate


def test_estimate_schema_version_3_gate_counts():
    program = _parse(
        "y q[0];\nz q[0];\ns q[1];\nsdg q[1];\nt q[2];\ntdg q[2];\nswap q[0],q[1];\n"
    )
    result = estimate(program, "state-vector")
    assert result["schema_version"] == 3
    assert list(result["gate_counts"]) == [
        "x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz",
        "y", "z", "s", "sdg", "t", "tdg", "swap",
    ]
    assert result["gate_counts"]["y"] == 1
    assert result["gate_counts"]["swap"] == 1
    assert result["gate_counts"]["x"] == 0
    assert result["gate_count"] == 7


def test_estimate_earlier_schemas_byte_preserved():
    v1 = estimate(_parse("x q[0];\n"), "state-vector")
    assert v1["schema_version"] == 1
    assert list(v1["gate_counts"]) == ["x", "h", "cx", "rx", "ry", "rz"]
    v2 = estimate(_parse("cz q[0],q[1];\n"), "state-vector")
    assert v2["schema_version"] == 2
    assert len(v2["gate_counts"]) == 10


def test_estimate_depth_and_count_with_swap():
    program = _parse("swap q[0],q[2];\ny q[0];\n")
    result = estimate(program, "state-vector")
    assert result["gate_count"] == 2
    assert result["circuit_depth"] == 2


# ----------------------------------------------------------------------- DSL


def test_dsl_chaining_and_round_trip():
    circuit = (
        Circuit(2, 2)
        .y(0)
        .z(1)
        .s(0)
        .sdg(1)
        .t(0)
        .tdg(1)
        .swap(0, 1)
        .measure(0, 0)
        .measure(1, 1)
    )
    text = circuit.to_qasm()
    assert "y q[0];" in text
    assert "swap q[0],q[1];" in text
    rebuilt = Circuit.from_qasm(text)
    assert rebuilt.to_qasm() == text
    assert rebuilt.probabilities() == circuit.probabilities()


def test_dsl_swap_same_operand_raises_and_keeps_circuit():
    circuit = Circuit(2, 2).h(0)
    with pytest.raises(ValueError):
        circuit.swap(1, 1)
    assert [op.kind for op in circuit.operations] == ["h"]


def test_dsl_new_gate_validation():
    circuit = Circuit(1, 1)
    with pytest.raises(TypeError):
        circuit.y(0.5)
    with pytest.raises(ValueError):
        circuit.z(1)
    circuit.measure(0, 0)
    for method in ("t", "tdg", "s", "sdg"):
        with pytest.raises(ValueError):
            getattr(circuit, method)(0)
    with pytest.raises(ValueError):
        circuit.swap(0, 0)


def test_dsl_swap_after_measurement_rejected():
    circuit = Circuit(2, 2).measure(0, 0)
    with pytest.raises(ValueError):
        circuit.swap(0, 1)
    assert len(circuit.operations) == 1


def test_dsl_sample_matches_cli(write_qasm, capsys):
    circuit = Circuit(2, 2).h(0).t(0).swap(0, 1).measure(0, 0).measure(1, 1)
    path = write_qasm("", "dsl.qasm")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(circuit.to_qasm())
    rc = cli.main(["simulate", path, "--shots", "64", "--seed", "7"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert circuit.sample(shots=64, seed=7) == out
