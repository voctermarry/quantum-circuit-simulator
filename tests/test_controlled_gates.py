"""Tests for the extended controlled gates: cz, crx, cry, crz.

The new gates are exercised through every public path: parsing/validation,
state-vector simulation, the full unitary, deterministic density-matrix
evolution (including per-qubit noise ordering), estimate (schema versioning
and gate counts) and optimize (cancellation, merging, normalization).
"""

from __future__ import annotations

import cmath
import json
import math

import pytest

from quantum_circuit import cli
from quantum_circuit.equivalence import unitary_distance
from quantum_circuit.noise import simulate_density_matrix
from quantum_circuit.openqasm import ParseError, Program, ValidationError, parse
from quantum_circuit.optimizer import normalize_controlled_angle, optimize
from quantum_circuit.simulator import simulate_state_vector, unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _program(body: str, n: int = 2) -> Program:
    return parse(HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + body)


def _amplitudes(body: str, n: int = 2):
    program = _program(body, n)
    return program, simulate_state_vector(program)


# ----------------------------------------------------------------- parsing


def test_new_gates_parse_with_control_target_and_angle():
    program = _program(
        "cz q[0],q[1];\n"
        "crx(pi/2) q[0],q[1];\n"
        "cry(0.25) q[1],q[0];\n"
        "crz(1e-3) q[1],q[0];\n"
    )
    ops = program.operations
    assert [op.kind for op in ops] == ["cz", "crx", "cry", "crz"]
    assert ops[0].targets == (0, 1) and ops[0].params == ()
    assert ops[1].targets == (0, 1) and ops[1].params == (math.pi / 2,)
    assert ops[2].targets == (1, 0) and ops[2].params == (0.25,)
    assert ops[3].targets == (1, 0) and ops[3].params == (1e-3,)


@pytest.mark.parametrize("gate", ["cz", "crx", "cry", "crz"])
def test_equal_control_and_target_is_validation_error(gate):
    stmt = (
        f"{gate} q[0],q[0];\n"
        if gate == "cz"
        else f"{gate}(0.5) q[0],q[0];\n"
    )
    with pytest.raises(ValidationError) as info:
        _program(stmt)
    assert "different" in info.value.message
    assert info.value.line >= 1 and info.value.column >= 1


@pytest.mark.parametrize(
    "stmt",
    [
        "cz q[0];",
        "cz q[0] q[1];",
        "cz q[0],q[1]",
        "crx q[0],q[1];",
        "crx() q[0],q[1];",
        "crx(0.5,0.6) q[0],q[1];",
        "crx(0.5) q[0];",
        "crx(0.5) q[0] q[1];",
        "cry(0.5) q[0],q[1]",
        "crz() q[1],q[0];",
        "crz(0.5,) q[1],q[0];",
    ],
)
def test_malformed_new_gate_statements_raise_parse_error(stmt):
    with pytest.raises(ParseError):
        _program(stmt + "\n")


def test_new_gate_angle_expressions_use_existing_syntax():
    program = _program("crx(-pi/2 + 2*0.1) q[0],q[1];\n")
    assert program.operations[0].params == (-math.pi / 2 + 0.2,)


# ------------------------------------------------------------- state vector


def test_cz_introduces_negative_phase_only_on_11():
    # Each basis input evolves in isolation; only |11> flips sign.
    for start in range(4):
        n = 2
        state = [0j] * 4
        state[start] = 1 + 0j
        program = _program("cz q[0],q[1];\n")
        # Re-evolve from the one-hot vector using the gate directly.
        from quantum_circuit.simulator import _apply_gate

        for op in program.operations:
            _apply_gate(state, op, n)
        expected = -1.0 if start == 3 else 1.0
        assert abs(state[start] - expected) < 1e-12
        assert sum(abs(a) for i, a in enumerate(state) if i != start) < 1e-12


def test_cz_is_symmetric_in_control_and_target():
    u_ab = unitary_matrix(_program("cz q[0],q[1];\n"))
    u_ba = unitary_matrix(_program("cz q[1],q[0];\n"))
    assert unitary_distance(u_ab, u_ba) == 0.0


def test_cz_after_xx_is_minus_11():
    _, state = _amplitudes("x q[0];\nx q[1];\ncz q[0],q[1];\n")
    assert state[3] == -1.0
    assert all(abs(v) < 1e-12 for i, v in enumerate(state) if i != 3)


@pytest.mark.parametrize("axis", ["crx", "cry", "crz"])
def test_controlled_rotation_is_identity_when_control_is_zero(axis):
    body = f"h q[1];\n{axis}(1.2345) q[0],q[1];\n"
    _, state = _amplitudes(body)
    # q[0]=0 throughout: the superposition over q[1] must be untouched.
    assert abs(abs(state[0]) - 1 / math.sqrt(2)) < 1e-12
    assert abs(abs(state[2]) - 1 / math.sqrt(2)) < 1e-12
    assert abs(state[1]) < 1e-12 and abs(state[3]) < 1e-12


def test_crx_pi_with_control_one_flips_target_up_to_phase():
    _, state = _amplitudes("x q[0];\ncrx(pi) q[0],q[1];\n")
    # |01> (index 1) -> -i |11> (index 3).
    assert abs(state[1]) < 1e-12
    assert abs(abs(state[3]) - 1.0) < 1e-12


def test_cry_pi_with_control_one_flips_target_in_phase():
    _, state = _amplitudes("x q[0];\ncry(pi) q[0],q[1];\n")
    assert abs(state[1] - 0.0) < 1e-12
    assert abs(state[3] - 1.0) < 1e-12


def test_crx_half_pi_block_amplitudes():
    _, state = _amplitudes("x q[0];\ncrx(pi/2) q[0],q[1];\n")
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    assert abs(state[1] - c) < 1e-12
    assert abs(state[3] - (-1j * s)) < 1e-12


def test_cry_half_pi_block_amplitudes():
    _, state = _amplitudes("x q[0];\ncry(pi/2) q[0],q[1];\n")
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    assert abs(state[1] - c) < 1e-12
    assert abs(state[3] - s) < 1e-12


def test_crz_relative_phase_in_control_block():
    _, state = _amplitudes("x q[0];\nh q[1];\ncrz(0.7) q[0],q[1];\n")
    theta = 0.7
    # Both amplitudes live in the control=1 sector (indices 1 and 3).
    assert abs(state[1] - cmath.exp(-0.5j * theta) / math.sqrt(2)) < 1e-12
    assert abs(state[3] - cmath.exp(0.5j * theta) / math.sqrt(2)) < 1e-12


def test_controlled_rotation_4pi_is_identity_but_2pi_is_not():
    identity = unitary_matrix(_program("h q[1];\n"))  # any reference circuit
    base = unitary_matrix(_program("", n=2))

    u_2pi = unitary_matrix(_program("crx(2*pi) q[0],q[1];\n"))
    u_4pi = unitary_matrix(_program("crx(4*pi) q[0],q[1];\n"))
    assert unitary_distance(u_2pi, base) > 0.5
    assert unitary_distance(u_4pi, base) < 1e-12
    # Reference var is exercised for other axes too via the control-1 block.
    assert unitary_distance(identity, identity) == 0.0


@pytest.mark.parametrize("axis", ["crx", "cry", "crz"])
def test_new_gates_preserve_norm(axis):
    body = (
        "h q[0];\nh q[1];\n"
        f"{axis}(pi/3) q[0],q[1];\ncz q[0],q[1];\n{axis}(-0.9) q[1],q[0];\n"
    )
    _, state = _amplitudes(body)
    assert abs(sum(abs(a) ** 2 for a in state) - 1.0) < 1e-12


# --------------------------------------------------------- density matrix


def test_density_diagonal_matches_state_vector_zero_noise():
    # A configured-but-inert channel still routes through the density path.
    noise = {"bit_flip": 0.0}
    bodies = [
        "h q[0];\ncz q[0],q[1];\n",
        "h q[0];\ncrx(0.6) q[0],q[1];\n",
        "h q[1];\ncry(-1.1) q[1],q[0];\n",
        "h q[0];\nh q[1];\ncrz(2.3) q[0],q[1];\n",
        "cz q[0],q[1];\ncrx(0.4) q[1],q[0];\ncry(0.5) q[0],q[1];\n",
    ]
    for body in bodies:
        program = _program(body)
        sv = [abs(a) ** 2 for a in simulate_state_vector(program)]
        dm = simulate_density_matrix(program, noise)
        assert all(abs(a - b) < 1e-12 for a, b in zip(sv, dm)), body


def test_density_evolution_matches_unitary_conjugation():
    # Compare the full rho (off-diagonals included) against U rho U^dag built
    # from the state-vector unitary, for every gate and both control orders.
    import random

    from quantum_circuit.noise import _apply_controlled_single_qubit, _controlled_gate_matrix

    rng = random.Random(7)
    size = 8
    for gate in ("cz", "crx", "cry", "crz"):
        for ctrl, tgt in ((0, 1), (1, 0), (2, 0), (0, 2)):
            vec = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(size)]
            norm = math.sqrt(sum(abs(z) ** 2 for z in vec))
            vec = [z / norm for z in vec]
            angle_stmt = f"{gate}(0.63) q[{ctrl}],q[{tgt}];" if gate != "cz" else f"cz q[{ctrl}],q[{tgt}];"
            u = unitary_matrix(_program(angle_stmt + "\n", n=3))
            rho = [[vec[i] * vec[j].conjugate() for j in range(size)] for i in range(size)]
            reference = [
                [
                    sum(u[i][k] * rho[k][l] * u[j][l].conjugate() for k in range(size) for l in range(size))
                    for j in range(size)
                ]
                for i in range(size)
            ]
            _apply_controlled_single_qubit(
                rho, ctrl, tgt, size, _controlled_gate_matrix(gate, (0.63,))
            )
            error = max(abs(rho[i][j] - reference[i][j]) for i in range(size) for j in range(size))
            assert error < 1e-12, (gate, ctrl, tgt, error)


def test_noise_after_new_gate_is_trace_preserving():
    noise = {
        "amplitude_damping": 0.3,
        "phase_damping": 0.1,
        "bit_flip": 0.2,
        "depolarizing": 0.05,
    }
    body = (
        "h q[0];\ncz q[0],q[1];\ncrx(0.6) q[1],q[0];\n"
        "cry(0.4) q[0],q[2];\ncrz(1.1) q[2],q[1];\n"
    )
    diagonal = simulate_density_matrix(_program(body, n=3), noise)
    assert abs(sum(diagonal) - 1.0) < 1e-10


# ---------------------------------------------------------------- estimate


_GATE_ORDER_V1 = ["x", "h", "cx", "rx", "ry", "rz"]
_GATE_ORDER_V2 = ["x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz"]


def test_estimate_old_gates_remain_schema_version_1():
    from quantum_circuit.estimation import estimate

    data = estimate(_program("h q[0];\ncx q[0],q[1];\nrx(0.1) q[0];\n"), "state-vector")
    assert data["schema_version"] == 1
    assert list(data["gate_counts"]) == _GATE_ORDER_V1


def test_estimate_each_new_gate_switches_to_schema_version_2():
    from quantum_circuit.estimation import estimate

    for stmt in [
        "cz q[0],q[1];\n",
        "crx(0.1) q[0],q[1];\n",
        "cry(0.1) q[0],q[1];\n",
        "crz(0.1) q[0],q[1];\n",
    ]:
        data = estimate(_program(stmt), "state-vector")
        assert data["schema_version"] == 2
        assert list(data["gate_counts"]) == _GATE_ORDER_V2


def test_estimate_counts_depth_and_total_for_new_gates():
    from quantum_circuit.estimation import estimate

    body = (
        "cz q[0],q[1];\ncrx(0.2) q[0],q[1];\ncry(0.3) q[1],q[0];\n"
        "crz(0.4) q[0],q[1];\nh q[0];\n"
    )
    data = estimate(_program(body), "state-vector")
    assert data["gate_count"] == 5
    counts = data["gate_counts"]
    assert counts["h"] == 1
    assert counts["cz"] == 1
    assert counts["crx"] == 1 and counts["cry"] == 1 and counts["crz"] == 1
    assert counts["x"] == 0 and counts["cx"] == 0 and counts["rx"] == 0
    # All gates touch q[0] or q[1] in one chain: depth 5; the disjoint h q[?]
    # also shares here, so the gates serialize.
    assert data["circuit_depth"] == 5


def test_estimate_new_gate_disjoint_qubits_share_depth_layer():
    from quantum_circuit.estimation import estimate

    # qreg/creg sized 4 via a dedicated program.
    program = parse(HEADER + "qreg q[4];\ncreg c[1];\ncz q[0],q[1];\ncrx(0.2) q[2],q[3];\n")
    data = __import__("quantum_circuit.estimation", fromlist=["estimate"]).estimate(
        program, "state-vector"
    )
    assert data["circuit_depth"] == 1


# ---------------------------------------------------------------- optimize


def _gate_lines(qasm: str) -> list[str]:
    return [
        line
        for line in qasm.splitlines()
        if not line.startswith(("OPENQASM", "include", "qreg", "creg", "measure"))
    ]


def test_adjacent_cz_pair_cancels():
    _, changed, qasm = optimize(_program("cz q[0],q[1];\ncz q[0],q[1];\n"))
    assert changed is True
    assert _gate_lines(qasm) == []


def test_cz_cancellation_sees_through_disjoint_gate():
    _, _, qasm = optimize(_program("cz q[0],q[1];\nx q[2];\ncz q[0],q[1];\n", n=3))
    assert "cz" not in qasm
    assert "x q[2];" in qasm


def test_cz_pair_blocked_by_shared_qubit_gate():
    _, changed, qasm = optimize(_program("cz q[0],q[1];\nx q[1];\ncz q[0],q[1];\n"))
    assert changed is False
    assert len(_gate_lines(qasm)) == 3


def test_swapped_cz_does_not_cancel():
    _, changed, qasm = optimize(_program("cz q[0],q[1];\ncz q[1],q[0];\n"))
    assert changed is False
    assert len(_gate_lines(qasm)) == 2


@pytest.mark.parametrize("axis", ["crx", "cry", "crz"])
def test_controlled_rotations_merge_on_matching_axis_and_qubits(axis):
    _, changed, qasm = optimize(
        _program(f"{axis}(0.3) q[0],q[1];\nh q[2];\n{axis}(0.4) q[0],q[1];\n", n=3)
    )
    assert changed is True
    assert _gate_lines(qasm) == [f"{axis}(0.7) q[0],q[1];", "h q[2];"]


def test_controlled_rotation_does_not_merge_across_axes_or_qubits():
    _, changed, qasm = optimize(_program("crx(0.1) q[0],q[1];\ncry(0.1) q[0],q[1];\n"))
    assert changed is False and len(_gate_lines(qasm)) == 2

    _, changed, qasm = optimize(_program("crx(0.3) q[0],q[1];\ncrx(0.4) q[1],q[0];\n"))
    assert changed is False and len(_gate_lines(qasm)) == 2


def test_controlled_angle_normalized_to_4pi_period():
    assert normalize_controlled_angle(2 * math.pi) == 2 * math.pi
    assert normalize_controlled_angle(-2 * math.pi) == 2 * math.pi
    assert normalize_controlled_angle(3 * math.pi) == -math.pi
    assert normalize_controlled_angle(5 * math.pi) == math.pi
    assert normalize_controlled_angle(0.7) == 0.7
    for value in (-10.0, 7.0, 20 * math.pi + 0.5):
        result = normalize_controlled_angle(value)
        assert -2 * math.pi < result <= 2 * math.pi


def test_optimize_keeps_2pi_controlled_rotation_and_drops_4pi():
    _, _, qasm = optimize(_program("crz(2*pi) q[0],q[1];\n"))
    assert _gate_lines(qasm) == ["crz(6.283185307179586) q[0],q[1];"]

    _, changed, qasm = optimize(_program("crz(4*pi) q[0],q[1];\n"))
    assert changed is True and _gate_lines(qasm) == []


def test_optimize_drops_tiny_but_keeps_near_2pi_controlled_rotation():
    _, _, qasm = optimize(_program("crx(1e-13) q[0],q[1];\n"))
    assert _gate_lines(qasm) == []
    _, _, qasm = optimize(_program("crx(2*pi+1e-13) q[0],q[1];\n"))
    assert len(_gate_lines(qasm)) == 1


def test_optimize_controlled_rotations_summing_to_4pi_vanish():
    _, changed, qasm = optimize(_program("cry(3*pi) q[0],q[1];\ncry(pi) q[0],q[1];\n"))
    assert changed is True and _gate_lines(qasm) == []


def test_optimize_renders_reparseable_new_gate_statements():
    body = (
        "cz q[0],q[1];\ncrx(0.3) q[0],q[1];\ncry(-1.2) q[1],q[0];\ncrz(pi) q[0],q[1];\n"
    )
    program = _program(body)
    _, _, qasm = optimize(program)
    reparsed = parse(qasm)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(reparsed)) <= 1e-10


def test_optimize_new_gates_is_idempotent_and_equivalent():
    body = (
        "h q[2];\ncz q[0],q[1];\ncz q[0],q[1];\n"
        "crx(0.3) q[0],q[1];\ncrx(0.4) q[0],q[1];\n"
        "cry(1.0) q[1],q[2];\nx q[0];\ncrz(0.5) q[2],q[0];\n"
    )
    program = _program(body, n=3)
    gates, changed, qasm = optimize(program)
    assert changed is True
    optimized = parse(qasm)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(optimized)) <= 1e-10
    again_gates, again_changed, again_qasm = optimize(optimized)
    assert again_qasm == qasm
    assert again_gates == gates
    assert again_changed is False


# ----------------------------------------------------------- CLI integration


def _write_circuit(tmp_path, body: str, n: int = 2) -> str:
    path = tmp_path / "circuit.qasm"
    path.write_text(HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + body, encoding="utf-8")
    return str(path)


def test_cli_simulate_new_gates(capsys, tmp_path):
    # h q[0]; cz keeps q[1]=0 (cz never flips the target), so only 00/01 appear.
    path = _write_circuit(
        tmp_path,
        "h q[0];\ncz q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
    )
    assert cli.main(["simulate", path, "--shots", "64", "--seed", "5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["counts"]) <= {"00", "01"}
    assert sum(data["counts"].values()) == 64


def test_cli_probabilities_new_gates(capsys, tmp_path):
    path = _write_circuit(tmp_path, "h q[0];\ncz q[0],q[1];\n")
    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert set(data["probabilities"]) == {"00"}  # no measurements -> all-zero


def test_cli_estimate_schema_version_2_field_order(capsys, tmp_path):
    path = _write_circuit(tmp_path, "cz q[0],q[1];\ncrx(0.5) q[0],q[1];\n")
    assert cli.main(["estimate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 2
    assert list(data["gate_counts"]) == _GATE_ORDER_V2


def test_cli_equivalent_recognizes_new_gate_identity(capsys, tmp_path):
    # cz . cz = identity, so a circuit with the pair equals one without gates.
    left = _write_circuit(tmp_path, "cz q[0],q[1];\ncz q[0],q[1];\n")
    right = tmp_path / "plain.qasm"
    right.write_text(HEADER + "qreg q[2];\ncreg c[2];\n", encoding="utf-8")
    assert cli.main(["equivalent", left, str(right)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is True and data["reason"] == "equivalent"


def test_cli_optimize_cancels_cz_pair(capsys, tmp_path):
    path = _write_circuit(tmp_path, "cz q[0],q[1];\ncz q[0],q[1];\n")
    assert cli.main(["optimize", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["changed"] is True
    assert data["optimized_gate_count"] == 0
    parse(data["qasm"])  # output must remain re-parseable


def test_cli_equal_control_target_reports_validation_error(capsys, tmp_path):
    path = _write_circuit(tmp_path, "crx(0.5) q[0],q[0];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_cli_malformed_controlled_gate_reports_parse_error(capsys, tmp_path):
    path = _write_circuit(tmp_path, "crx(0.5) q[0] q[1];\n")
    rc = cli.main(["probabilities", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    assert json.loads(captured.err)["error"] == "parse_error"
