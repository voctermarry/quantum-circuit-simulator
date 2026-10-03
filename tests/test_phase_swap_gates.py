"""Tests for the phase gates y, z, s, sdg, t, tdg and the swap gate."""

from __future__ import annotations

import cmath
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import Circuit, cli
from quantum_circuit.equivalence import unitary_distance
from quantum_circuit.estimation import estimate
from quantum_circuit.noise import CHANNEL_ORDER, _CHANNELS, evolve_density_matrix, simulate_density_matrix
from quantum_circuit.openqasm import ParseError, ValidationError, parse
from quantum_circuit.optimizer import optimize
from quantum_circuit.simulator import simulate_state_vector, unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent

PHASE_GATES = ("y", "z", "s", "sdg", "t", "tdg")
ALL_NEW_GATES = ("y", "z", "s", "sdg", "t", "tdg", "swap")
SCHEMA3_ORDER = (
    "x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz",
    "y", "z", "s", "sdg", "t", "tdg", "swap",
)


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


def _parse(body: str, qreg: str = "q[3]", creg: str = "c[3]"):
    return parse(HEADER + f"qreg {qreg};\ncreg {creg};\n{body}")


def _unitary(body: str, n: int = 2):
    return unitary_matrix(_parse(body, f"q[{n}]", f"c[{n}]"))


# ------------------------------------------------------------------- parsing


def test_new_gates_parse_with_positions():
    program = _parse(
        "y q[0];\nz q[1];\ns q[2];\nsdg q[0];\nt q[1];\ntdg q[2];\nswap q[0],q[1];\n"
    )
    assert [op.kind for op in program.operations] == [
        "y", "z", "s", "sdg", "t", "tdg", "swap",
    ]
    assert program.operations[-1].targets == (0, 1)
    assert all(op.params == () for op in program.operations)


def test_swap_operand_order_is_preserved():
    program = _parse("swap q[2],q[0];\n")
    assert program.operations[0].targets == (2, 0)


def test_swap_same_operand_is_validation_error():
    with pytest.raises(ValidationError) as info:
        _parse("swap q[1],q[1];\n")
    assert info.value.line >= 1 and info.value.column >= 1
    assert "swap" in info.value.message and "different qubits" in info.value.message


@pytest.mark.parametrize(
    "text",
    [
        "swap q[0];\n",  # missing second operand
        "swap q[0],q[1],q[2];\n",  # extra operand
        "swap(0.1) q[0],q[1];\n",  # swap takes no parameter
        "y q[0],q[1];\n",  # single-qubit gate with an extra operand
        "s(0.1) q[0];\n",  # phase gate takes no parameter
        "y;\n",  # no operand at all
    ],
)
def test_operand_structure_errors_are_parse_errors(text):
    with pytest.raises(ParseError) as info:
        _parse(text)
    assert info.value.line >= 1 and info.value.column >= 1


def test_unknown_gate_name_is_still_validation_error():
    with pytest.raises(ValidationError):
        _parse("w q[0];\n")


def test_new_gate_after_measurement_is_validation_error():
    with pytest.raises(ValidationError):
        _parse(
            "h q[0];\nmeasure q[0] -> c[0];\nz q[1];\n",
            qreg="q[2]",
            creg="c[2]",
        )


# ------------------------------------------------------------- state-vector


def test_y_is_pauli_y():
    state = simulate_state_vector(_parse("y q[0];", "q[1]", "c[1]"))
    assert state[0] == 0 and abs(state[1] - 1j) < 1e-12
    state = simulate_state_vector(_parse("x q[0];\ny q[0];", "q[1]", "c[1]"))
    assert abs(state[0] + 1j) < 1e-12 and state[1] == 0


def test_z_is_diag_one_minus_one():
    state = simulate_state_vector(_parse("h q[0];\nz q[0];", "q[1]", "c[1]"))
    half = 2.0**-0.5
    assert abs(state[0] - half) < 1e-12 and abs(state[1] + half) < 1e-12


@pytest.mark.parametrize(
    "gate,factor",
    [
        ("s", 1j),
        ("sdg", -1j),
        ("t", cmath.exp(1j * math.pi / 4)),
        ("tdg", cmath.exp(-1j * math.pi / 4)),
    ],
)
def test_phase_gates_on_excited_state(gate, factor):
    state = simulate_state_vector(_parse(f"x q[0];\n{gate} q[0];", "q[1]", "c[1]"))
    assert state[0] == 0 and abs(state[1] - factor) < 1e-12


def test_phase_gates_leave_zero_state_alone():
    for gate in ("z", "s", "sdg", "t", "tdg"):
        state = simulate_state_vector(_parse(f"{gate} q[0];", "q[1]", "c[1]"))
        assert state[0] == 1 and state[1] == 0


def test_swap_exchanges_basis_states():
    # |01> (q[0]=1, q[1]=0, index 1) -> |10> (index 2).
    state = simulate_state_vector(_parse("x q[0];\nswap q[0],q[1];", "q[2]", "c[2]"))
    assert abs(state[2] - 1) < 1e-12 and state[1] == 0
    state = simulate_state_vector(_parse("x q[1];\nswap q[0],q[1];", "q[2]", "c[2]"))
    assert abs(state[1] - 1) < 1e-12 and state[2] == 0


def test_gate_identities_via_unitary():
    assert unitary_distance(_unitary("s q[0];\ns q[0];", 1), _unitary("z q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("sdg q[0];\nsdg q[0];", 1), _unitary("z q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("t q[0];\nt q[0];", 1), _unitary("s q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("tdg q[0];\ntdg q[0];", 1), _unitary("sdg q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("s q[0];\nsdg q[0];", 1), _unitary("h q[0];\nh q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("h q[0];\nz q[0];\nh q[0];", 1), _unitary("x q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("h q[0];\ny q[0];\nh q[0];", 1), _unitary("y q[0];", 1)) <= 1e-10
    assert unitary_distance(_unitary("y q[0];\ny q[0];", 1), _unitary("z q[0];\nz q[0];", 1)) <= 1e-10


def test_swap_unitary_is_symmetric_in_operand_order():
    forward = _unitary("swap q[0],q[1];")
    backward = _unitary("swap q[1],q[0];")
    assert unitary_distance(forward, backward) <= 1e-10


def test_unitary_and_state_vector_share_semantics():
    body = "h q[0];\ny q[1];\ns q[0];\nt q[1];\nswap q[0],q[1];\n"
    program = _parse(body, "q[2]", "c[2]")
    state = simulate_state_vector(program)
    matrix = unitary_matrix(program)
    # The evolved |0...0> is column 0 of the full unitary.
    for row, amplitude in enumerate(state):
        assert abs(amplitude - matrix[row][0]) < 1e-12


# ------------------------------------------------------------ density matrix


def test_density_matches_state_vector_without_effective_noise():
    body = (
        "h q[0];\ny q[1];\ncz q[0],q[1];\ns q[0];\nt q[1];\n"
        "sdg q[0];\ntdg q[1];\nswap q[0],q[1];\n"
    )
    program = _parse(body, "q[2]", "c[2]")
    expected = [abs(a) ** 2 for a in simulate_state_vector(program)]
    diag = simulate_density_matrix(program, {"depolarizing": 0.0})
    assert all(abs(a - b) < 1e-12 for a, b in zip(expected, diag))


def test_single_qubit_gate_channel_is_applied_after_gate():
    # y|0> = i|1>; bit flip with probability p mixes the population back.
    program = _parse("y q[0];", "q[1]", "c[1]")
    diag = simulate_density_matrix(program, {"bit_flip": 0.25})
    assert abs(diag[0] - 0.25) < 1e-12 and abs(diag[1] - 0.75) < 1e-12


def test_swap_channels_applied_once_per_qubit_in_ascending_order():
    program = _parse("h q[0];\nswap q[0],q[1];", "q[2]", "c[2]")
    noise = {"amplitude_damping": 0.1, "phase_damping": 0.2, "bit_flip": 0.05, "depolarizing": 0.03}

    # Independent gate-by-gate reference: h on qubit 0 (channels on 0), then
    # the swap (channels in qubit order 0 then 1, CHANNEL_ORDER within each).
    from quantum_circuit.noise import _apply_single_qubit, _apply_swap, _gate_matrix

    size = 4
    reference = [[0j] * size for _ in range(size)]
    reference[0][0] = 1 + 0j
    _apply_single_qubit(reference, 0, size, _gate_matrix("h", ()))
    for channel in CHANNEL_ORDER:
        _CHANNELS[channel](reference, 0, size, noise[channel])
    _apply_swap(reference, 0, 1, size)
    for qubit in (0, 1):
        for channel in CHANNEL_ORDER:
            _CHANNELS[channel](reference, qubit, size, noise[channel])

    actual = evolve_density_matrix(program, noise)
    assert all(
        abs(actual[i][j] - reference[i][j]) < 1e-12
        for i in range(size)
        for j in range(size)
    )


def test_swap_operand_order_does_not_change_noise_order():
    body_forward = _parse("h q[0];\nh q[1];\nswap q[0],q[1];", "q[2]", "c[2]")
    body_backward = _parse("h q[0];\nh q[1];\nswap q[1],q[0];", "q[2]", "c[2]")
    noise = {"amplitude_damping": 0.15, "depolarizing": 0.1}
    forward = evolve_density_matrix(body_forward, noise)
    backward = evolve_density_matrix(body_backward, noise)
    assert all(
        abs(forward[i][j] - backward[i][j]) < 1e-12
        for i in range(4)
        for j in range(4)
    )


def test_density_noisy_evolution_stays_normalized():
    program = _parse(
        "y q[0];\ns q[1];\nt q[0];\nswap q[0],q[1];\n",
        "q[2]",
        "c[2]",
    )
    noise = {"amplitude_damping": 0.1, "phase_damping": 0.2, "bit_flip": 0.05, "depolarizing": 0.03}
    diag = simulate_density_matrix(program, noise)
    assert abs(sum(diag) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in diag)


# --------------------------------------------------------------------- DSL


def test_dsl_chaining_keeps_order_and_identity():
    circuit = Circuit(3, 3)
    result = (
        circuit.y(0).z(1).s(2).sdg(0).t(1).tdg(2)
        .swap(0, 2)
        .measure(0, 0).measure(1, 1).measure(2, 2)
    )
    assert result is circuit
    assert [op.kind for op in circuit.operations] == [
        "y", "z", "s", "sdg", "t", "tdg", "swap",
        "measure", "measure", "measure",
    ]
    assert circuit.operations[6].targets == (0, 2)


@pytest.mark.parametrize("value", [0.5, "0", None, True])
def test_dsl_new_gate_index_type_errors(value):
    circuit = Circuit(2, 2)
    with pytest.raises(TypeError):
        circuit.y(value)
    with pytest.raises(TypeError):
        circuit.swap(value, 1)
    with pytest.raises(TypeError):
        circuit.swap(0, value)


@pytest.mark.parametrize("value", [-1, 2])
def test_dsl_new_gate_index_range_errors(value):
    circuit = Circuit(2, 2)
    with pytest.raises(ValueError):
        circuit.t(value)
    with pytest.raises(ValueError):
        circuit.swap(0, value)


def test_dsl_swap_equal_operands_raises_and_keeps_circuit():
    circuit = Circuit(2, 2).h(0)
    with pytest.raises(ValueError):
        circuit.swap(1, 1)
    assert [op.kind for op in circuit.operations] == ["h"]


def test_dsl_new_gate_after_measurement_raises_and_keeps_circuit():
    circuit = Circuit(2, 2).measure(0, 0)
    with pytest.raises(ValueError):
        circuit.y(1)
    with pytest.raises(ValueError):
        circuit.swap(0, 1)
    assert [op.kind for op in circuit.operations] == ["measure"]


def test_dsl_to_qasm_exact_text_and_round_trip():
    circuit = (
        Circuit(3, 2)
        .y(0).z(1).s(2).sdg(0).t(1).tdg(2).swap(2, 0)
    )
    assert circuit.to_qasm() == (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[3];\n"
        "creg c[2];\n"
        "y q[0];\n"
        "z q[1];\n"
        "s q[2];\n"
        "sdg q[0];\n"
        "t q[1];\n"
        "tdg q[2];\n"
        "swap q[2],q[0];\n"
    )
    clone = Circuit.from_qasm(circuit.to_qasm())
    assert clone.operations == circuit.operations
    assert clone.to_qasm() == circuit.to_qasm()


def test_dsl_sample_and_probabilities_match_cli(write_qasm, capsys):
    circuit = (
        Circuit(2, 2)
        .h(0).y(1).s(0).t(1).swap(0, 1)
        .measure(0, 0).measure(1, 1)
    )
    body = circuit.to_qasm().split(HEADER, 1)[1]
    path = write_qasm(body)

    assert cli.main(["simulate", path, "--shots", "200", "--seed", "7"]) == 0
    expected = json.loads(capsys.readouterr().out)
    assert circuit.sample(shots=200, seed=7) == expected

    assert cli.main(["probabilities", path]) == 0
    expected_probs = json.loads(capsys.readouterr().out)
    assert circuit.probabilities() == expected_probs


def test_dsl_sample_with_noise_matches_cli(write_qasm, tmp_path, capsys):
    circuit = Circuit(2, 2).h(0).swap(0, 1).measure(0, 0).measure(1, 1)
    model = {"phase_damping": 0.2, "bit_flip": 0.1}
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    model_file = tmp_path / "noise.json"
    model_file.write_text(json.dumps(model), encoding="utf-8")
    assert cli.main(["simulate", path, "--noise-model", str(model_file), "--shots", "128", "--seed", "3"]) == 0
    expected = json.loads(capsys.readouterr().out)
    assert circuit.sample(shots=128, seed=3, noise_model=model) == expected


def test_dsl_repeated_calls_are_equal():
    circuit = Circuit(2, 2).h(0).y(1).swap(0, 1).measure(0, 0).measure(1, 1)
    first = circuit.sample(shots=64, seed=11)
    assert circuit.sample(shots=64, seed=11) == first
    probs = circuit.probabilities()
    assert circuit.probabilities() == probs


# ---------------------------------------------------------------- optimize


@pytest.mark.parametrize("gate", ["y", "z", "h", "swap"])
def test_self_pairs_cancel(gate):
    if gate == "swap":
        body = "swap q[0],q[1];\nx q[2];\nswap q[0],q[1];\n"
    else:
        body = f"{gate} q[0];\nx q[2];\n{gate} q[0];\n"
    _, changed, qasm = optimize(_parse(body))
    assert changed is True
    assert gate not in qasm
    assert "x q[2];" in qasm


@pytest.mark.parametrize("pair", [("s", "sdg"), ("sdg", "s"), ("t", "tdg"), ("tdg", "t")])
def test_inverse_phase_pairs_cancel_in_either_order(pair):
    first, second = pair
    body = f"{first} q[0];\nx q[2];\n{second} q[0];\n"
    _, changed, qasm = optimize(_parse(body))
    assert changed is True
    assert first + " q[0]" not in qasm and second + " q[0]" not in qasm
    assert "x q[2];" in qasm


def test_swap_pair_sees_through_disjoint_gates():
    body = "swap q[0],q[1];\nh q[2];\ns q[2];\nswap q[0],q[1];\n"
    _, changed, qasm = optimize(_parse(body))
    assert changed is True
    assert "swap" not in qasm


def test_swap_with_swapped_operands_does_not_cancel():
    _, changed, qasm = optimize(_parse("swap q[0],q[1];\nswap q[1],q[0];", "q[2]", "c[2]"))
    assert changed is False
    assert qasm.count("swap") == 2


def test_phase_gate_does_not_cancel_against_a_different_gate():
    # s and s merge-semantically as z, but the optimizer has no s+s rule:
    # they must simply remain.
    _, changed, qasm = optimize(_parse("s q[0];\ns q[0];", "q[1]", "c[1]"))
    assert changed is False
    assert qasm.count("s q[0];") == 2


def test_optimized_new_gates_reparse_and_stay_equivalent():
    body = (
        "h q[0];\ny q[0];\ny q[0];\nz q[1];\nh q[2];\nz q[1];\n"
        "s q[0];\nt q[1];\nswap q[0],q[1];\nsdg q[0];\ntdg q[1];\nswap q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    program = _parse(body)
    _, changed, qasm = optimize(program)
    assert changed is True
    optimized = parse(qasm)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(optimized)) <= 1e-10
    assert optimize(optimized)[2] == qasm  # fixed point


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_circuits_with_new_gates_stay_equivalent(seed):
    import random

    rng = random.Random(seed + 20)
    n = rng.randint(1, 5)
    bodies: list[str] = []
    for _ in range(rng.randint(0, 24)):
        choice = rng.randrange(9)
        qubit = rng.randrange(n)
        if choice < 6:
            bodies.append(f"{PHASE_GATES[choice]} q[{qubit}];")
        elif n >= 2:
            other = rng.randrange(n - 1)
            if other >= qubit:
                other += 1
            bodies.append(f"swap q[{qubit}],q[{other}];")
        else:
            bodies.append(f"y q[{qubit}];")
    measures = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n))
    program = parse(
        HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + "\n".join(bodies) + "\n" + measures
    )
    _, _, qasm = optimize(program)
    optimized = parse(qasm)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(optimized)) <= 1e-10
    assert optimize(optimized)[2] == qasm


# ---------------------------------------------------------------- estimate


def test_estimate_schema_3_appends_seven_counts():
    body = "y q[0];\nz q[1];\ns q[2];\nsdg q[0];\nt q[1];\ntdg q[2];\nswap q[0],q[1];\n"
    data = estimate(_parse(body), "state-vector")
    assert data["schema_version"] == 3
    assert list(data["gate_counts"]) == list(SCHEMA3_ORDER)
    counts = data["gate_counts"]
    assert counts["y"] == 1 and counts["z"] == 1 and counts["s"] == 1
    assert counts["sdg"] == 1 and counts["t"] == 1 and counts["tdg"] == 1
    assert counts["swap"] == 1
    assert data["gate_count"] == 7
    # Three disjoint layers: {y,z,s}, {sdg,t,tdg}, {swap}.
    assert data["circuit_depth"] == 3


def test_estimate_schema_3_with_only_one_new_gate():
    data = estimate(_parse("h q[0];\ncz q[0],q[1];\ntdg q[2];"), "state-vector")
    assert data["schema_version"] == 3
    assert len(data["gate_counts"]) == 17
    assert data["gate_counts"]["tdg"] == 1


def test_estimate_old_gate_sets_keep_schema_1_and_2_byte_shape():
    schema1 = estimate(_parse("h q[0];\ncx q[0],q[1];\nrx(0.5) q[0];"), "state-vector")
    assert schema1["schema_version"] == 1
    assert list(schema1["gate_counts"]) == ["x", "h", "cx", "rx", "ry", "rz"]

    schema2 = estimate(_parse("cz q[0],q[1];\ncrx(0.1) q[0],q[1];"), "state-vector")
    assert schema2["schema_version"] == 2
    assert list(schema2["gate_counts"]) == [
        "x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz",
    ]


def test_estimate_depth_counts_swap_as_one_layer():
    data = estimate(_parse("swap q[0],q[1];\ny q[2];\n", "q[3]", "c[1]"), "state-vector")
    assert data["circuit_depth"] == 1
    data = estimate(_parse("y q[0];\nswap q[0],q[1];", "q[2]", "c[1]"), "state-vector")
    assert data["circuit_depth"] == 2


# -------------------------------------------------------------- equivalence


def test_equivalent_sees_phase_decompositions_and_swap_symmetry(write_qasm, capsys):
    left = write_qasm(
        "qreg q[2];\ncreg c[2];\ns q[0];\ns q[0];\nswap q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        "left.qasm",
    )
    right = write_qasm(
        "qreg q[2];\ncreg c[2];\nz q[0];\nswap q[1],q[0];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        "right.qasm",
    )
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is True
    assert data["reason"] == "equivalent"


# ------------------------------------------------------- batch and reconcile


def test_batch_simulate_runs_new_gate_jobs(write_qasm, tmp_path, capsys):
    circuit = Circuit(2, 2).h(0).y(1).swap(0, 1).measure(0, 0).measure(1, 1)
    source = write_qasm(circuit.to_qasm().split(HEADER, 1)[1], "job.qasm")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "jobs": [
                    {"id": "a", "source": source, "shots": 50, "seed": 4},
                    {"id": "b", "source": source, "shots": 10, "seed": 9},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["batch-simulate", str(manifest)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["succeeded"] == 2 and data["failed"] == 0
    assert data["results"][0]["output"] == circuit.sample(shots=50, seed=4)
    assert data["results"][1]["output"] == circuit.sample(shots=10, seed=9)


def test_reconcile_matches_new_gate_baseline(write_qasm, tmp_path, capsys):
    circuit = Circuit(2, 2).h(0).s(1).swap(0, 1).measure(0, 0).measure(1, 1)
    source = write_qasm(circuit.to_qasm().split(HEADER, 1)[1], "job.qasm")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "jobs": [{"id": "a", "source": source}]}),
        encoding="utf-8",
    )
    assert cli.main(["batch-simulate", str(manifest)]) == 0
    baseline = capsys.readouterr().out
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(baseline, encoding="utf-8")

    assert cli.main(["reconcile", str(manifest), str(baseline_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["consistent"] is True
    assert report["matched"] == 1 and report["mismatched"] == 0
    assert report["results"][0]["reason"] == "identical"


# ------------------------------------------------------------------- CLI


def test_cli_swap_same_operand_is_positioned_validation_error(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\nswap q[1],q[1];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_cli_swap_structure_error_is_parse_error(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\nswap q[0];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    assert json.loads(captured.err)["error"] == "parse_error"


def test_cli_simulate_probabilities_and_estimate(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\n"
        "h q[0];\ny q[1];\nt q[0];\nswap q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["simulate", path, "--shots", "100", "--seed", "5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert sum(data["counts"].values()) == 100

    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-12

    assert cli.main(["estimate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 3
    assert list(data["gate_counts"]) == list(SCHEMA3_ORDER)


def test_cli_repeated_runs_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\ny q[0];\ns q[1];\nswap q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["simulate", path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def _run_process(*args: str, stdin: str | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "quantum_circuit.cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=env,
    )


def test_process_stdin_simulate_new_gates():
    source = (
        HEADER
        + "qreg q[2];\ncreg c[2];\nh q[0];\ny q[1];\nswap q[0],q[1];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    result_a = _run_process("simulate", "-", "--seed", "3", stdin=source)
    result_b = _run_process("simulate", "-", "--seed", "3", stdin=source)
    assert result_a.returncode == 0 and result_a.stderr == ""
    assert result_a.stdout == result_b.stdout
