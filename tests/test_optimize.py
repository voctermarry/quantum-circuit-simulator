"""Tests for the ``optimize`` subcommand and the canonical simplifier."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import cli
from quantum_circuit.equivalence import measurement_layout, unitary_distance
from quantum_circuit.openqasm import parse
from quantum_circuit.optimizer import normalize_angle, optimize
from quantum_circuit.simulator import unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm", header: str = HEADER):
        path = tmp_path / name
        path.write_text(header + body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_bytes(tmp_path):
    def _write(data: bytes, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return _write


def _optimize_body(body: str, qreg: str = "q[3]", creg: str = "c[3]"):
    program = parse(HEADER + f"qreg {qreg};\ncreg {creg};\n{body}")
    return optimize(program)


def _gate_lines(qasm: str) -> list[str]:
    return [
        line
        for line in qasm.splitlines()
        if not line.startswith(("OPENQASM", "include", "qreg", "creg", "measure"))
    ]


# ------------------------------------------------------------ angle helpers


def test_normalize_angle_interval_and_boundaries():
    assert normalize_angle(0.0) == 0.0
    assert normalize_angle(0.7) == 0.7
    assert normalize_angle(-0.7) == -0.7
    assert normalize_angle(math.pi) == math.pi
    # -pi is represented as +pi (they differ only by a global phase).
    assert normalize_angle(-math.pi) == math.pi
    assert normalize_angle(3 * math.pi) == math.pi
    assert normalize_angle(2 * math.pi) == 0.0
    assert normalize_angle(3 * math.pi / 2) == -math.pi / 2
    for value in (-10.0, -3.1, 2.3, 7.0, 20 * math.pi + 0.5):
        result = normalize_angle(value)
        assert -math.pi < result <= math.pi


# ------------------------------------------------------------- cancellation


def test_adjacent_x_h_cancel():
    for gate in ("x", "h"):
        _, changed, qasm = _optimize_body(f"{gate} q[0];\n{gate} q[0];\n")
        assert changed is True
        assert _gate_lines(qasm) == []


def test_cx_pair_cancels_when_control_and_target_match():
    _, changed, qasm = _optimize_body("cx q[0],q[1];\ncx q[0],q[1];\n")
    assert changed is True
    assert _gate_lines(qasm) == []


def test_cx_pair_with_swapped_bits_does_not_cancel():
    _, changed, qasm = _optimize_body("cx q[0],q[1];\ncx q[1],q[0];\n")
    assert changed is False  # distinct gates on shared qubits: neither cancels nor reorders
    assert _gate_lines(qasm) == ["cx q[0],q[1];", "cx q[1],q[0];"]


def test_cancellation_sees_through_disjoint_gates():
    body = (
        "x q[0];\n"
        "h q[1];\nrx(0.5) q[2];\ncx q[1],q[2];\n"
        "x q[0];\n"
    )
    _, _, qasm = _optimize_body(body)
    assert "x q[0]" not in qasm
    # The disjoint middle gates are preserved.
    assert "rx(0.5) q[2];" in qasm
    assert "cx q[1],q[2];" in qasm


def test_shared_qubit_gate_between_blocks_cancellation():
    # x; ry(0.5); x must remain because the rotation shares q[0] and does
    # not commute with x (only rx does).
    _, changed, qasm = _optimize_body("x q[0];\nry(0.5) q[0];\nx q[0];\n")
    assert changed is False
    assert _gate_lines(qasm) == ["x q[0];", "ry(0.5) q[0];", "x q[0];"]


def test_cascading_cancellation_after_disjoint_gate_removed():
    # Removing a disjoint blocker lets further gates meet and cancel.
    body = "h q[1];\nh q[1];\nx q[0];\nx q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == []


# ----------------------------------------------------------------- merging


def test_same_axis_rotations_merge():
    body = "rx(0.3) q[0];\nh q[1];\nrx(0.4) q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["rx(0.7) q[0];", "h q[1];"]


def test_all_three_axes_merge_independently():
    # A same-axis pair merges even with a gate on a disjoint qubit between,
    # while different-axis rotations sharing the qubit stay put.
    for axis in ("rx", "ry", "rz"):
        _, _, qasm = _optimize_body(f"{axis}(0.1) q[0];\nh q[1];\n{axis}(0.2) q[0];\n")
        assert _gate_lines(qasm) == [f"{axis}(0.30000000000000004) q[0];", "h q[1];"]

    # Interleaved different-axis rotations on the same qubit cannot merge.
    body = (
        "rz(0.1) q[0];\nry(0.1) q[0];\nrx(0.1) q[0];\n"
        "rz(0.2) q[0];\nry(0.2) q[0];\nrx(0.2) q[0];\n"
    )
    _, changed, qasm = _optimize_body(body)
    assert changed is False
    assert len(_gate_lines(qasm)) == 6


def test_rotations_on_different_qubits_do_not_merge():
    _, changed, qasm = _optimize_body("rx(0.3) q[0];\nrx(0.4) q[1];\n")
    assert changed is False
    assert "rx(0.3) q[0];" in qasm and "rx(0.4) q[1];" in qasm
    assert len(_gate_lines(qasm)) == 2


def test_merged_angle_normalized_to_interval():
    _, _, qasm = _optimize_body("rz(3*pi/2) q[0];\nrz(0) q[0];\n")
    assert "rz(-1.5707963267948966) q[0];" in qasm


def test_tiny_rotations_are_deleted():
    for angle in ("1e-13", "-1e-13", "0", "-0.0", "2*pi", "0.0000000000005"):
        _, _, qasm = _optimize_body(f"rx({angle}) q[0];\n")
        assert _gate_lines(qasm) == [], angle


def test_rotation_just_above_epsilon_is_kept():
    _, _, qasm = _optimize_body("rx(1e-10) q[0];\n")
    assert _gate_lines(qasm) == ["rx(1e-10) q[0];"]


def test_rx_near_pi_is_not_replaced_by_x():
    for angle in ("pi", "3.141592653589793", "pi-1e-13"):
        _, _, qasm = _optimize_body(f"rx({angle}) q[0];\n")
        lines = _gate_lines(qasm)
        assert all(line.startswith("rx") for line in lines)
        assert not any(line.startswith("x ") for line in lines)


def test_two_rotations_summing_to_zero_disappear():
    _, changed, qasm = _optimize_body("ry(0.6123) q[0];\nry(-0.6123) q[0];\n")
    assert changed is True
    assert _gate_lines(qasm) == []


# -------------------------------------------------------------- canonical order


def test_independent_gate_orderings_converge_byte_identically():
    measures = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(3))
    variants = [
        "x q[0];\nh q[1];\nx q[2];\n",
        "x q[2];\nx q[0];\nh q[1];\n",
        "h q[1];\nx q[2];\nx q[0];\n",
        "x q[0];\nx q[2];\nh q[1];\n",
    ]
    outputs = {optimize(parse(HEADER + "qreg q[3];\ncreg c[3];\n" + body + measures))[2] for body in variants}
    assert outputs == {
        HEADER
        + "qreg q[3];\ncreg c[3];\n"
        + "x q[0];\nh q[1];\nx q[2];\n"
        + measures
    }


def test_dependency_order_is_preserved():
    # h q[0] must stay before and after the cx on the shared qubit.
    body = "h q[0];\ncx q[0],q[1];\nh q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is False
    assert _gate_lines(qasm) == ["h q[0];", "cx q[0],q[1];", "h q[0];"]


def test_sort_key_min_then_max_qubit_then_name():
    # All three gates are pairwise disjoint; ordering is by minimum qubit,
    # so the two-qubit gate spanning q[3] sorts before the single-qubit gates.
    body = "h q[2];\nh q[1];\ncx q[0],q[3];\n"
    _, changed, qasm = _optimize_body(body, qreg="q[4]", creg="c[4]")
    assert changed is True
    assert _gate_lines(qasm) == ["cx q[0],q[3];", "h q[1];", "h q[2];"]


def test_sort_key_uses_name_for_same_qubit_set_but_dependency_blocks_moves():
    # x and h on the same qubit share it, so neither may cross the other even
    # though their sort keys are ordered (h < x).
    _, changed, qasm = _optimize_body("x q[0];\nh q[0];\n", qreg="q[1]", creg="c[1]")
    assert changed is False
    assert _gate_lines(qasm) == ["x q[0];", "h q[0];"]


# ------------------------------------------------------------- idempotence


def test_optimize_is_a_fixed_point():
    body = (
        "h q[2];\nx q[0];\nrx(0.3) q[0];\nrx(0.4) q[0];\n"
        "cx q[1],q[2];\nh q[1];\nh q[1];\ncx q[1],q[2];\n"
        "measure q[0] -> c[1];\nmeasure q[1] -> c[0];\nmeasure q[2] -> c[2];\n"
    )
    program = parse(HEADER + "qreg q[3];\ncreg c[3];\n" + body)
    gates, changed, qasm = optimize(program)
    assert changed is True
    again_gates, again_changed, again_qasm = optimize(parse(qasm))
    assert again_qasm == qasm
    assert again_gates == gates
    assert again_changed is False


# ------------------------------------------------- register names / measurements


def test_registers_renamed_q_and_c_with_sizes_preserved():
    text = (
        HEADER
        + "qreg qubits[2];\ncreg out[2];\nx qubits[0];\n"
        + "measure qubits[1] -> out[0];\nmeasure qubits[0] -> out[1];\n"
    )
    _, _, qasm = optimize(parse(text))
    assert "qreg q[2];" in qasm
    assert "creg c[2];" in qasm
    assert "qubits" not in qasm and "out" not in qasm
    # Sorted by clbit then qubit: c[0] measures q1, c[1] measures q0.
    measure_lines = [line for line in qasm.splitlines() if line.startswith("measure")]
    assert measure_lines == ["measure q[1] -> c[0];", "measure q[0] -> c[1];"]


def test_measurement_mapping_is_preserved():
    body = "x q[0];\nmeasure q[0] -> c[2];\nmeasure q[2] -> c[0];\nmeasure q[1] -> c[1];\n"
    program = parse(HEADER + "qreg q[3];\ncreg c[3];\n" + body)
    _, _, qasm = optimize(program)
    optimized = parse(qasm)
    assert measurement_layout(program) == measurement_layout(optimized)
    measure_lines = [line for line in qasm.splitlines() if line.startswith("measure")]
    assert measure_lines == [
        "measure q[2] -> c[0];",
        "measure q[1] -> c[1];",
        "measure q[0] -> c[2];",
    ]


def test_no_quantum_gates_still_succeeds():
    body = "measure q[0] -> c[0];\n"
    _, changed, qasm = _optimize_body(body, qreg="q[1]", creg="c[1]")
    assert changed is False
    assert _gate_lines(qasm) == []
    assert qasm.endswith("measure q[0] -> c[0];\n")


def test_qasm_is_reparseable_and_sectioned_one_item_per_line():
    body = "h q[0];\ncx q[0],q[1];\nrx(0.5) q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    program = parse(HEADER + "qreg q[2];\ncreg c[2];\n" + body)
    _, _, qasm = optimize(program)
    reparsed = parse(qasm)  # raises on any formatting problem
    assert reparsed.num_qubits == 2 and reparsed.num_clbits == 2
    assert qasm.endswith("\n")
    lines = qasm.splitlines()
    assert lines[0] == "OPENQASM 2.0;"
    assert lines[1] == 'include "qelib1.inc";'
    assert lines[2] == "qreg q[2];"
    assert lines[3] == "creg c[2];"
    assert qasm.count("\n") == len(lines)


def test_angle_formatting_negative_zero_and_scientific_notation():
    def lines_for(source):
        _, _, qasm = _optimize_body(f"rz({source}) q[0];\n", qreg="q[1]", creg="c[1]")
        parse(qasm)  # rendered text must be re-readable
        return _gate_lines(qasm)

    assert lines_for("-0.0") == []  # identity rotation is deleted
    assert lines_for("0.0000001") == ["rz(1e-07) q[0];"]
    assert lines_for("-0.0000001") == ["rz(-1e-07) q[0];"]
    assert lines_for("pi/2") == ["rz(1.5707963267948966) q[0];"]

    # No negative-zero angle text may ever appear.
    _, _, qasm = _optimize_body("rz(-2*pi) q[0];\nrz(0) q[0];\n", qreg="q[1]", creg="c[1]")
    assert "-0" not in "".join(_gate_lines(qasm))


def test_rendered_angles_round_trip():
    _, _, qasm = _optimize_body(
        "rx(0.123456789012345678) q[0];\nry(1e-9) q[0];\nrz(-1234567.89) q[0];\n",
        qreg="q[1]",
        creg="c[1]",
    )
    reparsed = parse(qasm)  # every rendered literal parses and stays finite
    assert reparsed.num_qubits == 1


# -------------------------------------------------------------- equivalence


def test_optimized_circuit_is_equivalent_to_original():
    body = (
        "h q[0];\ncx q[0],q[1];\nh q[0];\nh q[0];\n"
        "rx(0.3) q[2];\nrx(-0.3) q[2];\n"
        "rz(3*pi/2) q[1];\nrz(0) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[2];\nmeasure q[2] -> c[1];\n"
    )
    program = parse(HEADER + "qreg q[3];\ncreg c[3];\n" + body)
    _, _, qasm = optimize(program)
    optimized = parse(qasm)
    distance = unitary_distance(unitary_matrix(program), unitary_matrix(optimized))
    assert distance <= 1e-10
    assert measurement_layout(program) == measurement_layout(optimized)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_circuits_stay_equivalent(seed):
    import random

    rng = random.Random(seed)
    n = rng.randint(1, 6)
    bodies: list[str] = []
    for _ in range(rng.randint(0, 20)):
        choice = rng.randrange(6)
        qubit = rng.randrange(n)
        if choice == 0:
            bodies.append(f"x q[{qubit}];")
        elif choice == 1:
            bodies.append(f"h q[{qubit}];")
        elif choice == 2 and n >= 2:
            other = rng.randrange(n - 1)
            if other >= qubit:
                other += 1
            bodies.append(f"cx q[{qubit}],q[{other}];")
        else:
            axis = ("rx", "ry", "rz")[choice - 3]
            angle = rng.choice([0.3, -0.3, math.pi, math.pi / 2, 2 * math.pi, 1e-13, rng.uniform(-6, 6)])
            bodies.append(f"{axis}({angle!r}) q[{qubit}];")
    measures = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n))
    text = HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + "\n".join(bodies) + "\n" + measures
    program = parse(text)
    _, _, qasm = optimize(program)
    optimized = parse(qasm)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(optimized)) <= 1e-10
    assert measurement_layout(program) == measurement_layout(optimized)
    # Fixed point.
    assert optimize(optimized)[2] == qasm


# ------------------------------------------------------------------- CLI JSON


def test_cli_success_payload(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\n"
        "x q[0];\nx q[0];\nh q[1];\n"
        "measure q[0] -> c[1];\nmeasure q[1] -> c[0];\n",
    )
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1 and captured.out.endswith("\n")
    data = json.loads(captured.out)
    assert list(data) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "original_gate_count",
        "optimized_gate_count",
        "changed",
        "qasm",
    ]
    assert data["schema_version"] == 1
    assert (data["num_qubits"], data["num_clbits"]) == (3, 3)
    assert (data["original_gate_count"], data["optimized_gate_count"]) == (3, 1)
    assert data["changed"] is True
    assert data["qasm"].endswith("\n")
    assert parse(data["qasm"]).num_qubits == 3


def test_gate_counts_ignore_measurements(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\n"
        "h q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    assert cli.main(["optimize", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["original_gate_count"] == 1 and data["optimized_gate_count"] == 1
    assert data["changed"] is False


def test_changed_true_for_reordering_with_same_count(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\n"
        "x q[2];\nx q[0];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    assert cli.main(["optimize", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["changed"] is True
    assert data["original_gate_count"] == data["optimized_gate_count"] == 2


def test_repeated_calls_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\n"
        "h q[2];\nx q[0];\nrx(0.3) q[1];\nrx(0.4) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["optimize", path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_optimize_stdin(monkeypatch, capsys):
    import io

    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    assert cli.main(["optimize", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["optimized_gate_count"] == 0 and data["changed"] is True


# ------------------------------------------------------------------ errors


def test_missing_file_is_io_error(capsys):
    rc = cli.main(["optimize", "/nonexistent/path/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert captured.err.count("\n") == 1


def test_invalid_utf8_is_io_error(write_bytes, capsys):
    path = write_bytes(HEADER.encode() + b"\xff\xfe bad")
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 1 and captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_exit_two(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_exit_two(write_qasm, capsys):
    path = write_qasm("qreg q[1];\nqreg q2[1];\n")
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    assert json.loads(captured.err)["error"] == "validation_error"


def test_missing_argument_is_exit_two_without_reading(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["optimize"])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""


def test_extra_argument_is_exit_two(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["optimize", "a.qasm", "b.qasm"])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------- real-process test


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


def test_process_optimize_stdin_single_line():
    source = HEADER + "qreg q[2];\ncreg c[2];\nh q[1];\nh q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    result = _run_process("optimize", "-", stdin=source)
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["optimized_gate_count"] == 0
    assert data["qasm"].endswith("\n")


def test_process_missing_file_exit_one():
    result = _run_process("optimize", "/nonexistent/circuit.qasm")
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "io_error"
