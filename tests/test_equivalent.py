"""Tests for the ``equivalent`` subcommand and its supporting math."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import cli
from quantum_circuit.equivalence import (
    EQUIVALENCE_TOLERANCE,
    MAX_EQUIVALENCE_QUBITS,
    measurement_layout,
    unitary_distance,
)
from quantum_circuit.openqasm import parse
from quantum_circuit.simulator import simulate_state_vector, unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent

BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_bytes(tmp_path):
    def _write(data: bytes, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return _write


def _bell_paths(tmp_path, left_body=BELL, right_body=BELL):
    left = tmp_path / "left.qasm"
    right = tmp_path / "right.qasm"
    left.write_text(HEADER + left_body, encoding="utf-8")
    right.write_text(HEADER + right_body, encoding="utf-8")
    return str(left), str(right)


# ------------------------------------------------------------- unitary builder


def test_unitary_identity_circuit_is_identity_matrix():
    program = parse(HEADER + "qreg q[2];\ncreg c[2];\n")
    u = unitary_matrix(program)
    for i in range(4):
        for j in range(4):
            assert u[i][j] == (1 + 0j if i == j else 0j)


def test_unitary_zero_column_matches_state_vector():
    body = "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[2];\nrx(0.7) q[1];\n"
    program = parse(HEADER + body)
    u = unitary_matrix(program)
    state = simulate_state_vector(program)
    # Column 0 of U is U|000>.
    for row in range(8):
        assert abs(u[row][0] - state[row]) < 1e-15


def test_unitary_is_unitary():
    body = "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[2];\nrx(0.7) q[1];\nrz(1.1) q[0];\n"
    u = unitary_matrix(parse(HEADER + body))
    dim = 8
    for i in range(dim):
        for j in range(dim):
            got = sum(u[k][i].conjugate() * u[k][j] for k in range(dim))
            expected = 1.0 if i == j else 0.0
            assert abs(got - expected) < 1e-12


# ------------------------------------------------------------------ distance


def test_distance_zero_for_equal_and_global_phase():
    body = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    u = unitary_matrix(parse(HEADER + body))
    phase = 0.6 - 0.8j  # unit modulus
    assert abs(abs(phase) - 1.0) < 1e-15
    phased = [[phase * entry for entry in row] for row in u]
    assert unitary_distance(u, u) == 0.0
    assert unitary_distance(u, phased) < 1e-15


def test_distance_one_for_orthogonal_unitaries():
    i_mat = unitary_matrix(parse(HEADER + "qreg q[1];\ncreg c[1];\n"))
    x_mat = unitary_matrix(parse(HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\n"))
    assert abs(unitary_distance(i_mat, x_mat) - 1.0) < 1e-15


def test_distance_is_symmetric():
    a = unitary_matrix(parse(HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\n"))
    b = unitary_matrix(parse(HEADER + "qreg q[1];\ncreg c[1];\nrx(0.9) q[0];\n"))
    assert abs(unitary_distance(a, b) - unitary_distance(b, a)) < 1e-15


# ------------------------------------------------------------- measurement map


def test_measurement_layout_ignores_names_and_order():
    a = parse(HEADER + "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    b = parse(
        HEADER
        + "qreg qubits[2];\ncreg out[2];\nmeasure qubits[1] -> out[1];\nmeasure qubits[0] -> out[0];\n"
    )
    assert measurement_layout(a) == measurement_layout(b)
    assert measurement_layout(a) == (2, frozenset({(0, 0), (1, 1)}))


def test_measurement_layout_detects_width_and_mapping_differences():
    swapped = parse(HEADER + "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[1];\nmeasure q[1] -> c[0];\n")
    wide = parse(HEADER + "qreg q[2];\ncreg c[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    base = parse(HEADER + BELL)
    assert measurement_layout(swapped) != measurement_layout(base)
    assert measurement_layout(wide) != measurement_layout(base)


# -------------------------------------------------------------------- success


def test_equivalent_success_payload_and_field_order(tmp_path, capsys):
    left, right = _bell_paths(tmp_path)
    rc = cli.main(["equivalent", left, right])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    data = json.loads(captured.out)
    assert list(data) == [
        "schema_version",
        "equivalent",
        "reason",
        "left_num_qubits",
        "right_num_qubits",
        "distance",
        "tolerance",
    ]
    assert data == {
        "schema_version": 1,
        "equivalent": True,
        "reason": "equivalent",
        "left_num_qubits": 2,
        "right_num_qubits": 2,
        "distance": 0.0,
        "tolerance": 1e-10,
    }


def test_equivalent_ignores_whitespace_comments_and_register_names(tmp_path, capsys):
    renamed = (
        "// leading comment\n\nqreg  qubits[2];\ncreg out[2];\n"
        "h qubits[0]; cx qubits[0],qubits[1];\n"
        "measure qubits[0]->out[0]; measure qubits[1] -> out[1]; // tail\n"
    )
    left, right = _bell_paths(tmp_path, BELL, renamed)
    assert cli.main(["equivalent", left, right]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "equivalent"


def test_equivalent_allows_global_phase(tmp_path, capsys):
    # rz(2*pi) = -I on each qubit: only a global phase differs.
    phased = (
        "qreg q[2];\ncreg c[2];\nrz(2*pi) q[0];\nrz(2*pi) q[1];\n"
        "h q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    left, right = _bell_paths(tmp_path, BELL, phased)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is True and data["reason"] == "equivalent"
    assert data["distance"] <= data["tolerance"]


def test_equivalent_deterministic_across_runs(tmp_path, capsys):
    left, right = _bell_paths(tmp_path)
    outputs = set()
    for _ in range(3):
        assert cli.main(["equivalent", left, right]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_equivalent_without_measurements(tmp_path, capsys):
    body = "qreg q[1];\ncreg c[1];\nh q[0];\n"
    left, right = _bell_paths(tmp_path, body, body)
    rc = cli.main(["equivalent", left, right])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["reason"] == "equivalent"


def test_small_phase_below_tolerance_counts_as_equal(tmp_path, capsys):
    # rz(1e-12) is numerically indistinguishable from identity at tolerance.
    tiny = "qreg q[1];\ncreg c[1];\nrz(1e-12) q[0];\nmeasure q[0] -> c[0];\n"
    identity = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"
    left, right = _bell_paths(tmp_path, identity, tiny)
    assert cli.main(["equivalent", left, right]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "equivalent"


def test_phase_above_tolerance_is_unitary_distance(tmp_path, capsys):
    rotated = "qreg q[1];\ncreg c[1];\nrz(1e-6) q[0];\nmeasure q[0] -> c[0];\n"
    identity = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"
    left, right = _bell_paths(tmp_path, identity, rotated)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is False
    assert data["reason"] == "unitary_distance"
    assert data["distance"] > EQUIVALENCE_TOLERANCE


def test_different_transforms_report_distance(tmp_path, capsys):
    other = "qreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    left, right = _bell_paths(tmp_path, BELL, other)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is False
    assert data["reason"] == "unitary_distance"
    assert (
        data["schema_version"],
        data["left_num_qubits"],
        data["right_num_qubits"],
        data["tolerance"],
    ) == (1, 2, 2, 1e-10)
    assert data["distance"] == pytest.approx(0.8040190354753588, abs=1e-12)


# ------------------------------------------------------------- reason branches


def test_qubit_count_mismatch_null_distance(tmp_path, capsys):
    other = (
        "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    left, right = _bell_paths(tmp_path, BELL, other)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is False
    assert data["reason"] == "qubit_count_mismatch"
    assert data["distance"] is None
    assert (data["left_num_qubits"], data["right_num_qubits"]) == (2, 3)


def test_measurement_layout_mismatch_keeps_distance(tmp_path, capsys):
    swapped = (
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[1];\nmeasure q[1] -> c[0];\n"
    )
    left, right = _bell_paths(tmp_path, BELL, swapped)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["equivalent"] is False
    assert data["reason"] == "measurement_layout_mismatch"
    assert data["distance"] == 0.0


def test_clbit_width_mismatch_is_layout_mismatch(tmp_path, capsys):
    wide = (
        "qreg q[2];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    left, right = _bell_paths(tmp_path, BELL, wide)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reason"] == "measurement_layout_mismatch"


def test_unitary_distance_takes_precedence_over_layout(tmp_path, capsys):
    other = (
        "qreg q[2];\ncreg c[2];\nx q[0];\n"
        "measure q[0] -> c[1];\nmeasure q[1] -> c[0];\n"
    )
    left, right = _bell_paths(tmp_path, BELL, other)
    assert cli.main(["equivalent", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reason"] == "unitary_distance"
    assert data["distance"] > EQUIVALENCE_TOLERANCE


# ------------------------------------------------------------- 8-qubit limit


def test_eight_qubits_at_boundary_succeed(tmp_path, capsys):
    measures = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(8))
    body = f"qreg q[8];\ncreg c[8];\nh q[0];\n{measures}"
    left, right = _bell_paths(tmp_path, body, body)
    rc = cli.main(["equivalent", left, right])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reason"] == "equivalent" and data["left_num_qubits"] == 8


def test_nine_qubits_is_simulation_error(tmp_path, capsys):
    nine = "qreg q[9];\ncreg c[9];\n" + "".join(
        f"measure q[{i}] -> c[{i}];\n" for i in range(9)
    )
    two = "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    left, right = _bell_paths(tmp_path, nine, two)
    rc = cli.main(["equivalent", left, right])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "simulation_error"
    assert payload["input"] == "left"


def test_nine_qubits_on_either_side_is_simulation_error(tmp_path, capsys):
    nine = "qreg q[9];\ncreg c[9];\n"
    three = "qreg q[3];\ncreg c[3];\n"
    left, right = _bell_paths(tmp_path, three, nine)
    rc = cli.main(["equivalent", left, right])
    captured = capsys.readouterr()
    assert rc == 3
    payload = json.loads(captured.err)
    assert payload["error"] == "simulation_error"
    assert payload["input"] == "right"


def test_over_limit_takes_precedence_over_qubit_count_mismatch(tmp_path, capsys):
    nine = "qreg q[9];\ncreg c[9];\n"
    three = "qreg q[3];\ncreg c[3];\n"
    left, right = _bell_paths(tmp_path, nine, three)
    rc = cli.main(["equivalent", left, right])
    captured = capsys.readouterr()
    assert rc == 3
    payload = json.loads(captured.err)
    assert payload["error"] == "simulation_error"
    assert payload["input"] == "left"


def test_left_parse_error_reported_before_right_over_limit(write_qasm, capsys):
    bad_left = write_qasm("qreg q[1]\n", name="l.qasm")
    big_right = write_qasm("qreg q[9];\ncreg c[9];\n", name="r.qasm")
    rc = cli.main(["equivalent", bad_left, big_right])
    captured = capsys.readouterr()
    assert rc == 2
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error" and payload["input"] == "left"


# ---------------------------------------------------------------- stdin rules


def test_both_stdin_is_comparison_error_without_reading(monkeypatch, capsys):
    import io

    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"should not be read")))
    rc = cli.main(["equivalent", "-", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "comparison_error"
    assert captured.err.count("\n") == 1


def test_right_side_via_stdin(monkeypatch, tmp_path, capsys):
    import io

    left, _ = _bell_paths(tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + BELL).encode())))
    rc = cli.main(["equivalent", left, "-"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "equivalent"


def test_left_side_via_stdin(monkeypatch, tmp_path, capsys):
    import io

    _, right = _bell_paths(tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + BELL).encode())))
    rc = cli.main(["equivalent", "-", right])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "equivalent"


# ----------------------------------------------------------------- io errors


def test_io_error_on_left(tmp_path, capsys):
    _, right = _bell_paths(tmp_path)
    rc = cli.main(["equivalent", "/nonexistent/left.qasm", right])
    captured = capsys.readouterr()
    assert rc == 1 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error" and payload["input"] == "left"


def test_io_error_on_right(tmp_path, capsys):
    left, _ = _bell_paths(tmp_path)
    rc = cli.main(["equivalent", left, "/nonexistent/right.qasm"])
    captured = capsys.readouterr()
    assert rc == 1 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error" and payload["input"] == "right"


def test_invalid_utf8_is_io_error(write_bytes, write_qasm, capsys):
    bad = write_bytes(HEADER.encode() + b"\xff\xfe bad", name="bad.qasm")
    good = write_qasm(BELL)
    rc = cli.main(["equivalent", good, bad])
    captured = capsys.readouterr()
    assert rc == 1 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "right"


# ------------------------------------------------------- parse / validation


def test_parse_error_reports_side(write_qasm, capsys):
    bad = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n", name="bad.qasm")
    good = write_qasm(BELL)
    rc = cli.main(["equivalent", good, bad])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error" and payload["input"] == "right"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_reports_side(write_qasm, capsys):
    bad = write_qasm("qreg q[1];\nqreg q2[1];\ncreg c[1];\n", name="bad.qasm")
    good = write_qasm(BELL)
    rc = cli.main(["equivalent", bad, good])
    captured = capsys.readouterr()
    assert rc == 2 and captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error" and payload["input"] == "left"


def test_first_error_reported_left_before_right(write_qasm, capsys):
    bad_left = write_qasm("qreg q[1]\n", name="l.qasm")
    bad_right = write_qasm("qreg q[1];\nqreg q2[1];\n", name="r.qasm")
    rc = cli.main(["equivalent", bad_left, bad_right])
    captured = capsys.readouterr()
    assert rc == 2
    payload = json.loads(captured.err)
    assert payload["input"] == "left" and payload["error"] == "parse_error"
    assert captured.err.count("\n") == 1


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


def test_process_both_stdin_exit_code():
    result = _run_process("equivalent", "-", "-", stdin="")
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "comparison_error"


def test_process_equivalent_success_single_line(tmp_path):
    left = tmp_path / "a.qasm"
    right = tmp_path / "b.qasm"
    left.write_text(HEADER + BELL, encoding="utf-8")
    right.write_text(HEADER + BELL, encoding="utf-8")
    result = _run_process("equivalent", str(left), str(right))
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["equivalent"] is True and data["schema_version"] == 1
    assert MAX_EQUIVALENCE_QUBITS == 8
