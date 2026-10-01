"""End-to-end tests for the `equivalent` subcommand."""

from __future__ import annotations

import io
import json

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


def run_equivalent(capsys, left, right):
    rc = cli.main(["equivalent", left, right])
    captured = capsys.readouterr()
    return rc, captured


# ------------------------------------------------------------------ success


def test_identical_circuits_are_equivalent(write_qasm, capsys):
    body = (
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    left = write_qasm(body, "left.qasm")
    right = write_qasm(body, "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    assert captured.err == ""
    raw = captured.out
    assert raw.endswith("\n") and raw.count("\n") == 1
    data = json.loads(raw)
    assert list(data) == [
        "schema_version",
        "equivalent",
        "reason",
        "left_num_qubits",
        "right_num_qubits",
        "distance",
        "tolerance",
    ]
    assert data["schema_version"] == 1
    assert data["equivalent"] is True
    assert data["reason"] == "equivalent"
    assert data["left_num_qubits"] == 2
    assert data["right_num_qubits"] == 2
    assert data["distance"] <= 1e-10
    assert data["tolerance"] == 1e-10


def test_gate_cancelation_and_register_names_irrelevant(write_qasm, capsys):
    left = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nh q[0];\nmeasure q[0] -> c[0];\n", "left.qasm"
    )
    right = write_qasm(
        "// nothing happens here\nqreg reg_a[1];\ncreg reg_b[1];\nmeasure reg_a[0] -> reg_b[0];\n",
        "right.qasm",
    )
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is True
    assert data["reason"] == "equivalent"


def test_global_phase_is_equivalent(write_qasm, capsys):
    # rz(2*pi) = -I: differs from identity only by a global phase.
    left = write_qasm("qreg q[1];\ncreg c[1];\nrz(2*pi) q[0];\nmeasure q[0] -> c[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is True
    assert data["reason"] == "equivalent"


def test_comparison_covers_all_input_states(write_qasm, capsys):
    # x then h vs h then x: identical on |0> sampling but different unitaries.
    left = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nh q[0];\nmeasure q[0] -> c[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\nx q[0];\nmeasure q[0] -> c[0];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is False
    assert data["reason"] == "unitary_distance"
    assert data["distance"] > 1e-10


def test_different_unitary_reports_distance(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is False
    assert data["reason"] == "unitary_distance"
    # tr(X† H) = sqrt(2), so d = sqrt(1 - sqrt(2)/2).
    assert data["distance"] == pytest.approx((1 - 2**0.5 / 2) ** 0.5)


def test_qubit_count_mismatch_has_null_distance(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n", "left.qasm")
    right = write_qasm(
        "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n", "right.qasm"
    )
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is False
    assert data["reason"] == "qubit_count_mismatch"
    assert data["distance"] is None
    assert data["left_num_qubits"] == 1
    assert data["right_num_qubits"] == 2


def test_measurement_layout_mismatch_keeps_distance(write_qasm, capsys):
    left = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        "left.qasm",
    )
    right = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[1];\nmeasure q[1] -> c[0];\n",
        "right.qasm",
    )
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is False
    assert data["reason"] == "measurement_layout_mismatch"
    assert data["distance"] <= 1e-10


def test_creg_width_mismatch_is_layout_mismatch(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[0];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    data = json.loads(captured.out)
    assert data["equivalent"] is False
    assert data["reason"] == "measurement_layout_mismatch"


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    left = write_qasm(
        "qreg q[3];\ncreg c[3];\nrx(0.3+pi/4) q[0];\ncx q[0],q[2];\nry(1e-1*2) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n",
        "left.qasm",
    )
    right = write_qasm(
        "qreg q[3];\ncreg c[3];\nry(0.2) q[1];\nrx(1.0853981633974483) q[0];\ncx q[0],q[2];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n",
        "right.qasm",
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["equivalent", left, right]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_stdin_as_one_side(monkeypatch, capsys, write_qasm):
    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    rc, captured = run_equivalent(capsys, "-", right)
    assert rc == 0
    assert json.loads(captured.out)["equivalent"] is True


# -------------------------------------------------------------------- errors


def test_both_stdin_is_comparison_error(capsys):
    rc, captured = run_equivalent(capsys, "-", "-")
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "comparison_error"
    assert captured.err.count("\n") == 1


def test_missing_file_is_io_error(write_qasm, capsys):
    right = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc, captured = run_equivalent(capsys, "/nonexistent/left.qasm", right)
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_is_io_error(tmp_path, write_qasm, capsys):
    bad = tmp_path / "bad.qasm"
    bad.write_bytes(HEADER.encode() + b"\xff\xfe")
    right = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc, captured = run_equivalent(capsys, str(bad), right)
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_marks_input_side(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["input"] == "left"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_marks_right_side(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\nqreg q2[1];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["input"] == "right"


def test_left_error_reported_before_right(write_qasm, capsys):
    left = write_qasm("qreg q[1];\nqreg q2[1];\n", "left.qasm")
    right = write_qasm("totally not qasm\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 2
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["input"] == "left"


def test_too_many_qubits_is_simulation_error(write_qasm, capsys):
    left = write_qasm("qreg q[9];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "simulation_error"


def test_too_many_qubits_on_right_is_simulation_error(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[9];\ncreg c[1];\n", "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "simulation_error"


def test_eight_qubits_accepted(write_qasm, capsys):
    body = "qreg q[8];\ncreg c[1];\nh q[0];\nh q[0];\nmeasure q[0] -> c[0];\n"
    left = write_qasm(body, "left.qasm")
    right = write_qasm(body, "right.qasm")
    rc, captured = run_equivalent(capsys, left, right)
    assert rc == 0
    assert json.loads(captured.out)["equivalent"] is True


def test_missing_arguments_is_usage_error(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["equivalent", "only-one.qasm"])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""
