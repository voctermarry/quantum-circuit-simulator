"""End-to-end tests for the estimate subcommand."""

from __future__ import annotations

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


def run_estimate(*argv, stdin_bytes: bytes | None = None, monkeypatch=None, capsys=None):
    if stdin_bytes is not None:
        import io

        class _Stdin:
            buffer = io.BytesIO(stdin_bytes)

        monkeypatch.setattr("sys.stdin", _Stdin())
    rc = cli.main(["estimate", *argv])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# --------------------------------------------------------------- success output


def test_field_order_and_basic_counts(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "rx(0.5) q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )
    rc, out, err = run_estimate(path, capsys=capsys)
    assert rc == 0
    assert err == ""
    data = json.loads(out)
    assert list(data) == [
        "schema_version",
        "mode",
        "num_qubits",
        "num_clbits",
        "gate_count",
        "measurement_count",
        "gate_counts",
        "circuit_depth",
        "entry_count",
        "complex_payload_bytes",
        "supported",
        "qubit_limit",
    ]
    assert data["schema_version"] == 1
    assert data["mode"] == "state-vector"
    assert data["num_qubits"] == 2
    assert data["num_clbits"] == 2
    assert data["gate_count"] == 3
    assert data["measurement_count"] == 2
    assert list(data["gate_counts"]) == ["x", "h", "cx", "rx", "ry", "rz"]
    assert data["gate_counts"] == {"x": 0, "h": 1, "cx": 1, "rx": 1, "ry": 0, "rz": 0}
    assert data["circuit_depth"] == 3
    assert data["entry_count"] == 4
    assert data["complex_payload_bytes"] == 64
    assert data["supported"] is True
    assert data["qubit_limit"] == 20


def test_default_mode_is_state_vector(write_qasm, capsys):
    path = write_qasm("qreg q[3];\ncreg c[1];\nx q[0];\n")
    rc, out, _ = run_estimate(path, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "state-vector"
    assert data["entry_count"] == 8
    assert data["qubit_limit"] == 20


def test_density_matrix_mode_uses_four_to_the_n(write_qasm, capsys):
    path = write_qasm("qreg q[3];\ncreg c[1];\nx q[0];\n")
    rc, out, _ = run_estimate(path, "--mode", "density-matrix", capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "density-matrix"
    assert data["entry_count"] == 64
    assert data["complex_payload_bytes"] == 64 * 16
    assert data["qubit_limit"] == 10
    assert data["supported"] is True


def test_unitary_mode_uses_four_to_the_n(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[1];\nh q[0];\n")
    rc, out, _ = run_estimate(path, "--mode", "unitary", capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["mode"] == "unitary"
    assert data["entry_count"] == 16
    assert data["complex_payload_bytes"] == 256
    assert data["qubit_limit"] == 8


def test_disjoint_gates_share_layers(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[1];\n"
        "x q[0];\n"
        "h q[1];\n"   # disjoint from x q[0]: same layer
        "h q[0];\n"   # stacks on x q[0]
        "cx q[0],q[2];\n"
    )
    rc, out, _ = run_estimate(path, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["circuit_depth"] == 3


def test_measurements_do_not_count_toward_depth(write_qasm, capsys):
    path = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    )
    rc, out, _ = run_estimate(path, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["circuit_depth"] == 1
    assert data["measurement_count"] == 1


def test_empty_circuit_has_zero_depth_and_counts(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\n")
    rc, out, _ = run_estimate(path, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["gate_count"] == 0
    assert data["measurement_count"] == 0
    assert data["gate_counts"] == {"x": 0, "h": 0, "cx": 0, "rx": 0, "ry": 0, "rz": 0}
    assert data["circuit_depth"] == 0
    assert data["entry_count"] == 4


def test_over_limit_still_returns_estimate(write_qasm, capsys):
    path = write_qasm("qreg q[12];\ncreg c[1];\nh q[0];\n")
    rc, out, err = run_estimate(path, "--mode", "density-matrix", capsys=capsys)
    assert rc == 0
    assert err == ""
    data = json.loads(out)
    assert data["supported"] is False
    assert data["qubit_limit"] == 10
    assert data["entry_count"] == 4**12
    assert data["complex_payload_bytes"] == 4**12 * 16


def test_unitary_over_limit(write_qasm, capsys):
    path = write_qasm("qreg q[9];\ncreg c[1];\n")
    rc, out, _ = run_estimate(path, "--mode", "unitary", capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["supported"] is False
    assert data["qubit_limit"] == 8


def test_state_vector_limit_20_is_supported(write_qasm, capsys):
    path = write_qasm("qreg q[20];\ncreg c[1];\n")
    rc, out, _ = run_estimate(path, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["supported"] is True
    assert data["entry_count"] == 2**20


def test_stdin_source(monkeypatch, capsys):
    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\n").encode("utf-8")
    rc, out, _ = run_estimate("-", stdin_bytes=source, monkeypatch=monkeypatch, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["gate_counts"]["x"] == 1


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\nry(0.7) q[0];\ncx q[0],q[1];\n")
    rc1, out1, _ = run_estimate(path, capsys=capsys)
    rc2, out2, _ = run_estimate(path, capsys=capsys)
    assert rc1 == rc2 == 0
    assert out1 == out2


# ----------------------------------------------------------------- error paths


def test_missing_file_is_io_error(capsys):
    rc = cli.main(["estimate", "/nonexistent/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "io_error"


def test_invalid_utf8_is_io_error(tmp_path, capsys):
    bad = tmp_path / "bad.qasm"
    bad.write_bytes(b"\xff\xfe not utf-8")
    rc = cli.main(["estimate", str(bad)])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_exit_2(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nh q[0]\n")
    rc = cli.main(["estimate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "parse_error"
    assert "line" in data and "column" in data


def test_validation_error_exit_2(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\ny q[0];\n")
    rc = cli.main(["estimate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "validation_error"


def test_invalid_mode_rejected_without_reading_source(monkeypatch, capsys):
    import io

    class _Stdin:
        buffer = io.BytesIO(b"should not be read")

    monkeypatch.setattr("sys.stdin", _Stdin())
    with pytest.raises(SystemExit) as info:
        cli.main(["estimate", "-", "--mode", "bogus"])
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
