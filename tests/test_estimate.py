"""End-to-end tests for the ``estimate`` subcommand."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent

FIELD_ORDER = [
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
GATE_ORDER = ["x", "h", "cx", "rx", "ry", "rz"]


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


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


def test_default_mode_payload(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\nx q[2];\ncx q[1],q[2];\nrx(0.5) q[0];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    rc = cli.main(["estimate", path])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.count("\n") == 1 and captured.out.endswith("\n")
    data = json.loads(captured.out)
    assert list(data) == FIELD_ORDER
    assert list(data["gate_counts"]) == GATE_ORDER
    assert data["schema_version"] == 1
    assert data["mode"] == "state-vector"
    assert (data["num_qubits"], data["num_clbits"]) == (3, 3)
    assert data["gate_count"] == 5
    assert data["measurement_count"] == 3
    assert data["gate_counts"] == {"x": 1, "h": 1, "cx": 2, "rx": 1, "ry": 0, "rz": 0}
    assert data["circuit_depth"] == 3
    assert data["entry_count"] == 8
    assert data["complex_payload_bytes"] == 128
    assert data["supported"] is True
    assert data["qubit_limit"] == 20


def test_disjoint_gates_share_a_layer(write_qasm, capsys):
    path = write_qasm("qreg q[3];\ncreg c[1];\nx q[0];\nh q[1];\nrz(0.1) q[2];\n")
    assert cli.main(["estimate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["circuit_depth"] == 1


def test_measurements_excluded_from_depth_and_gate_count(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    assert cli.main(["estimate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["gate_count"] == 1
    assert data["measurement_count"] == 2
    assert data["circuit_depth"] == 1


def test_no_gates_has_depth_zero(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[1];\n")
    assert cli.main(["estimate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["circuit_depth"] == 0
    assert data["gate_count"] == 0
    assert data["entry_count"] == 4


@pytest.mark.parametrize(
    "mode,base,limit",
    [("state-vector", 2, 20), ("density-matrix", 4, 10), ("unitary", 4, 8)],
)
def test_mode_entry_counts_and_limits(write_qasm, capsys, mode, base, limit):
    path = write_qasm("qreg q[2];\ncreg c[1];\nh q[0];\n")
    assert cli.main(["estimate", path, "--mode", mode]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == mode
    assert data["entry_count"] == base**2
    assert data["complex_payload_bytes"] == data["entry_count"] * 16
    assert data["qubit_limit"] == limit
    assert data["supported"] is True


def test_over_limit_still_estimates(write_qasm, capsys):
    # 20 qubits exceeds the unitary limit of 8 but the estimate is returned.
    path = write_qasm("qreg q[20];\ncreg c[1];\n")
    rc = cli.main(["estimate", path, "--mode", "unitary"])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["supported"] is False
    assert data["qubit_limit"] == 8
    assert data["entry_count"] == 4**20
    assert data["complex_payload_bytes"] == 4**20 * 16


def test_stdin_source(monkeypatch, capsys):
    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    assert cli.main(["estimate", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["gate_count"] == 1
    assert data["entry_count"] == 2


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    outputs = set()
    for _ in range(3):
        assert cli.main(["estimate", path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_missing_file_is_io_error(capsys):
    rc = cli.main(["estimate", "/nonexistent/path/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_reports_position(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    rc = cli.main(["estimate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_reports_position(write_qasm, capsys):
    path = write_qasm("qreg q[1];\nqreg q2[1];\n")
    rc = cli.main(["estimate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"


def test_bad_mode_rejected_without_reading_source(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["estimate", "/nonexistent/never-read.qasm", "--mode", "bogus"])
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "io_error" not in captured.err


def test_process_bad_mode_exit_code():
    result = _run_process("estimate", "-", "--mode", "bogus", stdin="")
    assert result.returncode == 2
    assert result.stdout == ""
