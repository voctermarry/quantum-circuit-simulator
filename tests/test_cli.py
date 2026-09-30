"""End-to-end tests for the command line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import __version__, cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent


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


# ------------------------------------------------------------- legacy behavior


def test_version_subcommand_prints_version(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_no_subcommand_prints_help(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "simulate" in out


def test_help_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_simulate_help_flag(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["simulate", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--shots" in out and "--seed" in out


# --------------------------------------------------------------- success output


def test_success_json_shape_and_field_order(write_qasm):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["simulate", path, "--shots", "50", "--seed", "7"]) == 0


def test_success_payload(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    rc = cli.main(["simulate", path, "--shots", "50", "--seed", "7"])
    captured = capsys.readouterr()
    assert rc == 0
    raw = captured.out
    assert raw.endswith("\n")
    assert raw.count("\n") == 1

    data = json.loads(raw)
    assert data["schema_version"] == 1
    assert (data["shots"], data["seed"]) == (50, 7)
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    assert sum(data["counts"].values()) == 50

    # Field order is part of the contract.
    assert list(data) == ["schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts"]
    counts = data["counts"]
    assert list(counts) == sorted(counts)


def test_defaults_are_1024_shots_and_seed_zero(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["simulate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["shots"] == 1024 and data["seed"] == 0
    assert data["counts"] == {"0": 1024}


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\nh q[0];\nh q[1];\ncx q[1],q[2];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["simulate", path, "--shots", "128", "--seed", "123"]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_negative_seed_accepted(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["simulate", path, "--seed", "-999999"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["seed"] == -999999


def test_stdin_source(monkeypatch, capsys):
    import io

    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    rc = cli.main(["simulate", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"1": 1024}


# -------------------------------------------------------------------- io errors


def test_missing_file_is_io_error(capsys):
    rc = cli.main(["simulate", "/nonexistent/path/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert captured.err.count("\n") == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_unreadable_file_is_io_error(tmp_path, capsys):
    path = tmp_path / "secret.qasm"
    path.write_text(HEADER + "qreg q[1];\n")
    path.chmod(0o000)
    try:
        rc = cli.main(["simulate", str(path)])
    finally:
        path.chmod(0o644)
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_is_io_error(write_bytes, capsys):
    path = write_bytes(HEADER.encode() + b"\xff\xfe bad")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


# ------------------------------------------------------------- parse / validation


def test_parse_error_exit_code_and_payload(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert isinstance(payload["line"], int) and payload["line"] >= 1
    assert isinstance(payload["column"], int) and payload["column"] >= 1


def test_validation_error_exit_code_and_payload(write_qasm, capsys):
    path = write_qasm("qreg q[1];\nqreg q2[1];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


# ------------------------------------------------------------- argument errors


def test_shots_must_be_positive(capsys):
    for bad in ["0", "-1", "abc", "1.5"]:
        with pytest.raises(SystemExit) as info:
            cli.main(["simulate", "-", "--shots", bad])
        assert info.value.code == 2
        assert capsys.readouterr().out == ""


def test_seed_must_be_integer(capsys):
    for bad in ["abc", "1.5", "0x1"]:
        with pytest.raises(SystemExit) as info:
            cli.main(["simulate", "-", "--seed", bad])
        assert info.value.code == 2
        assert capsys.readouterr().out == ""


def test_argument_error_does_not_open_source(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["simulate", "/nonexistent/never-read.qasm", "--shots", "0"])
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert "io_error" not in captured.err


# ----------------------------------------------------------- real-process checks


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


def test_process_bad_argument_exit_code():
    result = _run_process("simulate", "-", "--shots", "x")
    assert result.returncode == 2
    assert result.stdout == ""


def test_process_io_error_exit_code():
    result = _run_process("simulate", "/nonexistent/circuit.qasm")
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "io_error"


def test_process_stdin_success_is_single_json_line():
    source = HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    result = _run_process("simulate", "-", "--shots", "8", "--seed", "2", stdin=source)
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["shots"] == 8
    assert sum(data["counts"].values()) == 8


def test_process_parameterized_gates_reproducible():
    source = (
        HEADER
        + "qreg q[2];\ncreg c[2];\nrx(pi/2) q[0];\nry(pi) q[1];\n"
        + "rz(-pi/4) q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    first = _run_process("simulate", "-", "--shots", "200", "--seed", "11", stdin=source)
    second = _run_process("simulate", "-", "--shots", "200", "--seed", "11", stdin=source)
    assert first.returncode == 0
    assert first.stdout == second.stdout
    data = json.loads(first.stdout)
    assert sum(data["counts"].values()) == 200
    # ry(pi) puts q[1] in |1>, so every observed key has the high bit set.
    assert all(key.startswith("1") for key in data["counts"])


def test_process_angle_validation_error_exit_code():
    source = HEADER + "qreg q[1];\ncreg c[1];\nrx(1/0) q[0];\n"
    result = _run_process("simulate", "-", stdin=source)
    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["error"] == "validation_error"
    assert payload["line"] == 5
    assert payload["column"] == 5


def test_process_angle_parse_error_exit_code():
    source = HEADER + "qreg q[1];\ncreg c[1];\nrx() q[0];\n"
    result = _run_process("simulate", "-", stdin=source)
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "parse_error"
