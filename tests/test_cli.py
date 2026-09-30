"""End-to-end tests for the quantum-circuit-simulator command."""

from __future__ import annotations

import json

import pytest

from quantum_circuit import __version__
from quantum_circuit.cli import main

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


BELL = (
    HEADER
    + "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\n"
    + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


class StdinStub:
    def __init__(self, data: bytes):
        self.buffer = _BufferStub(data)


class _BufferStub:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_version_command_prints_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == __version__
    assert out == "0.1.0\n"


def test_no_subcommand_prints_help_to_stdout(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "quantum-circuit-simulator" in captured.out
    assert captured.err == ""


def test_simulate_writes_one_ordered_json_object(tmp_path, capsys):
    source = tmp_path / "circuit.qasm"
    source.write_text(BELL, encoding="utf-8")
    assert main(["simulate", str(source), "--shots", "100", "--seed", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert list(result) == [
        "schema_version",
        "shots",
        "seed",
        "num_qubits",
        "num_clbits",
        "counts",
    ]
    assert result["schema_version"] == 1
    assert result["shots"] == 100
    assert result["seed"] == 1
    assert result["num_qubits"] == 2
    assert result["num_clbits"] == 2
    assert sum(result["counts"].values()) == 100
    keys = list(result["counts"])
    assert keys == sorted(keys)


def test_defaults_shots_1024_seed_0(tmp_path, capsys):
    source = tmp_path / "x.qasm"
    source.write_text(
        HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n",
        encoding="utf-8",
    )
    assert main(["simulate", str(source)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["shots"] == 1024
    assert result["seed"] == 0
    assert result["counts"] == {"1": 1024}


def test_stdin_source(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", StdinStub(BELL.encode("utf-8")))
    assert main(["simulate", "-", "--shots", "20", "--seed", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["shots"] == 20
    assert set(result["counts"]) <= {"00", "11"}


def test_repeated_runs_are_byte_identical(tmp_path, capsys):
    source = tmp_path / "bell.qasm"
    source.write_text(BELL, encoding="utf-8")
    outputs = []
    for _ in range(2):
        assert main(["simulate", str(source), "--shots", "256", "--seed", "9"]) == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]


def test_missing_file_is_io_error_exit_1(tmp_path, capsys):
    missing = tmp_path / "nope.qasm"
    assert main(["simulate", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"error": "io_error"}


def test_directory_source_is_io_error(tmp_path, capsys):
    assert main(["simulate", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip()) == {"error": "io_error"}


def test_invalid_utf8_is_io_error(tmp_path, capsys):
    source = tmp_path / "bad.qasm"
    source.write_bytes(b"OPENQASM 2.0;\n\xff\xfe\n")
    assert main(["simulate", str(source)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip()) == {"error": "io_error"}


def test_parse_error_exit_2_with_position(tmp_path, capsys):
    source = tmp_path / "bad.qasm"
    source.write_text(HEADER + "qreg q[1] creg c[1];\n", encoding="utf-8")
    assert main(["simulate", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err.strip())
    assert payload == {"error": "parse_error", "line": 3, "column": 11}


def test_validation_error_exit_2_with_position(tmp_path, capsys):
    source = tmp_path / "bad.qasm"
    source.write_text(HEADER + "qreg q[1];\nx q[5];\n", encoding="utf-8")
    assert main(["simulate", str(source)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err.strip())
    assert payload["error"] == "validation_error"
    assert payload["line"] == 4
    assert "column" in payload


@pytest.mark.parametrize("shots", ["0", "-1", "abc", "1.5", "1_0"])
def test_bad_shots_exits_2_without_reading_input(tmp_path, capsys, shots):
    # Nonexistent source would be io_error (1) if arguments were read first.
    missing = tmp_path / "nope.qasm"
    with pytest.raises(SystemExit) as info:
        main(["simulate", str(missing), "--shots", shots])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("seed", ["abc", "1.0", "0x5", ""])
def test_bad_seed_exits_2_without_reading_input(tmp_path, capsys, seed):
    missing = tmp_path / "nope.qasm"
    with pytest.raises(SystemExit) as info:
        main(["simulate", str(missing), "--seed", seed])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""


def test_negative_seed_accepted(tmp_path, capsys):
    source = tmp_path / "h.qasm"
    source.write_text(
        HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n",
        encoding="utf-8",
    )
    assert main(["simulate", str(source), "--shots", "10", "--seed", "-7"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["seed"] == -7


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    assert "quantum-circuit-simulator" in capsys.readouterr().out


def test_simulate_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as info:
        main(["simulate", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--shots" in out
    assert "--seed" in out


def test_unreadable_file_is_io_error(tmp_path, capsys, monkeypatch):
    source = tmp_path / "blocked.qasm"
    source.write_text(BELL, encoding="utf-8")

    def deny_open(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", deny_open)
    assert main(["simulate", str(source)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err.strip()) == {"error": "io_error"}
