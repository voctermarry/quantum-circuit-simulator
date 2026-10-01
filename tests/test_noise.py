"""End-to-end tests for the optional density-matrix noise simulation."""

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
def write_model(tmp_path):
    def _write(text: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


# --------------------------------------------------------------- success output


def test_noise_success_payload_shape_and_order(write_qasm, write_model, capsys):
    path = write_qasm(BELL)
    model = write_model('{"bit_flip": 0.1, "amplitude_damping": 0.2}')
    rc = cli.main(["simulate", path, "--shots", "100", "--seed", "7", "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    raw = captured.out
    assert raw.endswith("\n") and raw.count("\n") == 1

    data = json.loads(raw)
    assert list(data) == [
        "schema_version",
        "shots",
        "seed",
        "num_qubits",
        "num_clbits",
        "noise_model",
        "counts",
    ]
    assert data["schema_version"] == 2
    assert (data["shots"], data["seed"]) == (100, 7)
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    # Echoed in fixed channel order, only the channels that were given.
    assert list(data["noise_model"]) == ["amplitude_damping", "bit_flip"]
    assert data["noise_model"] == {"amplitude_damping": 0.2, "bit_flip": 0.1}
    counts = data["counts"]
    assert list(counts) == sorted(counts)
    assert sum(counts.values()) == 100


def test_noise_model_whitespace_and_key_order_do_not_matter(write_qasm, write_model, capsys):
    path = write_qasm(BELL)
    model_a = write_model('{"amplitude_damping": 0.2, "bit_flip": 0.1}', "a.json")
    model_b = write_model('  {\n  "bit_flip":   0.1,\n  "amplitude_damping": 0.2\n}\n', "b.json")
    outputs = set()
    for model in (model_a, model_b):
        assert cli.main(["simulate", path, "--shots", "128", "--seed", "5", "--noise-model", model]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_noise_repeated_runs_are_byte_identical(write_qasm, write_model, capsys):
    path = write_qasm(BELL)
    model = write_model('{"depolarizing": 0.3, "phase_damping": 0.1}')
    outputs = set()
    for _ in range(3):
        assert cli.main(["simulate", path, "--shots", "256", "--seed", "42", "--noise-model", model]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_noise_model_from_stdin(write_qasm, monkeypatch, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"bit_flip": 1}')))
    rc = cli.main(["simulate", path, "--shots", "16", "--noise-model", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 2
    assert data["counts"] == {"0": 16}  # x then a certain bit flip


def test_circuit_from_stdin_with_noise_model_file(write_qasm, write_model, monkeypatch, capsys):
    model = write_model('{"amplitude_damping": 1}')
    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    rc = cli.main(["simulate", "-", "--shots", "16", "--noise-model", model])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"0": 16}  # the excitation fully decays


def test_noise_channel_physics(write_qasm, write_model, capsys):
    # phase_damping preserves populations: x then full dephasing stays |1>.
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"phase_damping": 1}')
    assert cli.main(["simulate", path, "--shots", "32", "--noise-model", model]) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {"1": 32}


def test_noiseless_output_is_unchanged(write_qasm, capsys):
    path = write_qasm(BELL)
    assert cli.main(["simulate", path, "--shots", "50", "--seed", "7"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data) == ["schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts"]
    assert data["schema_version"] == 1


# -------------------------------------------------------------------- io errors


def test_missing_noise_model_file_is_io_error(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    rc = cli.main(["simulate", path, "--noise-model", "/nonexistent/noise.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert captured.err.count("\n") == 1


# --------------------------------------------------------- noise model errors


@pytest.mark.parametrize(
    "text",
    [
        "{",  # invalid JSON
        "[]",  # non-object
        '"bit_flip"',  # non-object
        "0.5",  # non-object
        "{}",  # empty object
        '{"unknown": 0.5}',  # unknown key
        '{"bit_flip": 0.1, "bit_flip": 0.2}',  # duplicate key
        '{"bit_flip": -0.1}',  # below range
        '{"bit_flip": 1.5}',  # above range
        '{"bit_flip": NaN}',  # non-finite
        '{"bit_flip": Infinity}',  # non-finite
        '{"bit_flip": 1e999}',  # overflows to infinity
        '{"bit_flip": true}',  # not a number
        '{"bit_flip": "0.5"}',  # not a number
        '{"bit_flip": null}',  # not a number
        '{"bit_flip": [0.5]}',  # not a number
    ],
)
def test_invalid_noise_models_are_rejected(write_qasm, write_model, capsys, text):
    path = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = write_model(text)
    rc = cli.main(["simulate", path, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "noise_model_error"
    assert captured.err.count("\n") == 1


def test_noise_model_invalid_utf8(write_qasm, tmp_path, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = tmp_path / "noise.json"
    model.write_bytes(b'{"bit_flip": 0.1}\xff\xfe')
    rc = cli.main(["simulate", path, "--noise-model", str(model)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


def test_both_inputs_stdin_is_rejected_without_reading_stdin(monkeypatch, capsys):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    monkeypatch.setattr(cli.sys, "stdin", ExplodingStdin())
    rc = cli.main(["simulate", "-", "--noise-model", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


# ----------------------------------------------------------- simulation errors


def test_noisy_circuit_over_ten_qubits_is_simulation_error(write_qasm, write_model, capsys):
    path = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    rc = cli.main(["simulate", path, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "simulation_error"
    assert captured.err.count("\n") == 1


def test_ten_qubit_noisy_circuit_is_allowed(write_qasm, write_model, capsys):
    path = write_qasm("qreg q[10];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    rc = cli.main(["simulate", path, "--shots", "8", "--noise-model", model])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_qubits"] == 10
    assert sum(data["counts"].values()) == 8


def test_large_circuit_without_noise_still_uses_state_vector_limit(write_qasm, capsys):
    path = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["simulate", path, "--shots", "8"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1


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


def test_process_noise_success(tmp_path):
    circuit = tmp_path / "circuit.qasm"
    circuit.write_text(HEADER + BELL, encoding="utf-8")
    model = tmp_path / "noise.json"
    model.write_text('{"amplitude_damping": 0.1, "depolarizing": 0.05}', encoding="utf-8")
    result = _run_process(
        "simulate", str(circuit), "--shots", "64", "--seed", "9", "--noise-model", str(model)
    )
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["schema_version"] == 2
    assert list(data["noise_model"]) == ["amplitude_damping", "depolarizing"]
    assert sum(data["counts"].values()) == 64


def test_process_noise_model_error_exit_code(tmp_path):
    circuit = tmp_path / "circuit.qasm"
    circuit.write_text(HEADER + "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n", encoding="utf-8")
    model = tmp_path / "noise.json"
    model.write_text("{}", encoding="utf-8")
    result = _run_process("simulate", str(circuit), "--noise-model", str(model))
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "noise_model_error"


def test_process_stdin_conflict_exit_code():
    result = _run_process("simulate", "-", "--noise-model", "-", stdin="")
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "noise_model_error"


def test_process_simulation_error_exit_code(tmp_path):
    circuit = tmp_path / "circuit.qasm"
    circuit.write_text(
        HEADER + "qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n", encoding="utf-8"
    )
    model = tmp_path / "noise.json"
    model.write_text('{"bit_flip": 0.1}', encoding="utf-8")
    result = _run_process("simulate", str(circuit), "--noise-model", str(model))
    assert result.returncode == 3
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "simulation_error"
