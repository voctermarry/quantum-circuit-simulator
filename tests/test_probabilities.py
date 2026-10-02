"""End-to-end tests for the ``probabilities`` subcommand."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import cli

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


@pytest.fixture
def write_noise(tmp_path):
    def _write(text: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
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


# --------------------------------------------------------------- success output


def test_noiseless_payload_shape_and_order(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["probabilities", path]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    raw = captured.out
    assert raw.endswith("\n") and raw.count("\n") == 1

    data = json.loads(raw)
    assert list(data) == ["schema_version", "num_qubits", "num_clbits", "probabilities"]
    assert data["schema_version"] == 1
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    probabilities = data["probabilities"]
    assert list(probabilities) == sorted(probabilities)
    assert probabilities == {"00": 0.5000000000000001, "11": 0.5000000000000001}
    assert abs(sum(probabilities.values()) - 1.0) <= 1e-12


def test_definite_outcome_is_reported_as_integer_one(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["probabilities", path]) == 0
    raw = capsys.readouterr().out
    assert json.loads(raw)["probabilities"] == {"1": 1}
    # The snapped value is the JSON integer 1, not 1.0.
    assert '"1": 1' in raw


def test_no_measurement_outputs_all_zero_key(write_qasm, capsys):
    path = write_qasm("qreg q[2];\ncreg c[3];\nh q[0];\nh q[1];\n")
    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["probabilities"] == {"000": 1}


def test_unwritten_clbits_stay_zero_and_mapping_is_honored(write_qasm, capsys):
    # q[0] is measured into c[1]; c[0] and c[2] are never written.
    path = write_qasm("qreg q[1];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n")
    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["probabilities"] == {"010": 1}


def test_basis_states_aggregate_by_measurement_mapping(write_qasm, capsys):
    # Only q[0] is measured, so the q[1] superposition aggregates away.
    path = write_qasm(
        "qreg q[2];\ncreg c[1];\nh q[1];\nmeasure q[0] -> c[0];\n"
    )
    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["probabilities"] == {"0": 1}


def test_tiny_probabilities_are_dropped_and_near_one_snapped(write_qasm, capsys):
    # rx(2*pi) leaves |0> with probability 1 - ~1.5e-32 and |1> with ~1.5e-32.
    path = write_qasm("qreg q[1];\ncreg c[1];\nrx(2*pi) q[0];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["probabilities"] == {"0": 1}


def test_probabilities_are_finite_numbers_in_unit_interval(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\nrx(pi/3) q[0];\nry(pi/5) q[1];\nrz(pi/7) q[2];\n"
        "cx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    assert cli.main(["probabilities", path]) == 0
    data = json.loads(capsys.readouterr().out)
    values = list(data["probabilities"].values())
    assert values
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
    assert abs(sum(values) - 1.0) <= 1e-12


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\nh q[0];\nry(pi/3) q[1];\ncx q[1],q[2];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["probabilities", path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_stdin_source(monkeypatch, capsys):
    source = HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"

    class _FakeStdin:
        buffer = __import__("io").BytesIO(source.encode("utf-8"))

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    assert cli.main(["probabilities", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert abs(sum(data["probabilities"].values()) - 1.0) <= 1e-12


def test_no_shots_or_seed_arguments(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    with pytest.raises(SystemExit) as info:
        cli.main(["probabilities", path, "--shots", "10"])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        cli.main(["probabilities", path, "--seed", "1"])
    assert info.value.code == 2


# ------------------------------------------------------------------ noise model


def test_noisy_payload_shape_and_channel_order(write_qasm, write_noise, capsys):
    circuit = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    )
    # Keys deliberately out of channel order with noisy whitespace.
    model = write_noise('{ "bit_flip": 0.1,\n  "amplitude_damping": 0.05 }')
    assert cli.main(["probabilities", circuit, "--noise-model", model]) == 0
    raw = capsys.readouterr().out
    assert raw.endswith("\n") and raw.count("\n") == 1
    data = json.loads(raw)
    assert list(data) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "noise_model",
        "probabilities",
    ]
    assert data["schema_version"] == 2
    assert list(data["noise_model"]) == ["amplitude_damping", "bit_flip"]
    assert data["noise_model"] == {"amplitude_damping": 0.05, "bit_flip": 0.1}
    probabilities = data["probabilities"]
    assert list(probabilities) == sorted(probabilities)
    assert abs(sum(probabilities.values()) - 1.0) <= 1e-12


def test_noisy_output_is_byte_identical_across_runs(write_qasm, write_noise, capsys):
    circuit = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    model = write_noise('{"depolarizing": 0.02, "phase_damping": 0.1}')
    outputs = set()
    for _ in range(3):
        assert cli.main(["probabilities", circuit, "--noise-model", model]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_noisy_too_many_qubits_is_simulation_error(write_qasm, write_noise, capsys):
    circuit = write_qasm("qreg q[11];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = write_noise('{"bit_flip": 0.1}')
    assert cli.main(["probabilities", circuit, "--noise-model", model]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "simulation_error"
    assert "10" in payload["message"] and "11" in payload["message"]


def test_noise_model_from_stdin(write_qasm, monkeypatch, capsys):
    circuit = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")

    class _FakeStdin:
        buffer = __import__("io").BytesIO(b'{"bit_flip": 1.0}')

    monkeypatch.setattr(sys, "stdin", _FakeStdin())
    assert cli.main(["probabilities", circuit, "--noise-model", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 2
    assert data["probabilities"] == {"0": 1}


def test_both_stdin_is_rejected_without_reading(monkeypatch, capsys):
    class _ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    monkeypatch.setattr(sys, "stdin", _ExplodingStdin())
    assert cli.main(["probabilities", "-", "--noise-model", "-"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


# ----------------------------------------------------------------- error paths


def test_missing_source_is_io_error(capsys):
    assert cli.main(["probabilities", "/nonexistent/circuit.qasm"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_source_is_io_error(write_bytes, capsys):
    path = write_bytes(b"\xff\xfe")
    assert cli.main(["probabilities", path]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_has_position(write_qasm, capsys):
    path = write_qasm("qreg q[1]\n")
    assert cli.main(["probabilities", path]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_has_position(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[1];\n")
    assert cli.main(["probabilities", path]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_missing_noise_model_is_io_error(write_qasm, capsys):
    circuit = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["probabilities", circuit, "--noise-model", "/nonexistent/noise.json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_noise_model_is_noise_model_error(write_qasm, write_bytes, capsys):
    circuit = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = write_bytes(b"\xff\xfe", name="noise.json")
    assert cli.main(["probabilities", circuit, "--noise-model", model]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "{}",
        '{"unknown_channel": 0.1}',
        '{"bit_flip": 0.1, "bit_flip": 0.2}',
        '{"bit_flip": "0.1"}',
        '{"bit_flip": 1.5}',
        '{"bit_flip": -0.1}',
    ],
)
def test_invalid_noise_models(write_qasm, write_noise, capsys, text):
    circuit = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = write_noise(text)
    assert cli.main(["probabilities", circuit, "--noise-model", model]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


# ----------------------------------------------------------- real-process checks


def test_process_success_is_single_json_line():
    source = HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    result = _run_process("probabilities", "-", stdin=source)
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["schema_version"] == 1
    assert abs(sum(data["probabilities"].values()) - 1.0) <= 1e-12


def test_process_io_error_exit_code():
    result = _run_process("probabilities", "/nonexistent/circuit.qasm")
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "io_error"


def test_process_simulation_error_exit_code(tmp_path):
    circuit = tmp_path / "big.qasm"
    circuit.write_text(
        HEADER + "qreg q[11];\ncreg c[1];\nmeasure q[0] -> c[0];\n", encoding="utf-8"
    )
    model = tmp_path / "noise.json"
    model.write_text('{"bit_flip": 0.1}', encoding="utf-8")
    result = _run_process("probabilities", str(circuit), "--noise-model", str(model))
    assert result.returncode == 3
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "simulation_error"


def test_process_bad_argument_exit_code():
    result = _run_process("probabilities", "-", "--shots", "10")
    assert result.returncode == 2
    assert result.stdout == ""
