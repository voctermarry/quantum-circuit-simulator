"""Tests for the deterministic ``probabilities`` subcommand."""

from __future__ import annotations

import io
import json
import math

import pytest

from quantum_circuit import cli
from quantum_circuit.openqasm import parse
from quantum_circuit.simulator import (
    measurement_probabilities,
    simulate_state_vector,
)

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


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


@pytest.fixture
def write_bytes(tmp_path):
    def _write(data: bytes, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return _write


def _program(body: str):
    return parse(HEADER + body)


# ------------------------------------------------- measurement_probabilities()


def test_bell_outcome_probabilities():
    program = _program(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    state = simulate_state_vector(program)
    probabilities = measurement_probabilities(program, [abs(a) ** 2 for a in state])
    assert probabilities["00"] == pytest.approx(0.5)
    assert probabilities["11"] == pytest.approx(0.5)
    assert abs(probabilities["01"]) < 1e-15 and abs(probabilities["10"]) < 1e-15


def test_partial_measurement_aggregates_basis_states():
    # Only q[0] is measured, onto c[0]: both q[1] basis states aggregate.
    program = _program(
        "qreg q[2];\ncreg c[1];\nh q[0];\nx q[1];\nmeasure q[0] -> c[0];\n"
    )
    state = simulate_state_vector(program)
    probabilities = measurement_probabilities(program, [abs(a) ** 2 for a in state])
    assert probabilities == {"0": pytest.approx(0.5), "1": pytest.approx(0.5)}


def test_unwritten_clbits_are_zero():
    program = _program("qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n")
    state = simulate_state_vector(program)
    probabilities = measurement_probabilities(program, [abs(a) ** 2 for a in state])
    assert probabilities["010"] == pytest.approx(1.0)
    assert probabilities["000"] == 0.0


def test_no_measurement_is_all_zero_outcome():
    program = _program("qreg q[2];\ncreg c[2];\nh q[0];\n")
    state = simulate_state_vector(program)
    probabilities = measurement_probabilities(program, [abs(a) ** 2 for a in state])
    assert probabilities == {"00": 1.0}


# ------------------------------------------------------------------- success v1


def test_noiseless_payload_shape_and_field_order(write_qasm, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    rc = cli.main(["probabilities", qasm])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1 and captured.out.endswith("\n")
    data = json.loads(captured.out)
    assert list(data) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "probabilities",
    ]
    assert data["schema_version"] == 1
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    assert list(data["probabilities"]) == sorted(data["probabilities"])
    assert data["probabilities"] == {
        "00": pytest.approx(0.5),
        "11": pytest.approx(0.5),
    }
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-12


def test_probability_values_are_finite_json_numbers_between_zero_and_one(write_qasm, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["probabilities", qasm]) == 0
    for value in json.loads(capsys.readouterr().out)["probabilities"].values():
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert 0.0 < value < 1.0


def test_certain_outcome_is_reported_as_exactly_one(write_qasm, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n")
    assert cli.main(["probabilities", qasm]) == 0
    data = json.loads(capsys.readouterr().out)
    # clbit 0 is the final character (highest clbit index first).
    assert data["probabilities"] == {"01": 1.0}


def test_no_measurement_outputs_single_all_zero_entry(write_qasm, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\n")
    assert cli.main(["probabilities", qasm]) == 0
    raw = capsys.readouterr().out
    assert json.loads(raw)["probabilities"] == {"00": 1.0}


def test_tiny_entries_omitted(write_qasm, capsys):
    # q[1] gets a ~3e-16 rotation: the |...1> outcomes have probability
    # around 1e-32 and must be omitted, leaving the two q[1]=0 entries.
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\nry(3e-16) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["probabilities", qasm]) == 0
    probabilities = json.loads(capsys.readouterr().out)["probabilities"]
    # clbit 1 (q[1]) is the first character; q[1]=1 entries vanish.
    assert set(probabilities) == {"00", "01"}
    assert all(abs(value - 0.5) < 1e-12 for value in probabilities.values())


def test_partial_measurement_mapping(write_qasm, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n")
    assert cli.main(["probabilities", qasm]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_clbits"] == 3
    assert data["probabilities"] == {"010": 1.0}


def test_twenty_qubits_noiseless_succeeds(write_qasm, capsys):
    qasm = write_qasm("qreg q[20];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    rc = cli.main(["probabilities", qasm])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["probabilities"] == {"1": 1.0}


def test_repeated_runs_are_byte_identical(write_qasm, write_model, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["probabilities", qasm]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1
    model = write_model('{"bit_flip": 0.1, "depolarizing": 0.2}')
    noisy = set()
    for _ in range(3):
        assert cli.main(["probabilities", qasm, "--noise-model", model]) == 0
        noisy.add(capsys.readouterr().out)
    assert len(noisy) == 1


# ------------------------------------------------------------------- success v2


def test_noisy_payload_shape_field_order_and_channel_echo(write_qasm, write_model, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    model = write_model('{"bit_flip": 0.1, "amplitude_damping": 0.2}')
    rc = cli.main(["probabilities", qasm, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert list(data) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "noise_model",
        "probabilities",
    ]
    assert data["schema_version"] == 2
    assert list(data["noise_model"]) == ["amplitude_damping", "bit_flip"]
    assert data["noise_model"] == {"amplitude_damping": 0.2, "bit_flip": 0.1}
    assert list(data["probabilities"]) == sorted(data["probabilities"])
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-12


def test_noisy_key_order_and_whitespace_irrelevant(write_qasm, write_model, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    model_a = write_model('{"bit_flip": 0.1, "depolarizing": 0.2}', "a.json")
    model_b = write_model('{\n  "depolarizing" : 0.2,\n  "bit_flip" : 0.1\n}', "b.json")
    outputs = set()
    for model in (model_a, model_b):
        assert cli.main(["probabilities", qasm, "--noise-model", model]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_zero_probability_channels_match_noiseless_probabilities(write_qasm, write_model, capsys):
    body = (
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    qasm = write_qasm(body)
    model = write_model('{"amplitude_damping": 0, "bit_flip": 0.0}')
    assert cli.main(["probabilities", qasm]) == 0
    clean = json.loads(capsys.readouterr().out)
    assert cli.main(["probabilities", qasm, "--noise-model", model]) == 0
    noisy = json.loads(capsys.readouterr().out)
    assert clean["schema_version"] == 1 and noisy["schema_version"] == 2
    assert set(clean["probabilities"]) == set(noisy["probabilities"])
    for key in clean["probabilities"]:
        assert abs(clean["probabilities"][key] - noisy["probabilities"][key]) < 1e-12


def test_noisy_no_measurement_is_all_zero(write_qasm, write_model, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\n")
    model = write_model('{"bit_flip": 0.25}')
    assert cli.main(["probabilities", qasm, "--noise-model", model]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["probabilities"] == {"00": 1.0}


def test_ten_qubits_with_noise_succeeds(write_qasm, write_model, capsys):
    qasm = write_qasm("qreg q[10];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"depolarizing": 0.1}')
    rc = cli.main(["probabilities", qasm, "--noise-model", model])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-12


# --------------------------------------------------------------- stdin and paths


def test_source_from_stdin(write_qasm, monkeypatch, capsys):
    body = (
        HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"
    ).encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(body)))
    rc = cli.main(["probabilities", "-"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["probabilities"] == {"1": 1.0}


def test_noise_model_from_stdin(write_qasm, monkeypatch, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"amplitude_damping": 1}')))
    rc = cli.main(["probabilities", qasm, "--noise-model", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 2
    assert data["probabilities"] == {"0": 1.0}


def test_both_stdin_is_noise_model_error_and_stdin_untouched(monkeypatch, capsys):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    monkeypatch.setattr(cli.sys, "stdin", ExplodingStdin())
    rc = cli.main(["probabilities", "-", "--noise-model", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "noise_model_error"
    assert captured.err.count("\n") == 1


def test_relative_paths_resolve_from_cwd(write_qasm, write_model, monkeypatch, capsys, tmp_path):
    qasm = write_qasm(
        "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n", name="c.qasm"
    )
    model = write_model('{"bit_flip": 0.0}', name="m.json")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["probabilities", "c.qasm", "--noise-model", "m.json"]) == 0
    assert json.loads(capsys.readouterr().out)["probabilities"] == {"1": 1.0}


# ----------------------------------------------------------------------- errors


def test_missing_source_is_io_error(capsys):
    rc = cli.main(["probabilities", "/nonexistent/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_source_is_io_error(write_bytes, capsys):
    path = write_bytes(b'OPENQASM 2.0;\n\xff')
    rc = cli.main(["probabilities", path])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_missing_noise_model_is_io_error(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc = cli.main(["probabilities", qasm, "--noise-model", "/nonexistent/noise.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_reports_position(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    rc = cli.main(["probabilities", qasm])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert isinstance(payload["line"], int) and isinstance(payload["column"], int)


def test_validation_error_reports_position(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[1];\n")
    rc = cli.main(["probabilities", qasm])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "[]",
        '{"nope": 0.5}',
        '{"bit_flip": 2}',
        '{"bit_flip": 0.1, "bit_flip": 0.2}',
        '{"bit_flip": "0.5"}',
        "not json",
    ],
)
def test_invalid_noise_model_content_is_noise_model_error(write_qasm, write_model, capsys, text):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = write_model(text)
    rc = cli.main(["probabilities", qasm, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


def test_invalid_utf8_noise_model_is_noise_model_error(write_qasm, tmp_path, capsys):
    qasm = tmp_path / "c.qasm"
    qasm.write_text(HEADER + "qreg q[1];\ncreg c[1];\n", encoding="utf-8")
    model = tmp_path / "noise.json"
    model.write_bytes(b'{"bit_flip": 0.5}\xff')
    rc = cli.main(["probabilities", str(qasm), "--noise-model", str(model)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


def test_noise_over_ten_qubits_is_simulation_error(write_qasm, write_model, capsys):
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    rc = cli.main(["probabilities", qasm, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "simulation_error"


# ------------------------------------------------------------- no sampling args


@pytest.mark.parametrize("flag", ["--shots", "--seed"])
def test_shots_and_seed_are_rejected(write_qasm, capsys, flag):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    with pytest.raises(SystemExit) as info:
        cli.main(["probabilities", qasm, flag, "5"])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""
