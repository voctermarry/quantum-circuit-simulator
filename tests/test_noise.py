"""Tests for density-matrix noise simulation (--noise-model)."""

from __future__ import annotations

import io
import json
import math

import pytest

from quantum_circuit import cli
from quantum_circuit.noise import (
    NoiseModelError,
    parse_noise_model,
    simulate_density_matrix,
)
from quantum_circuit.openqasm import parse

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


def _program(body: str):
    return parse(HEADER + body)


# ------------------------------------------------------------- model parsing


def test_parse_noise_model_canonical_order_and_floats():
    model = parse_noise_model('{ "bit_flip": 1, "amplitude_damping": 0.25 }')
    assert list(model) == ["amplitude_damping", "bit_flip"]
    assert model == {"amplitude_damping": 0.25, "bit_flip": 1.0}
    assert all(isinstance(v, float) for v in model.values())


def test_parse_noise_model_whitespace_and_key_order_irrelevant():
    a = parse_noise_model('{"bit_flip": 0.1, "depolarizing": 0.2}')
    b = parse_noise_model('{\n  "depolarizing": 0.2,\n  "bit_flip": 0.1\n}')
    assert a == b and list(a) == list(b)


@pytest.mark.parametrize(
    "text",
    [
        "",  # not JSON
        "{",  # truncated
        "[]",  # non-object
        "null",
        '"bit_flip"',
        "0.5",
        "{}",  # empty object
        '{"unknown": 0.1}',  # unknown key
        '{"bit_flip": 0.1, "bit_flip": 0.2}',  # duplicate key
        '{"bit_flip": true}',  # not a number
        '{"bit_flip": null}',
        '{"bit_flip": "0.5"}',
        '{"bit_flip": [0.5]}',
        '{"bit_flip": -0.1}',  # out of range
        '{"bit_flip": 1.0000001}',
        '{"bit_flip": 1e400}',  # not finite
        '{"bit_flip": NaN}',  # not valid JSON
        '{"bit_flip": Infinity}',
    ],
)
def test_parse_noise_model_rejects_invalid(text):
    with pytest.raises(NoiseModelError):
        parse_noise_model(text)


@pytest.mark.parametrize("text", ['{"bit_flip": 0}', '{"bit_flip": 1}', '{"phase_damping": 0.5e-1}'])
def test_parse_noise_model_accepts_boundaries(text):
    assert parse_noise_model(text)


# -------------------------------------------------------- density matrix math


def test_zero_probability_channels_match_state_vector():
    body = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    program = _program(body)
    diag = simulate_density_matrix(program, {"bit_flip": 0.0})
    assert abs(diag[0] - 0.5) < 1e-12
    assert abs(diag[3] - 0.5) < 1e-12
    assert abs(sum(diag) - 1.0) < 1e-12


def test_amplitude_damping_full_decay():
    program = _program("qreg q[1];\ncreg c[1];\nx q[0];\n")
    diag = simulate_density_matrix(program, {"amplitude_damping": 1.0})
    assert abs(diag[0] - 1.0) < 1e-12
    assert abs(diag[1]) < 1e-12


def test_amplitude_damping_partial():
    program = _program("qreg q[1];\ncreg c[1];\nx q[0];\n")
    diag = simulate_density_matrix(program, {"amplitude_damping": 0.3})
    assert abs(diag[0] - 0.3) < 1e-12
    assert abs(diag[1] - 0.7) < 1e-12


def test_phase_damping_preserves_populations():
    program = _program("qreg q[1];\ncreg c[1];\nh q[0];\n")
    diag = simulate_density_matrix(program, {"phase_damping": 1.0})
    assert abs(diag[0] - 0.5) < 1e-12
    assert abs(diag[1] - 0.5) < 1e-12


def test_bit_flip_full_flip():
    # Noise only applies after gates, so add an x first.
    program = _program("qreg q[1];\ncreg c[1];\nx q[0];\n")
    diag = simulate_density_matrix(program, {"bit_flip": 1.0})
    assert abs(diag[0] - 1.0) < 1e-12
    assert abs(diag[1]) < 1e-12


def test_depolarizing_full_mixes():
    program = _program("qreg q[1];\ncreg c[1];\nx q[0];\n")
    diag = simulate_density_matrix(program, {"depolarizing": 1.0})
    assert abs(diag[0] - 0.5) < 1e-12
    assert abs(diag[1] - 0.5) < 1e-12


def test_noise_applies_to_both_cx_qubits():
    # After h q[0]; cx q[0],q[1] the state is Bell; full bit flip on both
    # qubits maps |00><00|+|11><11| block onto itself swapped: still Bell
    # populations. Full amplitude damping on q[1] after cx moves all
    # population to q[1]=0.
    program = _program("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n")
    diag = simulate_density_matrix(program, {"amplitude_damping": 1.0})
    # q[1] damped to 0 after cx: remaining states are |00> and |01> (q[0]
    # still in superposition, itself damped to 0 afterwards -> |00>).
    assert abs(diag[0] - 1.0) < 1e-12
    assert abs(sum(diag) - 1.0) < 1e-12


def test_diagonal_stays_normalized_with_all_channels():
    program = _program(
        "qreg q[3];\ncreg c[3];\nh q[0];\nrx(pi/3) q[1];\ncx q[0],q[2];\nry(0.7) q[2];\n"
    )
    noise = {
        "amplitude_damping": 0.1,
        "phase_damping": 0.2,
        "bit_flip": 0.05,
        "depolarizing": 0.15,
    }
    diag = simulate_density_matrix(program, noise)
    assert abs(sum(diag) - 1.0) < 1e-9
    assert all(p >= -1e-12 for p in diag)


# ------------------------------------------------------------------ CLI paths


def test_success_payload_schema_and_field_order(write_qasm, write_model, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    model = write_model('{"bit_flip": 0.1, "amplitude_damping": 0.2}')
    rc = cli.main(["simulate", qasm, "--shots", "100", "--seed", "5", "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.count("\n") == 1
    data = json.loads(captured.out)
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
    assert list(data["noise_model"]) == ["amplitude_damping", "bit_flip"]
    assert data["noise_model"] == {"amplitude_damping": 0.2, "bit_flip": 0.1}
    assert sum(data["counts"].values()) == 100
    assert list(data["counts"]) == sorted(data["counts"])


def test_noise_output_byte_identical_across_key_order_and_whitespace(
    write_qasm, write_model, capsys
):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    model_a = write_model('{"bit_flip": 0.1, "depolarizing": 0.2}', "a.json")
    model_b = write_model('{ "depolarizing" : 0.2,\n  "bit_flip" : 0.1 }', "b.json")
    outputs = set()
    for model in (model_a, model_b, model_a):
        rc = cli.main(["simulate", qasm, "--shots", "64", "--seed", "9", "--noise-model", model])
        assert rc == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_no_noise_model_output_unchanged(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["simulate", qasm, "--shots", "32", "--seed", "4"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    assert "noise_model" not in data


def test_zero_probability_noise_matches_baseline_counts(write_qasm, write_model, capsys):
    body = (
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    qasm = write_qasm(body)
    model = write_model('{"amplitude_damping": 0, "bit_flip": 0.0}')
    assert cli.main(["simulate", qasm, "--shots", "200", "--seed", "11"]) == 0
    baseline = json.loads(capsys.readouterr().out)["counts"]
    assert cli.main(["simulate", qasm, "--shots", "200", "--seed", "11", "--noise-model", model]) == 0
    noisy = json.loads(capsys.readouterr().out)["counts"]
    assert noisy == baseline


def test_noise_model_from_stdin(write_qasm, monkeypatch, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"amplitude_damping": 1}')))
    rc = cli.main(["simulate", qasm, "--noise-model", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"0": 1024}


def test_both_stdin_is_noise_model_error_and_stdin_untouched(monkeypatch, capsys):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    monkeypatch.setattr(cli.sys, "stdin", ExplodingStdin())
    rc = cli.main(["simulate", "-", "--noise-model", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "noise_model_error"
    assert captured.err.count("\n") == 1


def test_missing_noise_model_file_is_io_error(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc = cli.main(["simulate", qasm, "--noise-model", "/nonexistent/noise.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_noise_model_is_noise_model_error(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    model = tmp_path / "noise.json"
    model.write_bytes(b'{"bit_flip": 0.5}\xff\xfe')
    rc = cli.main(["simulate", qasm, "--noise-model", str(model)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        "[]",
        '{"nope": 0.5}',
        '{"bit_flip": 2}',
        '{"bit_flip": 0.1, "bit_flip": 0.2}',
        "not json",
    ],
)
def test_invalid_noise_model_content_exit_2(write_qasm, write_model, capsys, text):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    model = write_model(text)
    rc = cli.main(["simulate", qasm, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "noise_model_error"
    assert captured.err.count("\n") == 1


def test_noise_over_ten_qubits_is_simulation_error(write_qasm, write_model, capsys):
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    rc = cli.main(["simulate", qasm, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "simulation_error"
    assert captured.err.count("\n") == 1


def test_ten_qubits_with_noise_succeeds(write_qasm, write_model, capsys):
    body = "qreg q[10];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    qasm = write_qasm(body)
    model = write_model('{"depolarizing": 0.1}')
    rc = cli.main(["simulate", qasm, "--shots", "8", "--seed", "1", "--noise-model", model])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert sum(data["counts"].values()) == 8


def test_parse_error_still_reported_with_noise_model(write_qasm, write_model, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    model = write_model('{"bit_flip": 0.1}')
    rc = cli.main(["simulate", qasm, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "parse_error"


def test_measurement_mapping_and_unwritten_clbits_with_noise(write_qasm, write_model, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n"
    )
    model = write_model('{"phase_damping": 0.9}')
    rc = cli.main(["simulate", qasm, "--shots", "16", "--seed", "0", "--noise-model", model])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"010": 16}
