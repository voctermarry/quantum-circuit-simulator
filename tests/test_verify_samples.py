"""Tests for the deterministic ``verify-samples`` subcommand."""

from __future__ import annotations

import io
import json
import math

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

BELL_BODY = (
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
def write_samples(tmp_path):
    def _write(text: str, name: str = "samples.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_model(tmp_path):
    def _write(text: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


# ------------------------------------------------------------------- success


def test_accepted_payload_shape_and_field_order(write_qasm, write_samples, capsys):
    qasm = write_qasm(BELL_BODY)
    samples = write_samples('{"schema_version": 1, "counts": {"00": 510, "11": 490}}')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1 and captured.out.endswith("\n")
    data = json.loads(captured.out)
    assert list(data) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "shots",
        "noise_model",
        "tolerance",
        "total_variation_distance",
        "accepted",
        "expected_probabilities",
        "observed_probabilities",
    ]
    assert data["schema_version"] == 1
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    assert data["shots"] == 1000
    assert data["noise_model"] is None
    assert data["tolerance"] == 0.05
    assert data["accepted"] is True
    assert data["total_variation_distance"] == pytest.approx(0.01)
    assert list(data["expected_probabilities"]) == sorted(data["expected_probabilities"])
    assert list(data["observed_probabilities"]) == sorted(data["observed_probabilities"])
    assert data["expected_probabilities"] == {
        "00": pytest.approx(0.5),
        "11": pytest.approx(0.5),
    }
    assert data["observed_probabilities"] == {"00": 0.51, "11": 0.49}


def test_rejected_still_reports_and_exits_3(write_qasm, write_samples, capsys):
    qasm = write_qasm(BELL_BODY)
    samples = write_samples('{"schema_version": 1, "counts": {"00": 1000}}')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.err == ""
    data = json.loads(captured.out)
    assert data["accepted"] is False
    # The unobserved "11" outcome contributes its full expected probability.
    assert data["total_variation_distance"] == pytest.approx(0.5)
    assert data["observed_probabilities"] == {"00": 1.0}


def test_custom_tolerance_flips_verdict(write_qasm, write_samples, capsys):
    qasm = write_qasm(BELL_BODY)
    samples = write_samples('{"schema_version": 1, "counts": {"00": 510, "11": 490}}')
    assert cli.main(["verify-samples", qasm, samples, "--tolerance", "0.001"]) == 3
    data = json.loads(capsys.readouterr().out)
    assert data["tolerance"] == 0.001
    assert data["accepted"] is False


def test_distance_equal_to_tolerance_is_accepted(write_qasm, write_samples, capsys):
    # Certain "0" outcome; one shot in sixteen lands on the impossible "1".
    # TVD = 0.5 * (0.0625 + 0.0625) = 0.0625 exactly (binary-exact values).
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"0": 15, "1": 1}}')
    rc = cli.main(["verify-samples", qasm, samples, "--tolerance", "0.0625"])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["total_variation_distance"] == 0.0625
    assert data["accepted"] is True
    assert cli.main(["verify-samples", qasm, samples, "--tolerance", "0.0624"]) == 3
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_impossible_outcome_counts_against_observed(write_qasm, write_samples, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"1": 10}}')
    assert cli.main(["verify-samples", qasm, samples]) == 3
    data = json.loads(capsys.readouterr().out)
    assert data["expected_probabilities"] == {"0": 1.0}
    assert data["observed_probabilities"] == {"1": 1.0}
    assert data["total_variation_distance"] == pytest.approx(1.0)


def test_no_measurement_circuit(write_qasm, write_samples, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"00": 7}}')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["total_variation_distance"] == 0.0
    assert data["expected_probabilities"] == {"00": 1.0}
    assert data["observed_probabilities"] == {"00": 1.0}


def test_partial_measurement_mapping(write_qasm, write_samples, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"010": 42}}')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["num_clbits"] == 3
    assert data["expected_probabilities"] == {"010": 1.0}


def test_noisy_verification_echoes_model(write_qasm, write_samples, write_model, capsys):
    qasm = write_qasm(BELL_BODY)
    model = write_model('{"bit_flip": 0.1, "amplitude_damping": 0.2}')
    # Build counts proportional to the exact noisy distribution.
    assert cli.main(["probabilities", qasm, "--noise-model", model]) == 0
    exact = json.loads(capsys.readouterr().out)["probabilities"]
    counts = {key: round(probability * 10000) for key, probability in exact.items()}
    samples = write_samples(json.dumps({"schema_version": 1, "counts": counts}))
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["noise_model"] == {"amplitude_damping": 0.2, "bit_flip": 0.1}
    assert list(data["noise_model"]) == ["amplitude_damping", "bit_flip"]
    assert abs(sum(data["expected_probabilities"].values()) - 1.0) < 1e-12
    assert data["accepted"] is True


def test_expected_probabilities_match_probabilities_command(
    write_qasm, write_samples, write_model, capsys
):
    qasm = write_qasm(BELL_BODY)
    model = write_model('{"bit_flip": 0.1, "depolarizing": 0.2}')
    samples = write_samples('{"schema_version": 1, "counts": {"00": 50, "11": 50}}')
    for extra in ([], ["--noise-model", model]):
        assert cli.main(["probabilities", qasm, *extra]) == 0
        probabilities = json.loads(capsys.readouterr().out)["probabilities"]
        cli.main(["verify-samples", qasm, samples, *extra])
        verified = json.loads(capsys.readouterr().out)
        assert verified["expected_probabilities"] == probabilities


def test_repeated_runs_are_byte_identical(write_qasm, write_samples, write_model, capsys):
    qasm = write_qasm(BELL_BODY)
    model = write_model('{"bit_flip": 0.1}')
    samples = write_samples(
        '{"schema_version": 1, "counts": {"00": 41, "01": 9, "10": 9, "11": 41}}'
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["verify-samples", qasm, samples, "--noise-model", model]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


# --------------------------------------------------------------- stdin and paths


def test_source_from_stdin(write_samples, monkeypatch, capsys):
    body = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    samples = write_samples('{"schema_version": 1, "counts": {"1": 25}}')
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(body)))
    rc = cli.main(["verify-samples", "-", samples])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True


def test_samples_from_stdin(write_qasm, monkeypatch, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(b'{"schema_version": 1, "counts": {"1": 25}}')),
    )
    rc = cli.main(["verify-samples", qasm, "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["shots"] == 25
    assert data["accepted"] is True


def test_noise_model_from_stdin(write_qasm, write_samples, monkeypatch, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"0": 100}}')
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"amplitude_damping": 1}')))
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["noise_model"] == {"amplitude_damping": 1.0}
    assert data["expected_probabilities"] == {"0": 1.0}


@pytest.mark.parametrize(
    "argv",
    [
        ["verify-samples", "-", "-"],
        ["verify-samples", "-", "SAMPLES", "--noise-model", "-"],
        ["verify-samples", "SOURCE", "-", "--noise-model", "-"],
    ],
)
def test_multiple_stdin_is_verification_error_and_stdin_untouched(
    monkeypatch, capsys, argv, tmp_path
):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    for name in ("SOURCE", "SAMPLES"):
        if name in argv:
            path = tmp_path / name.lower()
            path.write_text("placeholder", encoding="utf-8")
            argv[argv.index(name)] = str(path)
    monkeypatch.setattr(cli.sys, "stdin", ExplodingStdin())
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "verification_error"
    assert captured.err.count("\n") == 1


# ----------------------------------------------------------------------- errors


def test_missing_samples_is_io_error(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc = cli.main(["verify-samples", qasm, "/nonexistent/samples.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "samples"


def test_invalid_utf8_samples_is_io_error(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    samples = tmp_path / "samples.json"
    samples.write_bytes(b'{"schema_version": 1, "counts": {"0": 1}}\xff')
    rc = cli.main(["verify-samples", qasm, str(samples)])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "samples"


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        "{}",
        '{"schema_version": 1}',
        '{"counts": {"0": 1}}',
        '{"schema_version": 1, "counts": {"0": 1}, "extra": 1}',
        '{"schema_version": 2, "counts": {"0": 1}}',
        '{"schema_version": true, "counts": {"0": 1}}',
        '{"schema_version": "1", "counts": {"0": 1}}',
        '{"schema_version": 1, "counts": {}}',
        '{"schema_version": 1, "counts": []}',
        '{"schema_version": 1, "counts": {"0": 1, "0": 2}}',
        '{"schema_version": 1, "counts": {"0": 1}, "schema_version": 1}',
        '{"schema_version": 1, "counts": {"0": 0}}',
        '{"schema_version": 1, "counts": {"0": -3}}',
        '{"schema_version": 1, "counts": {"0": 1.5}}',
        '{"schema_version": 1, "counts": {"0": true}}',
        '{"schema_version": 1, "counts": {"0": "1"}}',
        '{"schema_version": 1, "counts": {"0": NaN}}',
    ],
)
def test_invalid_samples_document_is_sample_input_error(
    write_qasm, write_samples, capsys, text
):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    samples = write_samples(text)
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "sample_input_error"
    assert captured.err.count("\n") == 1


@pytest.mark.parametrize(
    "key",
    ["", "00", "2", "a", " 0", "0 "],
)
def test_counts_key_width_and_alphabet(write_qasm, write_samples, capsys, key):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    samples = write_samples(json.dumps({"schema_version": 1, "counts": {key: 1}}))
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "sample_input_error"


def test_circuit_errors_keep_existing_types(write_qasm, write_samples, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    samples = write_samples('{"schema_version": 1, "counts": {"0": 1}}')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "parse_error"

    rc = cli.main(["verify-samples", "/nonexistent/circuit.qasm", samples])
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_noise_model_is_noise_model_error(write_qasm, write_samples, write_model, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 2}')
    samples = write_samples('{"schema_version": 1, "counts": {"0": 1}}')
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


def test_noise_over_ten_qubits_is_simulation_error(
    write_qasm, write_samples, write_model, capsys
):
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    samples = write_samples('{"schema_version": 1, "counts": {"0": 10}}')
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "simulation_error"


def test_samples_validated_before_state_evolution(write_qasm, write_samples, capsys):
    # An 11-qubit noisy circuit would be a simulation_error, but an invalid
    # samples document is reported first (no state is evolved).
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {}}')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "sample_input_error"


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "inf", "-inf", "abc", ""])
def test_invalid_tolerance_is_usage_error(write_qasm, write_samples, capsys, value):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"0": 1}}')
    with pytest.raises(SystemExit) as info:
        cli.main(["verify-samples", qasm, samples, "--tolerance", value])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", ["0", "1", "0.25"])
def test_valid_tolerance_bounds(write_qasm, write_samples, capsys, value):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    samples = write_samples('{"schema_version": 1, "counts": {"0": 1}}')
    rc = cli.main(["verify-samples", qasm, samples, "--tolerance", value])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tolerance"] == float(value)
    assert math.isfinite(data["total_variation_distance"])


# -------------------------------------------------------------------- --output


def test_output_export_writes_report_and_keeps_stdout_empty(
    write_qasm, write_samples, tmp_path, capsys
):
    qasm = write_qasm(BELL_BODY)
    samples = write_samples('{"schema_version": 1, "counts": {"00": 510, "11": 490}}')
    target = tmp_path / "report.json"
    rc = cli.main(["verify-samples", qasm, samples, "--output", str(target)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "" and captured.err == ""
    exported = target.read_text(encoding="utf-8")
    assert exported.count("\n") == 1
    assert json.loads(exported)["accepted"] is True


def test_output_export_preserves_rejection_exit_code(
    write_qasm, write_samples, tmp_path, capsys
):
    qasm = write_qasm(BELL_BODY)
    samples = write_samples('{"schema_version": 1, "counts": {"00": 1000}}')
    target = tmp_path / "report.json"
    rc = cli.main(["verify-samples", qasm, samples, "--output", str(target)])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == "" and captured.err == ""
    assert json.loads(target.read_text(encoding="utf-8"))["accepted"] is False


def test_output_matching_an_input_is_output_error(
    write_qasm, write_samples, write_model, capsys
):
    qasm = write_qasm(BELL_BODY)
    model = write_model('{"bit_flip": 0.1}')
    samples = write_samples('{"schema_version": 1, "counts": {"00": 10}}')
    for target in (qasm, samples, model):
        rc = cli.main(
            ["verify-samples", qasm, samples, "--noise-model", model, "--output", target]
        )
        captured = capsys.readouterr()
        assert rc == 2
        assert captured.out == ""
        assert json.loads(captured.err)["error"] == "output_error"


def test_output_dash_is_output_error(write_qasm, write_samples, capsys):
    qasm = write_qasm(BELL_BODY)
    samples = write_samples('{"schema_version": 1, "counts": {"00": 10}}')
    rc = cli.main(["verify-samples", qasm, samples, "--output", "-"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "output_error"
