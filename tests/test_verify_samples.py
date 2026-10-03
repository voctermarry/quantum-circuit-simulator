"""Tests for the ``verify-samples`` external-count verification subcommand."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)
X_ONE = "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_samples(tmp_path):
    def _write(payload: object | str | bytes, name: str = "samples.json"):
        path = tmp_path / name
        if isinstance(payload, (bytes, str)) and not isinstance(payload, (dict, list)):
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            path.write_bytes(data)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_model(tmp_path):
    def _write(text: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


def _bell_samples(tmp_path, counts: dict[str, int], name: str = "samples.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "counts": counts}), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------- success


def test_exact_match_payload_shape_and_field_order(write_qasm, tmp_path, capsys):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 50, "11": 50})
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
    assert data["shots"] == 100
    assert data["noise_model"] is None
    assert data["tolerance"] == 0.05
    assert data["total_variation_distance"] == 0.0
    assert data["accepted"] is True
    assert list(data["expected_probabilities"]) == ["00", "11"]
    assert list(data["observed_probabilities"]) == ["00", "11"]
    assert data["expected_probabilities"] == {"00": 0.5, "11": 0.5}
    assert data["observed_probabilities"] == {"00": 0.5, "11": 0.5}


def test_probability_objects_are_sorted(write_qasm, tmp_path, capsys):
    qasm = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    )
    samples = _bell_samples(tmp_path, {"1": 5, "0": 5})
    assert cli.main(["verify-samples", qasm, samples]) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data["observed_probabilities"]) == ["0", "1"]


def test_tvd_is_half_l1_over_union_with_unseen_outcomes_zero(write_qasm, tmp_path, capsys):
    # Bell theory {00: .5, 11: .5}; samples only ever saw 00: observed
    # {00: 1}; unseen-but-legal 11 contributes |.5 - 0| on the union.
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 100})
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.err == ""
    data = json.loads(captured.out)
    assert data["total_variation_distance"] == 0.5
    assert data["accepted"] is False
    assert data["observed_probabilities"] == {"00": 1.0}
    # The full report is still emitted on failure.
    assert data["expected_probabilities"] == {"00": 0.5, "11": 0.5}


def test_samples_with_outcome_theory_omits_still_lands_in_union(write_qasm, tmp_path, capsys):
    # A near-1 outcome in the samples whose theoretical probability snaps to
    # zero/omitted still participates through the observed side.
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\nry(3e-16) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    samples = _bell_samples(tmp_path, {"00": 25, "01": 25, "10": 25, "11": 25})
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 3
    data = json.loads(captured.out)
    # Theory keeps only 00 and 01 (the q[1]=1 entries snap away); the TVD
    # over the full union stays a finite, deterministic number.
    assert math.isfinite(data["total_variation_distance"])
    assert set(data["expected_probabilities"]) == {"00", "01"}
    assert set(data["observed_probabilities"]) == {"00", "01", "10", "11"}


def test_boundary_distance_equal_to_tolerance_is_accepted(write_qasm, tmp_path, capsys):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 100})
    rc = cli.main(["verify-samples", qasm, samples, "--tolerance", "0.5"])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["accepted"] is True


def test_default_tolerance_is_0_05(write_qasm, tmp_path, capsys):
    # TVD 0.04 for bell: observed 00 at 0.54 -> |.04| on each of two keys,
    # half is 0.04, inside the default 0.05.
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 54, "11": 46})
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["tolerance"] == 0.05
    assert data["total_variation_distance"] == pytest.approx(0.04)
    assert data["accepted"] is True


def test_zero_tolerance_requires_exact_distribution(write_qasm, tmp_path, capsys):
    qasm = write_qasm(X_ONE)
    samples = _bell_samples(tmp_path, {"1": 100})
    assert cli.main(["verify-samples", qasm, samples, "--tolerance", "0"]) == 0
    capsys.readouterr()
    samples_off = _bell_samples(tmp_path, {"0": 1, "1": 99}, "off.json")
    rc = cli.main(["verify-samples", qasm, samples_off, "--tolerance", "0"])
    captured = capsys.readouterr()
    assert rc == 3
    assert json.loads(captured.out)["total_variation_distance"] == pytest.approx(0.01)


def test_certain_outcome_reported_as_exactly_one(write_qasm, tmp_path, capsys):
    qasm = write_qasm(X_ONE)
    samples = _bell_samples(tmp_path, {"1": 8})
    assert cli.main(["verify-samples", qasm, samples]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["expected_probabilities"] == {"1": 1.0}
    assert data["observed_probabilities"] == {"1": 1.0}
    assert data["total_variation_distance"] == 0.0


def test_no_measurements_uses_all_zero_result(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\n")
    samples = _bell_samples(tmp_path, {"00": 10})
    assert cli.main(["verify-samples", qasm, samples]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["expected_probabilities"] == {"00": 1.0}
    assert data["observed_probabilities"] == {"00": 1.0}


def test_partial_measurement_and_unwritten_clbits(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[3];\nx q[0];\nmeasure q[0] -> c[1];\n")
    samples = _bell_samples(tmp_path, {"010": 5})
    assert cli.main(["verify-samples", qasm, samples]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_clbits"] == 3
    assert data["expected_probabilities"] == {"010": 1.0}
    assert data["observed_probabilities"] == {"010": 1.0}


def test_expected_probabilities_match_probabilities_command(write_qasm, tmp_path, capsys):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 54, "11": 46})
    assert cli.main(["probabilities", qasm]) == 0
    probs = json.loads(capsys.readouterr().out)["probabilities"]
    assert cli.main(["verify-samples", qasm, samples]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["expected_probabilities"] == probs


def test_repeated_runs_are_byte_identical(write_qasm, tmp_path, capsys):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 60, "11": 40})
    outputs = set()
    for _ in range(3):
        assert cli.main(["verify-samples", qasm, samples, "--tolerance", "0.2"]) in (0, 3)
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


# --------------------------------------------------------------- noise support


def test_noise_model_path_echoes_canonical_channels(write_qasm, write_model, tmp_path, capsys):
    qasm = write_qasm(BELL)
    model = write_model('{"bit_flip": 0.1, "amplitude_damping": 0.2}')
    samples = _bell_samples(tmp_path, {"00": 25, "01": 25, "10": 25, "11": 25})
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc in (0, 3)
    data = json.loads(captured.out)
    assert list(data["noise_model"]) == ["amplitude_damping", "bit_flip"]
    assert data["noise_model"] == {"amplitude_damping": 0.2, "bit_flip": 0.1}
    assert list(data["expected_probabilities"]) == sorted(data["expected_probabilities"])
    assert math.isfinite(data["total_variation_distance"])


def test_noise_model_key_order_and_whitespace_irrelevant(write_qasm, write_model, tmp_path, capsys):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 50, "11": 50})
    a = write_model('{"bit_flip": 0.1, "depolarizing": 0.2}', "a.json")
    b = write_model('{\n "depolarizing":0.2,\n "bit_flip":0.1\n}', "b.json")
    outs = set()
    for model in (a, b):
        assert cli.main(["verify-samples", qasm, samples, "--noise-model", model]) in (0, 3)
        outs.add(capsys.readouterr().out)
    assert len(outs) == 1


def test_ten_qubits_with_noise_succeeds(write_qasm, write_model, tmp_path, capsys):
    qasm = write_qasm("qreg q[10];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"depolarizing": 0.1}')
    samples = _bell_samples(tmp_path, {"0": 5, "1": 5})
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    assert rc in (0, 3)
    data = json.loads(capsys.readouterr().out)
    assert math.isfinite(data["total_variation_distance"])


# ---------------------------------------------------------------- stdin routing


def test_samples_from_stdin(write_qasm, monkeypatch, capsys, tmp_path):
    qasm = write_qasm(BELL)
    body = json.dumps({"schema_version": 1, "counts": {"00": 50, "11": 50}}).encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(body)))
    rc = cli.main(["verify-samples", qasm, "-"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True


def test_source_from_stdin(monkeypatch, capsys, tmp_path):
    samples = _bell_samples(tmp_path, {"00": 50, "11": 50})
    body = (HEADER + BELL).encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(body)))
    rc = cli.main(["verify-samples", "-", samples])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True


def test_noise_model_from_stdin(write_qasm, monkeypatch, capsys, tmp_path):
    qasm = write_qasm(X_ONE)
    samples = _bell_samples(tmp_path, {"1": 10})
    monkeypatch.setattr(
        cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"amplitude_damping": 1}'))
    )
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", "-"])
    captured = capsys.readouterr()
    assert rc == 3
    data = json.loads(captured.out)
    assert data["noise_model"] == {"amplitude_damping": 1.0}
    # Full damping sends |1> to |0>, so the observed all-ones counts mismatch.
    assert data["expected_probabilities"] == {"0": 1.0}


@pytest.mark.parametrize(
    "argv",
    [
        ["-", "-"],
        ["-", "samples.json", "--noise-model", "-"],
        ["circuit.qasm", "-", "--noise-model", "-"],
    ],
)
def test_two_stdin_inputs_is_verification_error_and_stdin_untouched(
    write_qasm, write_samples, monkeypatch, capsys, argv
):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    qasm = write_qasm(BELL, "circuit.qasm")
    samples = write_samples({"schema_version": 1, "counts": {"00": 1}}, "samples.json")
    argv = [arg.replace("circuit.qasm", qasm).replace("samples.json", samples) for arg in argv]
    monkeypatch.setattr(cli.sys, "stdin", ExplodingStdin())
    rc = cli.main(["verify-samples", *argv])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "verification_error"
    assert captured.err.count("\n") == 1


# ------------------------------------------------------------ samples io errors


def test_missing_samples_file_is_tagged_io_error(write_qasm, capsys):
    qasm = write_qasm(BELL)
    rc = cli.main(["verify-samples", qasm, "/nonexistent/samples.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "samples"


def test_invalid_utf8_samples_is_tagged_io_error(write_qasm, write_samples, capsys):
    qasm = write_qasm(BELL)
    samples = write_samples(b'{"schema_version": 1, "counts": {"00": 1}}\xff')
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "samples"


# ------------------------------------------------------- samples document errors


@pytest.mark.parametrize(
    "document",
    [
        "not json",
        "{",
        "[]",
        "{}",
        "null",
        '"schema_version: 1"',
        '{"schema_version": 1}',
        '{"counts": {"00": 1}}',
        '{"schema_version": 2, "counts": {"00": 1}}',
        '{"schema_version": "1", "counts": {"00": 1}}',
        '{"schema_version": 1.0, "counts": {"00": 1}}',
        '{"schema_version": true, "counts": {"00": 1}}',
        '{"schema_version": 1, "counts": {}}',
        '{"schema_version": 1, "counts": []}',
        '{"schema_version": 1, "counts": null}',
        '{"schema_version": 1, "counts": {"00": 0}}',
        '{"schema_version": 1, "counts": {"00": -3}}',
        '{"schema_version": 1, "counts": {"00": true}}',
        '{"schema_version": 1, "counts": {"00": "1"}}',
        '{"schema_version": 1, "counts": {"00": 1.5}}',
        '{"schema_version": 1, "counts": {"0": 1}}',
        '{"schema_version": 1, "counts": {"000": 1}}',
        '{"schema_version": 1, "counts": {"0a": 1}}',
        '{"schema_version": 1, "counts": {"": 1}}',
        '{"schema_version": 1, "counts": {"00": 1}, "extra": 2}',
        '{"schema_version": 1, "schema_version": 1, "counts": {"00": 1}}',
        '{"schema_version": 1, "counts": {"00": 1, "00": 2}}',
        '{"schema_version": 1, "counts": {"00": NaN}}',
        '{"schema_version": 1, "counts": {"00": Infinity}}',
    ],
)
def test_invalid_samples_document_is_sample_input_error(
    write_qasm, write_samples, capsys, document
):
    qasm = write_qasm(BELL)
    samples = write_samples(document)
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "sample_input_error"
    assert captured.err.count("\n") == 1


def test_sample_validation_runs_before_state_evolution(write_qasm, write_model, write_samples, capsys):
    # A document problem must surface as sample_input_error even on a circuit
    # that would fail the noisy 10-qubit limit once evolution started.
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    samples = write_samples({"schema_version": 1, "counts": {}})
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "sample_input_error"


# ----------------------------------------------------- circuit and model errors


def test_missing_source_is_io_error(tmp_path, capsys):
    samples = _bell_samples(tmp_path, {"00": 1})
    rc = cli.main(["verify-samples", "/nonexistent/c.qasm", samples])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_reports_position(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    samples = _bell_samples(tmp_path, {"0": 1})
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_reports_position(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[1];\n")
    samples = _bell_samples(tmp_path, {"0": 1})
    rc = cli.main(["verify-samples", qasm, samples])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "validation_error"


def test_invalid_noise_model_is_noise_model_error(write_qasm, write_model, tmp_path, capsys):
    qasm = write_qasm(BELL)
    model = write_model('{"bit_flip": 2}')
    samples = _bell_samples(tmp_path, {"00": 1})
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


def test_noise_over_ten_qubits_is_simulation_error(write_qasm, write_model, tmp_path, capsys):
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    samples = _bell_samples(tmp_path, {"0": 5, "1": 5})
    rc = cli.main(["verify-samples", qasm, samples, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "simulation_error"


def test_noiseless_twenty_qubit_circuit_succeeds(write_qasm, tmp_path, capsys):
    qasm = write_qasm("qreg q[20];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    samples = _bell_samples(tmp_path, {"1": 4})
    assert cli.main(["verify-samples", qasm, samples]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["expected_probabilities"] == {"1": 1.0}


# -------------------------------------------------------------- argument errors


@pytest.mark.parametrize("value", ["-0.1", "1.01", "nan", "inf", "-inf", "abc", ""])
def test_bad_tolerance_is_usage_error(write_qasm, tmp_path, value):
    qasm = str(tmp_path / "c.qasm")
    Path(qasm).write_text(HEADER + BELL, encoding="utf-8")
    samples = _bell_samples(tmp_path, {"00": 1})
    with pytest.raises(SystemExit) as info:
        cli.main(["verify-samples", qasm, samples, "--tolerance", value])
    assert info.value.code == 2


@pytest.mark.parametrize("flag", ["--shots", "--seed"])
def test_shots_and_seed_are_rejected(write_qasm, tmp_path, capsys, flag):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 1})
    with pytest.raises(SystemExit) as info:
        cli.main(["verify-samples", qasm, samples, flag, "5"])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""


# -------------------------------------------------------------------- --output


def _run_main(argv):
    from contextlib import redirect_stderr, redirect_stdout

    out, err = __import__("io").StringIO(), __import__("io").StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def test_exported_bytes_equal_stdout_on_accept_and_reject(write_qasm, tmp_path):
    qasm = write_qasm(BELL)
    good = _bell_samples(tmp_path, {"00": 50, "11": 50}, "good.json")
    bad = _bell_samples(tmp_path, {"00": 100}, "bad.json")
    for samples, expected_rc in ((good, 0), (bad, 3)):
        rc_stdout, stdout_text, stderr_text = _run_main(["verify-samples", qasm, samples])
        assert rc_stdout == expected_rc and stderr_text == ""
        target = str(tmp_path / f"out-{expected_rc}.json")
        rc, out, err = _run_main(["verify-samples", qasm, samples, "--output", target])
        assert (rc, out, err) == (expected_rc, "", "")
        assert Path(target).read_text(encoding="utf-8") == stdout_text


def test_output_conflict_with_any_input(write_qasm, write_model, tmp_path):
    qasm = write_qasm(BELL, "c.qasm")
    model = write_model('{"bit_flip": 0.1}', "m.json")
    samples = _bell_samples(tmp_path, {"00": 1})

    for target in (qasm, samples, model):
        argv = ["verify-samples", qasm, samples, "--noise-model", model, "--output", target]
        rc, out, err = _run_main(argv)
        assert rc == 2 and out == ""
        assert json.loads(err)["error"] == "output_error"


def test_validation_failure_does_not_create_target(write_qasm, write_samples, tmp_path):
    qasm = write_qasm(BELL)
    samples = write_samples({"schema_version": 1, "counts": {}})
    target = tmp_path / "out.json"
    target.write_text("KEEP-ME", encoding="utf-8")
    rc, out, err = _run_main(["verify-samples", qasm, samples, "--output", str(target)])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "sample_input_error"
    assert target.read_text() == "KEEP-ME"


def test_output_dash_is_output_error(write_qasm, tmp_path):
    qasm = write_qasm(BELL)
    samples = _bell_samples(tmp_path, {"00": 1})
    rc, out, err = _run_main(["verify-samples", qasm, samples, "--output", "-"])
    assert rc == 1 and out == ""
    assert json.loads(err)["error"] == "output_error"
