"""End-to-end tests for the state-metrics subcommand."""

from __future__ import annotations

import json
import math

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


def run_metrics(*argv, stdin_bytes: bytes | None = None, monkeypatch=None, capsys=None):
    if stdin_bytes is not None:
        import io

        class _Stdin:
            buffer = io.BytesIO(stdin_bytes)

        monkeypatch.setattr("sys.stdin", _Stdin())
    rc = cli.main(["state-metrics", *argv])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# --------------------------------------------------------------- success output


def test_bell_state_against_itself(write_qasm, capsys):
    body = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    left = write_qasm(body, "left.qasm")
    right = write_qasm(body, "right.qasm")
    rc, out, err = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    assert err == ""
    data = json.loads(out)
    assert list(data) == [
        "schema_version",
        "left_num_qubits",
        "right_num_qubits",
        "reason",
        "fidelity",
        "left_single_qubit_entropy",
        "right_single_qubit_entropy",
    ]
    assert data["schema_version"] == 1
    assert data["reason"] == "compared"
    assert data["fidelity"] == 1.0
    assert data["left_single_qubit_entropy"] == [1.0, 1.0]
    assert data["right_single_qubit_entropy"] == [1.0, 1.0]


def test_product_state_has_zero_entropy(write_qasm, capsys):
    left = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\nh q[1];\n", "left.qasm")
    right = write_qasm("qreg q[2];\ncreg c[2];\nx q[0];\n", "right.qasm")
    rc, out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["left_single_qubit_entropy"] == [0.0, 0.0]
    assert data["right_single_qubit_entropy"] == [0.0, 0.0]
    # |++> vs |10> (q[0] LSB): overlap amplitude 1/2, fidelity 1/4.
    assert data["fidelity"] == pytest.approx(0.25)


def test_orthogonal_states_have_zero_fidelity(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    rc, out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == 0.0


def test_partial_entanglement_entropy(write_qasm, capsys):
    # ry(2*theta)|0> = cos(theta)|0> + sin(theta)|1> on q[0], then cx entangles.
    theta = 0.3
    body = (
        f"qreg q[2];\ncreg c[2];\nry({2 * theta}) q[0];\ncx q[0],q[1];\n"
    )
    left = write_qasm(body, "left.qasm")
    right = write_qasm(body, "right.qasm")
    rc, out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    p = math.cos(theta) ** 2
    expected = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    assert data["left_single_qubit_entropy"] == pytest.approx([expected, expected])
    assert 0.0 < data["left_single_qubit_entropy"][0] < 1.0


def test_measurements_are_ignored(write_qasm, capsys):
    left = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n", "left.qasm"
    )
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    rc, out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["reason"] == "compared"
    assert data["fidelity"] == 1.0


def test_qubit_count_mismatch(write_qasm, capsys):
    left = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    rc, out, err = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    assert err == ""
    data = json.loads(out)
    assert data["reason"] == "qubit_count_mismatch"
    assert data["fidelity"] is None
    assert data["left_num_qubits"] == 2
    assert data["right_num_qubits"] == 1
    # Entropy arrays are still computed for both sides.
    assert data["left_single_qubit_entropy"] == [1.0, 1.0]
    assert data["right_single_qubit_entropy"] == [0.0]


def test_stdin_on_one_side(write_qasm, monkeypatch, capsys):
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    source = (HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\n").encode("utf-8")
    rc, out, _ = run_metrics("-", right, stdin_bytes=source, monkeypatch=monkeypatch, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == 1.0


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    left = write_qasm("qreg q[2];\ncreg c[2];\nry(0.7) q[0];\ncx q[0],q[1];\n", "left.qasm")
    right = write_qasm("qreg q[2];\ncreg c[2];\nrx(1.1) q[1];\n", "right.qasm")
    rc1, out1, _ = run_metrics(left, right, capsys=capsys)
    rc2, out2, _ = run_metrics(left, right, capsys=capsys)
    assert rc1 == rc2 == 0
    assert out1 == out2


def test_no_nan_infinity_or_negative_zero(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    rc, out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    for token in ("NaN", "Infinity", "-0"):
        assert token not in out


# ----------------------------------------------------------------- error paths


def test_both_stdin_is_metrics_error(monkeypatch, capsys):
    import io

    class _Stdin:
        buffer = io.BytesIO(b"should not be read")

    monkeypatch.setattr("sys.stdin", _Stdin())
    rc = cli.main(["state-metrics", "-", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "metrics_error"


def test_missing_file_reports_io_error_with_side(capsys):
    rc = cli.main(["state-metrics", "/nonexistent/left.qasm", "/nonexistent/right.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "io_error"
    assert data["input"] == "left"


def test_left_validated_before_right(write_qasm, capsys):
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    rc = cli.main(["state-metrics", "/nonexistent/left.qasm", right])
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["input"] == "left"


def test_right_failure_reported_after_left_passes(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    rc = cli.main(["state-metrics", left, "/nonexistent/right.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["input"] == "right"


def test_invalid_utf8_is_io_error(write_qasm, tmp_path, capsys):
    bad = tmp_path / "bad.qasm"
    bad.write_bytes(b"\xff\xfe not utf-8")
    good = write_qasm("qreg q[1];\ncreg c[1];\n", "good.qasm")
    rc = cli.main(["state-metrics", str(bad), good])
    captured = capsys.readouterr()
    assert rc == 1
    data = json.loads(captured.err)
    assert data["error"] == "io_error"
    assert data["input"] == "left"


def test_parse_error_reports_side(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    bad = write_qasm("qreg q[1];\ncreg c[1];\nh q[0]\n", "bad.qasm")
    rc = cli.main(["state-metrics", left, bad])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "parse_error"
    assert data["input"] == "right"


def test_validation_error_reports_side(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\ny q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 2
    data = json.loads(captured.err)
    assert data["error"] == "validation_error"
    assert data["input"] == "left"


def test_register_size_limit_is_validation_error(write_qasm, capsys):
    left = write_qasm("qreg q[21];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 2
    data = json.loads(captured.err)
    assert data["error"] == "validation_error"
    assert data["input"] == "left"


def test_argument_error_exits_2_without_reading_stdin(monkeypatch, capsys):
    import io

    class _Stdin:
        buffer = io.BytesIO(b"should not be read")

    monkeypatch.setattr("sys.stdin", _Stdin())
    with pytest.raises(SystemExit) as info:
        cli.main(["state-metrics", "-"])
    assert info.value.code == 2


# ------------------------------------------------------------- noisy comparison


@pytest.fixture
def write_model(tmp_path):
    def _write(content: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    return _write


def test_noisy_success_schema_and_field_order(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 1.0}')
    rc, out, err = run_metrics(left, right, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    assert err == ""
    data = json.loads(out)
    assert list(data) == [
        "schema_version",
        "left_num_qubits",
        "right_num_qubits",
        "reason",
        "left_noise_model",
        "right_noise_model",
        "fidelity",
        "left_purity",
        "right_purity",
        "left_single_qubit_entropy",
        "right_single_qubit_entropy",
    ]
    assert data["schema_version"] == 2
    assert data["reason"] == "compared"
    assert data["left_noise_model"] == {"depolarizing": 1.0}
    assert data["right_noise_model"] is None
    # Fully depolarized single qubit: rho = I/2.
    assert data["left_purity"] == pytest.approx(0.5)
    assert data["right_purity"] == 1.0
    assert data["left_single_qubit_entropy"] == [1.0]
    assert data["right_single_qubit_entropy"] == [0.0]
    # F = <+|I/2|+> = 1/2.
    assert data["fidelity"] == 0.5


def test_noise_model_echoed_in_canonical_order(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    model = write_model('{ "depolarizing": 0.5,\n"bit_flip": 0.1 }')
    rc, out, _ = run_metrics(left, right, "--right-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["left_noise_model"] is None
    assert list(data["right_noise_model"]) == ["bit_flip", "depolarizing"]
    assert data["right_noise_model"] == {"bit_flip": 0.1, "depolarizing": 0.5}


def test_pure_mixed_fidelity_is_expectation_value(write_qasm, write_model, capsys):
    # rho = (1-p)|+><+| + p I/2 with p = 0.5; F = <+|rho|+> = 1 - p/2 = 0.75.
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.5}')
    rc, out, _ = run_metrics(left, right, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == pytest.approx(0.75)
    assert data["left_purity"] == pytest.approx(0.625)


def test_both_noisy_uhlmann_fidelity(write_qasm, write_model, capsys):
    # Both sides fully depolarized to I/2: F = (Tr(I/2))^2 = 1.
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 1.0}')
    rc, out, _ = run_metrics(
        left,
        right,
        "--left-noise-model",
        model,
        "--right-noise-model",
        model,
        capsys=capsys,
    )
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == 1.0
    assert data["left_purity"] == pytest.approx(0.5)
    assert data["right_purity"] == pytest.approx(0.5)
    assert data["left_single_qubit_entropy"] == [1.0]
    assert data["right_single_qubit_entropy"] == [1.0]


def test_both_noisy_distinct_states_fidelity(write_qasm, write_model, capsys):
    # Left: I/2. Right: eigenvalues 3/4, 1/4 (depolarizing 0.5 on |+>).
    # F = ((sqrt(3/4) + sqrt(1/4)) / sqrt(2))^2.
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    full = write_model('{"depolarizing": 1.0}', "full.json")
    half = write_model('{"depolarizing": 0.5}', "half.json")
    rc, out, _ = run_metrics(
        left,
        right,
        "--left-noise-model",
        full,
        "--right-noise-model",
        half,
        capsys=capsys,
    )
    assert rc == 0
    data = json.loads(out)
    expected = ((math.sqrt(0.75) + math.sqrt(0.25)) / math.sqrt(2)) ** 2
    assert data["fidelity"] == pytest.approx(expected)


def test_zero_probability_noise_matches_pure_metrics(write_qasm, write_model, capsys):
    body = "qreg q[2];\ncreg c[2];\nry(0.7) q[0];\ncx q[0],q[1];\n"
    left = write_qasm(body, "left.qasm")
    right = write_qasm(body, "right.qasm")
    model = write_model('{"bit_flip": 0.0, "depolarizing": 0.0}')
    rc, out, _ = run_metrics(left, right, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["schema_version"] == 2
    assert data["fidelity"] == 1.0
    assert data["left_purity"] == 1.0
    rc, pure_out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    pure = json.loads(pure_out)
    assert data["left_single_qubit_entropy"] == pytest.approx(pure["left_single_qubit_entropy"])
    assert data["right_single_qubit_entropy"] == pure["right_single_qubit_entropy"]


def test_noisy_qubit_count_mismatch(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 1.0}')
    rc, out, err = run_metrics(left, right, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    assert err == ""
    data = json.loads(out)
    assert data["schema_version"] == 2
    assert data["reason"] == "qubit_count_mismatch"
    assert data["fidelity"] is None
    # Purity and entropy arrays are still reported for both sides.
    assert data["left_purity"] == pytest.approx(0.25)
    assert data["right_purity"] == 1.0
    assert data["left_single_qubit_entropy"] == [1.0, 1.0]
    assert data["right_single_qubit_entropy"] == [0.0]


def test_noisy_measurements_are_ignored(write_qasm, write_model, capsys):
    left = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n", "left.qasm"
    )
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.5}')
    rc, out, _ = run_metrics(left, right, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["reason"] == "compared"
    assert data["fidelity"] == pytest.approx(0.75)


def test_noisy_repeated_runs_are_byte_identical(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[2];\ncreg c[2];\nry(0.7) q[0];\ncx q[0],q[1];\n", "left.qasm")
    right = write_qasm("qreg q[2];\ncreg c[2];\nrx(1.1) q[1];\n", "right.qasm")
    left_model = write_model('{"amplitude_damping": 0.3, "phase_damping": 0.2}', "left.json")
    right_model = write_model('{"bit_flip": 0.1, "depolarizing": 0.4}', "right.json")
    argv = (left, right, "--left-noise-model", left_model, "--right-noise-model", right_model)
    rc1, out1, _ = run_metrics(*argv, capsys=capsys)
    rc2, out2, _ = run_metrics(*argv, capsys=capsys)
    assert rc1 == rc2 == 0
    assert out1 == out2


def test_noisy_no_nan_infinity_or_negative_zero(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "right.qasm")
    model = write_model('{"amplitude_damping": 1.0}')
    rc, out, _ = run_metrics(
        left, right, "--left-noise-model", model, "--right-noise-model", model, capsys=capsys
    )
    assert rc == 0
    for token in ("NaN", "Infinity", "-0"):
        assert token not in out


def test_noise_model_from_stdin(write_qasm, monkeypatch, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    rc, out, _ = run_metrics(
        left,
        right,
        "--left-noise-model",
        "-",
        stdin_bytes=b'{"depolarizing": 1.0}',
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
    assert rc == 0
    assert json.loads(out)["left_noise_model"] == {"depolarizing": 1.0}


# --------------------------------------------------------- noisy error paths


def test_multiple_stdin_inputs_is_metrics_error_and_stdin_untouched(write_qasm, monkeypatch, capsys):
    import io

    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")

    class _Stdin:
        buffer = io.BytesIO(b"should not be read")

    monkeypatch.setattr("sys.stdin", _Stdin())
    rc = cli.main(["state-metrics", "-", right, "--right-noise-model", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "metrics_error"


def test_both_noise_models_stdin_is_metrics_error(write_qasm, monkeypatch, capsys):
    import io

    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")

    class _Stdin:
        buffer = io.BytesIO(b"should not be read")

    monkeypatch.setattr("sys.stdin", _Stdin())
    rc = cli.main(
        ["state-metrics", left, right, "--left-noise-model", "-", "--right-noise-model", "-"]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "metrics_error"


def test_missing_noise_model_file_is_io_error_with_side(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    rc = cli.main(
        ["state-metrics", left, right, "--right-noise-model", "/nonexistent/noise.json"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "io_error"
    assert data["input"] == "right"


def test_invalid_noise_model_is_noise_model_error_with_side(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    model = write_model('{"depolarizing": 1.5}')
    rc = cli.main(["state-metrics", left, right, "--left-noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "noise_model_error"
    assert data["input"] == "left"


def test_invalid_utf8_noise_model_is_noise_model_error(write_qasm, tmp_path, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe not utf-8")
    rc = cli.main(["state-metrics", left, right, "--left-noise-model", str(bad)])
    captured = capsys.readouterr()
    assert rc == 2
    data = json.loads(captured.err)
    assert data["error"] == "noise_model_error"
    assert data["input"] == "left"


def test_first_problem_reported_in_input_order(write_qasm, write_model, capsys):
    # Left source fails before anything else, even with a bad left model.
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    bad_model = write_model('{"depolarizing": 2.0}', "bad.json")
    rc = cli.main(
        ["state-metrics", "/nonexistent/left.qasm", right, "--left-noise-model", bad_model]
    )
    captured = capsys.readouterr()
    assert rc == 1
    data = json.loads(captured.err)
    assert data["error"] == "io_error"
    assert data["input"] == "left"

    # Left model fails before the right source is read.
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    rc = cli.main(
        ["state-metrics", left, "/nonexistent/right.qasm", "--left-noise-model", bad_model]
    )
    captured = capsys.readouterr()
    assert rc == 2
    data = json.loads(captured.err)
    assert data["error"] == "noise_model_error"
    assert data["input"] == "left"

    # Right source fails before the right model.
    rc = cli.main(
        ["state-metrics", left, "/nonexistent/right.qasm", "--right-noise-model", bad_model]
    )
    captured = capsys.readouterr()
    assert rc == 1
    data = json.loads(captured.err)
    assert data["error"] == "io_error"
    assert data["input"] == "right"


def test_noisy_side_over_ten_qubits_is_simulation_error(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[11];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.5}')
    rc = cli.main(["state-metrics", left, right, "--left-noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    data = json.loads(captured.err)
    assert data["error"] == "simulation_error"
    assert data["input"] == "left"


def test_noisy_right_side_over_ten_qubits_reports_right(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[11];\ncreg c[1];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.5}')
    rc = cli.main(["state-metrics", left, right, "--right-noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    data = json.loads(captured.err)
    assert data["error"] == "simulation_error"
    assert data["input"] == "right"


def test_noiseless_side_keeps_twenty_qubit_allowance(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[11];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.5}')
    rc = cli.main(["state-metrics", left, right, "--right-noise-model", model])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["reason"] == "qubit_count_mismatch"
    assert data["left_purity"] == 1.0


def test_ten_qubits_with_noise_succeeds(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[10];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[10];\ncreg c[1];\nh q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.1}')
    rc = cli.main(["state-metrics", left, right, "--left-noise-model", model])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["reason"] == "compared"


def test_output_conflict_with_noise_model_is_output_error(write_qasm, write_model, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "right.qasm")
    model = write_model('{"depolarizing": 0.5}')
    rc = cli.main(
        ["state-metrics", left, right, "--left-noise-model", model, "--output", model]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "output_error"


def test_noisy_output_option_writes_file(write_qasm, write_model, tmp_path, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "left.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "right.qasm")
    model = write_model('{"depolarizing": 1.0}')
    target = tmp_path / "result.json"
    rc = cli.main(
        ["state-metrics", left, right, "--left-noise-model", model, "--output", str(target)]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["fidelity"] == 0.5
