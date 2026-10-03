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


@pytest.fixture
def write_model(tmp_path):
    def _write(text: str, name: str = "model.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
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
    left = write_qasm("qreg q[1];\ncreg c[1];\nu q[0];\n", "left.qasm")
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


# ----------------------------------------------------------- noisy schema v2


V2_KEYS = [
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


def test_one_noisy_side_uses_schema_v2(write_qasm, write_model, capsys):
    q = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "x.qasm")
    model = write_model('{"depolarizing": 0.1}')
    rc, out, err = run_metrics(q, q, "--left-noise-model", model, capsys=capsys)
    assert rc == 0 and err == ""
    data = json.loads(out)
    assert list(data) == V2_KEYS
    assert data["schema_version"] == 2
    assert data["reason"] == "compared"
    assert data["left_noise_model"] == {"depolarizing": 0.1}
    assert data["right_noise_model"] is None
    # depol(p) on |1>: diag(p/2, 1-p/2) -> purity .905, overlap with |1> .95.
    assert data["left_purity"] == pytest.approx(0.905)
    assert data["right_purity"] == 1.0
    assert data["fidelity"] == pytest.approx(0.95)
    binary = -(0.05 * math.log2(0.05) + 0.95 * math.log2(0.95))
    assert data["left_single_qubit_entropy"] == [pytest.approx(binary)]
    assert data["right_single_qubit_entropy"] == [0.0]


def test_noise_model_is_echoed_in_canonical_channel_order(write_qasm, write_model, capsys):
    q = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    model = write_model('{"depolarizing": 0.1, "bit_flip": 0.05}')
    rc, out, _ = run_metrics(q, q, "--right-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["left_noise_model"] is None
    assert list(data["right_noise_model"]) == ["bit_flip", "depolarizing"]
    assert data["right_noise_model"] == {"bit_flip": 0.05, "depolarizing": 0.1}


def test_both_sides_noisy_identical_states_have_unit_fidelity(write_qasm, write_model, capsys):
    bell = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n", "bell.qasm")
    model = write_model('{"depolarizing": 0.1}')
    rc, out, _ = run_metrics(
        bell, bell, "--left-noise-model", model, "--right-noise-model", model, capsys=capsys
    )
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == 1.0
    assert data["left_purity"] == data["right_purity"]
    # depol on one Bell qubit leaves each qubit maximally mixed locally.
    assert data["left_single_qubit_entropy"] == [1.0, 1.0]


def test_mixed_mixed_fidelity_matches_closed_form(write_qasm, write_model, capsys):
    # |1> under amplitude damping gamma -> diag(gamma, 1-gamma). Two such
    # commuting states: F = (sqrt(ab) + sqrt((1-a)(1-b)))**2, exactly 0.9
    # for a=0.2, b=0.5.
    x = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    left_model = write_model('{"amplitude_damping": 0.2}', "left.json")
    right_model = write_model('{"amplitude_damping": 0.5}', "right.json")
    rc, out, _ = run_metrics(
        x, x, "--left-noise-model", left_model, "--right-noise-model", right_model, capsys=capsys
    )
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == pytest.approx(0.9, abs=1e-12)
    assert data["left_purity"] == pytest.approx(0.2**2 + 0.8**2)
    assert data["right_purity"] == pytest.approx(0.5)


def test_one_sided_amplitude_damping_fidelity(write_qasm, write_model, capsys):
    # Noisy left diag(0.2, 0.8) vs pure |1> on the right: F = 0.8.
    x = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    model = write_model('{"amplitude_damping": 0.2}')
    rc, out, _ = run_metrics(x, x, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == pytest.approx(0.8, abs=1e-12)
    assert data["right_purity"] == 1.0
    assert data["left_purity"] == pytest.approx(0.68)


def test_noisy_qubit_count_mismatch_keeps_purity_and_entropies(write_qasm, write_model, capsys):
    one = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "one.qasm")
    two = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n", "two.qasm")
    left_model = write_model('{"depolarizing": 0.1}', "l.json")
    right_model = write_model('{"depolarizing": 0.1}', "r.json")
    rc, out, _ = run_metrics(
        one, two, "--left-noise-model", left_model, "--right-noise-model", right_model, capsys=capsys
    )
    assert rc == 0
    data = json.loads(out)
    assert data["schema_version"] == 2
    assert data["reason"] == "qubit_count_mismatch"
    assert data["fidelity"] is None
    assert data["left_num_qubits"] == 1 and data["right_num_qubits"] == 2
    assert 0.0 < data["left_purity"] < 1.0
    assert 0.0 < data["right_purity"] < 1.0
    assert len(data["left_single_qubit_entropy"]) == 1
    assert len(data["right_single_qubit_entropy"]) == 2


def test_noise_model_on_gateless_circuit_has_no_effect(write_qasm, write_model, capsys):
    identity = write_qasm("qreg q[2];\ncreg c[2];\n", "id.qasm")
    model = write_model('{"bit_flip": 0.3, "depolarizing": 0.2}')
    rc, out, _ = run_metrics(identity, identity, "--left-noise-model", model, capsys=capsys)
    assert rc == 0
    data = json.loads(out)
    assert data["fidelity"] == 1.0
    assert data["left_purity"] == 1.0
    assert data["left_single_qubit_entropy"] == [0.0, 0.0]


def test_noisy_v2_is_byte_identical_across_runs(write_qasm, write_model, capsys):
    bell = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n")
    other = write_qasm("qreg q[2];\ncreg c[2];\nrx(1.1) q[1];\n")
    model = write_model('{"phase_damping": 0.07, "depolarizing": 0.04}')
    rc1, out1, _ = run_metrics(bell, other, "--right-noise-model", model, capsys=capsys)
    rc2, out2, _ = run_metrics(bell, other, "--right-noise-model", model, capsys=capsys)
    assert rc1 == rc2 == 0
    assert out1 == out2
    for token in ("NaN", "Infinity", "-0"):
        assert token not in out1


def test_measurements_do_not_affect_noisy_comparison(write_qasm, write_model, capsys):
    measured = write_qasm(
        "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n", "m.qasm"
    )
    bare = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\n", "b.qasm")
    model = write_model('{"bit_flip": 0.1}')
    rc1, out1, _ = run_metrics(measured, bare, "--left-noise-model", model, capsys=capsys)
    rc2, out2, _ = run_metrics(bare, bare, "--left-noise-model", model, capsys=capsys)
    assert rc1 == rc2 == 0
    assert json.loads(out1)["fidelity"] == json.loads(out2)["fidelity"]


def test_no_flags_still_emits_schema_v1(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "l.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n", "r.qasm")
    rc, out, _ = run_metrics(left, right, capsys=capsys)
    assert rc == 0
    assert json.loads(out)["schema_version"] == 1
    assert "noise_model" not in out and "purity" not in out


# ------------------------------------------------------------- noisy errors


def test_eleven_qubit_noisy_side_is_simulation_error(write_qasm, write_model, capsys):
    big = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\n", "big.qasm")
    small = write_qasm("qreg q[1];\ncreg c[1];\n", "small.qasm")
    model = write_model('{"bit_flip": 0.1}')
    rc, out, err = run_metrics(big, small, "--left-noise-model", model, capsys=capsys)
    assert rc == 3 and out == ""
    data = json.loads(err)
    assert data["error"] == "simulation_error"
    assert data["input"] == "left"


def test_eleven_qubit_noisy_right_is_simulation_error(write_qasm, write_model, capsys):
    big = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\n", "big.qasm")
    small = write_qasm("qreg q[1];\ncreg c[1];\n", "small.qasm")
    model = write_model('{"bit_flip": 0.1}')
    rc, _out, err = run_metrics(small, big, "--right-noise-model", model, capsys=capsys)
    assert rc == 3
    assert json.loads(err)["input"] == "right"


def test_noiseless_side_keeps_twenty_qubit_limit(write_qasm, write_model, capsys):
    # 11-20 qubits is fine without a model; the noisy other side stays <=10.
    big = write_qasm("qreg q[20];\ncreg c[1];\nx q[0];\n", "big.qasm")
    small = write_qasm("qreg q[1];\ncreg c[1];\n", "small.qasm")
    model = write_model('{"bit_flip": 0.1}')
    rc, out, err = run_metrics(big, small, "--right-noise-model", model, capsys=capsys)
    assert rc == 0 and err == ""
    data = json.loads(out)
    assert data["reason"] == "qubit_count_mismatch"


def test_eleven_qubit_noisy_side_rejected_after_model_validates(write_qasm, write_model, capsys):
    big = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\n", "big.qasm")
    small = write_qasm("qreg q[1];\ncreg c[1];\n", "small.qasm")
    model = write_model('{"bit_flip": 0.1}', "good.json")
    # Inputs read and validate first (all valid here); the size cap then fires.
    rc, _out, err = run_metrics(big, small, "--left-noise-model", model, capsys=capsys)
    assert rc == 3
    assert json.loads(err)["input"] == "left"


def test_missing_noise_model_is_io_error_with_side(write_qasm, capsys):
    q = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc, out, err = run_metrics(q, q, "--left-noise-model", "/no/such/model.json", capsys=capsys)
    assert rc == 1 and out == ""
    data = json.loads(err)
    assert data["error"] == "io_error"
    assert data["input"] == "left"


def test_invalid_noise_model_content_is_error_with_side(write_qasm, write_model, capsys):
    q = write_qasm("qreg q[1];\ncreg c[1];\n")
    model = write_model('{"bit_flip": 2.0}')
    rc, _out, err = run_metrics(q, q, "--right-noise-model", model, capsys=capsys)
    assert rc == 2
    data = json.loads(err)
    assert data["error"] == "noise_model_error"
    assert data["input"] == "right"


def test_noise_model_must_be_json_object(write_qasm, write_model, capsys):
    q = write_qasm("qreg q[1];\ncreg c[1];\n")
    model = write_model('[0.1]')
    rc, _out, err = run_metrics(q, q, "--left-noise-model", model, capsys=capsys)
    assert rc == 2
    assert json.loads(err)["error"] == "noise_model_error"


def test_validation_order_left_source_first(write_qasm, write_model, capsys):
    good = write_qasm("qreg q[1];\ncreg c[1];\n", "good.qasm")
    missing_model = write_model('{"bit_flip": 0.1}', "m.json")  # path exists
    # Left source unreadable beats every other (valid) input.
    rc, _out, err = run_metrics(
        "/no/such/left.qasm", good, "--right-noise-model", missing_model, capsys=capsys
    )
    assert rc == 1
    data = json.loads(err)
    assert data["error"] == "io_error" and data["input"] == "left"


def test_validation_order_left_model_before_right_source(write_qasm, write_model, capsys):
    good = write_qasm("qreg q[1];\ncreg c[1];\n", "good.qasm")
    rc, _out, err = run_metrics(
        good, "/no/such/right.qasm", "--left-noise-model", "/no/such/model.json", capsys=capsys
    )
    assert rc == 1
    data = json.loads(err)
    assert data["error"] == "io_error" and data["input"] == "left"


def test_validation_order_right_source_before_right_model(write_qasm, capsys):
    good = write_qasm("qreg q[1];\ncreg c[1];\n", "good.qasm")
    rc, _out, err = run_metrics(
        good,
        "/no/such/right.qasm",
        "--right-noise-model",
        "/no/such/model.json",
        capsys=capsys,
    )
    assert rc == 1
    data = json.loads(err)
    assert data["error"] == "io_error" and data["input"] == "right"
    assert "source" in data["message"]


@pytest.mark.parametrize(
    "extra",
    [
        ["-", "-"],
        ["-", "r.qasm", "--left-noise-model", "-"],
        ["l.qasm", "-", "--right-noise-model", "-"],
        ["l.qasm", "r.qasm", "--left-noise-model", "-", "--right-noise-model", "-"],
        ["-", "r.qasm", "--right-noise-model", "-"],
    ],
)
def test_multiple_stdin_inputs_is_metrics_error(write_qasm, monkeypatch, capsys, extra):
    import io

    class _Stdin:
        buffer = io.BytesIO(b"standard input must not be read")

    monkeypatch.setattr("sys.stdin", _Stdin())
    left = write_qasm("qreg q[1];\ncreg c[1];\n", "l.qasm")
    right = write_qasm("qreg q[1];\ncreg c[1];\n", "r.qasm")
    argv = [a.replace("l.qasm", left).replace("r.qasm", right) for a in extra]
    rc = cli.main(["state-metrics", *argv])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "metrics_error"


def test_single_stdin_noise_model_works(write_qasm, write_model, monkeypatch, capsys):
    import io

    q = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    class _Stdin:
        buffer = io.BytesIO(b'{"depolarizing": 0.1}')

    monkeypatch.setattr("sys.stdin", _Stdin())
    rc = cli.main(["state-metrics", q, q, "--left-noise-model", "-"])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert data["left_noise_model"] == {"depolarizing": 0.1}


def test_output_conflict_with_noise_model_file(write_qasm, write_model, capsys):
    q = write_qasm("qreg q[1];\ncreg c[1];\n")
    model = write_model('{"bit_flip": 0.1}')
    rc, out, err = run_metrics(
        q, q, "--left-noise-model", model, "--output", model, capsys=capsys
    )
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
