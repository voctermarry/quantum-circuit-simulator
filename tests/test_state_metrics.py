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
