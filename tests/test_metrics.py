"""Tests for the ``state-metrics`` subcommand and its supporting math."""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import cli
from quantum_circuit.metrics import single_qubit_entropies, state_fidelity
from quantum_circuit.openqasm import parse
from quantum_circuit.simulator import simulate_state_vector

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent

BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)
PRODUCT = "qreg q[2];\ncreg c[2];\nh q[0];\nx q[1];\n"


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


def _pair(tmp_path, left_body: str, right_body: str):
    left = tmp_path / "left.qasm"
    right = tmp_path / "right.qasm"
    left.write_text(HEADER + left_body, encoding="utf-8")
    right.write_text(HEADER + right_body, encoding="utf-8")
    return str(left), str(right)


# ------------------------------------------------------------------- metrics


def test_fidelity_identical_states_is_one():
    program = parse(HEADER + BELL)
    state = simulate_state_vector(program)
    assert state_fidelity(state, state) == 1.0


def test_fidelity_orthogonal_states_is_zero():
    zero = parse(HEADER + "qreg q[1];\ncreg c[1];\n")
    one = parse(HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\n")
    assert state_fidelity(simulate_state_vector(zero), simulate_state_vector(one)) == 0.0


def test_fidelity_known_value():
    # |+> vs |0>: |<+|0>|**2 = 1/2.
    plus = parse(HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\n")
    zero = parse(HEADER + "qreg q[1];\ncreg c[1];\n")
    assert state_fidelity(simulate_state_vector(plus), simulate_state_vector(zero)) == pytest.approx(0.5)


def test_product_state_has_zero_entropy():
    program = parse(HEADER + PRODUCT)
    assert single_qubit_entropies(simulate_state_vector(program), 2) == [0.0, 0.0]


def test_bell_state_has_maximal_entropy_on_both_qubits():
    program = parse(HEADER + BELL)
    assert single_qubit_entropies(simulate_state_vector(program), 2) == [1.0, 1.0]


def test_partial_entanglement_entropy():
    # cos(theta)|00> + sin(theta)|11> with cos^2(theta) = 0.75.
    theta = math.acos(math.sqrt(0.75))
    program = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\nry({2 * theta!r}) q[0];\ncx q[0],q[1];\n"
    )
    entropies = single_qubit_entropies(simulate_state_vector(program), 2)
    expected = -0.75 * math.log2(0.75) - 0.25 * math.log2(0.25)
    assert entropies[0] == pytest.approx(expected, abs=1e-12)
    assert entropies[1] == pytest.approx(expected, abs=1e-12)


def test_ghz_only_entangled_qubits_have_entropy():
    program = parse(
        HEADER + "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\nx q[2];\n"
    )
    assert single_qubit_entropies(simulate_state_vector(program), 3) == [1.0, 1.0, 0.0]


def test_metrics_stay_in_domain_and_are_plain_floats():
    program = parse(HEADER + BELL)
    state = simulate_state_vector(program)
    values = [state_fidelity(state, state)]
    values += single_qubit_entropies(state, 2)
    for value in values:
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)
        assert not (value == 0.0 and math.copysign(1.0, value) < 0)


# ------------------------------------------------------------------ CLI happy


def test_state_metrics_success_payload_and_field_order(tmp_path, capsys):
    left, right = _pair(tmp_path, BELL, BELL)
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert list(json.loads(captured.out)) == [
        "schema_version",
        "left_num_qubits",
        "right_num_qubits",
        "reason",
        "fidelity",
        "left_single_qubit_entropy",
        "right_single_qubit_entropy",
    ]
    data = json.loads(captured.out)
    assert data == {
        "schema_version": 1,
        "left_num_qubits": 2,
        "right_num_qubits": 2,
        "reason": "compared",
        "fidelity": 1.0,
        "left_single_qubit_entropy": [1.0, 1.0],
        "right_single_qubit_entropy": [1.0, 1.0],
    }


def test_state_metrics_ignores_measurement_layout(tmp_path, capsys):
    left_body = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[1];\n"
    right_body = "qreg q[2];\ncreg c[1];\nh q[0];\ncx q[0],q[1];\n"
    left, right = _pair(tmp_path, left_body, right_body)
    assert cli.main(["state-metrics", left, right]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["reason"] == "compared"
    assert data["fidelity"] == 1.0


def test_state_metrics_qubit_count_mismatch(tmp_path, capsys):
    left, right = _pair(tmp_path, BELL, "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\n")
    assert cli.main(["state-metrics", left, right]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    data = json.loads(captured.out)
    assert data["reason"] == "qubit_count_mismatch"
    assert data["fidelity"] is None
    assert data["left_num_qubits"] == 2
    assert data["right_num_qubits"] == 3
    # Entropy arrays are still computed for both sides.
    assert data["left_single_qubit_entropy"] == [1.0, 1.0]
    assert data["right_single_qubit_entropy"] == [1.0, 1.0, 0.0]


def test_state_metrics_deterministic_across_runs(tmp_path, capsys):
    left, right = _pair(
        tmp_path,
        "qreg q[3];\ncreg c[3];\nrx(0.7) q[0];\nh q[1];\ncx q[1],q[2];\n",
        "qreg q[3];\ncreg c[3];\nry(1.1) q[2];\ncx q[2],q[0];\nrz(0.3) q[1];\n",
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["state-metrics", left, right]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_state_metrics_no_nan_inf_or_negative_zero_in_output(tmp_path, capsys):
    left, right = _pair(tmp_path, BELL, PRODUCT)
    assert cli.main(["state-metrics", left, right]) == 0
    out = capsys.readouterr().out
    assert "NaN" not in out
    assert "Infinity" not in out
    assert "-0" not in out


# ------------------------------------------------------------------ stdin rules


def test_both_stdin_is_metrics_error_without_reading(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"should not be read")))
    rc = cli.main(["state-metrics", "-", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "metrics_error"


def test_right_side_via_stdin(monkeypatch, tmp_path, capsys):
    left = tmp_path / "left.qasm"
    left.write_text(HEADER + BELL, encoding="utf-8")
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + BELL).encode())))
    rc = cli.main(["state-metrics", str(left), "-"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["fidelity"] == 1.0


def test_left_side_via_stdin(monkeypatch, tmp_path, capsys):
    right = tmp_path / "right.qasm"
    right.write_text(HEADER + PRODUCT, encoding="utf-8")
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + PRODUCT).encode())))
    rc = cli.main(["state-metrics", "-", str(right)])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["fidelity"] == 1.0
    assert data["left_single_qubit_entropy"] == [0.0, 0.0]


# ------------------------------------------------------------------ errors


def test_missing_left_file_is_io_error(tmp_path, capsys):
    right = tmp_path / "right.qasm"
    right.write_text(HEADER + BELL, encoding="utf-8")
    rc = cli.main(["state-metrics", str(tmp_path / "missing.qasm"), str(right)])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "left"


def test_invalid_utf8_right_is_io_error(write_qasm, write_bytes, capsys):
    left = write_qasm(BELL, "left.qasm")
    right = write_bytes(b"\xff\xfe invalid", "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "right"


def test_parse_error_reports_side(write_qasm, capsys):
    left = write_qasm(BELL, "left.qasm")
    right = write_qasm("qreg q[1]\ncreg c[1];\n", "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["input"] == "right"
    assert payload["line"] >= 1
    assert payload["column"] >= 1


def test_validation_error_reports_side(write_qasm, capsys):
    left = write_qasm("qreg q[1];\ncreg c[1];\nh q[0];\nh q[1];\n", "left.qasm")
    right = write_qasm(BELL, "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["input"] == "left"


def test_left_failure_reported_before_right(write_qasm, capsys):
    left = write_qasm("qreg q[1]\ncreg c[1];\n", "left.qasm")
    right = write_qasm("qreg q[1]\ncreg c[1];\n", "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    payload = json.loads(capsys.readouterr().err)
    assert rc == 2
    assert payload["input"] == "left"


def test_oversized_register_is_validation_error(write_qasm, capsys):
    left = write_qasm("qreg q[21];\ncreg c[1];\n", "left.qasm")
    right = write_qasm(BELL, "right.qasm")
    rc = cli.main(["state-metrics", left, right])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["input"] == "left"


def test_argument_error_does_not_read_stdin(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"should not be read")))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["state-metrics", "-"])
    assert excinfo.value.code == 2
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------ process level


def _run_process(*args: str, stdin: str | None = None):
    return subprocess.run(
        [sys.executable, "-m", "quantum_circuit.cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        cwd=REPO_ROOT,
    )


def test_process_both_stdin_exit_code():
    result = _run_process("state-metrics", "-", "-", stdin="")
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "metrics_error"


def test_process_success_exit_code(tmp_path):
    left, right = _pair(tmp_path, BELL, BELL)
    result = _run_process("state-metrics", left, right)
    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout)["reason"] == "compared"
