"""End-to-end tests for the optimize subcommand."""

from __future__ import annotations

import json
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


def _optimize_payload(body: str, write_qasm, capsys):
    path = write_qasm(body)
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    raw = captured.out
    assert raw.endswith("\n") and raw.count("\n") == 1
    return json.loads(raw)


# --------------------------------------------------------------- success output


def test_success_payload_shape_and_field_order(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert list(data) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "original_gate_count",
        "optimized_gate_count",
        "changed",
        "qasm",
    ]
    assert data["schema_version"] == 1
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    assert data["original_gate_count"] == 2
    assert data["optimized_gate_count"] == 2
    assert data["changed"] is False
    assert data["qasm"].endswith("\n")


def test_canonical_qasm_is_reparseable_and_unchanged(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert data["qasm"] == (
        HEADER
        + "qreg q[2];\ncreg c[2];\n"
        + "h q[0];\ncx q[0],q[1];\n"
        + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )


def test_registers_are_renamed_to_q_and_c(write_qasm, capsys):
    data = _optimize_payload(
        "qreg r[1];\ncreg s[1];\nx r[0];\nmeasure r[0] -> s[0];\n",
        write_qasm,
        capsys,
    )
    assert "qreg q[1];\ncreg c[1];\n" in data["qasm"]
    assert "x q[0];\n" in data["qasm"]
    assert "measure q[0] -> c[0];\n" in data["qasm"]


def test_no_quantum_gates_is_ok(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert data["original_gate_count"] == 0
    assert data["optimized_gate_count"] == 0
    assert data["changed"] is False
    assert data["qasm"] == HEADER + "qreg q[2];\ncreg c[2];\nmeasure q[0] -> c[1];\n"


def test_measurements_sorted_by_clbit_then_qubit(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[3];\ncreg c[3];\n"
        "measure q[2] -> c[2];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    lines = data["qasm"].splitlines()
    assert lines[-3:] == [
        "measure q[0] -> c[0];",
        "measure q[1] -> c[1];",
        "measure q[2] -> c[2];",
    ]


def test_measurement_mapping_preserved_when_permuted(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nmeasure q[1] -> c[0];\nmeasure q[0] -> c[1];\n",
        write_qasm,
        capsys,
    )
    lines = data["qasm"].splitlines()
    assert lines[-2:] == ["measure q[1] -> c[0];", "measure q[0] -> c[1];"]


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\nrx(0.3) q[2];\nh q[0];\nrx(0.4) q[2];\n"
        "cx q[0],q[1];\nmeasure q[0] -> c[0];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["optimize", path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_stdin_source(monkeypatch, capsys):
    import io

    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    rc = cli.main(["optimize", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["optimized_gate_count"] == 0
    assert data["changed"] is True


# -------------------------------------------------------------------- cancellation


def test_x_and_h_pairs_cancel(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nx q[0];\nx q[0];\nh q[0];\nh q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 0
    assert data["changed"] is True
    assert "x q[0];" not in data["qasm"]


def test_cancellation_ignores_disjoint_gates_in_between(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nx q[0];\nh q[1];\nrx(0.5) q[1];\nx q[0];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 2
    assert "x q[0];" not in data["qasm"]


def test_gates_sharing_a_qubit_block_cancellation(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[1];\nx q[0];\nh q[0];\nx q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 3


def test_cx_pairs_with_same_control_and_target_cancel(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\ncx q[0],q[1];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 0
    assert data["changed"] is True


def test_cx_blocked_by_gate_on_either_qubit_does_not_cancel(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[2];\ncx q[0],q[1];\nh q[1];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 3


def test_cx_pair_cancels_across_disjoint_gate(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[3];\ncreg c[1];\ncx q[0],q[1];\nh q[2];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 1
    assert "cx" not in data["qasm"]
    assert "h q[2];" in data["qasm"]


def test_cx_with_swapped_control_and_target_does_not_cancel(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[1];\ncx q[0],q[1];\ncx q[1],q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 2


# -------------------------------------------------------------------- rotations


def test_same_axis_rotations_merge(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrx(0.3) q[0];\nrx(0.4) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 1
    assert f"rx({0.3 + 0.4!r}) q[0];" in data["qasm"]


def test_rotations_merge_across_disjoint_gates(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[1];\nrz(0.25) q[0];\nh q[1];\nrz(0.25) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 2
    assert "rz(0.5) q[0];" in data["qasm"]


def test_different_axes_do_not_merge(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrx(0.3) q[0];\nry(0.3) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 2


def test_angles_normalized_to_minus_pi_pi(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrz(3*pi) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert f"rz({3.141592653589793!r}) q[0];" in data["qasm"]
    assert data["changed"] is True


def test_negative_pi_normalizes_to_pi(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrz(-pi) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert "rz(3.141592653589793) q[0];" in data["qasm"]


def test_tiny_rotation_is_dropped(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrx(1e-13) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 0
    assert data["changed"] is True


def test_rotation_at_epsilon_boundary_is_kept(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrx(1e-11) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 1
    assert "rx(1e-11) q[0];" in data["qasm"]


def test_rotations_merging_to_zero_are_dropped(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nry(0.5) q[0];\nry(-0.5) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 0


def test_near_pi_rx_and_ry_are_not_replaced_by_x(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nrx(pi) q[0];\nry(pi) q[0];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert "rx(3.141592653589793) q[0];" in data["qasm"]
    assert "ry(3.141592653589793) q[0];" in data["qasm"]
    assert "x q[0];" not in data["qasm"]


def test_merge_chain_simplifies_to_fixpoint(write_qasm, capsys):
    # The rx pair merges to zero, exposing the x pair, which then cancels.
    data = _optimize_payload(
        "qreg q[1];\ncreg c[1];\nx q[0];\nrx(0.3) q[0];\nrx(-0.3) q[0];\nx q[0];\n"
        "measure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    assert data["optimized_gate_count"] == 0


# -------------------------------------------------------------------- reordering


def test_independent_gates_get_canonical_order(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[3];\ncreg c[1];\nh q[2];\nx q[0];\nh q[1];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    gate_lines = [line for line in data["qasm"].splitlines() if line.endswith(";")][4:-1]
    assert gate_lines == ["x q[0];", "h q[1];", "h q[2];"]
    assert data["changed"] is True


def test_independent_gate_orderings_produce_identical_qasm(write_qasm, capsys):
    first = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nh q[1];\nrx(0.5) q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    second = _optimize_payload(
        "qreg q[2];\ncreg c[2];\nrx(0.5) q[0];\nh q[1];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        write_qasm,
        capsys,
    )
    assert first["qasm"] == second["qasm"]


def test_dependent_gates_keep_dependency_order(write_qasm, capsys):
    data = _optimize_payload(
        "qreg q[2];\ncreg c[1];\nx q[1];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\n",
        write_qasm,
        capsys,
    )
    gate_lines = [line for line in data["qasm"].splitlines() if line.endswith(";")][4:-1]
    # h q[0] sorts before x q[1]; cx depends on both and stays last.
    assert gate_lines == ["h q[0];", "x q[1];", "cx q[0],q[1];"]


def test_optimized_output_is_a_fixpoint(write_qasm, capsys, tmp_path):
    data = _optimize_payload(
        "qreg q[3];\ncreg c[3];\nh q[2];\nx q[2];\nrx(0.2) q[0];\nrx(0.3) q[0];\n"
        "cx q[1],q[2];\ncx q[1],q[2];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n",
        write_qasm,
        capsys,
    )
    again = tmp_path / "again.qasm"
    again.write_text(data["qasm"], encoding="utf-8")
    assert cli.main(["optimize", str(again)]) == 0
    rerun = json.loads(capsys.readouterr().out)
    assert rerun["qasm"] == data["qasm"]
    assert rerun["changed"] is False


# -------------------------------------------------------------------- equivalence


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


@pytest.mark.parametrize(
    "body",
    [
        "qreg q[1];\ncreg c[1];\nx q[0];\nh q[0];\nx q[0];\nh q[0];\nmeasure q[0] -> c[0];\n",
        "qreg q[2];\ncreg c[2];\nrx(0.3) q[0];\nrx(0.7) q[0];\nry(pi) q[1];\n"
        "cx q[0],q[1];\ncx q[0],q[1];\nrz(3*pi) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        "qreg q[3];\ncreg c[3];\nh q[2];\nrx(1e-13) q[0];\ncx q[2],q[1];\n"
        "measure q[2] -> c[0];\nmeasure q[1] -> c[2];\n",
    ],
)
def test_optimized_circuit_is_equivalent_to_original(body, tmp_path):
    original = tmp_path / "original.qasm"
    original.write_text(HEADER + body, encoding="utf-8")
    result = _run_process("optimize", str(original))
    assert result.returncode == 0
    optimized = tmp_path / "optimized.qasm"
    optimized.write_text(json.loads(result.stdout)["qasm"], encoding="utf-8")

    comparison = _run_process("equivalent", str(original), str(optimized))
    assert comparison.returncode == 0
    verdict = json.loads(comparison.stdout)
    assert verdict["equivalent"] is True, verdict
    assert verdict["reason"] == "equivalent"


# -------------------------------------------------------------------- errors


def test_missing_file_is_io_error(capsys):
    rc = cli.main(["optimize", "/nonexistent/path/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"
    assert captured.err.count("\n") == 1


def test_invalid_utf8_is_io_error(write_bytes, capsys):
    path = write_bytes(HEADER.encode() + b"\xff\xfe bad")
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_parse_error_exit_code(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_validation_error_exit_code(write_qasm, capsys):
    path = write_qasm("qreg q[1];\nqreg q2[1];\n")
    rc = cli.main(["optimize", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "validation_error"


def test_argument_error_exits_2_without_reading_input(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["optimize", "/nonexistent/never-read.qasm", "--bogus"])
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "io_error" not in captured.err


def test_process_optimize_from_stdin():
    source = HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nh q[0];\nmeasure q[0] -> c[0];\n"
    result = _run_process("optimize", "-", stdin=source)
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["optimized_gate_count"] == 0
    assert data["changed"] is True


def test_process_io_error_exit_code():
    result = _run_process("optimize", "/nonexistent/circuit.qasm")
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "io_error"
