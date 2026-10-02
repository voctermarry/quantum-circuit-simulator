"""Tests for the unified ``--output PATH`` result-export option."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
BELL = (
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
def write_bytes(tmp_path):
    def _write(data: bytes, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return _write


@pytest.fixture
def bell_file(write_qasm):
    return write_qasm(BELL)


def _read_stdout_line(capsys) -> str:
    out = capsys.readouterr()
    assert out.err == ""
    assert out.out.endswith("\n") and "\n" not in out.out[:-1]
    return out.out


def _assert_output_error(capsys, code: int, actual: int) -> None:
    assert actual == code
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "output_error"
    assert isinstance(payload["message"], str) and payload["message"]


# --------------------------------------------------------------- happy paths


def test_simulate_output_matches_stdout(bell_file, tmp_path, capsys):
    assert cli.main(["simulate", bell_file, "--shots", "50", "--seed", "7"]) == 0
    expected = _read_stdout_line(capsys)

    target = tmp_path / "result.json"
    assert (
        cli.main(
            ["simulate", bell_file, "--shots", "50", "--seed", "7", "--output", str(target)]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert target.read_bytes() == expected.encode("utf-8")


def test_probabilities_output(bell_file, tmp_path, capsys):
    assert cli.main(["probabilities", bell_file]) == 0
    expected = _read_stdout_line(capsys)

    target = tmp_path / "probs.json"
    assert cli.main(["probabilities", bell_file, "--output", str(target)]) == 0
    assert capsys.readouterr().out == ""
    assert target.read_bytes() == expected.encode("utf-8")


def test_optimize_output(bell_file, tmp_path, capsys):
    assert cli.main(["optimize", bell_file]) == 0
    expected = _read_stdout_line(capsys)

    target = tmp_path / "opt.json"
    assert cli.main(["optimize", bell_file, "--output", str(target)]) == 0
    assert capsys.readouterr().out == ""
    assert target.read_bytes() == expected.encode("utf-8")


def test_estimate_output(bell_file, tmp_path, capsys):
    assert cli.main(["estimate", bell_file, "--mode", "unitary"]) == 0
    expected = _read_stdout_line(capsys)

    target = tmp_path / "est.json"
    assert (
        cli.main(["estimate", bell_file, "--mode", "unitary", "--output", str(target)]) == 0
    )
    assert capsys.readouterr().out == ""
    assert target.read_bytes() == expected.encode("utf-8")


def test_equivalent_output(write_qasm, tmp_path, capsys):
    left = write_qasm(BELL, "left.qasm")
    right = write_qasm(BELL, "right.qasm")
    assert cli.main(["equivalent", left, right]) == 0
    expected = _read_stdout_line(capsys)

    target = tmp_path / "eq.json"
    assert cli.main(["equivalent", left, right, "--output", str(target)]) == 0
    assert capsys.readouterr().out == ""
    assert target.read_bytes() == expected.encode("utf-8")


def test_state_metrics_output(write_qasm, tmp_path, capsys):
    left = write_qasm(BELL, "left.qasm")
    right = write_qasm("qreg q[2];\ncreg c[2];\nx q[0];\n", "right.qasm")
    assert cli.main(["state-metrics", left, right]) == 0
    expected = _read_stdout_line(capsys)

    target = tmp_path / "metrics.json"
    assert cli.main(["state-metrics", left, right, "--output", str(target)]) == 0
    assert capsys.readouterr().out == ""
    assert target.read_bytes() == expected.encode("utf-8")


def _write_manifest(tmp_path, jobs, name="manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "jobs": jobs}), encoding="utf-8")
    return str(path)


def test_batch_simulate_output_with_task_failure(write_qasm, tmp_path, capsys):
    good = write_qasm(BELL, "good.qasm")
    manifest = _write_manifest(
        tmp_path,
        [
            {"id": "a", "source": "good.qasm", "shots": 16, "seed": 3},
            {"id": "b", "source": "missing.qasm"},
        ],
    )
    assert cli.main(["batch-simulate", manifest]) == 3
    expected = _read_stdout_line(capsys)

    target = tmp_path / "batch.json"
    assert cli.main(["batch-simulate", manifest, "--output", str(target)]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert target.read_bytes() == expected.encode("utf-8")
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["failed"] == 1


def test_reconcile_output_with_mismatch(write_qasm, tmp_path, capsys):
    write_qasm(BELL, "good.qasm")
    manifest = _write_manifest(
        tmp_path, [{"id": "a", "source": "good.qasm", "shots": 16, "seed": 3}]
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_count": 1,
                "succeeded": 1,
                "failed": 0,
                "results": [
                    {
                        "id": "a",
                        "status": "succeeded",
                        "output": {
                            "schema_version": 1,
                            "shots": 16,
                            "seed": 3,
                            "num_qubits": 2,
                            "num_clbits": 2,
                            "counts": {"00": 16},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["reconcile", manifest, str(baseline)]) == 3
    expected = _read_stdout_line(capsys)

    target = tmp_path / "report.json"
    assert cli.main(["reconcile", manifest, str(baseline), "--output", str(target)]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert target.read_bytes() == expected.encode("utf-8")
    assert json.loads(target.read_text(encoding="utf-8"))["consistent"] is False


def test_output_overwrites_existing_regular_file(bell_file, tmp_path, capsys):
    target = tmp_path / "result.json"
    target.write_text("stale contents\n", encoding="utf-8")
    assert cli.main(["probabilities", bell_file, "--output", str(target)]) == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_output_relative_path_uses_cwd(bell_file, tmp_path, capsys, monkeypatch):
    assert cli.main(["probabilities", bell_file]) == 0
    expected = _read_stdout_line(capsys)

    monkeypatch.chdir(tmp_path)
    assert cli.main(["probabilities", bell_file, "--output", "rel.json"]) == 0
    assert capsys.readouterr().out == ""
    assert (tmp_path / "rel.json").read_bytes() == expected.encode("utf-8")


def test_same_input_exports_identical_bytes(bell_file, tmp_path, capsys):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert cli.main(["simulate", bell_file, "--shots", "32", "--seed", "5", "--output", str(first)]) == 0
    assert cli.main(["simulate", bell_file, "--shots", "32", "--seed", "5", "--output", str(second)]) == 0
    capsys.readouterr()
    assert first.read_bytes() == second.read_bytes()


# -------------------------------------------------------------- export errors


def test_output_parent_directory_must_exist(bell_file, tmp_path, capsys):
    target = tmp_path / "no-such-dir" / "result.json"
    code = cli.main(["simulate", bell_file, "--output", str(target)])
    _assert_output_error(capsys, 1, code)
    assert not target.exists()


def test_output_must_not_be_a_directory(bell_file, tmp_path, capsys):
    code = cli.main(["simulate", bell_file, "--output", str(tmp_path)])
    _assert_output_error(capsys, 1, code)


def test_output_must_not_be_dash(bell_file, capsys):
    code = cli.main(["simulate", bell_file, "--output", "-"])
    _assert_output_error(capsys, 1, code)


def test_failed_export_preserves_existing_target(bell_file, tmp_path, capsys):
    target = tmp_path / "result.json"
    target.write_text("keep me\n", encoding="utf-8")
    # Make the directory read-only so the atomic replace fails.
    os.chmod(tmp_path, 0o555)
    try:
        code = cli.main(["simulate", bell_file, "--output", str(target)])
    finally:
        os.chmod(tmp_path, 0o755)
    _assert_output_error(capsys, 1, code)
    assert target.read_text(encoding="utf-8") == "keep me\n"
    # No partial export files are left behind.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".output-")]


# ------------------------------------------------------------- path conflicts


def test_output_conflicting_with_source(bell_file, capsys):
    code = cli.main(["simulate", bell_file, "--output", bell_file])
    _assert_output_error(capsys, 2, code)


def test_output_conflicting_with_source_normalized(bell_file, capsys):
    spelled = os.path.join(os.path.dirname(bell_file), ".", os.path.basename(bell_file))
    code = cli.main(["probabilities", bell_file, "--output", spelled])
    _assert_output_error(capsys, 2, code)


def test_output_conflicting_with_noise_model(bell_file, tmp_path, capsys):
    noise = tmp_path / "noise.json"
    noise.write_text('{"bit_flip": 0.1}', encoding="utf-8")
    code = cli.main(["simulate", bell_file, "--noise-model", str(noise), "--output", str(noise)])
    _assert_output_error(capsys, 2, code)
    assert noise.read_text(encoding="utf-8") == '{"bit_flip": 0.1}'


def test_output_conflicting_with_manifest(write_qasm, tmp_path, capsys):
    write_qasm(BELL, "good.qasm")
    manifest = _write_manifest(tmp_path, [{"id": "a", "source": "good.qasm"}])
    code = cli.main(["batch-simulate", manifest, "--output", manifest])
    _assert_output_error(capsys, 2, code)


def test_output_conflicting_with_manifest_referenced_circuit(write_qasm, tmp_path, capsys):
    circuit = write_qasm(BELL, "good.qasm")
    manifest = _write_manifest(tmp_path, [{"id": "a", "source": "good.qasm"}])
    code = cli.main(["batch-simulate", manifest, "--output", circuit])
    _assert_output_error(capsys, 2, code)
    # The conflict is detected before any task runs; the circuit is untouched.
    assert Path(circuit).read_text(encoding="utf-8") == HEADER + BELL


def test_output_conflicting_with_reconcile_baseline(write_qasm, tmp_path, capsys):
    write_qasm(BELL, "good.qasm")
    manifest = _write_manifest(tmp_path, [{"id": "a", "source": "good.qasm"}])
    baseline = tmp_path / "baseline.json"
    baseline.write_text("not json", encoding="utf-8")
    # Baseline is invalid: the original reconcile_input_error wins, no conflict.
    assert cli.main(["reconcile", manifest, str(baseline), "--output", str(baseline)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"] == "reconcile_input_error"


def test_output_conflicting_with_equivalent_side(write_qasm, capsys):
    left = write_qasm(BELL, "left.qasm")
    right = write_qasm(BELL, "right.qasm")
    code = cli.main(["equivalent", left, right, "--output", right])
    _assert_output_error(capsys, 2, code)


def test_output_conflicting_with_metrics_side(write_qasm, capsys):
    left = write_qasm(BELL, "left.qasm")
    code = cli.main(["state-metrics", left, left, "--output", left])
    _assert_output_error(capsys, 2, code)


def test_output_conflicting_with_optimize_source(bell_file, capsys):
    code = cli.main(["optimize", bell_file, "--output", bell_file])
    _assert_output_error(capsys, 2, code)


def test_output_conflicting_with_estimate_source(bell_file, capsys):
    code = cli.main(["estimate", bell_file, "--output", bell_file])
    _assert_output_error(capsys, 2, code)


# ------------------------------------------------- original failures untouched


def test_parse_failure_does_not_create_output(write_bytes, tmp_path, capsys):
    source = write_bytes(b"OPENQASM 2.0;\nbogus\n", "bad.qasm")
    target = tmp_path / "result.json"
    assert cli.main(["simulate", source, "--output", str(target)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"] == "parse_error"
    assert not target.exists()


def test_io_failure_does_not_create_output(tmp_path, capsys):
    target = tmp_path / "result.json"
    assert cli.main(["simulate", str(tmp_path / "missing.qasm"), "--output", str(target)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"] == "io_error"
    assert not target.exists()


def test_simulation_failure_does_not_touch_output(write_qasm, tmp_path, capsys):
    noise = tmp_path / "noise.json"
    noise.write_text('{"depolarizing": 0.5}', encoding="utf-8")
    source = write_qasm("qreg q[11];\ncreg c[11];\n", "big.qasm")
    target = tmp_path / "result.json"
    target.write_text("keep me\n", encoding="utf-8")
    assert (
        cli.main(
            ["simulate", source, "--noise-model", str(noise), "--output", str(target)]
        )
        == 3
    )
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"] == "simulation_error"
    assert target.read_text(encoding="utf-8") == "keep me\n"


def test_invalid_manifest_does_not_create_output(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 1, "jobs": []}', encoding="utf-8")
    target = tmp_path / "result.json"
    assert cli.main(["batch-simulate", str(manifest), "--output", str(target)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"] == "batch_input_error"
    assert not target.exists()
