"""Layered command tests: public entry matrix and pure-logic independence.

These tests guard the command-layer refactor from both sides:

* ``test_public_entry_*`` drive every subcommand through the public
  ``quantum_circuit.cli.main`` entry point in the three representative
  situations: a successful JSON line, an input failure (stderr line and
  its exit code), and -- where the command can produce one -- a complete
  report returned with a non-zero exit code.
* ``test_pure_*`` run in an isolated interpreter that imports the shared
  logic but never the command module, proving the pure computation and
  document validation are callable without a terminal, file system input
  or standard streams.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def write(tmp_path):
    def _write(name: str, content: str | bytes) -> str:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        with open(path, "wb") as handle:
            handle.write(data)
        return str(path)

    return _write


BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)
ONE = "qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"
PARSE_BAD = "qreg q[1];\nh q[0]\n"
NOISE_OK = json.dumps({"depolarizing": 0.1, "amplitude_damping": 0.05})
OBS_OK = json.dumps(
    {
        "schema_version": 1,
        "observables": [
            {"id": "Z", "operators": [{"qubit": 0, "pauli": "Z"}]},
            {"id": "I", "operators": []},
        ],
    }
)


def _run(argv, capsys):
    rc = cli.main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# ----------------------------------------------------- success / input failure


def test_simulate_success(write, capsys):
    path = write("c.qasm", HEADER + ONE)
    rc, out, err = _run(["simulate", path, "--shots", "16", "--seed", "3"], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert list(data) == ["schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts"]
    assert (data["shots"], data["seed"], data["counts"]) == (16, 3, {"0": 16})
    assert out.endswith("\n") and out.count("\n") == 1


def test_simulate_input_failure(write, capsys):
    rc, out, err = _run(["simulate", write("c.qasm", HEADER + PARSE_BAD)], capsys)
    assert (rc, out) == (2, "")
    payload = json.loads(err)
    assert payload["error"] == "parse_error"
    assert {"error", "message", "line", "column"} == set(payload)


def test_simulate_simulation_error_exit_3(write, capsys):
    big = HEADER + "qreg q[11];\ncreg c[11];\nh q[0];\n" + "".join(
        f"measure q[{i}] -> c[{i}];\n" for i in range(11)
    )
    rc, out, err = _run(
        ["simulate", write("big.qasm", big), "--noise-model", write("n.json", NOISE_OK)],
        capsys,
    )
    assert (rc, out) == (3, "")
    assert json.loads(err)["error"] == "simulation_error"


def test_probabilities_success(write, capsys):
    rc, out, err = _run(["probabilities", write("c.qasm", HEADER + ONE)], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert data["probabilities"] == {"0": 1.0}


def test_probabilities_input_failure(write, capsys):
    rc, out, err = _run(["probabilities", write("c.qasm", HEADER + PARSE_BAD)], capsys)
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "parse_error"


def test_probabilities_simulation_error_exit_3(write, capsys):
    big = HEADER + "qreg q[11];\ncreg c[11];\nh q[0];\n" + "".join(
        f"measure q[{i}] -> c[{i}];\n" for i in range(11)
    )
    rc, out, err = _run(
        ["probabilities", write("big.qasm", big), "--noise-model", write("n.json", NOISE_OK)],
        capsys,
    )
    assert (rc, out) == (3, "")
    assert json.loads(err)["error"] == "simulation_error"


def test_expectation_success(write, capsys):
    rc, out, err = _run(
        ["expectation", write("c.qasm", HEADER + ONE), write("o.json", OBS_OK)], capsys
    )
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert [entry["id"] for entry in data["results"]] == ["Z", "I"]


def test_expectation_input_failure(write, capsys):
    bad_obs = json.dumps(
        {"schema_version": 1, "observables": [
            {"id": "o", "operators": [{"qubit": 9, "pauli": "Z"}]}]}
    )
    rc, out, err = _run(
        ["expectation", write("c.qasm", HEADER + ONE), write("o.json", bad_obs)], capsys
    )
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "observable_error"


def test_expectation_simulation_error_exit_3(write, capsys):
    big = HEADER + "qreg q[11];\ncreg c[11];\nh q[0];\n"
    rc, out, err = _run(
        [
            "expectation",
            write("big.qasm", big),
            write("o.json", OBS_OK),
            "--noise-model",
            write("n.json", NOISE_OK),
        ],
        capsys,
    )
    assert (rc, out) == (3, "")
    assert json.loads(err)["error"] == "simulation_error"


def test_verify_samples_success(write, capsys):
    samples = json.dumps({"schema_version": 1, "counts": {"0": 100}})
    rc, out, err = _run(
        ["verify-samples", write("c.qasm", HEADER + ONE), write("s.json", samples)],
        capsys,
    )
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert data["accepted"] is True


def test_verify_samples_input_failure(write, capsys):
    bad = json.dumps({"schema_version": 1, "counts": {"00": 1}})
    rc, out, err = _run(
        ["verify-samples", write("c.qasm", HEADER + ONE), write("s.json", bad)],
        capsys,
    )
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "sample_input_error"


def test_verify_samples_reports_full_report_at_exit_3(write, capsys):
    rejected = json.dumps({"schema_version": 1, "counts": {"1": 100}})
    rc, out, err = _run(
        ["verify-samples", write("c.qasm", HEADER + ONE), write("s.json", rejected)],
        capsys,
    )
    # A producible non-zero result is the complete report on stdout, code 3.
    assert (rc, err) == (3, "")
    data = json.loads(out)
    assert data["accepted"] is False
    assert 0.0 < data["total_variation_distance"] <= 1.0
    assert set(data) == {
        "schema_version", "num_qubits", "num_clbits", "shots", "noise_model",
        "tolerance", "total_variation_distance", "accepted",
        "expected_probabilities", "observed_probabilities",
    }


def _write_batch(write, name, jobs):
    return write(name, json.dumps({"schema_version": 1, "jobs": jobs}))


def test_batch_simulate_success(write, capsys):
    qasm = write("c.qasm", HEADER + ONE)
    manifest = _write_batch(write, "m.json", [{"id": "a", "source": qasm}])
    rc, out, err = _run(["batch-simulate", manifest], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert (data["succeeded"], data["failed"]) == (1, 0)
    assert data["results"][0]["status"] == "succeeded"


def test_batch_simulate_input_failure(write, capsys):
    manifest = _write_batch(write, "m.json", [{"id": "a"}])
    rc, out, err = _run(["batch-simulate", manifest], capsys)
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "batch_input_error"


def test_batch_simulate_embedded_failure_at_exit_3(write, capsys):
    qasm = write("c.qasm", HEADER + ONE)
    manifest = _write_batch(
        write,
        "m.json",
        [
            {"id": "ok", "source": qasm, "shots": 8, "seed": 1},
            {"id": "gone", "source": write("missing-ref", "x") + ".nope"},
        ],
    )
    rc, out, err = _run(["batch-simulate", manifest], capsys)
    assert (rc, err) == (3, "")
    data = json.loads(out)
    assert (data["succeeded"], data["failed"]) == (1, 1)
    # Task isolation: the good task keeps its full output.
    assert data["results"][0] == {
        "id": "ok",
        "status": "succeeded",
        "output": {
            "schema_version": 1, "shots": 8, "seed": 1,
            "num_qubits": 1, "num_clbits": 1, "counts": {"0": 8},
        },
    }
    assert data["results"][1]["status"] == "failed"
    assert data["results"][1]["error"]["error"] == "io_error"


def test_reconcile_success(write, capsys):
    qasm = write("c.qasm", HEADER + ONE)
    manifest = _write_batch(write, "m.json", [{"id": "a", "source": qasm, "seed": 5}])
    rc, out, err = _run(["batch-simulate", manifest], capsys)
    assert rc == 0
    baseline = write("b.json", out)
    rc, out, err = _run(["reconcile", manifest, baseline], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert (data["matched"], data["mismatched"], data["consistent"]) == (1, 0, True)


def test_reconcile_input_failure(write, capsys):
    qasm = write("c.qasm", HEADER + ONE)
    manifest = _write_batch(write, "m.json", [{"id": "a", "source": qasm}])
    rc, out, err = _run(["reconcile", manifest, write("b.json", "{not json")], capsys)
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "reconcile_input_error"


def test_reconcile_mismatch_report_at_exit_3(write, capsys):
    qasm = write("c.qasm", HEADER + ONE)
    manifest = _write_batch(write, "m.json", [{"id": "a", "source": qasm, "seed": 5}])
    rc, out, err = _run(["batch-simulate", manifest], capsys)
    assert rc == 0
    baseline_data = json.loads(out)
    # Corrupt the baseline's successful output to force an output_mismatch.
    baseline_data["results"][0]["output"]["counts"] = {"0": 512, "1": 512}
    baseline_data["results"][0]["output"]["shots"] = 1024
    baseline = write("b.json", json.dumps(baseline_data))
    rc, out, err = _run(["reconcile", manifest, baseline], capsys)
    assert (rc, err) == (3, "")
    data = json.loads(out)
    assert (data["matched"], data["mismatched"], data["consistent"]) == (0, 1, False)
    entry = data["results"][0]
    assert entry["reason"] == "output_mismatch"
    assert "expected" in entry and "actual" in entry


def test_equivalent_success(write, capsys):
    left = write("l.qasm", HEADER + ONE)
    right = write("r.qasm", HEADER + ONE)
    rc, out, err = _run(["equivalent", left, right], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert data["equivalent"] is True and data["reason"] == "equivalent"


def test_equivalent_input_failure(write, capsys):
    left = write("l.qasm", HEADER + ONE)
    rc, out, err = _run(["equivalent", left, write("r.qasm", HEADER + PARSE_BAD)], capsys)
    assert (rc, out) == (2, "")
    payload = json.loads(err)
    assert payload["error"] == "parse_error" and payload["input"] == "right"


def test_equivalent_size_limit_exit_3(write, capsys):
    big = HEADER + "qreg q[9];\ncreg c[9];\nh q[0];\n"
    one = write("r.qasm", HEADER + ONE)
    rc, out, err = _run(["equivalent", write("big.qasm", big), one], capsys)
    assert (rc, out) == (3, "")
    payload = json.loads(err)
    assert payload["error"] == "simulation_error" and payload["input"] == "left"


def test_optimize_success(write, capsys):
    source = HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nx q[0];\nmeasure q[0] -> c[0];\n"
    rc, out, err = _run(["optimize", write("c.qasm", source)], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert (data["original_gate_count"], data["optimized_gate_count"], data["changed"]) == (2, 0, True)


def test_optimize_input_failure(write, capsys):
    rc, out, err = _run(["optimize", write("c.qasm", HEADER + PARSE_BAD)], capsys)
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "parse_error"


def test_estimate_success(write, capsys):
    rc, out, err = _run(["estimate", write("c.qasm", HEADER + ONE)], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert data["mode"] == "state-vector" and data["entry_count"] == 2


def test_estimate_input_failure(write, capsys):
    rc, out, err = _run(["estimate", write("c.qasm", HEADER + PARSE_BAD)], capsys)
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "parse_error"


def test_state_metrics_success(write, capsys):
    left = write("l.qasm", HEADER + ONE)
    rc, out, err = _run(["state-metrics", left, write("r.qasm", HEADER + ONE)], capsys)
    assert (rc, err) == (0, "")
    data = json.loads(out)
    assert data["schema_version"] == 1 and data["fidelity"] == 1.0


def test_state_metrics_input_failure(capsys):
    # Standard-input conflict is rejected before any input is read.
    rc, out, err = _run(["state-metrics", "-", "-"], capsys)
    assert (rc, out) == (2, "")
    assert json.loads(err)["error"] == "metrics_error"


def test_state_metrics_simulation_error_exit_3(write, capsys):
    big = HEADER + "qreg q[11];\ncreg c[11];\nh q[0];\n"
    one = write("r.qasm", HEADER + ONE)
    rc, out, err = _run(
        [
            "state-metrics",
            write("big.qasm", big),
            one,
            "--left-noise-model",
            write("n.json", NOISE_OK),
        ],
        capsys,
    )
    assert (rc, out) == (3, "")
    payload = json.loads(err)
    assert payload["error"] == "simulation_error" and payload["input"] == "left"


# ------------------------------------------------------- pure-layer independence


_ISOLATED_PURE = """
import sys

from quantum_circuit import commands, documents, errors
from quantum_circuit.openqasm import parse

assert "quantum_circuit.cli" not in sys.modules

# Pure sampling/payload logic from an in-memory program.
program = parse(
    "OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\n"
    "qreg q[1];\\ncreg c[1];\\nmeasure q[0] -> c[0];\\n"
)
sample = commands.simulate_task(program, None, 32, 7)
assert sample["counts"] == {"0": 32}
assert commands.probabilities_task(program, None)["probabilities"] == {"0": 1.0}
optimized = commands.optimize_task(program)
assert optimized["optimized_gate_count"] == 0

# Document validation is pure and reports through CommandFailure data.
counts = documents.parse_samples_document(
    '{"schema_version": 1, "counts": {"0": 5}}', 1
)
assert counts == {"0": 5}
assert documents.json_equal({"b": 2, "a": [1]}, {"a": [1], "b": 2}) is True

jobs = commands.manifest_jobs(
    '{"schema_version": 1, "jobs": [{"id": "j", "source": "c.qasm"}]}',
    "batch_input_error",
)
assert jobs[0]["id"] == "j"

try:
    commands.samples_counts('{"schema_version": 1, "counts": {"00": 1}}', 1)
except errors.CommandFailure as failure:
    assert failure.exit_code == 2 and failure.error == "sample_input_error"
    payload = failure.to_payload()
    assert list(payload) == ["error", "message"]
else:
    raise AssertionError("expected CommandFailure")

# Nothing in this process may have pulled in the command entry point.
assert "quantum_circuit.cli" not in sys.modules
print("ok")
"""


def test_pure_logic_usable_without_command_module():
    result = subprocess.run(
        [sys.executable, "-c", _ISOLATED_PURE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
