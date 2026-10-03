"""Tests for the shared simulation core (``quantum_circuit.core``).

Covers the two structural guarantees of the simulate/probabilities
refactoring: the core and the DSL never depend on ``quantum_circuit.cli``,
and the CLI and the DSL produce identical results for the same circuit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from quantum_circuit import Circuit
from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_python(source: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )


# ------------------------------------------------------------- independence


def test_core_import_does_not_pull_in_cli():
    script = (
        "import sys\n"
        "import quantum_circuit.core\n"
        "import quantum_circuit\n"
        "assert 'quantum_circuit.cli' not in sys.modules, sys.modules.keys()\n"
    )
    completed = _run_python(script)
    assert completed.returncode == 0, completed.stderr


def test_circuit_simulation_works_without_cli_module():
    script = """
import sys


class _BlockCli:
    def find_spec(self, name, path=None, target=None):
        if name == "quantum_circuit.cli":
            raise ImportError(f"blocked import of {name}")
        return None


sys.meta_path.insert(0, _BlockCli())

import quantum_circuit
assert "quantum_circuit.cli" not in sys.modules

from quantum_circuit import Circuit

circuit = (
    Circuit(3, 3)
    .h(0)
    .rx(0.7, 1)
    .crx(0.3, 0, 2)
    .cx(1, 2)
    .measure(2, 0)
    .measure(0, 2)
)
model = {"amplitude_damping": 0.1, "bit_flip": 0.05}

plain_counts = circuit.sample(shots=97, seed=5)
noisy_counts = circuit.sample(shots=97, seed=5, noise_model=model)
plain_probs = circuit.probabilities()
noisy_probs = circuit.probabilities(noise_model=model)

assert sum(plain_counts["counts"].values()) == 97
assert sum(noisy_counts["counts"].values()) == 97
assert abs(sum(plain_probs["probabilities"].values()) - 1.0) < 1e-12
assert abs(sum(noisy_probs["probabilities"].values()) - 1.0) < 1e-12
assert "quantum_circuit.cli" not in sys.modules
print("ok")
"""
    completed = _run_python(script)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


# ---------------------------------------------------------- cross-consistency


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_model(tmp_path):
    def _write(model: dict, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(json.dumps(model), encoding="utf-8")
        return str(path)

    return _write


def _run_cli(argv, capsys):
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.err == ""
    return json.loads(captured.out)


def _rich_circuit() -> Circuit:
    # Parameterized, controlled and two-qubit gates, partial measurement and
    # a permuted qubit->clbit mapping (q2->c0, q0->c2, q1 unmeasured).
    return (
        Circuit(3, 3)
        .h(0)
        .ry(0.9, 1)
        .crz(0.4, 0, 2)
        .swap(1, 2)
        .t(2)
        .measure(2, 0)
        .measure(0, 2)
    )


def test_sample_and_probabilities_match_cli_without_noise(write_qasm, capsys):
    circuit = _rich_circuit()
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])

    expected_counts = _run_cli(["simulate", path, "--shots", "333", "--seed", "42"], capsys)
    result_counts = circuit.sample(shots=333, seed=42)
    assert result_counts == expected_counts
    assert list(result_counts) == list(expected_counts)
    assert list(result_counts["counts"]) == list(expected_counts["counts"])
    assert sum(result_counts["counts"].values()) == 333

    expected_probs = _run_cli(["probabilities", path], capsys)
    result_probs = circuit.probabilities()
    assert result_probs == expected_probs
    assert list(result_probs) == list(expected_probs)
    assert list(result_probs["probabilities"]) == list(expected_probs["probabilities"])


def test_sample_and_probabilities_match_cli_with_noise(write_qasm, write_model, capsys):
    circuit = _rich_circuit()
    model = {
        "amplitude_damping": 0.08,
        "phase_damping": 0.03,
        "bit_flip": 0.02,
        "depolarizing": 0.01,
    }
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    model_path = write_model(model)

    expected_counts = _run_cli(
        ["simulate", path, "--noise-model", model_path, "--shots", "256", "--seed", "9"],
        capsys,
    )
    result_counts = circuit.sample(shots=256, seed=9, noise_model=model)
    assert result_counts == expected_counts
    assert list(result_counts) == list(expected_counts)
    assert list(result_counts["counts"]) == list(expected_counts["counts"])
    assert sum(result_counts["counts"].values()) == 256

    expected_probs = _run_cli(["probabilities", path, "--noise-model", model_path], capsys)
    result_probs = circuit.probabilities(noise_model=model)
    assert result_probs == expected_probs
    assert list(result_probs) == list(expected_probs)
    assert list(result_probs["probabilities"]) == list(expected_probs["probabilities"])


def test_no_measurement_circuit_matches_cli(write_qasm, capsys):
    circuit = Circuit(2, 2).h(0).cx(0, 1)
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])

    expected = _run_cli(["probabilities", path], capsys)
    result = circuit.probabilities()
    assert result == expected
    assert result["probabilities"] == {"00": 1.0}

    expected_counts = _run_cli(["simulate", path, "--shots", "50", "--seed", "1"], capsys)
    assert circuit.sample(shots=50, seed=1) == expected_counts
