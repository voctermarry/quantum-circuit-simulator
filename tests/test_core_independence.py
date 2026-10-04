"""The DSL simulation entry points must not depend on the CLI module.

``Circuit.sample`` and ``Circuit.probabilities`` share the CLI-independent
simulation core, so a program that only imports the package must be able
to use them without ``quantum_circuit.cli`` ever being loaded.
"""

from __future__ import annotations

import subprocess
import sys


def _run_isolated(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )


def test_sample_and_probabilities_work_without_cli_loaded():
    result = _run_isolated(
        """
import sys

import quantum_circuit
from quantum_circuit import Circuit

assert "quantum_circuit.cli" not in sys.modules

circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)

sample = circuit.sample(shots=200, seed=7)
probabilities = circuit.probabilities()
noisy_sample = circuit.sample(
    shots=128, seed=3, noise_model={"depolarizing": 0.1, "amplitude_damping": 0.05}
)
noisy_probabilities = circuit.probabilities(
    noise_model={"phase_damping": 0.2, "bit_flip": 0.1}
)
observables = [
    {"id": "zz", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"}]},
    {"id": "identity", "operators": []},
]
expectation = circuit.expectation(observables)
noisy_expectation = circuit.expectation(
    observables, noise_model={"depolarizing": 0.1, "amplitude_damping": 0.05}
)

# No import of the command line may have happened at any point.
assert "quantum_circuit.cli" not in sys.modules

# Calls are deterministic given the same seed/model.
assert circuit.sample(shots=200, seed=7) == sample
assert circuit.probabilities() == probabilities
assert (
    circuit.sample(
        shots=128,
        seed=3,
        noise_model={"amplitude_damping": 0.05, "depolarizing": 0.1},
    )
    == noisy_sample
)
assert circuit.probabilities({"phase_damping": 0.2, "bit_flip": 0.1}) == noisy_probabilities
assert circuit.expectation(observables) == expectation
assert (
    circuit.expectation(
        observables, noise_model={"amplitude_damping": 0.05, "depolarizing": 0.1}
    )
    == noisy_expectation
)

# The canonical payload field order matches the documented commands.
assert list(sample) == [
    "schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts",
]
assert list(noisy_sample) == [
    "schema_version", "shots", "seed", "num_qubits", "num_clbits",
    "noise_model", "counts",
]
assert list(noisy_sample["noise_model"]) == ["amplitude_damping", "depolarizing"]
assert list(expectation) == ["schema_version", "num_qubits", "noise_model", "results"]
assert expectation["noise_model"] is None
assert [entry["id"] for entry in expectation["results"]] == ["zz", "identity"]
assert list(noisy_expectation["noise_model"]) == ["amplitude_damping", "depolarizing"]

print("ok")
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_core_module_does_not_import_cli():
    # Importing the shared core never pulls in the command line package.
    result = _run_isolated(
        """
import sys

from quantum_circuit import core

assert "quantum_circuit.cli" not in sys.modules
assert not hasattr(core, "cli")
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
