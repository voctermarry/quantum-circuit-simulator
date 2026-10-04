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

# The canonical payload field order matches the documented commands.
assert list(sample) == [
    "schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts",
]
assert list(noisy_sample) == [
    "schema_version", "shots", "seed", "num_qubits", "num_clbits",
    "noise_model", "counts",
]
assert list(noisy_sample["noise_model"]) == ["amplitude_damping", "depolarizing"]

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


_COMMAND_MODULES = ("quantum_circuit.cli", "quantum_circuit.commands", "quantum_circuit.cli_io")


def test_shared_pure_logic_works_without_command_modules():
    # Every command's pure computation and JSON document validation is
    # usable without the command layer (argument parsing, input preparation
    # and result delivery) ever being imported.
    result = _run_isolated(
        """
import sys

from quantum_circuit import core, documents
from quantum_circuit.openqasm import parse

for module in {modules!r}:
    assert module not in sys.modules, module

BELL = "OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\nqreg q[2];\\ncreg c[2];\\nh q[0];\\ncx q[0], q[1];\\nmeasure q[0] -> c[0];\\nmeasure q[1] -> c[1];\\n"
program = parse(BELL)

# simulate / probabilities / verify-samples
simulation = core.run_simulation(program, 64, 7)
probabilities = core.run_probabilities(program)
counts = documents.parse_samples_document(
    '{{"schema_version": 1, "counts": {{"00": 30, "11": 34}}}}', program.num_clbits
)
verification = core.run_verification(program, counts, None, 0.5)
assert verification["accepted"] is True

# expectation
observables = [("z0", [(0, "Z")])]
expectation = core.run_expectation(program, observables, None)
assert expectation["schema_version"] == 1

# equivalent / state-metrics
equivalence = core.run_equivalence(program, program)
assert equivalence["equivalent"] is True
metrics = core.run_state_metrics(program, program, None, None)
assert metrics["fidelity"] == 1.0

# batch-simulate / reconcile scheduling over prepared tasks
jobs = documents.parse_batch_manifest(
    '{{"schema_version": 1, "jobs": [{{"id": "a", "source": "bell.qasm", "shots": 16, "seed": 3}}]}}'
)
prepared = [(jobs[0], program, None, None)]
summary = core.run_batch(prepared)
assert summary["succeeded"] == 1 and summary["failed"] == 0
baseline = documents.parse_baseline_document(
    __import__("json").dumps(summary), jobs
)
report = core.run_reconciliation(prepared, baseline)
assert report["consistent"] is True and report["matched"] == 1
assert documents.json_equal(summary["results"], baseline)

for module in {modules!r}:
    assert module not in sys.modules, module

print("ok")
""".format(modules=_COMMAND_MODULES)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
