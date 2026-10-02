"""Tests for the Python circuit DSL (``quantum_circuit.Circuit``)."""

from __future__ import annotations

import io
import json
import math
import sys
import types

import pytest

from quantum_circuit import Circuit, NoiseModelError, ParseError, ValidationError
from quantum_circuit import cli
from quantum_circuit.openqasm import parse

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


# --------------------------------------------------------------- construction


def test_constructor_bounds_and_types():
    Circuit(1, 1)
    Circuit(20, 20)
    for bad in (0, 21, -5):
        with pytest.raises(ValueError):
            Circuit(bad, 2)
        with pytest.raises(ValueError):
            Circuit(2, bad)
    for bad in (True, False, 1.0, "2", None, 2.0):
        with pytest.raises(TypeError):
            Circuit(bad, 2)
        with pytest.raises(TypeError):
            Circuit(2, bad)


def test_chaining_returns_same_object_and_keeps_order():
    circuit = Circuit(3, 2)
    for call in (
        lambda: circuit.x(2),
        lambda: circuit.h(0),
        lambda: circuit.rx(1, 0.5),
        lambda: circuit.cx(0, 1),
        lambda: circuit.cz(2, 0),
        lambda: circuit.crx(0, 1, 0.25),
        lambda: circuit.cry(1, 2, 1.0),
        lambda: circuit.crz(2, 0, -0.5),
        lambda: circuit.ry(0, 1),
        lambda: circuit.rz(1, 2),
        lambda: circuit.measure(2, 1),
        lambda: circuit.measure(0, 0),
    ):
        assert call() is circuit
    lines = [line for line in circuit.to_qasm().splitlines() if line.endswith(";")][4:]
    assert lines == [
        "x q[2];",
        "h q[0];",
        "rx(0.5) q[1];",
        "cx q[0],q[1];",
        "cz q[2],q[0];",
        "crx(0.25) q[0],q[1];",
        "cry(1.0) q[1],q[2];",
        "crz(-0.5) q[2],q[0];",
        "ry(1.0) q[0];",
        "rz(2.0) q[1];",
        "measure q[2] -> c[1];",
        "measure q[0] -> c[0];",
    ]


def test_accessors():
    circuit = Circuit(3, 2)
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 2


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "append",
    [
        lambda c: c.x(1.0),
        lambda c: c.x(True),
        lambda c: c.h("0"),
        lambda c: c.cx(0.0, 1),
        lambda c: c.cz(0, False),
        lambda c: c.measure(0.0, 0),
        lambda c: c.measure(0, 0.0),
    ],
)
def test_index_type_errors(append):
    with pytest.raises(TypeError):
        append(Circuit(2, 2))


@pytest.mark.parametrize(
    "append",
    [
        lambda c: c.x(2),
        lambda c: c.x(-1),
        lambda c: c.cx(0, 2),
        lambda c: c.h(5),
        lambda c: c.measure(0, 2),
        lambda c: c.measure(2, 0),
        lambda c: c.measure(-1, 0),
    ],
)
def test_index_range_errors(append):
    with pytest.raises(ValueError):
        append(Circuit(2, 2))


def test_control_and_target_must_differ():
    with pytest.raises(ValueError):
        Circuit(2, 2).cx(1, 1)
    with pytest.raises(ValueError):
        Circuit(2, 2).cz(0, 0)
    with pytest.raises(ValueError):
        Circuit(2, 2).crx(1, 1, 0.5)
    with pytest.raises(ValueError):
        Circuit(2, 2).cry(0, 0, 0.5)
    with pytest.raises(ValueError):
        Circuit(2, 2).crz(1, 1, 0.5)


@pytest.mark.parametrize("gate", ["rx", "ry", "rz"])
def test_rotation_angle_types_and_finiteness(gate):
    getattr(Circuit, gate)(Circuit(1, 1), 0, 1)  # integer is accepted
    with pytest.raises(TypeError):
        getattr(Circuit, gate)(Circuit(1, 1), 0, "0.5")
    with pytest.raises(TypeError):
        getattr(Circuit, gate)(Circuit(1, 1), 0, True)
    with pytest.raises(ValueError):
        getattr(Circuit, gate)(Circuit(1, 1), 0, float("nan"))
    with pytest.raises(ValueError):
        getattr(Circuit, gate)(Circuit(1, 1), 0, float("inf"))


@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
def test_controlled_rotation_angle_validation(gate):
    getattr(Circuit, gate)(Circuit(2, 2), 0, 1, 1)  # integer angle accepted
    with pytest.raises(TypeError):
        getattr(Circuit, gate)(Circuit(2, 2), 0, 1, None)
    with pytest.raises(ValueError):
        getattr(Circuit, gate)(Circuit(2, 2), 0, 1, float("-inf"))


def test_measurement_rules():
    circuit = Circuit(2, 2).measure(0, 0)
    with pytest.raises(ValueError):
        circuit.h(1)  # gate after measurement
    with pytest.raises(ValueError):
        circuit.crx(0, 1, 0.5)
    with pytest.raises(ValueError):
        circuit.measure(0, 1)  # qubit measured twice
    with pytest.raises(ValueError):
        circuit.measure(1, 0)  # clbit written twice
    # A different qubit/clbit pair is still allowed after measuring starts.
    circuit.measure(1, 1)


def test_failed_append_does_not_mutate_circuit():
    circuit = Circuit(2, 2).h(0)
    snapshot = circuit.to_qasm()
    bad_calls = (
        lambda: circuit.x(5),
        lambda: circuit.rx(0, float("nan")),
        lambda: circuit.cx(0, 0),
        lambda: circuit.measure(9, 0),
        lambda: circuit.h("x"),
    )
    for call in bad_calls:
        with pytest.raises((TypeError, ValueError)):
            call()
        assert circuit.to_qasm() == snapshot
    assert len(circuit.to_qasm().splitlines()) == len(snapshot.splitlines())


# -------------------------------------------------------------- qasm emit/parse


def test_to_qasm_fixed_shape():
    qasm = (
        Circuit(2, 2)
        .h(0)
        .cx(0, 1)
        .measure(0, 0)
        .measure(1, 1)
        .to_qasm()
    )
    assert qasm == (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )
    assert qasm.endswith("\n")


def test_angle_serialization_round_trips_and_canonicalizes_zero():
    for angle in (0.1 + 0.2, 1.0, -2.5, 3e8, 7e-300, 123456789012345678.0):
        qasm = Circuit(1, 1).rx(0, angle).to_qasm()
        rendered = next(line for line in qasm.splitlines() if line.startswith("rx("))
        token = rendered[len("rx(") :].split(")")[0]
        assert float(token) == float(angle)
        # The rendered text is Python's shortest round-trip repr.
        assert token == repr(float(angle))
    assert Circuit(1, 1).rz(0, -0.0).to_qasm().splitlines()[-1] == "rz(0) q[0];"


def test_from_qasm_round_trip_every_gate():
    source = (
        HEADER
        + "qreg q[3];\ncreg c[3];\n"
        "x q[0];\nh q[1];\ncx q[0],q[1];\ncz q[1],q[2];\n"
        "rx(1.5) q[0];\nry(-2.0) q[1];\nrz(0.3) q[2];\n"
        "crx(0.5) q[0],q[1];\ncry(1.0) q[1],q[2];\ncrz(-0.25) q[2],q[0];\n"
        "measure q[0] -> c[2];\nmeasure q[1] -> c[1];\n"
    )
    circuit = Circuit.from_qasm(source)
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 3
    re_parsed = Circuit.from_qasm(circuit.to_qasm())
    assert re_parsed.to_qasm() == circuit.to_qasm()


def test_from_qasm_preserves_parse_error_position():
    with pytest.raises(ParseError) as excinfo:
        Circuit.from_qasm("not qasm")
    assert excinfo.value.line == 1
    assert excinfo.value.column == 1


def test_from_qasm_preserves_validation_error_position():
    source = HEADER + "qreg q[1];\ncreg c[1];\nx q[2];\n"
    with pytest.raises(ValidationError) as excinfo:
        Circuit.from_qasm(source)
    assert (excinfo.value.line, excinfo.value.column) == (5, 5)


def test_from_qasm_requires_string():
    with pytest.raises(TypeError):
        Circuit.from_qasm(123)


# ----------------------------------------------------------- CLI result parity


def _run_cli(command, source, noise=None, shots=None, seed=None):
    args = [command, "-"]
    if shots is not None:
        args += ["--shots", str(shots)]
    if seed is not None:
        args += ["--seed", str(seed)]
    if noise is not None:
        import tempfile

        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(noise, handle)
        handle.close()
        args += ["--noise-model", handle.name]
    stdin = types.SimpleNamespace(buffer=io.BytesIO(source.encode("utf-8")))
    stdout = io.StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = stdin, stdout
    try:
        code = cli.main(args)
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
    assert code == 0, stdout.getvalue()
    return json.loads(stdout.getvalue())


CIRCUITS = [
    lambda: Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1),
    lambda: Circuit(2, 2).rx(0, 0.7).cry(1, 0, 1.2).cz(0, 1)
    .measure(0, 1)
    .measure(1, 0),
    lambda: Circuit(3, 3).h(0).cx(0, 1).crx(0, 2, math.pi / 2)
    .measure(0, 0)
    .measure(1, 1),
    lambda: Circuit(1, 1),  # no measurements
]


@pytest.mark.parametrize("build", CIRCUITS)
def test_sample_matches_cli_simulate(build):
    circuit = build()
    source = circuit.to_qasm()
    for shots, seed in ((100, 3), (250, -7), (1, 0), (1024, 42)):
        expected = _run_cli("simulate", source, shots=shots, seed=seed)
        assert circuit.sample(shots=shots, seed=seed) == expected


@pytest.mark.parametrize("build", CIRCUITS)
def test_probabilities_matches_cli_probabilities(build):
    circuit = build()
    source = circuit.to_qasm()
    expected = _run_cli("probabilities", source)
    assert circuit.probabilities() == expected


NOISE = {"bit_flip": 0.1, "amplitude_damping": 0.2, "depolarizing": 0.05}


def test_noisy_sample_and_probabilities_match_cli():
    circuit = (
        Circuit(2, 2)
        .h(0)
        .cx(0, 1)
        .measure(0, 0)
        .measure(1, 1)
    )
    source = circuit.to_qasm()
    expected_sample = _run_cli("simulate", source, noise=NOISE, shots=77, seed=11)
    assert circuit.sample(shots=77, seed=11, noise_model=NOISE) == expected_sample
    assert expected_sample["schema_version"] == 2
    assert list(expected_sample["noise_model"]) == [
        "amplitude_damping",
        "bit_flip",
        "depolarizing",
    ]
    expected_probs = _run_cli("probabilities", source, noise=NOISE)
    assert circuit.probabilities(noise_model=NOISE) == expected_probs


def test_defaults_shot_count_and_no_measurement_outcome():
    result = Circuit(2, 2).sample()
    assert result["shots"] == 1024
    assert result["seed"] == 0
    assert result["counts"] == {"00": 1024}
    probs = Circuit(2, 2).probabilities()
    assert probs["probabilities"] == {"00": 1.0}


def test_repeated_calls_equal_and_isolated():
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    first = circuit.sample(shots=100, seed=5)
    for seed in (1, 999, -3):
        circuit.sample(shots=100, seed=seed)
    assert circuit.sample(shots=100, seed=5) == first
    # probabilities are deterministic too
    assert circuit.probabilities() == circuit.probabilities()


def test_simulation_does_not_mutate_circuit():
    circuit = Circuit(2, 2).h(0).cx(0,1).measure(0, 0).measure(1, 1)
    snapshot = circuit.to_qasm()
    circuit.sample(shots=10, seed=1)
    circuit.sample(shots=10, seed=2, noise_model={"bit_flip": 0.1})
    circuit.probabilities()
    circuit.probabilities(noise_model={"phase_damping": 0.3})
    assert circuit.to_qasm() == snapshot


# -------------------------------------------------------------- noise models


@pytest.mark.parametrize(
    "model",
    [
        {},
        {"unknown": 0.1},
        {"bit_flip": 2},
        {"bit_flip": -0.01},
        {"bit_flip": True},
        {"bit_flip": "0.5"},
        {"bit_flip": float("nan")},
        {"amplitude_damping": float("inf")},
        [],
    ],
)
def test_invalid_noise_model(model):
    with pytest.raises(NoiseModelError):
        Circuit(1, 1).sample(noise_model=model)
    with pytest.raises(NoiseModelError):
        Circuit(1, 1).probabilities(noise_model=model)


def test_noise_qubit_limit():
    circuit = Circuit(11, 11)
    with pytest.raises(ValueError):
        circuit.sample(noise_model={"bit_flip": 0.1})
    with pytest.raises(ValueError):
        circuit.probabilities(noise_model={"bit_flip": 0.1})
    # Noiseless simulation still supports the 20-qubit register.
    Circuit(20, 20).sample(shots=1)


# ---------------------------------------------------------------- sample args


def test_shots_validation():
    circuit = Circuit(1, 1)
    with pytest.raises(ValueError):
        circuit.sample(shots=0)
    with pytest.raises(ValueError):
        circuit.sample(shots=-1)
    with pytest.raises(TypeError):
        circuit.sample(shots=1.5)
    with pytest.raises(TypeError):
        circuit.sample(shots=True)
    with pytest.raises(TypeError):
        circuit.sample(shots="10")


def test_seed_validation():
    circuit = Circuit(1, 1)
    with pytest.raises(TypeError):
        circuit.sample(seed=1.0)
    with pytest.raises(TypeError):
        circuit.sample(seed=False)
    with pytest.raises(TypeError):
        circuit.sample(seed="0")
    circuit.sample(seed=-123456789)
