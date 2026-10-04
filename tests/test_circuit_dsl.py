"""Tests for the Python circuit DSL (``quantum_circuit.Circuit``)."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

import quantum_circuit
from quantum_circuit import Circuit, NoiseModelError, ParseError, ValidationError
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
    def _write(text: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


def _bell() -> Circuit:
    return Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)


# ------------------------------------------------------------- construction


def test_package_exports_circuit_and_keeps_version():
    assert quantum_circuit.Circuit is Circuit
    assert quantum_circuit.__version__ == "0.1.0"


def test_valid_register_sizes_accepted():
    circuit = Circuit(1, 20)
    assert circuit.num_qubits == 1
    assert circuit.num_clbits == 20
    assert circuit.operations == ()


@pytest.mark.parametrize("value", [True, False, 1.0, "2", None, [2]])
def test_register_size_type_errors(value):
    with pytest.raises(TypeError):
        Circuit(value, 2)
    with pytest.raises(TypeError):
        Circuit(2, value)


@pytest.mark.parametrize("value", [0, -1, 21, 100])
def test_register_size_range_errors(value):
    with pytest.raises(ValueError):
        Circuit(value, 2)
    with pytest.raises(ValueError):
        Circuit(2, value)


# ------------------------------------------------------- chaining and order


def test_chaining_returns_same_object_and_keeps_order():
    circuit = Circuit(3, 3)
    result = circuit.x(0).h(1).cx(0, 1).cz(1, 2).rx(0.5, 0).ry(0.6, 1).rz(0.7, 2)
    result = result.crx(0.8, 0, 1).cry(0.9, 1, 2).crz(1.0, 2, 0)
    result = result.measure(0, 0).measure(1, 1).measure(2, 2)
    assert result is circuit
    assert [op.kind for op in circuit.operations] == [
        "x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz",
        "measure", "measure", "measure",
    ]


# ----------------------------------------------------------- gate validation


@pytest.mark.parametrize("value", [True, 0.5, "0", None])
def test_gate_index_type_errors(value):
    circuit = Circuit(2, 2)
    with pytest.raises(TypeError):
        circuit.x(value)
    with pytest.raises(TypeError):
        circuit.cx(0, value)
    with pytest.raises(TypeError):
        circuit.measure(value, 0)
    with pytest.raises(TypeError):
        circuit.measure(0, value)


@pytest.mark.parametrize("value", [-1, 2, 5])
def test_gate_index_range_errors(value):
    circuit = Circuit(2, 2)
    with pytest.raises(ValueError):
        circuit.h(value)
    with pytest.raises(ValueError):
        circuit.cz(0, value)
    with pytest.raises(ValueError):
        circuit.measure(0, value)


@pytest.mark.parametrize(
    "append",
    [
        lambda c: c.cx(1, 1),
        lambda c: c.cz(0, 0),
        lambda c: c.crx(0.5, 1, 1),
        lambda c: c.cry(0.5, 0, 0),
        lambda c: c.crz(0.5, 1, 1),
    ],
)
def test_control_and_target_must_differ(append):
    with pytest.raises(ValueError):
        append(Circuit(2, 2))


@pytest.mark.parametrize("value", [True, "0.5", None, [0.5]])
def test_angle_type_errors(value):
    circuit = Circuit(2, 2)
    with pytest.raises(TypeError):
        circuit.rx(value, 0)
    with pytest.raises(TypeError):
        circuit.crx(value, 0, 1)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_angle_must_be_finite(value):
    circuit = Circuit(2, 2)
    with pytest.raises(ValueError):
        circuit.ry(value, 0)
    with pytest.raises(ValueError):
        circuit.crz(value, 0, 1)


def test_failed_append_does_not_change_circuit():
    circuit = Circuit(2, 2).h(0)
    before = circuit.operations
    for bad_call in (
        lambda: circuit.x(5),
        lambda: circuit.cx(0, 0),
        lambda: circuit.rx(math.nan, 0),
        lambda: circuit.rx("a", 0),
        lambda: circuit.measure(0, 7),
    ):
        with pytest.raises((TypeError, ValueError)):
            bad_call()
        assert circuit.operations == before


# ------------------------------------------------------- measurement semantics


def test_qubit_and_clbit_measured_at_most_once():
    circuit = Circuit(2, 2).measure(0, 0)
    with pytest.raises(ValueError):
        circuit.measure(0, 1)
    with pytest.raises(ValueError):
        circuit.measure(1, 0)
    assert [op.kind for op in circuit.operations] == ["measure"]


def test_no_gate_after_measurement():
    circuit = Circuit(2, 2).measure(0, 0)
    with pytest.raises(ValueError):
        circuit.h(1)
    with pytest.raises(ValueError):
        circuit.rx(0.5, 1)
    assert [op.kind for op in circuit.operations] == ["measure"]


# ------------------------------------------------------------------- to_qasm


def test_to_qasm_exact_text():
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    assert circuit.to_qasm() == (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0],q[1];\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
    )


def test_to_qasm_all_gate_forms_and_insertion_order():
    circuit = (
        Circuit(3, 2)
        .x(2).h(0).cz(2, 0)
        .rx(0.5, 0).ry(1.25, 1).rz(-2.5, 2)
        .crx(0.1, 0, 1).cry(0.2, 1, 2).crz(0.3, 2, 0)
        .measure(2, 1).measure(0, 0)
    )
    assert circuit.to_qasm() == (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[3];\n"
        "creg c[2];\n"
        "x q[2];\n"
        "h q[0];\n"
        "cz q[2],q[0];\n"
        "rx(0.5) q[0];\n"
        "ry(1.25) q[1];\n"
        "rz(-2.5) q[2];\n"
        "crx(0.1) q[0],q[1];\n"
        "cry(0.2) q[1],q[2];\n"
        "crz(0.3) q[2],q[0];\n"
        "measure q[2] -> c[1];\n"
        "measure q[0] -> c[0];\n"
    )


def test_to_qasm_angle_formatting_round_trips():
    circuit = Circuit(1, 1).rx(-0.0, 0).ry(2, 0).rz(0.1, 0)
    text = circuit.to_qasm()
    # Negative zero renders as 0; integers as floats; repr round-trips.
    assert "rx(0) q[0];" in text
    assert "ry(2.0) q[0];" in text
    assert "rz(0.1) q[0];" in text
    parsed = Circuit.from_qasm(text)
    assert parsed.to_qasm() == text
    assert parsed.operations[1].params == (2.0,)


# ----------------------------------------------------------------- from_qasm


def test_from_qasm_round_trip():
    circuit = (
        Circuit(3, 3)
        .h(0).cx(0, 2).rz(0.25, 1).crx(0.75, 2, 1)
        .measure(0, 0).measure(1, 2).measure(2, 1)
    )
    clone = Circuit.from_qasm(circuit.to_qasm())
    assert clone.num_qubits == 3 and clone.num_clbits == 3
    assert clone.operations == circuit.operations
    assert clone.to_qasm() == circuit.to_qasm()


def test_from_qasm_uses_parser_semantics():
    circuit = Circuit.from_qasm(
        HEADER
        + "qreg q[2];\ncreg c[2];\n"
        + "rx(pi/2) q[0]; // a comment\n"
        + "measure q[0] -> c[1];\n"
    )
    assert circuit.operations[0].kind == "rx"
    assert circuit.operations[0].params == (math.pi / 2,)
    assert circuit.operations[1].targets == (0, 1)
    # Measurement already started: gates are refused afterwards.
    with pytest.raises(ValueError):
        circuit.h(1)


def test_from_qasm_parse_error_position_preserved():
    with pytest.raises(ParseError) as excinfo:
        Circuit.from_qasm(HEADER + "qreg q[1];\ncreg c[1];\n&\n")
    assert (excinfo.value.line, excinfo.value.column) == (5, 1)


def test_from_qasm_validation_error_position_preserved():
    with pytest.raises(ValidationError) as excinfo:
        Circuit.from_qasm(HEADER + "qreg q[1];\ncreg c[1];\nx q[1];\n")
    assert (excinfo.value.line, excinfo.value.column) == (5, 5)


def test_from_qasm_rejects_non_string():
    with pytest.raises(TypeError):
        Circuit.from_qasm(42)


# --------------------------------------------------------- sample vs simulate


def _run_cli(argv, capsys):
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    return json.loads(captured.out)


def test_sample_matches_simulate_command(write_qasm, capsys):
    circuit = _bell()
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    expected = _run_cli(["simulate", path, "--shots", "200", "--seed", "7"], capsys)
    result = circuit.sample(shots=200, seed=7)
    assert result == expected
    assert list(result) == list(expected)


def test_sample_defaults_match_simulate_defaults(write_qasm, capsys):
    circuit = _bell()
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    expected = _run_cli(["simulate", path], capsys)
    assert circuit.sample() == expected


def test_sample_with_noise_matches_simulate(write_qasm, write_model, capsys):
    circuit = _bell()
    model = {"depolarizing": 0.1, "amplitude_damping": 0.05}
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    model_path = write_model(json.dumps(model))
    expected = _run_cli(
        ["simulate", path, "--noise-model", model_path, "--shots", "128", "--seed", "3"],
        capsys,
    )
    result = circuit.sample(shots=128, seed=3, noise_model=model)
    assert result == expected
    assert list(result["noise_model"]) == ["amplitude_damping", "depolarizing"]


def test_sample_is_deterministic_and_shares_no_random_state():
    circuit = _bell()
    first = circuit.sample(shots=64, seed=11)
    # Interleave calls on another circuit with a different seed.
    other = _bell().sample(shots=64, seed=99)
    assert circuit.sample(shots=64, seed=11) == first
    assert _bell().sample(shots=64, seed=99) == other
    assert circuit.sample(shots=64, seed=11) == first


def test_sample_does_not_modify_circuit():
    circuit = _bell()
    before = circuit.operations
    circuit.sample()
    circuit.sample(shots=10, seed=1, noise_model={"bit_flip": 0.5})
    assert circuit.operations == before


@pytest.mark.parametrize("shots", [True, 1.5, "10", None])
def test_sample_shots_type_errors(shots):
    with pytest.raises(TypeError):
        _bell().sample(shots=shots)


@pytest.mark.parametrize("shots", [0, -3])
def test_sample_shots_must_be_positive(shots):
    with pytest.raises(ValueError):
        _bell().sample(shots=shots)


@pytest.mark.parametrize("seed", [True, 0.5, "0", None])
def test_sample_seed_type_errors(seed):
    with pytest.raises(TypeError):
        _bell().sample(seed=seed)


def test_sample_noise_qubit_limit():
    circuit = Circuit(11, 1).h(0).measure(0, 0)
    with pytest.raises(ValueError):
        circuit.sample(noise_model={"bit_flip": 0.1})
    with pytest.raises(ValueError):
        circuit.probabilities(noise_model={"bit_flip": 0.1})
    # Without a noise model the state-vector path allows 20 qubits.
    wide = Circuit(11, 1).h(0).measure(0, 0)
    assert wide.probabilities()["schema_version"] == 1


# ------------------------------------------- probabilities vs probabilities


def test_probabilities_matches_command(write_qasm, capsys):
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    expected = _run_cli(["probabilities", path], capsys)
    assert circuit.probabilities() == expected


def test_probabilities_with_noise_matches_command(write_qasm, write_model, capsys):
    circuit = Circuit(2, 2).x(0).h(1).cz(0, 1).measure(0, 0).measure(1, 1)
    model = {"phase_damping": 0.2, "bit_flip": 0.1}
    path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    model_path = write_model(json.dumps(model))
    expected = _run_cli(["probabilities", path, "--noise-model", model_path], capsys)
    assert circuit.probabilities(noise_model=model) == expected


def test_probabilities_without_measurement_is_all_zero():
    circuit = Circuit(2, 3).h(0).x(1)
    result = circuit.probabilities()
    assert result["probabilities"] == {"000": 1.0}


def test_probabilities_measurement_bit_order():
    # q[0] -> c[1]: the measured bit lands in the second classical bit.
    circuit = Circuit(1, 2).x(0).measure(0, 1)
    assert circuit.probabilities()["probabilities"] == {"10": 1.0}


def test_probabilities_does_not_modify_circuit_and_is_repeatable():
    circuit = _bell()
    before = circuit.operations
    first = circuit.probabilities()
    assert circuit.probabilities() == first
    assert circuit.operations == before


# -------------------------------------------------------------- noise models


def test_noise_model_mapping_validation():
    circuit = _bell()
    with pytest.raises(NoiseModelError):
        circuit.sample(noise_model={})
    with pytest.raises(NoiseModelError):
        circuit.sample(noise_model={"unknown": 0.1})
    with pytest.raises(NoiseModelError):
        circuit.sample(noise_model={"bit_flip": 1.5})
    with pytest.raises(NoiseModelError):
        circuit.sample(noise_model={"bit_flip": math.nan})
    with pytest.raises(NoiseModelError):
        circuit.sample(noise_model={"bit_flip": True})
    with pytest.raises(NoiseModelError):
        circuit.sample(noise_model="bit_flip")
    with pytest.raises(NoiseModelError):
        circuit.probabilities(noise_model=[("bit_flip", 0.1)])


def test_noise_model_accepts_int_probabilities_and_canonical_order():
    circuit = _bell()
    result = circuit.sample(noise_model={"depolarizing": 0, "amplitude_damping": 1})
    assert list(result["noise_model"]) == ["amplitude_damping", "depolarizing"]
    assert result["noise_model"] == {"amplitude_damping": 1.0, "depolarizing": 0.0}


# ------------------------------------------------------------- expectation


def _observables():
    return [
        {"id": "zz", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"}]},
        {"id": "identity", "operators": []},
        {"id": "x0", "operators": [{"qubit": 0, "pauli": "X"}]},
    ]


def _write_observables(tmp_path, observables, name="observables.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "observables": observables}), encoding="utf-8"
    )
    return str(path)


def test_expectation_matches_command(write_qasm, tmp_path, capsys):
    circuit = _bell()
    observables = _observables()
    qasm_path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    obs_path = _write_observables(tmp_path, observables)
    expected = _run_cli(["expectation", qasm_path, obs_path], capsys)
    result = circuit.expectation(observables)
    assert result == expected
    assert list(result) == list(expected)
    assert [entry["id"] for entry in result["results"]] == ["zz", "identity", "x0"]
    assert result["results"][0]["expectation"] == 1
    assert result["results"][1]["expectation"] == 1
    assert result["noise_model"] is None


def test_expectation_with_noise_matches_command(write_qasm, write_model, tmp_path, capsys):
    circuit = _bell()
    observables = _observables()
    model = {"depolarizing": 0.1, "amplitude_damping": 0.05}
    qasm_path = write_qasm(circuit.to_qasm().split(HEADER, 1)[1])
    obs_path = _write_observables(tmp_path, observables)
    model_path = write_model(json.dumps(model))
    expected = _run_cli(
        ["expectation", qasm_path, obs_path, "--noise-model", model_path], capsys
    )
    result = circuit.expectation(observables, noise_model=model)
    assert result == expected
    assert list(result["noise_model"]) == ["amplitude_damping", "depolarizing"]


def test_expectation_known_values_and_range():
    circuit = Circuit(1, 1).x(0)
    result = circuit.expectation(
        [
            {"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]},
            {"id": "x", "operators": [{"qubit": 0, "pauli": "X"}]},
            {"id": "y", "operators": [{"qubit": 0, "pauli": "Y"}]},
        ]
    )
    assert result["num_qubits"] == 1
    values = {entry["id"]: entry["expectation"] for entry in result["results"]}
    assert values == {"z": -1, "x": 0.0, "y": 0.0}
    for value in values.values():
        assert -1 <= value <= 1
        assert math.isfinite(value)
        assert not (value == 0 and math.copysign(1, value) < 0)


def test_expectation_accepts_tuples_and_ignores_measurement_layout():
    circuit = Circuit(2, 3).h(0).cx(0, 1).measure(0, 2).measure(1, 0)
    result = circuit.expectation(
        ({"id": "zz", "operators": ({"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"})},)
    )
    assert result["results"] == [{"id": "zz", "expectation": 1}]
    assert "num_clbits" not in result


def test_expectation_is_repeatable_and_modifies_nothing():
    circuit = _bell()
    observables = _observables()
    snapshot = json.loads(json.dumps(observables))
    before = circuit.operations
    first = circuit.expectation(observables)
    assert circuit.expectation(observables) == first
    assert circuit.expectation(list(observables)) == first
    assert observables == snapshot
    assert circuit.operations == before


def test_expectation_without_importing_cli():
    code = (
        "import sys\n"
        "from quantum_circuit import Circuit\n"
        'assert "quantum_circuit.cli" not in sys.modules\n'
        "result = Circuit(1, 1).x(0).expectation("
        '[{"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]}])\n'
        'assert "quantum_circuit.cli" not in sys.modules\n'
        'assert result["results"] == [{"id": "z", "expectation": -1}]\n'
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("observables", [True, "observables", 42, None, {"id": "a"}])
def test_expectation_observables_type_errors(observables):
    with pytest.raises(TypeError):
        _bell().expectation(observables)


@pytest.mark.parametrize(
    "observables",
    [
        [],
        tuple(),
        [{"id": str(i), "operators": []} for i in range(101)],
    ],
)
def test_expectation_observables_count_errors(observables):
    with pytest.raises(ValueError):
        _bell().expectation(observables)


@pytest.mark.parametrize("item", [True, 1, "x", [], None])
def test_expectation_item_type_errors(item):
    with pytest.raises(TypeError):
        _bell().expectation([item])


@pytest.mark.parametrize(
    "observables",
    [
        [{"operators": []}],
        [{"id": "a"}],
        [{"id": "a", "operators": [], "extra": 1}],
        [{"id": "", "operators": []}],
        [{"id": "a", "operators": []}, {"id": "a", "operators": []}],
    ],
)
def test_expectation_item_value_errors(observables):
    with pytest.raises(ValueError):
        _bell().expectation(observables)


@pytest.mark.parametrize("observable_id", [True, 1, 1.5, None, ["a"]])
def test_expectation_id_type_errors(observable_id):
    with pytest.raises(TypeError):
        _bell().expectation([{"id": observable_id, "operators": []}])


@pytest.mark.parametrize("operators", [True, "ops", 1, None, {}])
def test_expectation_operators_type_errors(operators):
    with pytest.raises(TypeError):
        _bell().expectation([{"id": "a", "operators": operators}])


@pytest.mark.parametrize("operator", [True, 1, "x", [], None])
def test_expectation_operator_type_errors(operator):
    with pytest.raises(TypeError):
        _bell().expectation([{"id": "a", "operators": [operator]}])


@pytest.mark.parametrize("qubit", [True, 0.5, "0", None])
def test_expectation_qubit_type_errors(qubit):
    with pytest.raises(TypeError):
        _bell().expectation([{"id": "a", "operators": [{"qubit": qubit, "pauli": "X"}]}])


@pytest.mark.parametrize("qubit", [-1, 2, 5])
def test_expectation_qubit_range_errors(qubit):
    with pytest.raises(ValueError):
        _bell().expectation([{"id": "a", "operators": [{"qubit": qubit, "pauli": "X"}]}])


@pytest.mark.parametrize("pauli", [True, 1, 1.5, None, ["X"]])
def test_expectation_pauli_type_errors(pauli):
    with pytest.raises(TypeError):
        _bell().expectation([{"id": "a", "operators": [{"qubit": 0, "pauli": pauli}]}])


@pytest.mark.parametrize("pauli", ["x", "I", "XX", "", "z"])
def test_expectation_pauli_value_errors(pauli):
    with pytest.raises(ValueError):
        _bell().expectation([{"id": "a", "operators": [{"qubit": 0, "pauli": pauli}]}])


@pytest.mark.parametrize(
    "operators",
    [
        [{"qubit": 0, "pauli": "X", "extra": 1}],
        [{"pauli": "X"}],
        [{"qubit": 0}],
        [{"qubit": 0, "pauli": "X"}, {"qubit": 0, "pauli": "Z"}],
    ],
)
def test_expectation_operator_value_errors(operators):
    with pytest.raises(ValueError):
        _bell().expectation([{"id": "a", "operators": operators}])


def test_expectation_noise_model_errors():
    circuit = _bell()
    observables = _observables()
    with pytest.raises(NoiseModelError):
        circuit.expectation(observables, noise_model={})
    with pytest.raises(NoiseModelError):
        circuit.expectation(observables, noise_model={"unknown": 0.1})
    with pytest.raises(NoiseModelError):
        circuit.expectation(observables, noise_model="bit_flip")


def test_expectation_noise_qubit_limit():
    circuit = Circuit(11, 1).h(0).measure(0, 0)
    observables = [{"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]}]
    with pytest.raises(ValueError):
        circuit.expectation(observables, noise_model={"bit_flip": 0.1})
    # Without a noise model the state-vector path allows 20 qubits.
    assert circuit.expectation(observables)["schema_version"] == 1
