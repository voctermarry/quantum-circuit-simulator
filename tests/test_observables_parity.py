"""Rule parity between the two observable-validation entry points.

The ``expectation`` command's strict JSON document and
``Circuit.expectation``'s native lists/tuples/mappings are two carriers
over one carrier-independent rule set (``_validate_observables`` in
:mod:`quantum_circuit.observables`). Every semantic case here is written
once as a plain Python value and then driven through both carriers:

* valid batches parse to identical domain content on both paths and, end
  to end, produce identical result JSON;
* invalid batches are rejected by both paths -- the command with
  ``observable_error``/exit code 2/empty stdout/a single stderr line, and
  the DSL with :class:`TypeError` for type substitutions or
  :class:`ValueError` for size/field/identifier/range/uniqueness/Pauli
  violations.

Only the envelope differs: JSON syntax, ``schema_version`` and strict
``dict``/``list`` containers belong to the command alone; tuples and
arbitrary mappings belong to the DSL alone.
"""

from __future__ import annotations

import json

import pytest

from quantum_circuit import Circuit, cli
from quantum_circuit.observables import (
    ObservableError,
    parse_observables,
    validate_observables,
)

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _write_inputs(tmp_path, observables_value):
    """Write a two-qubit circuit and one observables JSON document."""
    qasm = tmp_path / "circuit.qasm"
    qasm.write_text(HEADER + "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n")
    document = {"schema_version": 1, "observables": observables_value}
    obs = tmp_path / "observables.json"
    obs.write_text(json.dumps(document), encoding="utf-8")
    return str(qasm), str(obs)


def _assert_command_observable_error(tmp_path, capsys, batch):
    qasm, obs = _write_inputs(tmp_path, batch)
    rc = cli.main(["expectation", qasm, obs])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert json.loads(captured.err)["error"] == "observable_error"


# ------------------------------------------------------------- valid batches


_VALID_BATCHES: list[tuple[int, object]] = [
    # The global identity: an empty operator list.
    (2, [{"id": "I", "operators": []}]),
    # Several entries with out-of-order, non-sorted ids preserved as given.
    (
        2,
        [
            {"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]},
            {"id": "a", "operators": []},
            {"id": "m", "operators": [{"qubit": 1, "pauli": "Y"}]},
            {"id": "id with spaces", "operators": [
                {"qubit": 1, "pauli": "Z"}, {"qubit": 0, "pauli": "X"}
            ]},
            {"id": "c", "operators": [{"qubit": 0, "pauli": "Z"}]},
        ],
    ),
    # Operators need not be in ascending qubit order.
    (3, [{"id": "rev", "operators": [{"qubit": 2, "pauli": "Z"}, {"qubit": 0, "pauli": "X"}]}]),
    # The inclusive upper bound of the batch size.
    (
        1,
        [{"id": f"o-{index}", "operators": [{"qubit": 0, "pauli": "Z"}]} for index in range(100)],
    ),
    # The DSL accepts tuples wherever the document has arrays; the parsed
    # domain content must be exactly the same.
    (
        2,
        (
            {"id": "tuple", "operators": ({"qubit": 0, "pauli": "Z"},)},
            {"id": "identity", "operators": ()},
        ),
    ),
]


@pytest.mark.parametrize("num_qubits,batch", _VALID_BATCHES)
def test_valid_batch_has_identical_domain_content(num_qubits, batch):
    from_document = parse_observables(
        json.dumps({"schema_version": 1, "observables": batch}), num_qubits
    )
    from_python = validate_observables(batch, num_qubits)
    assert from_document == from_python
    # Input order and the 1..100 count are part of the accepted content.
    assert [observable_id for observable_id, _ in from_python] == [
        entry["id"] for entry in batch
    ]
    assert 1 <= len(from_python) <= 100


def test_valid_results_are_byte_identical_across_paths(tmp_path, capsys):
    batch = _VALID_BATCHES[1][1]
    qasm, obs = _write_inputs(tmp_path, batch)
    assert cli.main(["expectation", qasm, obs]) == 0
    command_output = capsys.readouterr().out

    circuit = Circuit(2, 2).h(0).cx(0, 1)
    dsl_output = json.dumps(circuit.expectation(batch)) + "\n"
    assert dsl_output == command_output
    payload = json.loads(command_output)
    assert [entry["id"] for entry in payload["results"]] == [
        "z", "a", "m", "id with spaces", "c"
    ]


def test_noisy_valid_results_are_byte_identical_across_paths(tmp_path, capsys):
    batch = [
        {"id": "I", "operators": []},
        {"id": "X0", "operators": [{"qubit": 0, "pauli": "X"}]},
        {"id": "ZZ", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"}]},
    ]
    qasm, obs = _write_inputs(tmp_path, batch)
    model = tmp_path / "noise.json"
    model.write_text('{"depolarizing": 0.25, "bit_flip": 0.1}', encoding="utf-8")
    assert cli.main(["expectation", qasm, obs, "--noise-model", str(model)]) == 0
    command_output = capsys.readouterr().out

    circuit = Circuit(2, 2).h(0).cx(0, 1)
    dsl_output = json.dumps(
        circuit.expectation(batch, noise_model={"depolarizing": 0.25, "bit_flip": 0.1})
    ) + "\n"
    assert dsl_output == command_output
    # The model is echoed in canonical key order on both paths.
    assert '"noise_model": {"bit_flip": 0.1, "depolarizing": 0.25}' in command_output


def test_measurements_are_ignored_identically(tmp_path, capsys):
    batch = [{"id": "ZZ", "operators": [{"qubit": q, "pauli": "Z"} for q in (0, 1)]}]
    qasm = tmp_path / "m.qasm"
    qasm.write_text(
        HEADER + "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
        encoding="utf-8",
    )
    obs = tmp_path / "observables.json"
    obs.write_text(json.dumps({"schema_version": 1, "observables": batch}), encoding="utf-8")
    assert cli.main(["expectation", str(qasm), str(obs)]) == 0
    command_output = capsys.readouterr().out

    measured = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    unmeasured = Circuit(2, 2).h(0).cx(0, 1)
    assert json.dumps(measured.expectation(batch)) + "\n" == command_output
    assert measured.expectation(batch) == unmeasured.expectation(batch)


# ----------------------------------------------------------- invalid batches


# JSON-representable type substitutions: driven end to end through both the
# command document and the DSL.
_TYPE_CASES: list[tuple[str, object]] = [
    ("batch is a string", "observables"),
    ("batch is a boolean", True),
    ("batch is an integer", 42),
    ("batch is None", None),
    ("batch is a mapping", {"id": "a"}),
    ("item is a boolean", [True]),
    ("item is an integer", [1]),
    ("item is a string", ["x"]),
    ("item is a list", [[]]),
    ("item is None", [None]),
    ("id is a boolean", [{"id": True, "operators": []}]),
    ("id is an integer", [{"id": 1, "operators": []}]),
    ("id is a float", [{"id": 1.5, "operators": []}]),
    ("id is None", [{"id": None, "operators": []}]),
    ("id is a list", [{"id": ["a"], "operators": []}]),
    ("operators is a boolean", [{"id": "a", "operators": True}]),
    ("operators is a string", [{"id": "a", "operators": "ops"}]),
    ("operators is an integer", [{"id": "a", "operators": 1}]),
    ("operators is None", [{"id": "a", "operators": None}]),
    # An empty mapping is not a list/tuple of operators.
    ("operators is a mapping", [{"id": "a", "operators": {}}]),
    ("operator is a boolean", [{"id": "a", "operators": [True]}]),
    ("operator is an integer", [{"id": "a", "operators": [1]}]),
    ("operator is a string", [{"id": "a", "operators": ["x"]}]),
    ("operator is a list", [{"id": "a", "operators": [[]]}]),
    ("operator is None", [{"id": "a", "operators": [None]}]),
    ("qubit is a boolean", [{"id": "a", "operators": [{"qubit": True, "pauli": "X"}]}]),
    ("qubit is a float", [{"id": "a", "operators": [{"qubit": 0.5, "pauli": "X"}]}]),
    ("qubit is a string", [{"id": "a", "operators": [{"qubit": "0", "pauli": "X"}]}]),
    ("qubit is None", [{"id": "a", "operators": [{"qubit": None, "pauli": "X"}]}]),
    ("pauli is a boolean", [{"id": "a", "operators": [{"qubit": 0, "pauli": True}]}]),
    ("pauli is an integer", [{"id": "a", "operators": [{"qubit": 0, "pauli": 1}]}]),
    ("pauli is a float", [{"id": "a", "operators": [{"qubit": 0, "pauli": 1.5}]}]),
    ("pauli is None", [{"id": "a", "operators": [{"qubit": 0, "pauli": None}]}]),
    ("pauli is a list", [{"id": "a", "operators": [{"qubit": 0, "pauli": ["X"]}]}]),
]


# Values JSON cannot express as such (other iterables, bytes): the shared
# Python core and the DSL must still refuse them; they must never be
# silently treated as lists/tuples.
_PYTHON_ONLY_TYPE_CASES: list[tuple[str, object]] = [
    ("batch is a range", range(3)),
    ("batch is a generator", (entry for entry in ({"id": "a", "operators": []},))),
    ("batch is bytes", b"observables"),
    ("operators is a range", [{"id": "a", "operators": range(2)}]),
    ("operators is a generator", [
        {"id": "a", "operators": (entry for entry in ({"qubit": 0, "pauli": "X"},))}
    ]),
]


_VALUE_CASES: list[tuple[str, object]] = [
    ("empty batch", []),
    ("empty tuple batch", tuple()),
    ("101 entries", [{"id": str(index), "operators": []} for index in range(101)]),
    ("missing id", [{"operators": []}]),
    ("missing operators", [{"id": "a"}]),
    ("unknown observable field", [{"id": "a", "operators": [], "extra": 1}]),
    # The DSL never learns the document-only schema_version key.
    ("schema_version is an unknown field", [{"id": "a", "operators": [], "schema_version": 1}]),
    ("empty id", [{"id": "", "operators": []}]),
    ("duplicate id", [{"id": "a", "operators": []}, {"id": "a", "operators": []}]),
    ("unknown operator field", [{"id": "a", "operators": [{"qubit": 0, "pauli": "X", "extra": 1}]}]),
    ("missing qubit", [{"id": "a", "operators": [{"pauli": "X"}]}]),
    ("missing pauli", [{"id": "a", "operators": [{"qubit": 0}]}]),
    ("negative qubit", [{"id": "a", "operators": [{"qubit": -1, "pauli": "X"}]}]),
    ("qubit at register size", [{"id": "a", "operators": [{"qubit": 2, "pauli": "X"}]}]),
    ("qubit beyond register", [{"id": "a", "operators": [{"qubit": 5, "pauli": "X"}]}]),
    ("repeated qubit in one entry", [
        {"id": "a", "operators": [{"qubit": 0, "pauli": "X"}, {"qubit": 0, "pauli": "Z"}]}
    ]),
    ("lowercase pauli", [{"id": "a", "operators": [{"qubit": 0, "pauli": "x"}]}]),
    ("identity pauli", [{"id": "a", "operators": [{"qubit": 0, "pauli": "I"}]}]),
    ("multi-letter pauli", [{"id": "a", "operators": [{"qubit": 0, "pauli": "XX"}]}]),
    ("empty pauli", [{"id": "a", "operators": [{"qubit": 0, "pauli": ""}]}]),
    ("lowercase z pauli", [{"id": "a", "operators": [{"qubit": 0, "pauli": "z"}]}]),
]


def _assert_dsl_type_error(batch):
    circuit = Circuit(2, 2)
    with pytest.raises(TypeError) as info:
        circuit.expectation(batch)
    # The exact built-in type (not a shared ObservableError subclass) and
    # never the ValueError family.
    assert type(info.value) is TypeError
    assert not isinstance(info.value, ObservableError)
    assert circuit.operations == ()


def _assert_dsl_value_error(batch):
    circuit = Circuit(2, 2)
    with pytest.raises(ValueError) as info:
        circuit.expectation(batch)
    assert type(info.value) is ValueError
    assert not isinstance(info.value, ObservableError)
    assert circuit.operations == ()


@pytest.mark.parametrize("label,batch", _TYPE_CASES, ids=[c[0] for c in _TYPE_CASES])
def test_type_substitution_rejected_by_both_paths(tmp_path, capsys, label, batch):
    snapshot = json.dumps(batch)
    _assert_dsl_type_error(batch)
    # A rejected batch is never mutated by validation.
    assert json.dumps(batch) == snapshot
    _assert_command_observable_error(tmp_path, capsys, batch)


@pytest.mark.parametrize(
    "label,batch", _PYTHON_ONLY_TYPE_CASES, ids=[c[0] for c in _PYTHON_ONLY_TYPE_CASES]
)
def test_non_sequence_iterable_rejected_by_dsl_and_shared_core(label, batch):
    _assert_dsl_type_error(batch)
    # The shared core itself refuses these on either carrier: only genuine
    # lists/tuples (DSL) or lists (JSON) are sequences, never arbitrary
    # iterables or bytes.
    from quantum_circuit.observables import _JSON_CARRIER, _validate_observables

    with pytest.raises(TypeError):
        validate_observables(batch, 2)
    with pytest.raises(ObservableError):
        _validate_observables(_JSON_CARRIER, batch, 2)


@pytest.mark.parametrize("label,batch", _VALUE_CASES, ids=[c[0] for c in _VALUE_CASES])
def test_value_violation_rejected_by_both_paths(tmp_path, capsys, label, batch):
    snapshot = json.dumps(batch)
    _assert_dsl_value_error(batch)
    assert json.dumps(batch) == snapshot
    _assert_command_observable_error(tmp_path, capsys, batch)


# --------------------------------------------------------- carrier envelopes


def test_dsl_does_not_inherit_json_envelope():
    # No schema_version on the DSL: it is an unknown field, not a recognized one.
    with pytest.raises(ValueError):
        Circuit(1, 1).expectation(
            [{"id": "a", "operators": [], "schema_version": 1}]
        )
    # Feeding the whole document object as the batch is a container type
    # error, rather than being accepted as a document-shaped shortcut.
    with pytest.raises(TypeError):
        Circuit(1, 1).expectation(
            {"schema_version": 1, "observables": [{"id": "a", "operators": []}]}
        )


def test_command_keeps_its_own_json_envelope():
    # The document envelope stays command-specific: parse_observables still
    # demands schema_version 1 and strict JSON containers.
    with pytest.raises(ObservableError):
        parse_observables(json.dumps({"observables": []}), 1)
    with pytest.raises(ObservableError):
        parse_observables(
            json.dumps({"schema_version": 2, "observables": [{"id": "a", "operators": []}]}), 1
        )
    # An object substituted for an array.
    with pytest.raises(ObservableError):
        parse_observables(json.dumps({"schema_version": 1, "observables": {}}), 1)
    # A non-standard JSON constant inside an otherwise valid document.
    with pytest.raises(ObservableError):
        parse_observables(
            '{"schema_version": 1, "observables": '
            '[{"id": "a", "operators": [{"qubit": NaN, "pauli": "X"}]}]}',
            1,
        )
