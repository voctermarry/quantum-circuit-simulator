"""Cross-carrier parity for the expectation observable rules.

The ``expectation`` command (strict JSON document) and
:meth:`Circuit.expectation` (Python lists/tuples and mappings) share one
carrier-independent rule set. These tests feed both entry points the same
observable *content* and assert the two paths accept exactly the same
domain content and classify every rejection identically:

* valid content produces equal payloads (order, values, noise echo);
* invalid content is ``observable_error`` (exit 2, empty stdout, one
  stderr line) on the command and the matching ``TypeError``/``ValueError``
  in the DSL;
* strict-JSON envelope concerns stay command-specific, while the DSL keeps
  accepting tuples and abstract mappings and rejecting bare iterables.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from quantum_circuit import Circuit, cli
from quantum_circuit.observables import ObservableError, parse_observables

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


# --------------------------------------------------------------- shared cases


def _bell_content():
    return [
        {"id": "zz", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"}]},
        {"id": "identity", "operators": []},
        {"id": "x0", "operators": [{"qubit": 0, "pauli": "X"}]},
    ]


# (QASM body, DSL circuit, content, noise model) — body and DSL describe the
# same circuit; content is fed verbatim to both entry points.
VALID_CASES = [
    (
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\n",
        Circuit(2, 2).h(0).cx(0, 1).measure(0, 0),
        _bell_content(),
        None,
    ),
    (
        "qreg q[1];\ncreg c[1];\nx q[0];\n",
        Circuit(1, 1).x(0),
        [
            {"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]},
            {"id": "y", "operators": [{"qubit": 0, "pauli": "Y"}]},
        ],
        None,
    ),
    (
        "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\nry(0.6) q[2];\n",
        Circuit(3, 3).h(0).cx(0, 1).ry(0.6, 2),
        [
            {"id": "I", "operators": []},
            {"id": "XZX", "operators": [
                {"qubit": 2, "pauli": "X"},
                {"qubit": 1, "pauli": "Z"},
                {"qubit": 0, "pauli": "X"},
            ]},
        ],
        {"depolarizing": 0.1, "bit_flip": 0.05},
    ),
]


# Invalid content shared by both paths on a 2-qubit register; the DSL
# exception class records the shared type/value classification.
INVALID_CASES = [
    ([], ValueError),
    ([{}], ValueError),
    ([{"operators": []}], ValueError),
    ([{"id": "a"}], ValueError),
    ([{"id": "", "operators": []}], ValueError),
    ([{"id": 1, "operators": []}], TypeError),
    ([{"id": "a", "operators": {}}], TypeError),
    ([{"id": "a", "operators": []}, {"id": "a", "operators": []}], ValueError),
    ([{"id": "a", "operators": [], "extra": 1}], ValueError),
    ([{"id": "a", "operators": [{}]}], ValueError),
    ([{"id": "a", "operators": [{"pauli": "Z"}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": 0}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": True, "pauli": "Z"}]}], TypeError),
    ([{"id": "a", "operators": [{"qubit": "0", "pauli": "Z"}]}], TypeError),
    ([{"id": "a", "operators": [{"qubit": -1, "pauli": "Z"}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": 2, "pauli": "Z"}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": 0, "pauli": 0}]}], TypeError),
    ([{"id": "a", "operators": [{"qubit": 0, "pauli": "x"}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": 0, "pauli": "I"}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": 0, "pauli": "Z", "extra": 1}]}], ValueError),
    ([{"id": "a", "operators": [{"qubit": 0, "pauli": "Z"},
                                {"qubit": 0, "pauli": "X"}]}], ValueError),
    ([[]], TypeError),
    (["x"], TypeError),
    ([True], TypeError),
    ([None], TypeError),
    ({}, TypeError),
    ("observables", TypeError),
]


# ------------------------------------------------------------------- helpers


def _run_cli(tmp_path, body, content, capsys, noise_model=None):
    qasm = tmp_path / "c.qasm"
    qasm.write_text(HEADER + body, encoding="utf-8")
    obs = tmp_path / "observables.json"
    obs.write_text(json.dumps({"schema_version": 1, "observables": content}), encoding="utf-8")
    args = ["expectation", str(qasm), str(obs)]
    if noise_model is not None:
        model = tmp_path / "noise.json"
        model.write_text(json.dumps(noise_model), encoding="utf-8")
        args += ["--noise-model", str(model)]
    rc = cli.main(args)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


class _Map(Mapping):
    """A non-dict mapping: accepted by the DSL, impossible in JSON input."""

    def __init__(self, data):
        self._d = dict(data)

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)


def _as_tuples_and_maps(value):
    if isinstance(value, list):
        return tuple(_as_tuples_and_maps(item) for item in value)
    if isinstance(value, dict):
        return _Map({key: _as_tuples_and_maps(item) for key, item in value.items()})
    return value


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("body,circuit,content,noise", VALID_CASES)
def test_equivalent_valid_content_gives_equal_payloads(
    tmp_path, capsys, body, circuit, content, noise
):
    rc, out, err = _run_cli(tmp_path, body, content, capsys, noise)
    assert rc == 0, err
    assert err == ""
    cli_payload = json.loads(out)

    dsl_payload = circuit.expectation(content, noise_model=noise)
    assert dsl_payload == cli_payload
    assert list(dsl_payload) == list(cli_payload)
    assert [entry["id"] for entry in dsl_payload["results"]] == [
        entry["id"] for entry in content
    ]

    # The DSL's tuple/abstract-mapping carrier shape carries the same content.
    assert circuit.expectation(_as_tuples_and_maps(content), noise_model=noise) == cli_payload


@pytest.mark.parametrize("content,expected_error", INVALID_CASES)
def test_equivalent_invalid_content_classified_identically(
    tmp_path, capsys, content, expected_error
):
    body = "qreg q[2];\ncreg c[1];\n"
    rc, out, err = _run_cli(tmp_path, body, content, capsys)
    assert rc == 2
    assert out == ""
    assert err.count("\n") == 1
    assert json.loads(err)["error"] == "observable_error"

    with pytest.raises(expected_error):
        Circuit(2, 1).expectation(content)


def test_count_boundary_matches_on_both_paths(tmp_path, capsys):
    body = "qreg q[1];\ncreg c[1];\n"
    for size, ok in ((1, True), (100, True), (0, False), (101, False)):
        content = [{"id": f"o-{i}", "operators": []} for i in range(size)]
        rc, _, err = _run_cli(tmp_path, body, content, capsys)
        assert (rc == 0) is ok, (size, err)
        if ok:
            Circuit(1, 1).expectation(content)
        else:
            with pytest.raises(ValueError):
                Circuit(1, 1).expectation(content)


def test_dsl_rejects_bare_iterables_booleans_and_strings():
    circuit = Circuit(1, 1)
    good = [{"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]}]
    bare_iterables = [
        (entry for entry in good),
        "observables",
        "XYZ",
        b"XYZ",
        True,
        42,
        None,
        {"id": "a"},
    ]
    for bad in bare_iterables:
        with pytest.raises(TypeError):
            circuit.expectation(bad)
    # A string must not be iterated as a one-character operator sequence.
    with pytest.raises(TypeError):
        circuit.expectation([{"id": "a", "operators": "X"}])


def test_strict_json_envelope_remains_command_specific():
    valid = json.dumps({"schema_version": 1, "observables": [{"id": "a", "operators": []}]})
    assert parse_observables(valid, 2) == [("a", [])]

    for document in (
        "[]",
        "{}",
        '{"schema_version": 2, "observables": []}',
        '{"schema_version": "1", "observables": []}',
        '{"schema_version": true, "observables": []}',
        '{"schema_version": 1, "observables": {}}',
        '{"schema_version": 1, "observables": [], "extra": 1}',
        '{"schema_version": 1, "observables": [], "observables": []}',
        '{"schema_version": 1, "observables": NaN}',
        '{"schema_version": 1, "observables": Infinity}',
        '{"schema_version": 1, "observables": []} trailing',
    ):
        with pytest.raises(ObservableError):
            parse_observables(document, 2)
