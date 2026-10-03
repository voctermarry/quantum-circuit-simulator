"""Tests for the ``expectation`` subcommand.

The expectation math is additionally cross-checked against an independent
Pauli-matrix construction (``reference_expectation``) and against the
density-matrix path with all-zero channels, which must agree with the
state-vector path.
"""

from __future__ import annotations

import io
import itertools
import json
import math

import pytest

from quantum_circuit import cli
from quantum_circuit.noise import evolve_density_matrix
from quantum_circuit.observables import (
    density_matrix_expectation,
    parse_observables,
    snap_expectation,
    state_vector_expectation,
)
from quantum_circuit.openqasm import parse
from quantum_circuit.simulator import simulate_state_vector

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_obs(tmp_path):
    def _write(data, name: str = "observables.json"):
        if isinstance(data, str):
            text = data
        else:
            text = json.dumps(data)
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_bytes(tmp_path):
    def _write(data: bytes, name: str = "observables.json"):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return _write


@pytest.fixture
def write_model(tmp_path):
    def _write(text: str, name: str = "noise.json"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write


def _program(body: str):
    return parse(HEADER + body)


def _expect(program, observable: dict):
    """Run one observable document through the CLI-independent math path."""
    text = json.dumps({"schema_version": 1, "observables": [observable]})
    return parse_observables(text, program.num_qubits)


# ------------------------------------------------------------ reference check


_PAULI_MATRICES = {
    "X": ((0 + 0j, 1 + 0j), (1 + 0j, 0 + 0j)),
    "Y": ((0 + 0j, 0 - 1j), (0 + 1j, 0 + 0j)),
    "Z": ((1 + 0j, 0 + 0j), (0 + 0j, -1 + 0j)),
}


def reference_expectation(state, num_qubits, operators):
    """Brute-force <psi|P|psi> with explicit Kronecker Pauli matrices."""

    def kron(a, b):
        a_rows, a_cols = len(a), len(a[0])
        b_rows, b_cols = len(b), len(b[0])
        result = []
        for i in range(a_rows * b_rows):
            row = []
            for j in range(a_cols * b_cols):
                row.append(a[i // b_rows][j // b_cols] * b[i % b_rows][j % b_cols])
            result.append(row)
        return result

    factors = {q: "I" for q in range(num_qubits)}
    for qubit, pauli in operators:
        factors[qubit] = pauli
    matrix = ((1.0,),)
    # qubit 0 is the least significant bit, so build in reverse order.
    for qubit in range(num_qubits - 1, -1, -1):
        factor = ((1.0, 0.0), (0.0, 1.0)) if factors[qubit] == "I" else _PAULI_MATRICES[factors[qubit]]
        matrix = kron(matrix, factor)
    applied = [
        sum(matrix[i][j] * state[j] for j in range(len(state))) for i in range(len(state))
    ]
    return sum(state[i].conjugate() * applied[i] for i in range(len(state))).real


@pytest.mark.parametrize(
    "body",
    [
        "qreg q[1];\ncreg c[1];\nh q[0];\n",
        "qreg q[1];\ncreg c[1];\nry(0.7) q[0];\n",
        "qreg q[1];\ncreg c[1];\nrx(1.3) q[0];\n",
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n",
        "qreg q[2];\ncreg c[2];\nh q[0];\nry(0.9) q[1];\ncz q[0],q[1];\n",
        "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\nry(0.5) q[2];\ncz q[1],q[2];\n",
    ],
)
def test_state_vector_expectation_matches_reference(body):
    program = _program(body)
    state = simulate_state_vector(program)

    candidates: list[list[tuple[int, str]]] = [[]]
    for qubit in range(program.num_qubits):
        for pauli in "XYZ":
            candidates.append([(qubit, pauli)])
    # Every Pauli product over the first min(n, 3) qubits: 1, 9 or 27 items.
    active = min(program.num_qubits, 3)
    for assignment in itertools.product("XYZ", repeat=active):
        candidates.append([(qubit, pauli) for qubit, pauli in enumerate(assignment)])

    for operators in candidates:
        observable = {
            "id": "o",
            "operators": [
                {"qubit": qubit, "pauli": pauli} for qubit, pauli in operators
            ],
        }
        parsed_operators = _expect(program, observable)[0][1]
        got = state_vector_expectation(state, parsed_operators)
        expected = reference_expectation(state, program.num_qubits, parsed_operators)
        assert got == pytest.approx(expected, abs=1e-12), operators
        assert -1.0 - 1e-12 <= got <= 1.0 + 1e-12


def test_global_identity_is_one(write_qasm, write_obs, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\nry(0.4) q[1];\n")
    obs = write_obs({"schema_version": 1, "observables": [{"id": "I", "operators": []}]})
    assert cli.main(["expectation", qasm, obs]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["results"] == [{"id": "I", "expectation": 1}]


def test_state_vector_and_zero_noise_density_paths_agree(write_qasm, write_model, write_obs):
    body = "qreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\nry(0.6) q[2];\ncz q[1],q[2];\n"
    program = _program(body)
    state = simulate_state_vector(program)
    rho = evolve_density_matrix(
        program, {"amplitude_damping": 0.0, "phase_damping": 0.0, "bit_flip": 0.0, "depolarizing": 0.0}
    )
    document = {
        "schema_version": 1,
        "observables": [
            {"id": "I", "operators": []},
            {"id": "ZZ", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"}]},
            {"id": "XY", "operators": [{"qubit": 0, "pauli": "X"}, {"qubit": 2, "pauli": "Y"}]},
            {"id": "YYY", "operators": [{"qubit": q, "pauli": "Y"} for q in range(3)]},
            {"id": "XZX", "operators": [{"qubit": 2, "pauli": "X"}, {"qubit": 1, "pauli": "Z"}, {"qubit": 0, "pauli": "X"}]},
        ],
    }
    parsed = parse_observables(json.dumps(document), program.num_qubits)
    for observable_id, operators in parsed:
        sv = state_vector_expectation(state, operators)
        dm = density_matrix_expectation(rho, operators)
        assert sv == pytest.approx(dm, abs=1e-12), observable_id


# --------------------------------------------------------------------- payload


def test_noiseless_payload_shape_and_field_order(write_qasm, write_obs, capsys):
    qasm = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\n"
    )
    obs = write_obs(
        {
            "schema_version": 1,
            "observables": [
                {"id": "ZZ", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 1, "pauli": "Z"}]},
                {"id": "X0", "operators": [{"qubit": 0, "pauli": "X"}]},
            ],
        }
    )
    rc = cli.main(["expectation", qasm, obs])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1 and captured.out.endswith("\n")
    data = json.loads(captured.out)
    assert list(data) == ["schema_version", "num_qubits", "noise_model", "results"]
    assert data["schema_version"] == 1
    assert data["num_qubits"] == 2
    assert data["noise_model"] is None
    assert list(data["results"][0]) == ["id", "expectation"]
    assert [entry["id"] for entry in data["results"]] == ["ZZ", "X0"]
    assert data["results"][0]["expectation"] == 1
    assert abs(data["results"][1]["expectation"]) <= 1e-15


def test_measurements_do_not_affect_expectations(write_qasm, write_obs, capsys):
    body = "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    measured = write_qasm(body + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n", "m.qasm")
    unmeasured = write_qasm(body, "u.qasm")
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "ZZ", "operators": [{"qubit": q, "pauli": "Z"} for q in (0, 1)]}
        ]}
    )
    assert cli.main(["expectation", measured, obs]) == 0
    first = capsys.readouterr().out
    assert cli.main(["expectation", unmeasured, obs]) == 0
    second = capsys.readouterr().out
    assert first == second


def test_noisy_payload_echoes_canonical_model(write_qasm, write_model, write_obs, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nry(0.5) q[0];\n")
    model = write_model('{"depolarizing": 0.25, "bit_flip": 0.1}')
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "X", "operators": [{"qubit": 0, "pauli": "X"}]}
        ]}
    )
    assert cli.main(["expectation", qasm, obs, "--noise-model", model]) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data) == ["schema_version", "num_qubits", "noise_model", "results"]
    assert data["noise_model"] == {"bit_flip": 0.1, "depolarizing": 0.25}
    assert list(data["noise_model"]) == ["bit_flip", "depolarizing"]


def test_noisy_expectation_values(write_qasm, write_model, write_obs, capsys):
    # ry(theta)|0> = c|0> + s|1> has <X>=sin(theta), <Z>=cos(theta).
    # A bit_flip channel with probability p preserves X and scales Z by
    # 1-2p; identity is untouched.
    theta = 0.5
    qasm = write_qasm(f"qreg q[1];\ncreg c[1];\nry({theta}) q[0];\n")
    model = write_model('{"bit_flip": 0.25}')
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "I", "operators": []},
            {"id": "X", "operators": [{"qubit": 0, "pauli": "X"}]},
            {"id": "Y", "operators": [{"qubit": 0, "pauli": "Y"}]},
            {"id": "Z", "operators": [{"qubit": 0, "pauli": "Z"}]},
        ]}
    )
    assert cli.main(["expectation", qasm, obs, "--noise-model", model]) == 0
    results = {entry["id"]: entry["expectation"] for entry in json.loads(capsys.readouterr().out)["results"]}
    assert results["I"] == 1
    assert results["X"] == pytest.approx(math.sin(theta), abs=1e-12)
    assert abs(results["Y"]) <= 1e-12
    assert results["Z"] == pytest.approx(0.5 * math.cos(theta), abs=1e-12)


def test_results_preserve_input_order(write_qasm, write_obs, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    ids = ["z", "a", "m", "id with spaces", "c"]
    obs = write_obs(
        {
            "schema_version": 1,
            "observables": [
                {"id": observable_id, "operators": [{"qubit": 0, "pauli": "Z"}]}
                for observable_id in ids
            ],
        }
    )
    assert cli.main(["expectation", qasm, obs]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [entry["id"] for entry in data["results"]] == ids
    assert all(entry["expectation"] == -1 for entry in data["results"])


def test_byte_identical_repeated_runs(write_qasm, write_model, write_obs, capsys):
    qasm = write_qasm("qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "XX", "operators": [{"qubit": q, "pauli": "X"} for q in (0, 1)]},
            {"id": "ZZ", "operators": [{"qubit": q, "pauli": "Z"} for q in (0, 1)]},
        ]}
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["expectation", qasm, obs]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1
    model = write_model('{"depolarizing": 0.07}')
    noisy = set()
    for _ in range(3):
        assert cli.main(["expectation", qasm, obs, "--noise-model", model]) == 0
        noisy.add(capsys.readouterr().out)
    assert len(noisy) == 1


# ------------------------------------------------------------------ snapping


def test_snap_expectation_endpoints_and_zero():
    assert snap_expectation(0.0) == 0.0
    assert snap_expectation(-0.0) == 0.0
    assert snap_expectation(5e-16) == 0.0
    assert snap_expectation(-5e-16) == 0.0
    assert snap_expectation(1.0) == 1
    assert snap_expectation(1.0 - 5e-16) == 1
    assert snap_expectation(-1.0) == -1
    assert snap_expectation(-1.0 + 5e-16) == -1
    assert isinstance(snap_expectation(1.0), int)
    assert isinstance(snap_expectation(-1.0), int)
    assert snap_expectation(0.5) == pytest.approx(0.5)
    assert snap_expectation(1.2) == 1
    assert snap_expectation(-1.2) == -1


def test_zero_is_serialized_without_negative_zero(write_qasm, write_obs, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "X", "operators": [{"qubit": 0, "pauli": "X"}]}
        ]}
    )
    assert cli.main(["expectation", qasm, obs]) == 0
    raw = capsys.readouterr().out
    assert '"expectation": 0.0' in raw
    assert "-0.0" not in raw


# --------------------------------------------------------------------- limits


def test_twenty_qubits_noiseless_succeeds(write_qasm, write_obs, capsys):
    qasm = write_qasm("qreg q[20];\ncreg c[1];\nx q[0];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "Z0", "operators": [{"qubit": 0, "pauli": "Z"}]},
            {"id": "I", "operators": []},
        ]}
    )
    assert cli.main(["expectation", qasm, obs]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["results"][0]["expectation"] == -1
    assert data["results"][1]["expectation"] == 1


def test_ten_qubits_noisy_succeeds(write_qasm, write_model, write_obs, capsys):
    qasm = write_qasm("qreg q[10];\ncreg c[1];\nh q[0];\n")
    model = write_model('{"depolarizing": 0.1}')
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "Z9", "operators": [{"qubit": 9, "pauli": "Z"}]}
        ]}
    )
    assert cli.main(["expectation", qasm, obs, "--noise-model", model]) == 0
    assert capsys.readouterr().err == ""


def test_noise_over_ten_qubits_is_simulation_error(write_qasm, write_model, write_obs, capsys):
    qasm = write_qasm("qreg q[11];\ncreg c[1];\nh q[0];\n")
    model = write_model('{"bit_flip": 0.1}')
    obs = write_obs(
        {"schema_version": 1, "observables": [{"id": "I", "operators": []}]}
    )
    rc = cli.main(["expectation", qasm, obs, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "simulation_error"


# --------------------------------------------------------------- stdin/errors


def test_observables_from_stdin(write_qasm, write_obs, monkeypatch, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    obs = json.dumps(
        {"schema_version": 1, "observables": [
            {"id": "Z", "operators": [{"qubit": 0, "pauli": "Z"}]}
        ]}
    ).encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(obs)))
    assert cli.main(["expectation", qasm, "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["results"] == [{"id": "Z", "expectation": -1}]


def test_noise_model_from_stdin(write_qasm, write_obs, monkeypatch, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [{"id": "I", "operators": []}]}
    )
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b'{"bit_flip": 1}')))
    assert cli.main(["expectation", qasm, obs, "--noise-model", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["noise_model"] == {"bit_flip": 1.0}


@pytest.mark.parametrize(
    "args",
    [
        ["-", "-"],
        ["-", "-", "--noise-model", "/tmp/whatever.json"],
        ["-", "obs.json", "--noise-model", "-"],
        ["src.qasm", "-", "--noise-model", "-"],
        ["-", "-", "--noise-model", "-"],
    ],
)
def test_multiple_stdin_inputs_is_expectation_error(monkeypatch, capsys, args):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin must not be read")

    monkeypatch.setattr(cli.sys, "stdin", ExplodingStdin())
    rc = cli.main(["expectation", *args])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "expectation_error"
    assert captured.err.count("\n") == 1


def test_missing_observables_is_tagged_io_error(write_qasm, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    rc = cli.main(["expectation", qasm, "/nonexistent/observables.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "observables"


def test_missing_source_is_io_error(write_obs, capsys):
    obs = write_obs({"schema_version": 1, "observables": [{"id": "I", "operators": []}]})
    rc = cli.main(["expectation", "/nonexistent/circuit.qasm", obs])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_observables_is_tagged_io_error(write_qasm, write_bytes, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    obs = write_bytes(b'{"schema_version": 1}\xff')
    rc = cli.main(["expectation", qasm, obs])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == "observables"


def test_circuit_parse_error_is_unchanged(write_obs, tmp_path, capsys):
    qasm = tmp_path / "c.qasm"
    qasm.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    obs = write_obs({"schema_version": 1, "observables": [{"id": "I", "operators": []}]})
    rc = cli.main(["expectation", str(qasm), obs])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_invalid_noise_model_is_noise_model_error(write_qasm, write_model, write_obs, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    model = write_model('{"bit_flip": 2}')
    obs = write_obs({"schema_version": 1, "observables": [{"id": "I", "operators": []}]})
    rc = cli.main(["expectation", qasm, obs, "--noise-model", model])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "noise_model_error"


_INVALID_DOCUMENTS = [
    "not json",
    "[]",
    "{}",
    '{"schema_version": 2, "observables": []}',
    '{"schema_version": "1", "observables": []}',
    '{"schema_version": true, "observables": []}',
    '{"schema_version": 1, "observables": {}}',
    '{"schema_version": 1, "observables": [], "extra": 1}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": []}], "observables": []}',
    '{"schema_version": 1, "observables": [{"id": "a"}]}',
    '{"schema_version": 1, "observables": [{"operators": []}]}',
    '{"schema_version": 1, "observables": [{"id": "", "operators": []}]}',
    '{"schema_version": 1, "observables": [{"id": 1, "operators": []}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": {}}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": []}, {"id": "a", "operators": []}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"pauli": "Z"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 0}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": true, "pauli": "Z"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": "0", "pauli": "Z"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": -1, "pauli": "Z"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 2, "pauli": "Z"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 0, "pauli": "x"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 0, "pauli": "I"}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 0, "pauli": "Z", "extra": 1}]}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "extra": 1, "operators": []}]}',
    '{"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 0, "pauli": "Z"}, {"qubit": 0, "pauli": "X"}]}]}',
    '{"schema_version": 1, "observables": [[]]}',
    '{"schema_version": 1, "observables": "NaN"}',
]


@pytest.mark.parametrize("document", _INVALID_DOCUMENTS)
def test_invalid_observables_document_is_observable_error(
    write_qasm, write_obs, capsys, document
):
    # Two-qubit circuit so the out-of-range qubit case (qubit 2) fails.
    qasm = write_qasm("qreg q[2];\ncreg c[1];\n")
    obs = write_obs(document)
    rc = cli.main(["expectation", qasm, obs])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "observable_error"
    assert captured.err.count("\n") == 1


def test_one_hundred_observables_allowed(write_qasm, write_obs, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    obs = write_obs(
        {
            "schema_version": 1,
            "observables": [
                {"id": f"o-{index}", "operators": [{"qubit": 0, "pauli": "Z"}]}
                for index in range(100)
            ],
        }
    )
    assert cli.main(["expectation", qasm, obs]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["results"]) == 100


def test_observable_error_emitted_before_evolution(write_qasm, write_obs, monkeypatch, capsys):
    qasm = write_qasm("qreg q[20];\ncreg c[1];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [{"id": "a", "operators": [{"qubit": 25, "pauli": "Z"}]}]}
    )
    # A 20-qubit state vector would be huge; validation must fail without it.
    with monkeypatch.context() as patched:
        patched.setattr(cli, "simulate_state_vector", lambda program: pytest.fail("state evolved"))
        rc = cli.main(["expectation", qasm, obs])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "observable_error"


# -------------------------------------------------------------------- output


def test_export_matches_stdout(write_qasm, write_obs, tmp_path, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [
            {"id": "Z", "operators": [{"qubit": 0, "pauli": "Z"}]}
        ]}
    )
    assert cli.main(["expectation", qasm, obs]) == 0
    stdout_bytes = capsys.readouterr().out
    target = str(tmp_path / "result.json")
    assert cli.main(["expectation", qasm, obs, "--output", target]) == 0
    assert capsys.readouterr().out == ""
    with open(target, encoding="utf-8") as handle:
        assert handle.read() == stdout_bytes


def test_output_conflict_with_any_input(write_qasm, write_obs, write_model, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [{"id": "I", "operators": []}]}
    )
    model = write_model('{"bit_flip": 0.0}')
    for target in (qasm, obs, model):
        args = ["expectation", qasm, obs, "--output", target]
        if target == model:
            args[3:3] = ["--noise-model", model]
        rc = cli.main(args)
        captured = capsys.readouterr()
        assert rc == 2
        assert captured.out == ""
        assert json.loads(captured.err)["error"] == "output_error"


def test_unsupported_arguments_are_usage_errors(write_qasm, write_obs, capsys):
    qasm = write_qasm("qreg q[1];\ncreg c[1];\n")
    obs = write_obs(
        {"schema_version": 1, "observables": [{"id": "I", "operators": []}]}
    )
    with pytest.raises(SystemExit) as info:
        cli.main(["expectation", qasm, obs, "--shots", "5"])
    assert info.value.code == 2
    assert capsys.readouterr().out == ""
