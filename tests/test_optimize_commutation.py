"""Tests for the same-qubit commutation relations in the canonical optimizer.

The optimizer recognizes exactly the relations decidable from the gate
semantics: ``x``/``rx`` and ``y``/``ry`` on the same qubit, and the
computational-basis diagonal gates ``z``, ``s``, ``sdg``, ``t``, ``tdg``,
``rz``, ``cz`` and ``crz`` whenever they share a qubit. Cancellation,
inverse cancellation and rotation merging see through any number of
commuting gates; commuting gates are reordered into a canonical order so
permutations of a commuting multiset optimize to byte-identical qasm.
Every other shared-qubit combination keeps its dependency order.
"""

from __future__ import annotations

import json

import pytest

from quantum_circuit import cli
from quantum_circuit.equivalence import measurement_layout, unitary_distance
from quantum_circuit.openqasm import parse
from quantum_circuit.optimizer import optimize
from quantum_circuit.simulator import unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm", header: str = HEADER):
        path = tmp_path / name
        path.write_text(header + body, encoding="utf-8")
        return str(path)

    return _write


def _optimize_body(body: str, qreg: str = "q[3]", creg: str = "c[3]"):
    program = parse(HEADER + f"qreg {qreg};\ncreg {creg};\n{body}")
    return optimize(program)


def _gate_lines(qasm: str) -> list[str]:
    return [
        line
        for line in qasm.splitlines()
        if not line.startswith(("OPENQASM", "include", "qreg", "creg", "measure"))
    ]


def _assert_fixed_point(qasm: str) -> None:
    _gates, changed, again = optimize(parse(qasm))
    assert changed is False
    assert again == qasm


def _assert_equivalent(original_body: str, qasm: str, qreg: str = "q[3]", creg: str = "c[3]") -> None:
    original = parse(HEADER + f"qreg {qreg};\ncreg {creg};\n{original_body}")
    optimized = parse(qasm)
    assert unitary_distance(unitary_matrix(original), unitary_matrix(optimized)) <= 1e-10
    assert measurement_layout(original) == measurement_layout(optimized)


# ------------------------------------------------- cancellation across commutes


def test_x_pair_cancels_across_rx():
    _, changed, qasm = _optimize_body("x q[0];\nrx(0.5) q[0];\nx q[0];\n")
    assert changed is True
    assert _gate_lines(qasm) == ["rx(0.5) q[0];"]
    _assert_fixed_point(qasm)


def test_y_pair_cancels_across_ry():
    _, changed, qasm = _optimize_body("y q[1];\nry(0.25) q[1];\ny q[1];\n")
    assert changed is True
    assert _gate_lines(qasm) == ["ry(0.25) q[1];"]
    _assert_fixed_point(qasm)


def test_x_pair_cancels_across_many_commuting_gates():
    body = "x q[0];\nrx(0.1) q[0];\nrx(0.2) q[0];\nrx(0.3) q[0];\nx q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    # The rx chain merges (left to right) and the x pair then cancels.
    assert _gate_lines(qasm) == [f"rx({0.1 + 0.2 + 0.3!r}) q[0];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


def test_s_sdg_cancel_across_cz():
    body = "s q[0];\ncz q[0],q[1];\nsdg q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["cz q[0],q[1];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


def test_t_tdg_cancel_across_multiple_diagonal_gates():
    body = "t q[0];\nz q[0];\ns q[0];\ncrz(0.5) q[0],q[1];\ntdg q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["s q[0];", "z q[0];", "crz(0.5) q[0],q[1];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


def test_z_pair_cancels_across_diagonal_gates():
    body = "z q[1];\ns q[1];\nt q[1];\nz q[1];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["s q[1];", "t q[1];"]
    _assert_fixed_point(qasm)


# ------------------------------------------------------ merging across commutes


def test_rz_merges_across_single_qubit_diagonal_gates():
    body = "rz(0.3) q[0];\nz q[0];\ns q[0];\nrz(0.4) q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["rz(0.7) q[0];", "s q[0];", "z q[0];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


def test_rz_merges_across_controlled_diagonal_gates():
    body = "rz(0.3) q[1];\ncz q[0],q[1];\ncrz(0.5) q[0],q[1];\nrz(0.4) q[1];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["crz(0.5) q[0],q[1];", "cz q[0],q[1];", "rz(0.7) q[1];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


def test_rx_merges_across_x():
    body = "rx(0.3) q[0];\nx q[0];\nrx(0.4) q[0];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["rx(0.7) q[0];", "x q[0];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


def test_crz_merges_across_diagonal_gates():
    body = "crz(0.3) q[0],q[1];\ns q[0];\nrz(0.2) q[1];\ncrz(0.4) q[0],q[1];\n"
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    assert _gate_lines(qasm) == ["s q[0];", "crz(0.7) q[0],q[1];", "rz(0.2) q[1];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm)


# ------------------------------------------------------- commutation reordering


def test_commuting_pair_is_sorted_and_marks_changed():
    _, changed, qasm = _optimize_body("s q[0];\nrz(0.5) q[0];\n", qreg="q[1]", creg="c[1]")
    assert changed is True
    assert _gate_lines(qasm) == ["rz(0.5) q[0];", "s q[0];"]
    _assert_fixed_point(qasm)


def test_already_sorted_commuting_pair_is_unchanged():
    _, changed, qasm = _optimize_body("rz(0.5) q[0];\ns q[0];\n", qreg="q[1]", creg="c[1]")
    assert changed is False
    assert _gate_lines(qasm) == ["rz(0.5) q[0];", "s q[0];"]


def test_x_rx_pair_is_sorted():
    _, changed, qasm = _optimize_body("x q[0];\nrx(0.5) q[0];\n", qreg="q[1]", creg="c[1]")
    assert changed is True
    assert _gate_lines(qasm) == ["rx(0.5) q[0];", "x q[0];"]
    _assert_fixed_point(qasm)


@pytest.mark.parametrize(
    "lines",
    [
        ["z q[0];", "s q[0];", "t q[0];", "rz(0.5) q[0];", "cz q[0],q[1];"],
        ["x q[0];", "rx(0.3) q[0];", "rx(0.4) q[0];"],
        ["y q[1];", "ry(0.3) q[1];"],
        ["cz q[0],q[1];", "cz q[1],q[0];"],
        ["crz(0.5) q[0],q[1];", "crz(0.5) q[1],q[0];"],
        ["s q[0];", "crz(0.5) q[0],q[1];", "cz q[0],q[1];", "rz(0.3) q[1];", "sdg q[0];"],
        # Three or more diagonal rotations: the merged angle must not depend
        # on the written order (floating-point addition is not associative).
        ["rz(0.1) q[0];", "rz(0.2) q[0];", "rz(0.3) q[0];"],
        ["rz(2.5) q[0];", "rz(-2.8) q[0];", "sdg q[0];", "rz(-0.5) q[0];", "z q[0];", "s q[0];"],
        ["crz(0.11) q[0],q[1];", "crz(0.22) q[0],q[1];", "crz(0.33) q[0],q[1];", "cz q[0],q[1];"],
    ],
)
def test_commuting_multiset_permutations_converge_byte_identically(lines):
    import itertools

    measures = "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    outputs = set()
    for order in itertools.permutations(lines):
        body = "".join(order)
        _, _, qasm = _optimize_body(body + measures, qreg="q[2]", creg="c[2]")
        outputs.add(qasm)
    assert len(outputs) == 1
    _assert_fixed_point(outputs.pop())


def test_cz_operand_direction_is_preserved_not_merged():
    # cz keeps the operand direction written in the source; the two
    # directions are distinct gates ordered by the canonical sort key.
    _, changed, qasm = _optimize_body("cz q[1],q[0];\ncz q[0],q[1];\n", qreg="q[2]", creg="c[2]")
    assert changed is True
    assert _gate_lines(qasm) == ["cz q[0],q[1];", "cz q[1],q[0];"]
    _assert_fixed_point(qasm)
    # A single directed cz is already canonical.
    _, changed, qasm = _optimize_body("cz q[1],q[0];\n", qreg="q[2]", creg="c[2]")
    assert changed is False
    assert _gate_lines(qasm) == ["cz q[1],q[0];"]


def test_crz_operand_direction_is_preserved():
    _, changed, qasm = _optimize_body("crz(0.5) q[1],q[0];\n", qreg="q[2]", creg="c[2]")
    assert changed is False
    assert _gate_lines(qasm) == ["crz(0.5) q[1],q[0];"]


# ---------------------------------------------------- pairs that must not move


@pytest.mark.parametrize(
    "body",
    [
        "x q[0];\nry(0.5) q[0];\n",      # x commutes with rx only
        "y q[0];\nrx(0.5) q[0];\n",      # y commutes with ry only
        "rx(0.1) q[0];\nry(0.2) q[0];\n",  # different axes
        "z q[0];\nx q[0];\n",            # diagonal vs Pauli x
        "x q[0];\nrz(0.5) q[0];\n",
        "h q[0];\nrz(0.5) q[0];\n",      # h commutes with nothing
        "h q[0];\ns q[0];\n",
        "cx q[0],q[1];\nz q[0];\n",      # cx is not diagonal
        "cx q[0],q[1];\nrz(0.5) q[1];\n",
        "crx(0.5) q[0],q[1];\nrz(0.5) q[1];\n",  # crx is not diagonal
        "cry(0.5) q[0],q[1];\ns q[0];\n",
        "swap q[0],q[1];\nrz(0.5) q[0];\n",
        "cz q[0],q[1];\nx q[1];\n",      # diagonal vs non-commuting Pauli
        "crz(0.5) q[0],q[1];\nrx(0.5) q[0];\n",
    ],
)
def test_non_listed_pairs_keep_dependency_order(body):
    _, changed, qasm = _optimize_body(body)
    assert changed is False
    assert _gate_lines(qasm) == [line for line in body.splitlines()]


def test_blocking_gate_still_prevents_cancellation():
    # An h between two commuting-separated x gates blocks the cancellation.
    body = "x q[0];\nrx(0.5) q[0];\nh q[0];\nx q[0];\n"
    _, changed, qasm = _optimize_body(body, qreg="q[1]", creg="c[1]")
    assert _gate_lines(qasm) == ["rx(0.5) q[0];", "x q[0];", "h q[0];", "x q[0];"]
    _assert_fixed_point(qasm)
    _assert_equivalent(body, qasm, qreg="q[1]", creg="c[1]")


# ------------------------------------------------------- counts and equivalence


def test_commutation_reorder_keeps_gate_counts(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\n"
        "t q[0];\nrz(0.5) q[0];\ncz q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n",
    )
    assert cli.main(["optimize", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["changed"] is True
    assert data["original_gate_count"] == 3
    assert data["optimized_gate_count"] == 3
    assert data["schema_version"] == 1
    _assert_fixed_point(data["qasm"])


def test_commuting_circuit_stays_equivalent_with_measurements():
    body = (
        "s q[0];\ncz q[0],q[1];\nsdg q[0];\nt q[1];\nrz(0.3) q[1];\n"
        "x q[2];\nrx(0.4) q[2];\n"
        "measure q[0] -> c[2];\nmeasure q[1] -> c[0];\nmeasure q[2] -> c[1];\n"
    )
    _, changed, qasm = _optimize_body(body)
    assert changed is True
    _assert_equivalent(body, qasm)
    _assert_fixed_point(qasm)
