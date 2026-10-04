"""Acceptance tests for the converged gate-semantics layer.

These tests pin the guarantees of the internal refactor without extending
the gate set or the public package surface:

* every supported gate has one definition in
  :mod:`quantum_circuit.gates`, and the state-vector, density-matrix and
  unitary paths apply it as the same unitary;
* operations produced by the parser and the DSL are equal, immutable
  values carrying the registered semantics;
* the operand orientation of directed controlled gates and the symmetry of
  swap are not confused;
* optimization stays equivalent up to a global phase and keeps the
  measurement layout, and estimation keeps its historical counts and
  schema versions;
* the semantics module is a dependency leaf and the public package never
  imports the command line.
"""

from __future__ import annotations

import math

import pytest

from quantum_circuit import Circuit
from quantum_circuit.gates import (
    CONTROLLED,
    GATE_NAMES,
    GATE_SPECS,
    SINGLE_QUBIT,
    SWAP,
    GateSpec,
    Operation,
    gate_operation,
    measurement_operation,
    rotation_matrix,
    single_qubit_unitary,
)
from quantum_circuit.noise import evolve_density_matrix
from quantum_circuit.openqasm import parse
from quantum_circuit.optimizer import optimize
from quantum_circuit.simulator import simulate_state_vector, unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

ALL_GATE_NAMES = (
    "x", "h", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz",
    "cx", "cz", "crx", "cry", "crz", "swap",
)


def _qasm(body: str, nq: int = 3, nc: int = 3) -> str:
    return f"{HEADER}qreg q[{nq}];\ncreg c[{nc}];\n{body}"


# ------------------------------------------------------------- registry shape


def test_registry_covers_exactly_seventeen_gates():
    assert len(ALL_GATE_NAMES) == 17
    assert set(ALL_GATE_NAMES) == set(GATE_NAMES) == set(GATE_SPECS)


def test_registry_categories_are_disjoint_and_complete():
    singles = {n for n, s in GATE_SPECS.items() if s.style == SINGLE_QUBIT}
    controlled = {n for n, s in GATE_SPECS.items() if s.style == CONTROLLED}
    swaps = {n for n, s in GATE_SPECS.items() if s.style == SWAP}
    assert singles == {"x", "h", "y", "z", "s", "sdg", "t", "tdg",
                       "rx", "ry", "rz"}
    assert controlled == {"cx", "cz", "crx", "cry", "crz"}
    assert swaps == {"swap"}
    assert singles.isdisjoint(controlled)
    assert singles | controlled | swaps == set(GATE_NAMES)


def test_parameter_and_axis_rules_match_names():
    for name in ("rx", "ry", "rz", "crx", "cry", "crz"):
        assert GATE_SPECS[name].parameterized is True
        assert GATE_SPECS[name].axis == name[-1]
    for name in set(GATE_NAMES) - {"rx", "ry", "rz", "crx", "cry", "crz"}:
        assert GATE_SPECS[name].parameterized is False
        assert GATE_SPECS[name].axis is None


def test_registry_and_operations_are_immutable():
    spec = GATE_SPECS["x"]
    with pytest.raises((TypeError, AttributeError)):
        spec.name = "z"  # type: ignore[misc]
    with pytest.raises(TypeError):
        GATE_SPECS["x"] = GATE_SPECS["h"]  # type: ignore[index]
    op = gate_operation("cx", (0, 1))
    with pytest.raises((TypeError, AttributeError)):
        op.targets = (1, 0)  # type: ignore[misc]
    with pytest.raises((TypeError, AttributeError)):
        op.kind = "cz"  # type: ignore[misc]
    measure = measurement_operation(0, 1)
    with pytest.raises((TypeError, AttributeError)):
        measure.params = (1.0,)  # type: ignore[misc]


def test_spec_rotation_periods():
    two_pi = 2.0 * math.pi
    for name in ("rx", "ry", "rz"):
        assert GATE_SPECS[name].rotation_period == pytest.approx(two_pi)
    for name in ("crx", "cry", "crz"):
        assert GATE_SPECS[name].rotation_period == pytest.approx(2.0 * two_pi)
    for name in ("x", "h", "s", "sdg", "cx", "cz", "swap"):
        assert GATE_SPECS[name].rotation_period is None


# ------------------------------------------- parser and DSL share one Operation


def test_parser_and_dsl_emit_equal_operations():
    source = _qasm(
        "h q[0];\ns q[1];\nrx(0.5) q[2];\ncx q[0],q[1];\n"
        "crz(pi/4) q[2],q[0];\nswap q[1],q[2];\n"
        "measure q[0]->c[0];\n"
    )
    program = parse(source)
    circuit = Circuit.from_qasm(source)
    assert program.operations == circuit.operations
    assert all(isinstance(op, Operation) for op in program.operations)
    # Every gate op resolves back to the same immutable spec.
    for op in program.operations:
        if op.kind != "measure":
            assert op.spec is GATE_SPECS[op.kind]


def test_operations_have_registered_arity_and_params():
    program = parse(
        _qasm("x q[0];\nrx(0.3) q[1];\ncx q[0],q[1];\n"
              "crx(0.4) q[1],q[0];\nswap q[0],q[2];\n")
    )
    by_name = {op.kind: op for op in program.operations}
    assert by_name["x"].targets == (0,) and by_name["x"].params == ()
    assert by_name["rx"].targets == (1,) and by_name["rx"].params == pytest.approx((0.3,))
    assert by_name["cx"].targets == (0, 1)
    assert by_name["crx"].targets == (1, 0)
    assert by_name["swap"].targets == (0, 2)


# ----------------------------------------------- operand orientation and order


def test_cx_direction_is_control_then_target():
    # cx q[0],q[1] and cx q[1],q[0] act on different basis states.
    forward = parse(_qasm("x q[0];\ncx q[0],q[1];\n"))
    reverse = parse(_qasm("x q[0];\ncx q[1],q[0];\n"))
    sf = simulate_state_vector(forward)
    sr = simulate_state_vector(reverse)
    # x on 0 then cx(0->1): |11> ; x on 0 then cx(1->0): |01>.
    assert abs(sf[0b11]) == pytest.approx(1.0)
    assert abs(sr[0b01]) == pytest.approx(1.0)


def test_controlled_rotation_direction_and_swap_symmetry():
    for name in ("crx", "cry", "crz"):
        with_target_one = parse(_qasm(f"x q[1];\n{name}(0.7) q[0],q[1];\n"))
        with_control_one = parse(_qasm(f"x q[0];\n{name}(0.7) q[0],q[1];\n"))
        t_state = simulate_state_vector(with_target_one)
        c_state = simulate_state_vector(with_control_one)
        # Control is 0 and target is 1: rotation is blocked, state unchanged.
        assert abs(t_state[0b010]) == pytest.approx(1.0)
        if name == "crz":
            # Rz is diagonal: the control-1 amplitude stays at |001> but
            # picks up the rotation phase, so it differs from the bare x.
            assert c_state[0b001] != pytest.approx(1.0)
        else:
            # crx/cry transfer population between |001> and |011>.
            mixed = abs(c_state[0b001]) ** 2 + abs(c_state[0b011]) ** 2
            assert mixed == pytest.approx(1.0)
            assert abs(c_state[0b011]) > 0.1

    # swap is symmetric: swapping in either operand order yields the same state.
    a = parse(_qasm("x q[0];\nswap q[0],q[2];\n"))
    b = parse(_qasm("x q[0];\nswap q[2],q[0];\n"))
    assert simulate_state_vector(a) == simulate_state_vector(b)
    assert GATE_SPECS["swap"].symmetric is True
    assert GATE_SPECS["cx"].symmetric is False


def test_single_qubit_unitary_matches_registered_matrix():
    # x/h/y/z fixed matrices against their known matrix constants.
    assert single_qubit_unitary("x") == (0j, 1 + 0j, 1 + 0j, 0j)
    h00 = single_qubit_unitary("h")[0]
    assert h00.real == pytest.approx(2**-0.5)
    # rotation helper agrees with the spec target unitary for crx.
    assert rotation_matrix("x", 0.6) == GATE_SPECS["crx"].target_unitary((0.6,))


# --------------------- state-vector / density / unitary path consistency


@pytest.mark.parametrize("name", ALL_GATE_NAMES)
def test_three_paths_apply_each_gate_as_one_unitary(name):
    angle = 0.73
    if name in ("x", "h", "y", "z", "s", "sdg", "t", "tdg"):
        body = f"{name} q[1];\n"
    elif name in ("rx", "ry", "rz"):
        body = f"{name}({angle}) q[1];\n"
    elif name in ("cx", "cz"):
        body = f"x q[0];\n{name} q[0],q[1];\n"
    elif name in ("crx", "cry", "crz"):
        body = f"x q[0];\n{name}({angle}) q[0],q[1];\n"
    else:  # swap
        body = f"x q[0];\nh q[2];\nswap q[0],q[1];\n"
    program = parse(_qasm(body))

    vector = simulate_state_vector(program)
    # The unitary's column 0 must equal the state evolved from |0>.
    unitary = unitary_matrix(program)
    column_zero = [unitary[row][0] for row in range(len(unitary))]
    for evolved, column in zip(vector, column_zero):
        assert evolved == pytest.approx(column, abs=1e-15)

    # Without noise the density diagonal equals the squared amplitudes and
    # the density matrix itself is the pure |psi><psi|, so U rho U dagger
    # here exercises the same registered unitaries as the other two paths.
    rho = evolve_density_matrix(program, {})
    for index, amplitude in enumerate(vector):
        assert rho[index][index].real == pytest.approx(abs(amplitude) ** 2, abs=1e-15)
        for j_index, other in enumerate(vector):
            assert rho[index][j_index] == pytest.approx(
                amplitude * other.conjugate(), abs=1e-15
            )


def test_all_gates_together_stay_consistent_across_paths():
    body = (
        "x q[0];\nh q[1];\ny q[2];\nz q[0];\ns q[1];\nsdg q[2];\nt q[0];\ntdg q[1];\n"
        "rx(0.2) q[2];\nry(0.3) q[0];\nrz(0.4) q[1];\n"
        "cx q[0],q[1];\ncz q[1],q[2];\nswap q[0],q[2];\n"
        "crx(0.5) q[2],q[0];\ncry(0.6) q[1],q[0];\ncrz(0.7) q[2],q[1];\n"
    )
    program = parse(_qasm(body))
    vector = simulate_state_vector(program)
    rho = evolve_density_matrix(program, {})
    for index, amplitude in enumerate(vector):
        assert rho[index][index].real == pytest.approx(abs(amplitude) ** 2, abs=1e-14)


# ----------------------------------------------------------- optimization rules


def test_optimization_preserves_unitary_and_measurement_layout():
    source = _qasm(
        "z q[2];\ny q[0];\nx q[1];\nh q[2];\nh q[0];\ncx q[0],q[1];\n"
        "rx(0.4) q[0];\nrx(0.6) q[0];\ncrx(2*pi) q[1],q[2];\n"
        "measure q[0]->c[2];\nmeasure q[1]->c[0];\nmeasure q[2]->c[1];\n"
    )
    program = parse(source)
    _gates, _changed, qasm = optimize(program)
    optimized = parse(qasm)

    n = program.num_qubits
    u_before = unitary_matrix(program)
    u_after = unitary_matrix(optimized)
    trace = sum(
        u_before[j][i].conjugate() * u_after[j][i]
        for i in range(1 << n)
        for j in range(1 << n)
    )
    assert abs(abs(trace) - (1 << n)) <= 1e-9

    # Measurement layout (qubit -> clbit) is preserved exactly.
    def layout(p):
        return sorted(
            (op.targets[0], op.targets[1])
            for op in p.operations
            if op.kind == "measure"
        )

    assert layout(optimized) == layout(program)


def test_optimization_is_idempotent_and_reaches_fixed_point():
    source = _qasm(
        "x q[0];\nh q[2];\nx q[0];\ns q[1];\nsdg q[1];\n"
        "swap q[0],q[1];\nswap q[1],q[0];\n"
        "ry(0.9) q[2];\nry(-0.9) q[2];\n"
        "measure q[0]->c[0];\nmeasure q[1]->c[1];\nmeasure q[2]->c[2];\n"
    )
    first_gates, _changed, first_qasm = optimize(parse(source))
    second_gates, changed_again, _second_qasm = optimize(parse(first_qasm))
    assert changed_again is False
    assert second_gates == first_gates


# ------------------------------------------------------------- estimation


def test_estimation_counts_and_schema_versions_track_registry():
    from quantum_circuit.estimation import estimate

    v1 = parse(_qasm("x q[0];\nh q[1];\ncx q[0],q[1];\nrx(0.1) q[0];\n"))
    payload = estimate(v1, "state-vector")
    assert payload["schema_version"] == 1
    assert set(payload["gate_counts"]) == {"x", "h", "cx", "rx", "ry", "rz"}
    assert payload["gate_count"] == 4

    v2 = parse(_qasm("x q[0];\ncz q[0],q[1];\ncrx(0.2) q[0],q[1];\n"))
    payload = estimate(v2, "state-vector")
    assert payload["schema_version"] == 2
    assert len(payload["gate_counts"]) == 10
    assert payload["gate_counts"]["cz"] == 1
    assert payload["gate_counts"]["crx"] == 1

    v3 = parse(_qasm("y q[0];\nswap q[0],q[1];\n"))
    payload = estimate(v3, "state-vector")
    assert payload["schema_version"] == 3
    assert len(payload["gate_counts"]) == 17
    assert payload["gate_counts"]["y"] == 1
    assert payload["gate_counts"]["swap"] == 1

    # Mode qubit limits and entry-count accounting are unchanged.
    dense = estimate(v3, "density-matrix")
    unit = estimate(v3, "unitary")
    assert dense["qubit_limit"] == 10 and dense["supported"]
    assert unit["qubit_limit"] == 8 and unit["supported"]
    assert dense["entry_count"] == 4 ** v3.num_qubits
    assert unit["complex_payload_bytes"] == unit["entry_count"] * 16


# ------------------------------------------------------------- layering rules


def test_gates_module_is_a_dependency_leaf():
    # gates.py may import only the standard library: it must not pull in any
    # sibling package module (relative imports or quantum_circuit.* imports).
    import ast
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "quantum_circuit" / "gates.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module != "quantum_circuit"
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("quantum_circuit") for alias in node.names)
    assert len(GATE_SPECS) == 17


def test_package_root_does_not_export_new_semantics_symbols():
    import quantum_circuit

    assert set(quantum_circuit.__all__) == {
        "Circuit",
        "NoiseModelError",
        "ParseError",
        "ValidationError",
        "__version__",
    }
    assert not hasattr(quantum_circuit, "GateSpec")
    assert not hasattr(quantum_circuit, "Operation")
    assert not hasattr(quantum_circuit, "GATE_SPECS")


def test_package_root_does_not_import_cli():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import quantum_circuit\n"
         "from quantum_circuit import Circuit\n"
         "assert 'quantum_circuit.cli' not in sys.modules\n"
         "Circuit(1, 1).h(0).sample(shots=4, seed=0)\n"
         "assert 'quantum_circuit.cli' not in sys.modules\n"
         "print('ok')\n"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ----------------------------------------------------------------- validation


def test_validation_semantics_are_unchanged():
    from quantum_circuit.openqasm import ParseError, ValidationError

    def expect_error(body, error, nq=3):
        with pytest.raises(error):
            parse(_qasm(body, nq=nq))

    expect_error("foo q[0];\n", ValidationError)  # unknown gate
    expect_error("cx q[0];\n", ParseError)  # arity: missing operand -> syntax
    expect_error("x q[2];\n", ValidationError, nq=2)  # qubit out of bounds
    expect_error("cx q[0],q[0];\n", ValidationError)  # control == target
    expect_error("swap q[1],q[1];\n", ValidationError)  # swap operands equal
    expect_error("rx(1/0) q[0];\n", ValidationError)  # non-finite angle
    expect_error("measure q[0]->c[0];\nx q[0];\n", ValidationError)  # gate after measure
    expect_error("rx(0.5, 0.5) q[0];\n", ParseError)  # two angle parameters

    # DSL mirrors the same rules with its documented exception types.
    with pytest.raises(ValueError):
        Circuit(1, 1).cx(0, 0)
    with pytest.raises(ValueError):
        Circuit(2, 2).swap(0, 0)
    with pytest.raises(ValueError):
        Circuit(1, 1).rx(float("inf"), 0)
    with pytest.raises(TypeError):
        Circuit(1, 1).x(0.5)  # type: ignore[arg-type]
