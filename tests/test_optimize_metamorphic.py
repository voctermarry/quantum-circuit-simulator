"""Deterministic metamorphic tests for the canonical optimizer.

Where ``test_optimize.py`` relies on hand-written circuits and a handful of
random seeds, this module builds a *reproducible* corpus of small legal
circuits from fixed seeds and proves three metamorphic properties across the
whole supported gate set (``x h y z s sdg t tdg cx cz swap rx ry rz
crx cry crz``):

1. **Semantic preservation** -- the full pre-measurement unitary of the
   original circuit and of the re-parsed optimization result differ by no
   more than :data:`EQUIVALENCE_TOLERANCE`. The distance is computed on the
   complete unitaries (every basis input), not just on ``|0...0>``, so phase
   errors and control-block errors stay visible. The classical register
   width and the qubit-to-clbit mapping are identical.
2. **Canonical stability** -- optimizing the rendered result a second time
   returns byte-identical qasm with ``changed is False``.
3. **Determinism** -- the same seed rebuilds the identical case sequence and
   the recorded assertion results match exactly, distances included.

The generator only emits gates, angles and operands the OpenQASM subset
accepts: finite decimal/``pi`` angle expressions, distinct operands for
two-qubit gates, one to five qubits (inside the eight-qubit equivalence
limit). Any generation or re-parse failure fails the test with the seed and
case index; nothing is skipped.

Directed boundary cases pin the single-qubit ``2*pi`` rotation period, the
controlled ``4*pi`` period (a ``2*pi`` shift is *not* a global phase of a
controlled rotation), angles straddling the cancellation epsilon and the
symmetric ``swap`` operand canonicalization. A final group proves the
full-unitary oracle actually has teeth: mistakes that only show up on a
non-zero basis input (dropping ``rz(pi)``'s relative phase, dropping a
``crz(2*pi)`` control block, or swapping a ``cx``'s control and target)
produce distances far above the tolerance.
"""

from __future__ import annotations

import random
import re

import pytest

from quantum_circuit.equivalence import (
    EQUIVALENCE_TOLERANCE,
    MAX_EQUIVALENCE_QUBITS,
    measurement_layout,
    unitary_distance,
)
from quantum_circuit.openqasm import parse
from quantum_circuit.optimizer import optimize
from quantum_circuit.simulator import unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

# 17 fixed seeds x 8 cases. Slot 0 of seed ``s`` forces hardware gate
# ALL_GATE_KINDS[s % 17], so the seeds collectively guarantee every
# supported gate kind appears in the corpus regardless of random draws.
SEEDS = tuple(range(17))
CASES_PER_SEED = 8
MIN_QUBITS, MAX_QUBITS = 1, 5
SLOTS_PER_CASE = 8

SINGLE_QUBIT_KINDS = ("x", "h", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz")
TWO_QUBIT_KINDS = ("cx", "cz", "swap", "crx", "cry", "crz")
ALL_GATE_KINDS = SINGLE_QUBIT_KINDS + TWO_QUBIT_KINDS
ROTATION_KINDS = ("rx", "ry", "rz")
CONTROLLED_ROTATION_KINDS = ("crx", "cry", "crz")
SELF_INVERSE_HARDWARE = ("x", "h", "y", "z", "cx", "cz", "swap")

# Finite angle expressions the parser accepts. Single-qubit rotations
# include ``2*pi`` (a global phase, normalized away) and values around the
# cancellation epsilon; the controlled palette keeps ``2*pi``, which is a
# genuine operation on the control-1 block.
SINGLE_ANGLE_TEXTS = (
    "0.3",
    "-0.7",
    "1.1",
    "pi/2",
    "-pi/3",
    "pi",
    "-pi",
    "3*pi/2",
    "2*pi",
    "1e-13",
    "1e-12",
    "1.0001e-12",
)
CONTROLLED_ANGLE_TEXTS = (
    "0.3",
    "-0.7",
    "pi/2",
    "-pi/3",
    "pi",
    "-pi",
    "2*pi",
    "-2*pi",
    "2*pi+0.3",
    "3*pi",
    "1e-13",
)
# Non-tiny angles for motifs that must survive the epsilon deletion rule.
ORDINARY_ANGLE_TEXTS = ("0.3", "-0.7", "1.1", "pi/2", "-pi/3")


# ------------------------------------------------------------- case generation


def _case_rng(seed: int, case_index: int) -> random.Random:
    # String seeding is deterministic across runs and independent of
    # PYTHONHASHSEED (unlike seeding from a hashed tuple).
    return random.Random(f"metamorphic:{seed}:{case_index}")


def _distinct_qubits(rng: random.Random, n: int) -> tuple[int, int]:
    """Two distinct qubit indices in ``range(n)`` (n must be at least 2)."""
    a = rng.randrange(n)
    b = rng.randrange(n - 1)
    if b >= a:
        b += 1
    return a, b


def _hardware_line(rng: random.Random, kind: str, n: int) -> str:
    """One legal application of *kind* with random legal operands/angle."""
    if kind in ROTATION_KINDS:
        qubit = rng.randrange(n)
        return f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];"
    if kind in CONTROLLED_ROTATION_KINDS:
        control, target = _distinct_qubits(rng, n)
        return (
            f"{kind}({rng.choice(CONTROLLED_ANGLE_TEXTS)}) "
            f"q[{control}],q[{target}];"
        )
    if kind in TWO_QUBIT_KINDS:
        control, target = _distinct_qubits(rng, n)
        return f"{kind} q[{control}],q[{target}];"
    qubit = rng.randrange(n)
    return f"{kind} q[{qubit}];"


def _inverse_pair(rng: random.Random, n: int) -> list[str]:
    """A cancelable pair: self-inverse hardware or s/sdg, t/tdg inverses."""
    if n >= 2:
        first = rng.choice(SELF_INVERSE_HARDWARE + ("s", "t"))
    else:
        first = rng.choice(("x", "h", "y", "z", "s", "t"))
    second = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}.get(first, first)

    if first in TWO_QUBIT_KINDS:
        control, target = _distinct_qubits(rng, n)
        operands = f" q[{control}],q[{target}];"
        return [f"{first}{operands}", f"{second}{operands}"]
    qubit = rng.randrange(n)
    return [f"{first} q[{qubit}];", f"{second} q[{qubit}];"]


def _merging_pair(rng: random.Random, n: int) -> list[str]:
    """Two same-axis, same-operand rotations the optimizer must combine."""
    if n >= 2 and rng.random() < 0.5:
        kind = rng.choice(CONTROLLED_ROTATION_KINDS)
        control, target = _distinct_qubits(rng, n)
        left = rng.choice(CONTROLLED_ANGLE_TEXTS)
        right = rng.choice(CONTROLLED_ANGLE_TEXTS)
        operands = f" q[{control}],q[{target}];"
        return [f"{kind}({left}){operands}", f"{kind}({right}){operands}"]
    kind = rng.choice(ROTATION_KINDS)
    qubit = rng.randrange(n)
    left = rng.choice(SINGLE_ANGLE_TEXTS)
    right = rng.choice(SINGLE_ANGLE_TEXTS)
    return [f"{kind}({left}) q[{qubit}];", f"{kind}({right}) q[{qubit}];"]


def _noncommuting_pair(rng: random.Random, n: int) -> list[str]:
    """Two different shared-qubit gates: they may neither cancel nor merge.

    The pool is pairwise non-commuting under every recognized same-qubit
    relation (none of x/h/y/s belong to one another's commutation groups),
    so sampled pairs keep their dependency order through the optimizer.
    """
    first_kind, second_kind = rng.sample(("x", "h", "y", "s"), 2)

    def line(kind: str, qubit: int) -> str:
        if kind in ROTATION_KINDS:
            return f"{kind}({rng.choice(ORDINARY_ANGLE_TEXTS)}) q[{qubit}];"
        return f"{kind} q[{qubit}];"

    qubit = rng.randrange(n)
    return [line(first_kind, qubit), line(second_kind, qubit)]


def _disjoint_block(rng: random.Random, n: int) -> list[str]:
    """Two to four gates on pairwise disjoint qubits, emitted in random order.

    A two-qubit gate is only used when at least three qubits are free, so
    the block always contains at least two mutually independent gates.
    """
    free = list(range(n))
    rng.shuffle(free)
    gates: list[str] = []

    if n >= 3 and rng.random() < 0.5:
        control, target = free.pop(), free.pop()
        kind = rng.choice(TWO_QUBIT_KINDS)
        if kind in CONTROLLED_ROTATION_KINDS:
            gates.append(
                f"{kind}({rng.choice(CONTROLLED_ANGLE_TEXTS)}) "
                f"q[{control}],q[{target}];"
            )
        else:
            gates.append(f"{kind} q[{control}],q[{target}];")

    target_count = rng.randint(2, min(4, max(2, len(free) + len(gates))))
    while free and len(gates) < target_count:
        qubit = free.pop()
        kind = rng.choice(SINGLE_QUBIT_KINDS)
        if kind in ROTATION_KINDS:
            gates.append(f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];")
        else:
            gates.append(f"{kind} q[{qubit}];")

    rng.shuffle(gates)
    return gates


def _occupied_qubits(lines: list[str]) -> set[int]:
    return {int(value) for value in re.findall(r"q\[(\d+)\]", "\n".join(lines))}


def _scattered_pair(rng: random.Random, n: int) -> list[str]:
    """A canceling/merging pair with disjoint-qubit gates inserted between."""
    merge = rng.random() < 0.5

    if n >= 2 and rng.random() < 0.5:
        pair = _merging_pair(rng, n) if merge else _inverse_pair(rng, n)
    else:
        # A single-qubit pair placed on one random qubit.
        qubit = rng.randrange(n)
        if merge:
            kind = rng.choice(ROTATION_KINDS)
            pair = [
                f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];",
                f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];",
            ]
        else:
            first = rng.choice(("x", "h", "y", "z", "s", "t"))
            second = {"s": "sdg", "t": "tdg"}.get(first, first)
            pair = [f"{first} q[{qubit}];", f"{second} q[{qubit}];"]

    fillers: list[str] = []
    available = [q for q in range(n) if q not in _occupied_qubits(pair)]
    rng.shuffle(available)
    while available and len(fillers) < rng.randint(1, 3):
        qubit = available.pop()
        kind = rng.choice(SINGLE_QUBIT_KINDS)
        if kind in ROTATION_KINDS:
            fillers.append(f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];")
        else:
            fillers.append(f"{kind} q[{qubit}];")

    return [pair[0], *fillers, pair[1]]


def _measurements(rng: random.Random, n: int) -> str:
    """A sparse, order-shuffled qubit-to-clbit measurement section."""
    if rng.random() < 0.15:
        return ""  # measurements are optional
    count = rng.randint(1, n)
    pairs = list(zip(rng.sample(range(n), count), rng.sample(range(n), count)))
    rng.shuffle(pairs)
    return "".join(f"measure q[{q}] -> c[{c}];\n" for q, c in pairs)


def _build_source(seed: int, case_index: int) -> str:
    """Generate one complete, parser-legal OpenQASM source string."""
    rng = _case_rng(seed, case_index)
    forced_kind = ALL_GATE_KINDS[seed % len(ALL_GATE_KINDS)]
    n = rng.randint(2, MAX_QUBITS) if forced_kind in TWO_QUBIT_KINDS else rng.randint(
        MIN_QUBITS, MAX_QUBITS
    )

    blocks: list[list[str]] = []
    for slot in range(SLOTS_PER_CASE):
        if slot == 0:
            blocks.append([_hardware_line(rng, forced_kind, n)])
            continue
        motif = rng.randrange(5)
        if motif == 0:
            pool = ALL_GATE_KINDS if n >= 2 else SINGLE_QUBIT_KINDS
            blocks.append([_hardware_line(rng, rng.choice(pool), n)])
        elif motif == 1:
            blocks.append(_inverse_pair(rng, n))
        elif motif == 2:
            blocks.append(_merging_pair(rng, n))
        elif motif == 3:
            blocks.append(_noncommuting_pair(rng, n))
        elif motif == 4 and rng.random() < 0.5:
            blocks.append(_disjoint_block(rng, n))
        else:
            blocks.append(_scattered_pair(rng, n))

    gates = "\n".join(line for block in blocks for line in block)
    return HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + gates + "\n" + _measurements(rng, n)


def _all_sources() -> list[str]:
    return [
        _build_source(seed, case_index)
        for seed in SEEDS
        for case_index in range(CASES_PER_SEED)
    ]


# --------------------------------------------------------------- case checking


def _check_case(seed: int, case_index: int) -> tuple:
    """Run every metamorphic assertion for one generated case.

    Returns an exact signature ``(source, qasm, changed, distance,
    again_qasm, again_changed)`` so the determinism test can compare whole
    runs byte-for-byte and value-for-value.
    """
    label = f"seed={seed} case={case_index}"
    try:
        source = _build_source(seed, case_index)
    except Exception as exc:  # generation failures must fail, not skip
        pytest.fail(f"{label}: generating the original failed: {exc}")
    try:
        program = parse(source)
    except Exception as exc:
        pytest.fail(f"{label}: parsing the original failed: {exc}\n{source}")

    assert MIN_QUBITS <= program.num_qubits <= MAX_QUBITS <= MAX_EQUIVALENCE_QUBITS

    try:
        _gates, changed, qasm = optimize(program)
        optimized = parse(qasm)  # the rendered qasm must re-parse verbatim
        _again_gates, again_changed, again_qasm = optimize(optimized)
    except Exception as exc:
        pytest.fail(f"{label}: optimize or re-parse failed: {exc}\n{source}")

    distance = unitary_distance(unitary_matrix(program), unitary_matrix(optimized))
    assert distance <= EQUIVALENCE_TOLERANCE, (
        f"{label}: unitary distance {distance!r} exceeds tolerance "
        f"{EQUIVALENCE_TOLERANCE!r}\n{source}\n--- optimized ---\n{qasm}"
    )
    assert measurement_layout(program) == measurement_layout(optimized), (
        f"{label}: measurement layout changed\n{source}\n--- optimized ---\n{qasm}"
    )
    assert again_qasm == qasm, f"{label}: second optimization changed the qasm"
    assert again_changed is False, f"{label}: changed is not False at the fixed point"
    return source, qasm, changed, distance, again_qasm, again_changed


# ----------------------------------------------------------------- corpus tests


@pytest.mark.parametrize(
    "seed,case_index",
    [
        pytest.param(seed, case_index, id=f"seed{seed:02d}-case{case_index}")
        for seed in SEEDS
        for case_index in range(CASES_PER_SEED)
    ],
)
def test_generated_case_is_equivalent_and_fixed(seed, case_index):
    _check_case(seed, case_index)


def test_corpus_covers_every_supported_gate_kind():
    corpus = "\n".join(_all_sources())
    for kind in ALL_GATE_KINDS:
        assert re.search(rf"(?<![a-z]){re.escape(kind)}\b", corpus), kind


def test_corpus_stays_within_equivalence_qubit_limit():
    for source in _all_sources():
        assert parse(source).num_qubits <= MAX_EQUIVALENCE_QUBITS


def test_corpus_includes_sparse_and_shuffled_measurements():
    # The generator must actually exercise the measurement-layout path:
    # at least one circuit omits a measurement and one uses a non-identity
    # qubit-to-clbit permutation.
    layouts = [measurement_layout(parse(source)) for source in _all_sources()]
    assert any(len(pairs) < width for width, pairs in layouts)
    assert any(any(qubit != clbit for qubit, clbit in pairs) for _width, pairs in layouts)


def test_same_seed_reproduces_case_sequence_and_results():
    first = [
        _check_case(seed, case_index)
        for seed in SEEDS
        for case_index in range(CASES_PER_SEED)
    ]
    second = [
        _check_case(seed, case_index)
        for seed in SEEDS
        for case_index in range(CASES_PER_SEED)
    ]
    assert second == first  # exact: sources, qasm bytes, flags and distances


# --------------------------------------------- disjoint-gate permutation oracle


def _disjoint_gate_set(rng: random.Random, n: int) -> list[str]:
    free = list(range(n))
    rng.shuffle(free)
    gates: list[str] = []
    if n >= 3 and rng.random() < 0.5:
        control, target = free.pop(), free.pop()
        kind = rng.choice(TWO_QUBIT_KINDS)
        if kind in CONTROLLED_ROTATION_KINDS:
            gates.append(
                f"{kind}({rng.choice(CONTROLLED_ANGLE_TEXTS)}) q[{control}],q[{target}];"
            )
        else:
            gates.append(f"{kind} q[{control}],q[{target}];")
    while free:
        qubit = free.pop()
        kind = rng.choice(SINGLE_QUBIT_KINDS)
        if kind in ROTATION_KINDS:
            gates.append(f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];")
        else:
            gates.append(f"{kind} q[{qubit}];")
    return gates


@pytest.mark.parametrize("seed", range(8))
def test_disjoint_gate_permutations_converge_byte_identically(seed):
    rng = random.Random(f"disjoint-permutation:{seed}")
    n = rng.randint(2, 5)
    gates = _disjoint_gate_set(rng, n)
    assert len(gates) >= 2

    measures = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n))

    def optimized_for(order: list[str]) -> tuple[str, bool, str]:
        body = "\n".join(order) + "\n"
        program = parse(HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + body + measures)
        _g, changed, qasm = optimize(program)
        reparsed = parse(qasm)
        distance = unitary_distance(unitary_matrix(program), unitary_matrix(reparsed))
        assert distance <= EQUIVALENCE_TOLERANCE
        assert measurement_layout(program) == measurement_layout(reparsed)
        return qasm, changed, body

    orders = [gates, list(reversed(gates)), sorted(gates)]
    # Add two seeded shuffles as extra distinct permutations.
    shuffled_a = gates[:]
    rng.shuffle(shuffled_a)
    shuffled_b = gates[:]
    rng.shuffle(shuffled_b)
    orders.extend([shuffled_a, shuffled_b])

    unique_orders: list[list[str]] = []
    for order in orders:
        if order not in unique_orders:
            unique_orders.append(order)
    assert len(unique_orders) >= 2

    results = {optimized_for(order)[0] for order in unique_orders}
    assert len(results) == 1, "different permutations of disjoint gates diverged"

    # The shared fixed point must itself be stable.
    canonical_qasm = results.pop()
    _, again_changed, again_qasm = optimize(parse(canonical_qasm))
    assert again_qasm == canonical_qasm
    assert again_changed is False


# ----------------------------------------------------- directed boundary cases


def _program(n: int, body: str):
    return parse(HEADER + f"qreg q[{n}];\ncreg c[{n}];\n" + body)


def _distance_between(n: int, left_body: str, right_body: str) -> float:
    return unitary_distance(unitary_matrix(_program(n, left_body)), unitary_matrix(_program(n, right_body)))


def test_single_qubit_rotations_have_2pi_period():
    for axis in ROTATION_KINDS:
        # theta + 2*pi is the same unitary up to a global phase.
        assert _distance_between(1, f"{axis}(0.3) q[0];\n", f"{axis}(0.3+2*pi) q[0];\n") <= EQUIVALENCE_TOLERANCE
        assert _distance_between(1, f"{axis}(-1.7) q[0];\n", f"{axis}(-1.7-2*pi) q[0];\n") <= EQUIVALENCE_TOLERANCE
        # A whole 2*pi rotation is identity up to a global phase and is
        # removed by the optimizer; the empty circuit stays at the fixed point.
        _, changed, qasm = optimize(_program(1, f"{axis}(2*pi) q[0];\n"))
        assert changed is True
        assert all(not line.startswith(axis) for line in qasm.splitlines())
        _, again_changed, again_qasm = optimize(parse(qasm))
        assert again_changed is False and again_qasm == qasm


def test_controlled_rotations_have_4pi_but_not_2pi_period():
    for axis in CONTROLLED_ROTATION_KINDS:
        # 4*pi period: theta + 4*pi is the same controlled unitary.
        assert _distance_between(2, f"{axis}(0.3) q[0],q[1];\n", f"{axis}(0.3+4*pi) q[0],q[1];\n") <= EQUIVALENCE_TOLERANCE
        assert _distance_between(2, f"{axis}(4*pi) q[0],q[1];\n", "\n") <= EQUIVALENCE_TOLERANCE
        # 2*pi is NOT a period: it changes the control-1 block.
        assert _distance_between(2, f"{axis}(0.3) q[0],q[1];\n", f"{axis}(0.3+2*pi) q[0],q[1];\n") > EQUIVALENCE_TOLERANCE
        assert _distance_between(2, f"{axis}(2*pi) q[0],q[1];\n", "\n") > EQUIVALENCE_TOLERANCE

        # The optimizer reflects this: 4*pi is deleted, 2*pi is kept.
        _, changed, qasm = optimize(_program(2, f"{axis}(4*pi) q[0],q[1];\n"))
        assert changed is True
        assert f"{axis}(" not in qasm
        _, kept_changed, kept_qasm = optimize(_program(2, f"{axis}(2*pi) q[0],q[1];\n"))
        assert kept_changed is False
        assert f"{axis}(6.283185307179586) q[0],q[1];" in kept_qasm
        _, again_changed, again_qasm = optimize(parse(kept_qasm))
        assert again_changed is False and again_qasm == kept_qasm


def test_angles_straddling_the_cancellation_epsilon():
    # Exactly at the epsilon the single rotation is deleted.
    _, changed, qasm = optimize(_program(1, "rx(1e-12) q[0];\n"))
    assert changed is True and "rx(" not in qasm
    # Just above it the rotation survives and the circuit is already canonical.
    _, changed, qasm = optimize(_program(1, "rx(1.0001e-12) q[0];\n"))
    assert changed is False
    assert "rx(1.0001e-12) q[0];" in qasm
    # Two sub-epsilon halves merge into an at-epsilon rotation that is deleted.
    _, changed, qasm = optimize(_program(1, "rx(5e-13) q[0];\nrx(5e-13) q[0];\n"))
    assert changed is True and "rx(" not in qasm
    # Merging just across the epsilon keeps one merged rotation. The parser
    # evaluates 5.0001e-12 - 4e-12 in binary, so compare the reparsed angle
    # numerically rather than hard-coding a decimal literal.
    _, changed, qasm = optimize(
        _program(1, "rx(5.0001e-12) q[0];\nrx(-4e-12) q[0];\n")
    )
    optimized = parse(qasm)
    rotations = [op for op in optimized.operations if op.kind == "rx"]
    assert len(rotations) == 1
    assert rotations[0].params[0] == pytest.approx(1.0001e-12, abs=1e-24)
    assert rotations[0].params[0] > 1e-12
    # The controlled epsilon deletes an exactly-epsilon controlled rotation,
    # and doing so is semantically sound on the full unitary.
    deleted = _program(2, "crx(1e-12) q[0],q[1];\n")
    _, _changed, deleted_qasm = optimize(deleted)
    assert "crx(" not in deleted_qasm
    assert _distance_between(2, "crx(1e-12) q[0],q[1];\n", "\n") <= EQUIVALENCE_TOLERANCE


def test_swap_operands_are_canonicalized_and_pair_cancels():
    # swap is symmetric: both operand orders render the same canonical line,
    # and operand-order normalization alone does not count as a change.
    forward = _program(2, "swap q[0],q[1];\n")
    reversed_order = _program(2, "swap q[1],q[0];\n")
    _, changed_f, qasm_f = optimize(forward)
    _, changed_r, qasm_r = optimize(reversed_order)
    assert qasm_f == qasm_r
    assert "swap q[0],q[1];" in qasm_f
    assert changed_f is False and changed_r is False
    assert _distance_between(2, "swap q[0],q[1];\n", "swap q[1],q[0];\n") <= EQUIVALENCE_TOLERANCE
    # A swap pair cancels even when the two applications write operands in
    # opposite orders.
    _, changed, qasm = optimize(_program(2, "swap q[0],q[1];\nswap q[1],q[0];\n"))
    assert changed is True and "swap" not in qasm
    # Canonicalization must not confuse swap with a directed cx.
    assert _distance_between(2, "swap q[0],q[1];\n", "cx q[0],q[1];\n") > EQUIVALENCE_TOLERANCE
    # cx itself is not symmetric: reversed control/target is a different gate.
    _, cx_changed, cx_qasm = optimize(_program(2, "cx q[1],q[0];\n"))
    assert cx_changed is False and "cx q[1],q[0];" in cx_qasm
    assert _distance_between(2, "cx q[0],q[1];\n", "cx q[1],q[0];\n") > EQUIVALENCE_TOLERANCE


def test_boundary_circuits_preserve_sparse_measurement_layout():
    body = "crz(2*pi) q[0],q[1];\nrx(0.3+2*pi) q[2];\nswap q[1],q[0];\n"
    measures = (
        "measure q[2] -> c[0];\nmeasure q[0] -> c[2];\n"
    )
    program = _program(3, body + measures)
    _g, _changed, qasm = optimize(program)
    optimized = parse(qasm)
    assert measurement_layout(program) == measurement_layout(optimized)
    assert unitary_distance(unitary_matrix(program), unitary_matrix(optimized)) <= EQUIVALENCE_TOLERANCE


# ------------------------------------------------- oracle effectiveness guards


def test_full_unitary_oracle_catches_relative_phase_dropped_by_zero_state_check():
    # Acting on |0> alone, rz(pi) only contributes a removable phase; the
    # |1> branch carries the opposite phase. A zero-state-only comparison
    # would call rz(pi) identical to the empty circuit, but the full unitary
    # distance is maximal.
    state_zero = _program(1, "rz(pi) q[0];\n")
    vector = [column[0] for column in unitary_matrix(state_zero)]
    assert abs(abs(vector[0]) - 1.0) < 1e-15  # |0> stays |0> ...
    assert abs(unitary_matrix(_program(1, "\n"))[0][0]) == 1.0
    assert _distance_between(1, "rz(pi) q[0];\n", "\n") > EQUIVALENCE_TOLERANCE
    # The optimizer correctly keeps the gate.
    _, changed, qasm = optimize(state_zero)
    assert changed is False and "rz(" in qasm


def test_full_unitary_oracle_catches_controlled_branch_errors():
    # crz(2*pi) leaves |00> exactly untouched, so evolving only the zero
    # state cannot distinguish it from the identity; the control-1 block
    # differs and the full-unitary distance is 1.
    controlled = _program(2, "crz(2*pi) q[0],q[1];\n")
    zero_column = [column[0] for column in unitary_matrix(controlled)]
    assert zero_column == [1 + 0j, 0j, 0j, 0j]
    assert _distance_between(2, "crz(2*pi) q[0],q[1];\n", "\n") > EQUIVALENCE_TOLERANCE
    # Reversing a cx's control and target likewise leaves the zero state
    # fixed but is plainly a different unitary.
    assert _distance_between(2, "cx q[0],q[1];\n", "cx q[1],q[0];\n") > EQUIVALENCE_TOLERANCE
