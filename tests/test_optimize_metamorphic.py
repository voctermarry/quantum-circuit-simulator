"""Deterministic metamorphic tests for the canonical optimizer.

These tests do not hand-pick equivalence pairs: a fixed-seed generator
constructs small legal OpenQASM programs on one to five qubits and checks
three metamorphic properties on every generated case:

1. **Semantics preservation** -- the full pre-measurement unitary of the
   original program and of the re-parsed optimized QASM differ by no more
   than the publicly exported ``EQUIVALENCE_TOLERANCE``; the classical
   register width and the qubit-to-clbit measurement mapping are identical.
2. **Canonical stability** -- optimizing the optimized QASM again returns
   byte-identical QASM with ``changed is False``, and two source programs
   that differ only in the ordering of a block of gates on pairwise
   disjoint qubits optimize to byte-identical output.
3. **Determinism** -- rebuilding the cases from the same seed reproduces
   every source text, and rerunning every assertion reproduces its result.

Targeted boundary tests pin single-qubit ``2*pi`` periodicity, controlled
rotation ``4*pi`` periodicity (a ``2*pi`` shift is *not* a global phase on
a controlled gate), angles immediately around the deletion threshold and
swap-operand canonicalization. All comparisons use full unitary matrices
rather than the all-zero initial state, so control-branch phases and global
phases cannot hide.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, replace

import pytest

from quantum_circuit.equivalence import (
    EQUIVALENCE_TOLERANCE,
    MAX_EQUIVALENCE_QUBITS,
    measurement_layout,
    unitary_distance,
)
from quantum_circuit.openqasm import parse
from quantum_circuit.optimizer import optimize
from quantum_circuit.simulator import simulate_state_vector, unitary_matrix

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

SINGLE_QUBIT_GATES = ("x", "h", "y", "z", "s", "sdg", "t", "tdg")
ROTATION_GATES = ("rx", "ry", "rz")
CONTROLLED_ROTATION_GATES = ("crx", "cry", "crz")
TWO_QUBIT_GATES = ("cx", "cz", "swap")
ALL_GATES = frozenset(
    SINGLE_QUBIT_GATES
    + ROTATION_GATES
    + CONTROLLED_ROTATION_GATES
    + TWO_QUBIT_GATES
)

# Finite angle expressions legal in the parsed OpenQASM subset. Decimal
# literals and pi-expressions are all finite; controlled angles include
# 2*pi, which must be kept rather than folded away.
SINGLE_ANGLE_TEXTS = (
    "0.3", "-0.3", "0.5", "-1.25", "pi/2", "-pi/2", "pi/4",
    "pi/3", "-0.9", "2*pi",
)
CONTROLLED_ANGLE_TEXTS = (
    "0.3", "-0.3", "pi/2", "-pi/2", "pi/4", "0.75", "-1.1", "2*pi",
)

# Inverse of a gate that participates in adjacent-pair cancellation.
_INVERSE = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}

SEEDS = (101, 202, 303)
CASES_PER_SEED = 4
MAX_GENERATED_QUBITS = 5
assert MAX_GENERATED_QUBITS <= MAX_EQUIVALENCE_QUBITS


# ------------------------------------------------------------- generator


@dataclass(frozen=True)
class GeneratedCase:
    """One generated circuit, with a disjoint-block permutation variant."""

    seed: int
    index: int
    num_qubits: int
    prefix: tuple[str, ...]          # gate lines before the shuffle block
    block_a: tuple[str, ...]         # one permutation of the disjoint block
    block_b: tuple[str, ...]         # another permutation of the same gates
    measurements: tuple[tuple[int, int], ...]  # (qubit, clbit) pairs

    def _assemble(self, block: tuple[str, ...]) -> str:
        lines = [
            HEADER,
            f"qreg q[{self.num_qubits}];",
            f"creg c[{self.num_qubits}];",
        ]
        lines.extend(self.prefix)
        lines.extend(block)
        lines.extend(f"measure q[{q}] -> c[{c}];" for q, c in self.measurements)
        return "\n".join(lines) + "\n"

    @property
    def source(self) -> str:
        return self._assemble(self.block_a)

    @property
    def alternative(self) -> str:
        return self._assemble(self.block_b)


def _distinct_qubits(rng: random.Random, n: int) -> tuple[int, int]:
    """Return two distinct qubit indices (valid two-qubit operands)."""
    a = rng.randrange(n)
    b = rng.randrange(n - 1)
    if b >= a:
        b += 1
    return a, b


def _single_line(rng: random.Random, kind: str, qubit: int) -> str:
    if kind in ROTATION_GATES:
        return f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];"
    return f"{kind} q[{qubit}];"


def _rotation_pair_chunk(
    rng: random.Random, n: int, kind: str
) -> tuple[list[str], frozenset[str]]:
    """Two same-axis single-qubit rotations, optionally split by a gate on
    a disjoint qubit, which the merge pass must see through."""
    qubit = rng.randrange(n)
    lines = [f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];"]
    if n >= 2 and rng.random() < 0.7:
        other = rng.randrange(n - 1)
        if other >= qubit:
            other += 1
        middle_kind = next(GATE_POOL["filler"])
        lines.append(_single_line(rng, middle_kind, other))
    lines.append(f"{kind}({rng.choice(SINGLE_ANGLE_TEXTS)}) q[{qubit}];")
    return lines, frozenset({kind})


def _controlled_pair_chunk(
    rng: random.Random, n: int, kind: str
) -> tuple[list[str], frozenset[str]]:
    """Two same-axis controlled rotations with identical operands."""
    control, target = _distinct_qubits(rng, n)
    lines = [
        f"{kind}({rng.choice(CONTROLLED_ANGLE_TEXTS)}) "
        f"q[{control}],q[{target}];"
    ]
    # A middle gate must be disjoint from both operands to be see-through.
    if n >= 3 and rng.random() < 0.7:
        outside = [q for q in range(n) if q not in (control, target)]
        middle_kind = next(GATE_POOL["filler"])
        lines.append(_single_line(rng, middle_kind, rng.choice(outside)))
    lines.append(
        f"{kind}({rng.choice(CONTROLLED_ANGLE_TEXTS)}) "
        f"q[{control}],q[{target}];"
    )
    return lines, frozenset({kind})


def _cancel_single_chunk(
    rng: random.Random, n: int, kind: str
) -> tuple[list[str], frozenset[str]]:
    """A cancellable inverse pair on one qubit with a disjoint gate between."""
    qubit = rng.randrange(n)
    closing = _INVERSE.get(kind, kind)
    lines = [f"{kind} q[{qubit}];"]
    used = {kind}
    if n >= 2 and rng.random() < 0.8:
        other = rng.randrange(n - 1)
        if other >= qubit:
            other += 1
        middle_kind = next(GATE_POOL["filler"])
        lines.append(_single_line(rng, middle_kind, other))
        used.add(middle_kind)
    lines.append(f"{closing} q[{qubit}];")
    used.add(closing)
    return lines, frozenset(used)


def _cancel_two_qubit_chunk(
    rng: random.Random, n: int, kind: str, *, reverse_swap: bool = False
) -> tuple[list[str], frozenset[str]]:
    """A self-inverse cx/cz/swap pair; swap may use reversed operands on
    the closing gate, which operand canonicalization must reconcile."""
    control, target = _distinct_qubits(rng, n)
    lines = [f"{kind} q[{control}],q[{target}];"]
    used = {kind}
    if n >= 3 and rng.random() < 0.8:
        outside = [q for q in range(n) if q not in (control, target)]
        middle_kind = next(GATE_POOL["filler"])
        lines.append(_single_line(rng, middle_kind, rng.choice(outside)))
        used.add(middle_kind)
    if kind == "swap" and (reverse_swap or rng.random() < 0.5):
        lines.append(f"{kind} q[{target}],q[{control}];")
    else:
        lines.append(f"{kind} q[{control}],q[{target}];")
    return lines, frozenset(used)


def _shared_qubit_chain_chunk(
    rng: random.Random, n: int
) -> tuple[list[str], frozenset[str]]:
    """Gates that share a qubit and must not be reordered: single -
    two-qubit - single, where the middle gate blocks the outer pair from
    cancelling even when the outer gates match."""
    shared, other = _distinct_qubits(rng, n)
    first_kind = next(GATE_POOL["filler"])
    middle_kind = next(GATE_POOL["twoq"])
    last_kind = next(GATE_POOL["filler"])
    if rng.random() < 0.5:
        middle = f"{middle_kind} q[{shared}],q[{other}];"
    else:
        middle = f"{middle_kind} q[{other}],q[{shared}];"
    lines = [
        _single_line(rng, first_kind, shared),
        middle,
        _single_line(rng, last_kind, shared),
    ]
    return lines, frozenset({first_kind, middle_kind, last_kind})


def _shuffle_block(
    rng: random.Random, n: int
) -> tuple[tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Gates on pairwise disjoint qubits, returned in two different orders.

    Every qubit carries at most one gate, so the gates commute; the
    optimizer's canonical order must converge regardless of the order.
    """
    remaining = list(range(n))
    rng.shuffle(remaining)
    plan: list[str] = []
    used: set[str] = set()
    # A two-qubit gate needs two qubits and leaves the rest carrying
    # single-qubit gates; it is only worth adding when at least one qubit
    # remains, so for two-qubit registers the block is always two distinct
    # single-qubit gates (guaranteeing a genuine permutation there).
    if len(remaining) >= 3 and rng.random() < 0.8:
        a = remaining.pop()
        b = remaining.pop()
        kind = next(GATE_POOL["twoq"])
        plan.append(f"{kind} q[{a}],q[{b}];")
        used.add(kind)
    while remaining:
        qubit = remaining.pop()
        kind = next(GATE_POOL["disjoint"])
        plan.append(_single_line(rng, kind, qubit))
        used.add(kind)

    first = plan.copy()
    rng.shuffle(first)
    second = plan.copy()
    rng.shuffle(second)
    if second == first and len(second) > 1:
        second.reverse()
    return tuple(first), tuple(second), frozenset(used)


def _measurement_pairs(rng: random.Random, n: int) -> tuple[tuple[int, int], ...]:
    """A sparse, shuffled qubit-to-clbit mapping.

    A random subset of qubits is measured onto a same-sized random subset
    of classical bits; emission order is shuffled as well. The empty
    mapping is a legal outcome.
    """
    qubits = list(range(n))
    rng.shuffle(qubits)
    measured = qubits[: rng.randrange(n + 1)]
    clbits = list(range(n))
    rng.shuffle(clbits)
    pairs = list(zip(measured, clbits[: len(measured)]))
    rng.shuffle(pairs)
    return tuple(pairs)


# Iterators are created per seed so a re-generated sequence draws from the
# same positions; each entry is a cycle over its gate palette.
GATE_POOL = {}


def _reset_pools() -> None:
    GATE_POOL.clear()
    GATE_POOL["single"] = itertools.cycle(SINGLE_QUBIT_GATES)
    GATE_POOL["rotation"] = itertools.cycle(ROTATION_GATES)
    GATE_POOL["crotation"] = itertools.cycle(CONTROLLED_ROTATION_GATES)
    GATE_POOL["twoq"] = itertools.cycle(TWO_QUBIT_GATES)
    GATE_POOL["filler"] = itertools.cycle(SINGLE_QUBIT_GATES)
    GATE_POOL["disjoint"] = itertools.cycle(
        SINGLE_QUBIT_GATES + ROTATION_GATES
    )


def _build_case(rng: random.Random, seed: int, index: int) -> GeneratedCase:
    # Spread register sizes deterministically across the four cases: small,
    # large, anything 1-5, and guaranteed at least two qubits.
    if index == 0:
        n = 1 + rng.randrange(2)
    elif index == 1:
        n = 3 + rng.randrange(3)
    elif index == 2:
        n = 1 + rng.randrange(MAX_GENERATED_QUBITS)
    else:
        n = max(2, 1 + rng.randrange(MAX_GENERATED_QUBITS))

    prefix: list[str] = []
    chunk_count = 6 + rng.randrange(3)
    for _ in range(chunk_count):
        choices = [0, 1, 2, 3, 4] if n >= 2 else [0, 2]
        kind = rng.choice(choices)
        if kind == 0:
            chunk, _ = _cancel_single_chunk(rng, n, next(GATE_POOL["single"]))
        elif kind == 1:
            chunk, _ = _cancel_two_qubit_chunk(rng, n, next(GATE_POOL["twoq"]))
        elif kind == 2:
            chunk, _ = _rotation_pair_chunk(rng, n, next(GATE_POOL["rotation"]))
        elif kind == 3:
            chunk, _ = _controlled_pair_chunk(
                rng, n, next(GATE_POOL["crotation"])
            )
        else:
            chunk, _ = _shared_qubit_chain_chunk(rng, n)
        prefix.extend(chunk)

    block_a, block_b, _ = _shuffle_block(rng, n)
    measurements = _measurement_pairs(rng, n)
    return GeneratedCase(seed, index, n, tuple(prefix), block_a, block_b, measurements)


def _gate_kinds(case: GeneratedCase) -> frozenset[str]:
    kinds = set()
    for line in case.prefix + case.block_a:
        kinds.add(line.split(None, 1)[0].split("(", 1)[0])
    return frozenset(kinds)


def _coverage_chunk(
    rng: random.Random, n: int, gate: str
) -> tuple[str, ...]:
    """Build a chunk that deterministically exercises *gate*."""
    if gate in SINGLE_QUBIT_GATES:
        lines, _ = _cancel_single_chunk(rng, n, gate)
    elif gate in ROTATION_GATES:
        lines, _ = _rotation_pair_chunk(rng, n, gate)
    elif gate in CONTROLLED_ROTATION_GATES:
        lines, _ = _controlled_pair_chunk(rng, n, gate)
    else:
        lines, _ = _cancel_two_qubit_chunk(rng, n, gate, reverse_swap=True)
    return tuple(lines)


def generate_cases(seed: int) -> tuple[GeneratedCase, ...]:
    """Reproducibly build the seed's case sequence; never skips a case."""
    rng = random.Random(seed)
    _reset_pools()
    cases: list[GeneratedCase] = []
    for index in range(CASES_PER_SEED):
        try:
            cases.append(_build_case(rng, seed, index))
        except Exception as exc:  # generator defect: fail loudly with location
            pytest.fail(f"generator failure at seed={seed}, case={index}: {exc!r}")

    # Guarantee whole-gate-set coverage deterministically: append a chunk
    # for any gate the random draw did not reach. Two-qubit gates land on a
    # case with at least two qubits.
    used = frozenset().union(*(_gate_kinds(case) for case in cases))
    two_qubit_index = next(i for i, c in enumerate(cases) if c.num_qubits >= 2)
    for gate in sorted(ALL_GATES - used):
        target_index = (
            0
            if gate in SINGLE_QUBIT_GATES + ROTATION_GATES
            else two_qubit_index
        )
        target = cases[target_index]
        chunk = _coverage_chunk(rng, target.num_qubits, gate)
        cases[target_index] = replace(
            target, prefix=target.prefix + chunk
        )
    return tuple(cases)


# ------------------------------------------------------------- verification


@dataclass(frozen=True)
class CheckResult:
    distance: float
    alt_distance: float
    optimized_qasm: str
    fixed_point_qasm: str
    fixed_point_changed: bool


def _verify_case(case: GeneratedCase) -> CheckResult:
    program = parse(case.source)
    _, _, qasm = optimize(program)
    optimized = parse(qasm)

    # Full unitary distance, not a comparison on the zero initial state.
    distance = unitary_distance(
        unitary_matrix(program), unitary_matrix(optimized)
    )
    assert distance <= EQUIVALENCE_TOLERANCE, (
        f"unitary distance {distance!r} exceeds tolerance "
        f"{EQUIVALENCE_TOLERANCE!r}"
    )

    # Classical register width plus the qubit-to-clbit mapping exactly.
    assert measurement_layout(program) == measurement_layout(optimized)
    assert optimized.num_clbits == program.num_clbits

    # The disjoint-block permutation is the same transformation and
    # converges to byte-identical canonical QASM.
    alternative = parse(case.alternative)
    alt_distance = unitary_distance(
        unitary_matrix(program), unitary_matrix(alternative)
    )
    assert alt_distance <= EQUIVALENCE_TOLERANCE
    _, _, alt_qasm = optimize(alternative)
    assert alt_qasm == qasm, "canonical output depends on disjoint-gate order"

    # Fixed point: a second optimization pass changes nothing.
    _, changed_again, qasm_again = optimize(optimized)
    assert qasm_again == qasm, "optimized QASM is not a byte-identical fixed point"
    assert changed_again is False, "changed must be False when re-optimizing"

    return CheckResult(distance, alt_distance, qasm, qasm_again, changed_again)


def _verify_with_location(case: GeneratedCase) -> CheckResult:
    try:
        return _verify_case(case)
    except Exception as exc:
        pytest.fail(
            f"metamorphic case failed (seed={case.seed}, case={case.index}): "
            f"{exc!r}\n--- generated source ---\n{case.source}"
        )


# ------------------------------------------------------------- generated suite


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_cases_preserve_semantics_and_reach_fixed_point(seed):
    cases = generate_cases(seed)
    # Rebuilding from the same seed must reproduce every text and mapping.
    rebuilt = generate_cases(seed)
    assert rebuilt == cases
    assert [c.source for c in rebuilt] == [c.source for c in cases]
    assert [c.alternative for c in rebuilt] == [c.alternative for c in cases]

    for index, case in enumerate(cases):
        assert 1 <= case.num_qubits <= MAX_GENERATED_QUBITS
        # Running the assertions twice gives identical results.
        first = _verify_with_location(case)
        second = _verify_with_location(case)
        assert first == second, (
            f"assertion results not reproducible (seed={seed}, case={index})"
        )


def test_generated_corpus_covers_every_gate_and_real_permutations():
    seen = set()
    nontrivial_blocks = 0
    for seed in SEEDS:
        for case in generate_cases(seed):
            seen |= _gate_kinds(case)
            if len(case.block_a) >= 2 and case.block_a != case.block_b:
                nontrivial_blocks += 1
    assert seen == ALL_GATES, f"generator never exercised: {sorted(ALL_GATES - seen)}"
    assert nontrivial_blocks > 0


# ------------------------------------------------------------- boundary: 2*pi


@pytest.mark.parametrize("axis", ROTATION_GATES)
def test_single_qubit_rotation_has_two_pi_period(axis):
    base = parse(HEADER + f"qreg q[1];\ncreg c[1];\n{axis}(0.6) q[0];\n")
    identity = parse(HEADER + "qreg q[1];\ncreg c[1];\n")

    for expression in ("0.6+2*pi", "0.6-2*pi", "0.6+4*pi"):
        shifted = parse(
            HEADER + f"qreg q[1];\ncreg c[1];\n{axis}({expression}) q[0];\n"
        )
        gates, changed, qasm = optimize(shifted)
        # One normalized rotation remains; pure angle normalization is not
        # reported as a structural change.
        assert len(gates) == 1
        assert changed is False
        reparsed = parse(qasm)
        assert (
            unitary_distance(unitary_matrix(shifted), unitary_matrix(reparsed))
            <= EQUIVALENCE_TOLERANCE
        )
        assert (
            unitary_distance(unitary_matrix(base), unitary_matrix(reparsed))
            <= EQUIVALENCE_TOLERANCE
        )
        # Fixed point of the normalized literal.
        assert optimize(reparsed)[2] == qasm
        assert optimize(reparsed)[1] is False

    # A full 2*pi turn is the identity up to a global phase, which only the
    # full unitary comparison acknowledges.
    full_turn = parse(
        HEADER + f"qreg q[1];\ncreg c[1];\n{axis}(2*pi) q[0];\n"
    )
    _, changed, qasm = optimize(full_turn)
    assert changed is True
    assert _gate_lines(qasm) == []
    assert (
        unitary_distance(unitary_matrix(full_turn), unitary_matrix(identity))
        <= EQUIVALENCE_TOLERANCE
    )


# ------------------------------------------------------ boundary: 4*pi / 2*pi


@pytest.mark.parametrize("axis", CONTROLLED_ROTATION_GATES)
def test_controlled_rotation_has_four_pi_period(axis):
    base = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\n{axis}(0.6) q[0],q[1];\n"
    )
    shifted = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\n{axis}(0.6+4*pi) q[0],q[1];\n"
    )
    gates, changed, qasm = optimize(shifted)
    assert len(gates) == 1
    assert changed is False
    reparsed = parse(qasm)
    assert (
        unitary_distance(unitary_matrix(base), unitary_matrix(reparsed))
        <= EQUIVALENCE_TOLERANCE
    )
    assert optimize(reparsed)[2] == qasm

    # 4*pi alone is the identity and must be deleted.
    full = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\n{axis}(4*pi) q[0],q[1];\n"
    )
    _, changed, qasm = optimize(full)
    assert changed is True
    assert _gate_lines(qasm) == []
    identity = parse(HEADER + "qreg q[2];\ncreg c[2];\n")
    assert (
        unitary_distance(unitary_matrix(full), unitary_matrix(identity))
        <= EQUIVALENCE_TOLERANCE
    )


@pytest.mark.parametrize("axis", CONTROLLED_ROTATION_GATES)
def test_controlled_rotation_two_pi_is_not_a_global_phase(axis):
    base = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\n{axis}(0.6) q[0],q[1];\n"
    )
    shifted = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\n{axis}(0.6+2*pi) q[0],q[1];\n"
    )
    # The 2*pi shift flips only the control-1 block: full matrices are
    # orthogonal even though the all-zero input state never sees it.
    distance = unitary_distance(
        unitary_matrix(base), unitary_matrix(shifted)
    )
    assert distance > EQUIVALENCE_TOLERANCE
    assert distance > 0.5

    # The optimizer keeps a bare 2*pi controlled turn rather than deleting
    # it, and the kept gate is genuinely non-trivial on the full matrix.
    kept = parse(
        HEADER + f"qreg q[2];\ncreg c[2];\n{axis}(2*pi) q[0],q[1];\n"
    )
    gates, changed, qasm = optimize(kept)
    assert len(gates) == 1 and changed is False
    identity = parse(HEADER + "qreg q[2];\ncreg c[2];\n")
    assert (
        unitary_distance(unitary_matrix(parse(qasm)), unitary_matrix(identity))
        > EQUIVALENCE_TOLERANCE
    )
    assert optimize(parse(qasm))[2] == qasm


def test_zero_initial_state_cannot_detect_control_branch_phase():
    # crz(2*pi) leaves |00> exactly unchanged (control is 0), yet its full
    # unitary differs from the identity: the control-1 block picks up -1.
    program = parse(
        HEADER + "qreg q[2];\ncreg c[2];\ncrz(2*pi) q[0],q[1];\n"
    )
    state = simulate_state_vector(program)
    assert state[0] == 1 + 0j
    assert all(amplitude == 0j for amplitude in state[1:])
    identity = parse(HEADER + "qreg q[2];\ncreg c[2];\n")
    assert (
        unitary_distance(unitary_matrix(program), unitary_matrix(identity))
        > EQUIVALENCE_TOLERANCE
    )


# ------------------------------------------------------- boundary: threshold


def test_angles_immediately_around_cancellation_threshold():
    # At or inside the epsilon interval the rotation is deleted.
    for expression in ("1e-12", "-1e-12", "5e-13", "-5e-13"):
        program = parse(
            HEADER + f"qreg q[1];\ncreg c[1];\nrx({expression}) q[0];\n"
        )
        _, changed, qasm = optimize(program)
        assert changed is True
        assert _gate_lines(qasm) == []
        assert parse(qasm).num_qubits == 1

    # Just outside the interval it survives, even though an equivalence
    # check would already consider it identity: the deletion threshold is
    # tighter than the comparison tolerance.
    for expression in ("1.5e-12", "1.0000000000001e-12"):
        program = parse(
            HEADER + f"qreg q[1];\ncreg c[1];\nrx({expression}) q[0];\n"
        )
        _, changed, qasm = optimize(program)
        assert changed is False
        lines = _gate_lines(qasm)
        assert len(lines) == 1 and lines[0].startswith("rx")
        assert optimize(parse(qasm))[2] == qasm

    # Two above-epsilon pieces merge into one surviving rotation whose
    # angle is still outside the deletion interval (a single sub-epsilon
    # piece, by contrast, is dropped on sight before any merge happens).
    merged = parse(
        HEADER + "qreg q[1];\ncreg c[1];\nrx(1.05e-12) q[0];\nrx(1.05e-12) q[0];\n"
    )
    _, changed, qasm = optimize(merged)
    assert changed is True
    lines = _gate_lines(qasm)
    assert len(lines) == 1 and lines[0].startswith("rx")

    # A sub-epsilon piece cancels against an above-epsilon one and leaves
    # only the surviving difference.
    partial = parse(
        HEADER + "qreg q[1];\ncreg c[1];\nrx(5e-13) q[0];\nrx(1.5e-12) q[0];\n"
    )
    _, changed, qasm = optimize(partial)
    assert changed is True
    assert _gate_lines(qasm) == ["rx(1.5e-12) q[0];"]

    # Two sub-epsilon pieces that sum to zero vanish entirely.
    cancel = parse(
        HEADER + "qreg q[1];\ncreg c[1];\nrx(5e-13) q[0];\nrx(-5e-13) q[0];\n"
    )
    _, changed, qasm = optimize(cancel)
    assert changed is True and _gate_lines(qasm) == []

    # The same threshold governs controlled rotations.
    tiny = parse(
        HEADER + "qreg q[2];\ncreg c[2];\ncrz(5e-13) q[0],q[1];\n"
    )
    _, changed, qasm = optimize(tiny)
    assert changed is True and "crz" not in qasm
    kept = parse(
        HEADER + "qreg q[2];\ncreg c[2];\ncrz(1.5e-12) q[0],q[1];\n"
    )
    _, changed, qasm = optimize(kept)
    assert changed is False and "crz(1.5e-12)" in qasm

    # Every deletion/merge keeps the full unitary within tolerance.
    for source in (merged, partial, cancel, tiny, kept):
        _, _, qasm = optimize(source)
        assert (
            unitary_distance(
                unitary_matrix(source), unitary_matrix(parse(qasm))
            )
            <= EQUIVALENCE_TOLERANCE
        )


# ----------------------------------------------------- boundary: swap operands


def test_swap_operands_are_canonicalized():
    ascending = parse(
        HEADER + "qreg q[2];\ncreg c[2];\nswap q[0],q[1];\n"
    )
    reversed_order = parse(
        HEADER + "qreg q[2];\ncreg c[2];\nswap q[1],q[0];\n"
    )
    # Both textual orders name the same gate: no structural change, but the
    # rendered operands are canonical.
    gates, changed, qasm = optimize(reversed_order)
    assert changed is False
    assert _gate_lines(qasm) == ["swap q[0],q[1];"]
    assert (
        unitary_distance(
            unitary_matrix(ascending), unitary_matrix(reversed_order)
        )
        == 0.0
    )
    assert optimize(parse(qasm))[2] == qasm

    # A pair written with opposite operand orders still cancels.
    pair = parse(
        HEADER
        + "qreg q[2];\ncreg c[2];\nswap q[0],q[1];\nswap q[1],q[0];\n"
    )
    _, changed, qasm = optimize(pair)
    assert changed is True and _gate_lines(qasm) == []

    # Inside a larger circuit the reversed swap pair cancels through the
    # disjoint h gate, which in turn lets the outer cx pair cancel.
    nested = parse(
        HEADER
        + "qreg q[3];\ncreg c[3];\n"
        + "cx q[0],q[1];\nswap q[1],q[0];\nh q[2];\n"
        + "swap q[0],q[1];\ncx q[0],q[1];\n"
    )
    _, changed, qasm = optimize(nested)
    assert changed is True
    assert _gate_lines(qasm) == ["h q[2];"]
    assert (
        unitary_distance(
            unitary_matrix(nested), unitary_matrix(parse(qasm))
        )
        <= EQUIVALENCE_TOLERANCE
    )


# ------------------------------------------------------------------ helpers


def _gate_lines(qasm: str) -> list[str]:
    return [
        line
        for line in qasm.splitlines()
        if not line.startswith(("OPENQASM", "include", "qreg", "creg", "measure"))
    ]
