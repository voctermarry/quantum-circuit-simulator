"""Deterministic canonical simplification for the OpenQASM 2.0 subset.

The optimizer rewrites the pre-measurement gate sequence into a canonical
form while preserving the circuit's transformation up to a global phase and
keeping the measurement layout (qubit-to-clbit mapping) intact:

* adjacent ``x``/``x``, ``h``/``h``, ``y``/``y``, ``z``/``z`` and
  ``swap``/``swap`` pairs, ``cx``/``cz`` pairs with identical control and
  target, and inverse pairs ``s``/``sdg`` and ``t``/``tdg`` on the same
  qubit cancel ("adjacent" allows intervening gates that act on disjoint
  qubits or commute with the moving gate);
* consecutive same-axis ``rx``/``ry``/``rz`` rotations on one qubit merge
  into a single rotation, as do consecutive same-axis ``crx``/``cry``/``crz``
  rotations with identical control and target;
* rotations whose normalized angle has magnitude at most ``1e-12`` are
  deleted;
* gates are moved past each other whenever they act on disjoint qubits or
  fall under the recognized same-qubit commutation relations, until the
  sequence is sorted by a stable key (minimum qubit, maximum qubit, gate
  name, parameter text and operand order).

The recognized commutation relations are exactly the ones decidable from
the gate semantics: ``x`` commutes with ``rx`` and ``y`` with ``ry`` on the
same qubit, and the computational-basis diagonal gates ``z``, ``s``,
``sdg``, ``t``, ``tdg``, ``rz``, ``cz`` and ``crz`` commute with each other
whenever they share a qubit. Cancellation, inverse cancellation and
rotation merging therefore see through any number of commuting gates, so
e.g. two ``x`` gates separated by an ``rx`` still cancel. Every other
shared-qubit combination keeps its dependency order, and ``cz``/``crz``
keep the operand direction written in the source.

Simplification passes repeat until the sequence reaches a fixed point, so
inputs that differ only in the ordering of independent or commuting gates
produce byte-identical output. Single-qubit rotation angles are normalized
to ``(-pi, pi]`` (rendering a value at either boundary as ``pi`` when it is
positive). Controlled rotations are normalized to ``(-2*pi, 2*pi]`` instead:
their angle is only ``4*pi``-periodic (a ``2*pi`` shift is *not* a global
phase of the controlled unitary), so ``2*pi`` is kept rather than deleted.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .gates import (
    COMMUTING_PAIRS as _REGISTRY_COMMUTING,
    CONTROLLED_ROTATION_GATES,
    DIAGONAL_GATES,
    GATE_SPECS,
    INVERSE_PAIRS as _REGISTRY_INVERSE_PAIRS,
    ROTATION_GATES,
    SELF_INVERSE_GATES,
    Operation,
    format_angle,
    gate_qasm,
)
from .openqasm import Program

# Rotations this small (after normalization to (-pi, pi]) act as the
# identity within floating-point precision and are removed.
_ANGLE_EPSILON = 1e-12

_ROTATIONS = ROTATION_GATES
_CONTROLLED_ROTATIONS = CONTROLLED_ROTATION_GATES
_ANY_ROTATIONS = frozenset(_ROTATIONS + _CONTROLLED_ROTATIONS)

# Rotation gates that commute with themselves (the diagonal ones, rz and
# crz): any multiset of them must merge to one canonical angle regardless
# of written order, so they accumulate sorted constituent angles instead of
# summing sequentially. Other rotations only ever merge in dependency
# order and keep the sequential sum.
_CANONICAL_SUM_ROTATIONS = frozenset(
    name for name in _ANY_ROTATIONS if name in DIAGONAL_GATES
)

# Backwards-compatible private alias: the DSL serializer used to take its
# angle rendering from here; the single rendering rule now lives in the
# gate-semantics registry.
_format_angle = format_angle


@dataclass(frozen=True)
class Gate:
    """A pre-measurement gate with its canonical (normalized) angle."""

    kind: str
    qubits: tuple[int, ...]
    angle: float | None = None
    # Constituent angles of a merged diagonal rotation (rz/crz), kept
    # sorted so the summed angle depends only on the multiset of merged
    # rotations, never on the order they were written or merged in.
    # Excluded from equality: it is summation state, not gate identity.
    parts: tuple[float, ...] | None = field(default=None, compare=False)


def normalize_angle(angle: float) -> float:
    """Reduce *angle* modulo ``2*pi`` into the interval ``(-pi, pi]``.

    Angles already inside the interval are returned untouched so ordinary
    values (e.g. ``0.7``) are not perturbed by floating-point modulo.
    """
    if -math.pi < angle <= math.pi:
        return angle
    reduced = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if reduced == -math.pi:
        return math.pi
    return reduced


def normalize_controlled_angle(angle: float) -> float:
    """Reduce a controlled-rotation angle modulo ``4*pi`` into ``(-2*pi, 2*pi]``.

    A controlled rotation by ``theta + 2*pi`` differs from one by ``theta``
    by more than a global phase (the unshifted rotation picks up a sign on
    the control-1 block only), so the period is ``4*pi`` and ``2*pi`` itself
    is kept. Angles already inside the interval are returned untouched.
    """
    if -2.0 * math.pi < angle <= 2.0 * math.pi:
        return angle
    reduced = (angle + 2.0 * math.pi) % (4.0 * math.pi) - 2.0 * math.pi
    if reduced == -2.0 * math.pi:
        return 2.0 * math.pi
    return reduced


def _gate_from_operation(op: Operation) -> Gate:
    spec = GATE_SPECS[op.kind]
    if spec.symmetric:
        # swap is symmetric in its operands; canonicalize the operand order
        # so swap q[a],q[b] and swap q[b],q[a] share one canonical form.
        return Gate(op.kind, tuple(sorted(op.targets)))
    if spec.axis is not None:
        period = spec.rotation_period
        assert period is not None
        normalize = normalize_controlled_angle if period == 4.0 * math.pi else normalize_angle
        angle = normalize(op.params[0])
        parts = (angle,) if op.kind in _CANONICAL_SUM_ROTATIONS else None
        return Gate(op.kind, tuple(op.targets), angle, parts)
    return Gate(op.kind, tuple(op.targets))


def _sort_key(gate: Gate) -> tuple[int, int, str, str, tuple[int, ...]]:
    # Ties that survive every field (identical gates) keep their relative
    # order via the original-position tiebreak in the canonical ordering.
    # The operand tuple separates commuting directed gates that share every
    # other field (e.g. cz q[0],q[1] versus cz q[1],q[0]) so their order is
    # fixed by the key rather than by how the input happened to be written;
    # each gate's own operand direction is rendered unchanged.
    parameter = _format_angle(gate.angle) if gate.angle is not None else ""
    return (min(gate.qubits), max(gate.qubits), gate.kind, parameter, gate.qubits)


# Inverse pairs that cancel when adjacent on the same qubit (both
# directions), taken from the single gate registry.
_INVERSE_PAIRS = _REGISTRY_INVERSE_PAIRS

# Same-qubit commutation relations, taken from the single gate registry:
# any two diagonal gates, plus the explicit Pauli/rotation pairs (stored
# ordered, both directions, so the check is a single tuple lookup).
_COMMUTING_PAIRS = frozenset(
    (first, second)
    for pair in _REGISTRY_COMMUTING
    for first, second in (tuple(pair), tuple(pair)[::-1])
)


def _commutes(first: Gate, second: Gate) -> bool:
    """True when two gates that share a qubit may be swapped.

    Covers exactly the relations decidable from the gate semantics: both
    gates diagonal in the computational basis, or a Pauli gate with a
    rotation about its own axis (``x``/``rx``, ``y``/``ry``). Gates acting
    on disjoint qubits are independent regardless of this predicate.
    """
    if first.kind in DIAGONAL_GATES and second.kind in DIAGONAL_GATES:
        return True
    if first.kind == second.kind:
        return False
    return (first.kind, second.kind) in _COMMUTING_PAIRS


# The kinds whose gates commute with a gate of each kind on a shared qubit
# (for x/rx and y/ry the pair partner only: a gate never commutes with
# another application of itself unless it is diagonal).
_COMMUTING_KINDS = {
    name: (
        DIAGONAL_GATES
        if name in DIAGONAL_GATES
        else next((pair - {name} for pair in _REGISTRY_COMMUTING if name in pair), frozenset())
    )
    for name in GATE_SPECS
}

# The kinds whose gates conflict with (do not commute with) a gate of each
# kind on a shared qubit.
_NONCOMMUTING_KINDS = {
    name: frozenset(kind for kind in GATE_SPECS if kind not in commuting)
    for name, commuting in _COMMUTING_KINDS.items()
}

# The unique partner kind a gate can cancel or merge with: itself for
# self-inverse gates and rotations, the inverse gate for s/sdg and t/tdg.
# Cancellation and merging both require matching operands, so the latest
# application of (partner kind, qubits) is the only rewrite candidate.
_PARTNER_KIND = {
    name: (
        name
        if spec.self_inverse or name in _ANY_ROTATIONS
        else spec.inverse_name
    )
    for name, spec in GATE_SPECS.items()
}


def _cancels(first: Gate, second: Gate) -> bool:
    """True when *second* immediately follows *first* and both vanish."""
    if first.qubits != second.qubits:
        return False
    if first.kind in SELF_INVERSE_GATES:
        return first.kind == second.kind
    return (first.kind, second.kind) in _INVERSE_PAIRS


def _merge(first: Gate, second: Gate) -> Gate | None:
    """Merge two same-axis rotations on the same qubits, if applicable.

    Returns the merged :class:`Gate`, or ``None`` when the gates cannot be
    merged. Controlled rotations additionally require identical control and
    target and are normalized onto their ``4*pi`` period. Diagonal
    rotations (``rz``/``crz``) commute with themselves, so their angles are
    summed over the sorted constituents: the result must not depend on the
    order a commuting multiset was written in.
    """
    if first.kind != second.kind or first.qubits != second.qubits:
        return None
    if first.kind not in _ANY_ROTATIONS:
        return None
    assert first.angle is not None and second.angle is not None
    normalize = (
        normalize_controlled_angle if first.kind in _CONTROLLED_ROTATIONS else normalize_angle
    )
    if first.kind in _CANONICAL_SUM_ROTATIONS:
        assert first.parts is not None and second.parts is not None
        parts = tuple(sorted(first.parts + second.parts))
        return Gate(first.kind, first.qubits, normalize(math.fsum(parts)), parts)
    return Gate(first.kind, first.qubits, normalize(first.angle + second.angle))


def _is_identity_rotation(gate: Gate) -> bool:
    return (
        gate.kind in _ANY_ROTATIONS
        and gate.angle is not None
        and abs(gate.angle) <= _ANGLE_EPSILON
    )


def _simplify(gates: list[Gate]) -> tuple[list[Gate], bool]:
    """Simplify *gates* to a fixed point; return ``(gates, changed)``.

    Single left-to-right pass. Each gate either rewrites with the latest
    preceding application of its partner kind on the same qubits — the only
    gate it could cancel or merge with — or is kept. The rewrite fires only
    when every gate between the pair acts on disjoint qubits or commutes
    with the moving gate, i.e. when no conflicting gate sits between them;
    the nearest conflicting gate on any of the gate's qubits blocks it.
    Kept gates are indexed by kind and by (kind, qubits) application so
    each lookup is constant time; consumed gates become tombstones and are
    compacted away at the end. A gate strictly between the partner and the
    current gate commutes with the partner's kind (otherwise it would have
    blocked the rewrite), so consumed partners never change what earlier
    gates can rewrite with — no rescanning is needed and the rewrite
    sequence is identical to repeatedly applying the first available
    rewrite.
    """
    changed = False
    work: list[Gate | None] = list(gates)
    # Tombstoned indices are left in the stacks and popped lazily on read.
    applications: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    by_kind: dict[int, dict[str, list[int]]] = {}

    def alive_top(stack: list[int] | None) -> int:
        if not stack:
            return -1
        while stack and work[stack[-1]] is None:
            stack.pop()
        return stack[-1] if stack else -1

    for index in range(len(work)):
        gate = work[index]
        assert gate is not None  # only earlier positions are tombstoned
        if _is_identity_rotation(gate):
            work[index] = None
            changed = True
            continue

        partner_kind = _PARTNER_KIND[gate.kind]
        partner = alive_top(applications.get((partner_kind, gate.qubits)))
        blocker = -1
        for qubit in gate.qubits:
            stacks = by_kind.get(qubit)
            if stacks is None:
                continue
            for kind in _NONCOMMUTING_KINDS[gate.kind]:
                top = alive_top(stacks.get(kind))
                if top <= blocker:
                    continue
                if kind == partner_kind and work[top].qubits == gate.qubits:
                    continue  # the partner candidate itself, not a blocker
                blocker = top

        if partner > blocker:
            changed = True
            previous = work[partner]
            assert previous is not None
            if _cancels(previous, gate):
                work[partner] = None
            else:
                merged = _merge(previous, gate)
                assert merged is not None
                # A merged rotation may itself collapse to the identity.
                work[partner] = None if _is_identity_rotation(merged) else merged
            work[index] = None
            continue

        applications.setdefault((gate.kind, gate.qubits), []).append(index)
        for qubit in gate.qubits:
            by_kind.setdefault(qubit, {}).setdefault(gate.kind, []).append(index)

    return [gate for gate in work if gate is not None], changed


def _canonical_order(gates: list[Gate]) -> list[Gate]:
    """Return *gates* in canonical order.

    Builds the dependency DAG in which each gate depends on the preceding
    gates that share a qubit and do not commute with it (gates on disjoint
    qubits, or pairs covered by the recognized commutation relations, are
    mutually independent), then takes the unique linear extension that
    minimizes :func:`_sort_key` among all available gates at each step.
    That extension is independent of how independent gates were written,
    which is what makes reordering canonical.
    """
    count = len(gates)
    # predecessors[i]: gates that must precede gate i (share a qubit, come
    # earlier and do not commute with it). Per qubit, gate indices are
    # grouped by kind so commuting kinds can be skipped wholesale, and the
    # latest barrier is tracked — a gate that commutes with no earlier gate
    # on the qubit, so every earlier gate reaches it transitively. A new
    # gate only needs edges from conflicting gates back to that barrier; a
    # barrier that commutes with the new gate covers nothing (older gates
    # may conflict with the new gate without reaching it), so the scan
    # then runs to the start of the qubit's history.
    predecessors: list[set[int]] = [set() for _ in range(count)]
    on_qubit: dict[int, dict[str, list[int]]] = {}
    barrier: dict[int, int] = {}
    for index, gate in enumerate(gates):
        commuting = _COMMUTING_KINDS[gate.kind]
        for qubit in gate.qubits:
            by_kind = on_qubit.setdefault(qubit, {})
            stop = barrier.get(qubit, -1)
            if stop >= 0 and _commutes(gates[stop], gate):
                stop = -1
            for kind, indices in by_kind.items():
                if kind in commuting:
                    continue
                for earlier in reversed(indices):
                    if earlier <= stop:
                        break
                    predecessors[index].add(earlier)
            if stop >= 0:
                predecessors[index].add(stop)
            if all(kind not in by_kind for kind in commuting):
                # Nothing on this qubit commutes with the new gate, so
                # every earlier gate reaches it: it is the new barrier.
                barrier[qubit] = index
            by_kind.setdefault(gate.kind, []).append(index)

    remaining = [len(preds) for preds in predecessors]
    dependents: list[list[int]] = [[] for _ in range(count)]
    for index, preds in enumerate(predecessors):
        for pred in preds:
            dependents[pred].append(index)

    # Heap entries break key ties by original position, preserving the
    # relative (dependency) order of gates that the sort key cannot separate.
    ready: list[tuple[tuple, int]] = [
        (_sort_key(gates[index]), index) for index in range(count) if remaining[index] == 0
    ]
    heapq.heapify(ready)

    ordered: list[Gate] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered.append(gates[index])
        for dependent in dependents[index]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, (_sort_key(gates[dependent]), dependent))
    return ordered


def canonical_gates(operations: tuple[Operation, ...]) -> tuple[tuple[Gate, ...], bool]:
    """Return ``(gates, changed)``: the canonical gate sequence and whether
    any cancellation, merge, deletion or reordering was applied to the
    extracted gates.

    Angle normalization alone (the same gates in the same order) does not
    count as a change.
    """
    gates = [_gate_from_operation(op) for op in operations if op.kind != "measure"]
    changed = False
    while True:
        gates, simplified = _simplify(gates)
        if simplified:
            changed = True
        ordered = _canonical_order(gates)
        if ordered != gates:
            gates = ordered
            changed = True
            continue
        return tuple(gates), changed


def _gate_line(gate: Gate) -> str:
    return gate_qasm(gate.kind, gate.qubits, gate.angle)


def render(program: Program, gates: tuple[Gate, ...]) -> str:
    """Render the canonical circuit as complete, re-parseable OpenQASM."""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{program.num_qubits}];",
        f"creg c[{program.num_clbits}];",
    ]
    lines.extend(_gate_line(gate) for gate in gates)
    # Measurements sorted by (clbit, qubit), preserving the qubit-to-clbit
    # mapping from the source program.
    measurements = sorted(
        (op.targets[1], op.targets[0])
        for op in program.operations
        if op.kind == "measure"
    )
    lines.extend(f"measure q[{qubit}] -> c[{clbit}];" for clbit, qubit in measurements)
    return "\n".join(lines) + "\n"


def optimize(program: Program) -> tuple[tuple[Gate, ...], bool, str]:
    """Return ``(canonical_gates, changed, canonical_qasm)`` for *program*.

    *changed* is true only when the pre-measurement gate sequence changed
    (cancellation, merge, deletion or reordering); re-rendering the same
    gates with normalized angles does not set it.
    """
    gates, changed = canonical_gates(program.operations)
    return gates, changed, render(program, gates)
