"""Deterministic canonical simplification for the OpenQASM 2.0 subset.

The optimizer rewrites the pre-measurement gate sequence into a canonical
form while preserving the circuit's transformation up to a global phase and
keeping the measurement layout (qubit-to-clbit mapping) intact:

* adjacent ``x``/``x``, ``h``/``h``, ``y``/``y``, ``z``/``z`` and
  ``swap``/``swap`` pairs, ``cx``/``cz`` pairs with identical control and
  target, and inverse pairs ``s``/``sdg`` and ``t``/``tdg`` on the same
  qubit cancel ("adjacent" allows intervening gates on disjoint qubits);
* consecutive same-axis ``rx``/``ry``/``rz`` rotations on one qubit merge
  into a single rotation, as do consecutive same-axis ``crx``/``cry``/``crz``
  rotations with identical control and target;
* rotations whose normalized angle has magnitude at most ``1e-12`` are
  deleted;
* gates acting on disjoint qubits are moved past each other until the
  sequence is sorted by a stable key (minimum qubit, maximum qubit, gate
  name and parameter text).

Simplification passes repeat until the sequence reaches a fixed point, so
inputs that differ only in the ordering of independent gates produce
byte-identical output. Single-qubit rotation angles are normalized to
``(-pi, pi]`` (rendering a value at either boundary as ``pi`` when it is
positive). Controlled rotations are normalized to ``(-2*pi, 2*pi]`` instead:
their angle is only ``4*pi``-periodic (a ``2*pi`` shift is *not* a global
phase of the controlled unitary), so ``2*pi`` is kept rather than deleted.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .gates import (
    CONTROLLED_ROTATION_PERIOD,
    GATES,
    ROTATION_PERIOD,
    format_angle,
    render_statement,
)
from .openqasm import Operation, Program

# Rotations this small (after normalization to (-pi, pi]) act as the
# identity within floating-point precision and are removed.
_ANGLE_EPSILON = 1e-12


@dataclass(frozen=True)
class Gate:
    """A pre-measurement gate with its canonical (normalized) angle."""

    kind: str
    qubits: tuple[int, ...]
    angle: float | None = None

    @property
    def qubit_set(self) -> frozenset[int]:
        return frozenset(self.qubits)


def _normalize_periodic(angle: float, period: float) -> float:
    """Reduce *angle* modulo *period* into the interval ``(-period/2, period/2]``.

    Angles already inside the interval are returned untouched so ordinary
    values (e.g. ``0.7``) are not perturbed by floating-point modulo.
    """
    half = 0.5 * period
    if -half < angle <= half:
        return angle
    reduced = (angle + half) % period - half
    if reduced == -half:
        return half
    return reduced


def normalize_angle(angle: float) -> float:
    """Reduce *angle* modulo ``2*pi`` into the interval ``(-pi, pi]``.

    A single-qubit rotation by ``theta + 2*pi`` differs only by a global
    phase, so the period is ``2*pi``.
    """
    return _normalize_periodic(angle, ROTATION_PERIOD)


def normalize_controlled_angle(angle: float) -> float:
    """Reduce a controlled-rotation angle modulo ``4*pi`` into ``(-2*pi, 2*pi]``.

    A controlled rotation by ``theta + 2*pi`` differs from one by ``theta``
    by more than a global phase (the unshifted rotation picks up a sign on
    the control-1 block only), so the period is ``4*pi`` and ``2*pi`` itself
    is kept rather than deleted.
    """
    return _normalize_periodic(angle, CONTROLLED_ROTATION_PERIOD)


# Kept importable from here for existing internal callers; the definition
# lives in the gate registry.
_format_angle = format_angle


def _gate_from_operation(op: Operation) -> Gate:
    spec = GATES[op.kind]
    # swap is symmetric in its operands; canonicalize the operand order so
    # swap q[a],q[b] and swap q[b],q[a] share one canonical form.
    qubits = tuple(sorted(op.targets)) if spec.symmetric else op.targets
    if spec.angle_period is None:
        return Gate(op.kind, qubits)
    return Gate(op.kind, qubits, _normalize_periodic(op.params[0], spec.angle_period))


def _sort_key(gate: Gate) -> tuple[int, int, str, str]:
    # Ties (including two commuting gates on the same set, e.g. rz/rx) keep
    # their relative order, which is the dependency order the semantics need.
    parameter = _format_angle(gate.angle) if gate.angle is not None else ""
    return (min(gate.qubits), max(gate.qubits), gate.kind, parameter)


def _cancels(first: Gate, second: Gate) -> bool:
    """True when *second* immediately follows *first* and both vanish."""
    if first.qubits != second.qubits:
        return False
    # Self-inverse gates (x, h, y, z, cx, cz, swap) cancel their own kind;
    # s/sdg and t/tdg cancel each other. Rotations merge instead.
    return GATES[first.kind].inverse == second.kind


def _merge(first: Gate, second: Gate) -> Gate | None:
    """Merge two same-axis rotations on the same qubits, if applicable.

    Returns the merged :class:`Gate`, or ``None`` when the gates cannot be
    merged. Controlled rotations additionally require identical control and
    target and are normalized onto their ``4*pi`` period.
    """
    if first.kind != second.kind or first.qubits != second.qubits:
        return None
    spec = GATES[first.kind]
    if spec.angle_period is None:
        return None
    assert first.angle is not None and second.angle is not None
    return Gate(
        first.kind,
        first.qubits,
        _normalize_periodic(first.angle + second.angle, spec.angle_period),
    )


def _is_identity_rotation(gate: Gate) -> bool:
    return (
        GATES[gate.kind].angle_period is not None
        and gate.angle is not None
        and abs(gate.angle) <= _ANGLE_EPSILON
    )


def _simplify_once(gates: list[Gate]) -> list[Gate] | None:
    """Run one simplification sweep.

    Scans left to right; whenever a gate can cancel or merge with the
    nearest preceding gate that shares a qubit (gates between them acting on
    disjoint qubits are ignored), the rewrite is performed and the scan
    restarts so the newly adjacent gates become visible. Identity rotations
    are dropped on sight. Returns the rewritten list, or ``None`` when no
    rewrite was possible.
    """
    for index in range(len(gates)):
        gate = gates[index]
        if _is_identity_rotation(gate):
            return gates[:index] + gates[index + 1 :]

        for earlier in range(index - 1, -1, -1):
            previous = gates[earlier]
            if previous.qubit_set & gate.qubit_set:
                if _cancels(previous, gate):
                    return gates[:earlier] + gates[earlier + 1 : index] + gates[index + 1 :]
                merged = _merge(previous, gate)
                if merged is not None:
                    return gates[:earlier] + [merged] + gates[earlier + 1 : index] + gates[index + 1 :]
                break
    return None


def _canonical_order(gates: list[Gate]) -> list[Gate]:
    """Return *gates* in canonical order.

    Builds the dependency DAG in which each gate depends on the latest
    preceding gate touching any of the same qubits (so gates on disjoint
    qubits are mutually independent), then takes the unique linear
    extension that minimizes :func:`_sort_key` among all available gates at
    each step. That extension is independent of how independent gates were
    written, which is what makes reordering canonical.
    """
    count = len(gates)
    # predecessors[i]: gates that must precede gate i (share a qubit and come
    # earlier in the source). Only the latest such gate per qubit is needed:
    # transitively it already requires everything before it.
    predecessors: list[set[int]] = [set() for _ in range(count)]
    last_on_qubit: dict[int, int] = {}
    for index, gate in enumerate(gates):
        for qubit in gate.qubits:
            if qubit in last_on_qubit:
                predecessors[index].add(last_on_qubit[qubit])
            last_on_qubit[qubit] = index

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
        simplified = _simplify_once(gates)
        if simplified is not None:
            gates = simplified
            changed = True
            continue
        ordered = _canonical_order(gates)
        if ordered != gates:
            gates = ordered
            changed = True
            continue
        return tuple(gates), changed


def _gate_line(gate: Gate) -> str:
    params = () if gate.angle is None else (gate.angle,)
    return render_statement(gate.kind, gate.qubits, params)


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
