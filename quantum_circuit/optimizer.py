"""Deterministic canonical simplification for the OpenQASM 2.0 subset.

The optimizer rewrites the pre-measurement gate sequence into a canonical
form while preserving the circuit's transformation up to a global phase and
keeping the measurement layout (qubit-to-clbit mapping) intact:

* adjacent ``x``/``x``, ``h``/``h``, ``y``/``y``, ``z``/``z`` and
  ``swap``/``swap`` pairs, ``cx``/``cz`` pairs with identical control and
  target, and inverse pairs ``s``/``sdg`` and ``t``/``tdg`` on the same
  qubit cancel ("adjacent" allows intervening gates on disjoint qubits, or
  intervening gates that commute on the shared qubit -- see below);
* consecutive same-axis ``rx``/``ry``/``rz`` rotations on one qubit merge
  into a single rotation, as do consecutive same-axis ``crx``/``cry``/``crz``
  rotations with identical control and target;
* rotations whose normalized angle has magnitude at most ``1e-12`` are
  deleted;
* gates acting on disjoint qubits are moved past each other, and so are a
  fixed set of same-qubit commuting pairs: ``x`` with ``rx``, ``y`` with
  ``ry``, and any two computational-basis-diagonal gates
  (``z``/``s``/``sdg``/``t``/``tdg``/``rz``/``cz``/``crz``). Gates are
  sorted by a stable key (minimum qubit, maximum qubit, gate name and
  parameter text).

Cancellations and merges therefore reach across any number of commuting
gates (two ``x`` gates separated by an ``rx``, ``s``/``sdg`` separated by a
``cz``, two ``rz`` rotations separated by a single-qubit diagonal gate).
Every other shared-qubit pair keeps its dependency order; the operand
orientation of directed gates (``cx``/``cz``/``crx``/``cry``/``crz``) is
never rewritten.

Simplification passes repeat until the sequence reaches a fixed point, so
inputs that differ only in the ordering of interchangeable gates produce
byte-identical output. Single-qubit rotation angles are normalized to
``(-pi, pi]`` (rendering a value at either boundary as ``pi`` when it is
positive). Controlled rotations are normalized to ``(-2*pi, 2*pi]`` instead:
their angle is only ``4*pi``-periodic (a ``2*pi`` shift is *not* a global
phase of the controlled unitary), so ``2*pi`` is kept rather than deleted.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .gates import (
    CONTROLLED_ROTATION_GATES,
    GATE_SPECS,
    INVERSE_PAIRS as _REGISTRY_INVERSE_PAIRS,
    ROTATION_GATES,
    SELF_INVERSE_GATES,
    Operation,
    commute_group,
    format_angle,
    gate_qasm,
)
from .openqasm import Program

# Rotations this small (after normalization to (-pi, pi]) act as the
# identity within floating-point precision and are removed.
_ANGLE_EPSILON = 1e-12

_ROTATIONS = ROTATION_GATES
_CONTROLLED_ROTATIONS = CONTROLLED_ROTATION_GATES

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

    @property
    def qubit_set(self) -> frozenset[int]:
        return frozenset(self.qubits)

    @property
    def commute_group(self) -> str | None:
        """Same-qubit commutation group label, or ``None`` for no crossing.

        Two gates that share a qubit and carry the same label may change
        places (``x``/``rx``, ``y``/``ry`` and the diagonal family); a gate
        with ``None`` keeps its dependency order against every shared-qubit
        gate, and gates with different labels keep theirs against each
        other.
        """
        return commute_group(self.kind)


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
        return Gate(op.kind, tuple(op.targets), normalize(op.params[0]))
    return Gate(op.kind, tuple(op.targets))


def _sort_key(gate: Gate) -> tuple[int, int, str, str]:
    # Canonical order of mutually interchangeable gates: minimum qubit,
    # maximum qubit, gate name, then parameter text. Gates that may not cross
    # never become simultaneously ready, so the key only orders independent
    # gates (disjoint qubits or one shared commutation group).
    parameter = _format_angle(gate.angle) if gate.angle is not None else ""
    return (min(gate.qubits), max(gate.qubits), gate.kind, parameter)


# Inverse pairs that cancel when adjacent on the same qubit (both
# directions), taken from the single gate registry.
_INVERSE_PAIRS = _REGISTRY_INVERSE_PAIRS


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
    target and are normalized onto their ``4*pi`` period.
    """
    if first.kind != second.kind or first.qubits != second.qubits:
        return None
    if first.kind in _ROTATIONS:
        assert first.angle is not None and second.angle is not None
        return Gate(first.kind, first.qubits, normalize_angle(first.angle + second.angle))
    if first.kind in _CONTROLLED_ROTATIONS:
        assert first.angle is not None and second.angle is not None
        return Gate(
            first.kind,
            first.qubits,
            normalize_controlled_angle(first.angle + second.angle),
        )
    return None


def _is_identity_rotation(gate: Gate) -> bool:
    return (
        gate.kind in _ROTATIONS + _CONTROLLED_ROTATIONS
        and gate.angle is not None
        and abs(gate.angle) <= _ANGLE_EPSILON
    )


def _crosses(left: Gate, right: Gate) -> bool:
    """Whether *right* may move left across *left* though they share a qubit.

    This is exactly the recognized same-qubit commutation relation: both
    gates belong to the same named commutation group. Disjoint gates are
    handled by the callers (they never constrain one another); every other
    shared-qubit pair is immovable.
    """
    group = left.commute_group
    return group is not None and group == right.commute_group


def _simplify_once(gates: list[Gate]) -> list[Gate] | None:
    """Run one simplification sweep.

    Scans left to right; whenever a gate can cancel or merge with a
    preceding gate that shares a qubit, the rewrite is performed and the
    scan restarts so the newly adjacent gates become visible. Gates between
    the two are ignored when each of them may be crossed: it acts on
    disjoint qubits, or it shares a qubit but belongs to the same
    commutation group (``x``/``rx``, ``y``/``ry`` or two diagonal gates).
    The first gate that genuinely blocks the pair stops the look-back.
    Identity rotations are dropped on sight. Returns the rewritten list,
    or ``None`` when no rewrite was possible.
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
                if _crosses(previous, gate):
                    # Commuting shared-qubit gate: slide the look-back past
                    # it; the pair sought may cancel/merge further left.
                    continue
                break
    return None


def _canonical_order(gates: list[Gate]) -> list[Gate]:
    """Return *gates* in canonical order.

    Builds the dependency DAG of the partial order "must stay before": gate
    *u* must precede a later gate *v* when they share a qubit and may not
    cross, i.e. they are not in the same named commutation group (a gate
    without a group blocks every shared-qubit gate, including another one).
    Gates on disjoint qubits and same-group gates on a shared qubit are
    mutually independent.

    On one qubit the gates split into runs: a maximal stretch of gates with
    one shared non-empty commutation label. Gates inside a run are mutually
    interchangeable; a gate without a label always starts a singleton run.
    A run as a whole must precede the next run on the qubit: e.g. for
    ``z s x`` both diagonal gates must be emitted before the ``x`` even
    though ``z`` and ``s`` carry no order between them. Two-qubit gates take
    part in one run on every qubit they touch and wait on each.

    The runs are synchronized by completion barriers in linear time: a gate
    is ready once the predecessor run on each qubit it touches is fully
    emitted; completing a run opens its single successor. Among all ready
    gates the smallest :func:`_sort_key` is emitted, which makes the result
    independent of how interchangeable gates were written.
    """
    count = len(gates)

    # Build the per-qubit runs. Each run has one qubit owner and a single
    # predecessor (the previous run on that qubit); a gate records every
    # run it belongs to (one per operand).
    run_members: list[list[int]] = []
    run_prev: list[int | None] = []
    gate_runs: list[list[int]] = [[] for _ in range(count)]
    # qubit -> (id of its latest run, label of that run's gates)
    last_run_on_qubit: dict[int, tuple[int, str | None]] = {}
    for index, gate in enumerate(gates):
        group = gate.commute_group
        for qubit in gate.qubits:
            state = last_run_on_qubit.get(qubit)
            if state is not None:
                previous_run, previous_group = state
                if group is not None and group == previous_group:
                    run_id = previous_run
                else:
                    run_id = len(run_members)
                    run_members.append([])
                    run_prev.append(previous_run)
            else:
                run_id = len(run_members)
                run_members.append([])
                run_prev.append(None)
            run_members[run_id].append(index)
            gate_runs[index].append(run_id)
            last_run_on_qubit[qubit] = (run_id, group)

    run_next: dict[int, int] = {}
    for run_id, predecessor in enumerate(run_prev):
        if predecessor is not None:
            run_next[predecessor] = run_id

    # A gate carries one barrier per operand whose run is not first on its
    # qubit; a barrier is removed when that run's predecessor drains, so the
    # gate becomes ready only after the preceding run on *every* touched
    # qubit has fully emitted.
    remaining = [0] * count
    run_remaining = [len(members) for members in run_members]
    ready: list[tuple[tuple, int]] = []
    for index, run_ids in enumerate(gate_runs):
        blocked = sum(1 for run_id in run_ids if run_prev[run_id] is not None)
        remaining[index] = blocked
        if blocked == 0:
            heapq.heappush(ready, (_sort_key(gates[index]), index))

    ordered: list[Gate] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered.append(gates[index])
        for run_id in gate_runs[index]:
            run_remaining[run_id] -= 1
            if run_remaining[run_id] != 0:
                continue
            successor = run_next.get(run_id)
            if successor is None:
                continue
            for member in run_members[successor]:
                remaining[member] -= 1
                if remaining[member] == 0:
                    heapq.heappush(ready, (_sort_key(gates[member]), member))
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
