"""Circuit simplification and canonical rendering for the OpenQASM subset.

The optimizer rewrites a parsed circuit into a deterministic canonical
form:

* pairs of ``x``/``h`` on the same qubit, and pairs of ``cx`` with the same
  control and target, cancel when only gates on disjoint qubits sit
  between them;
* same-axis rotations on the same qubit merge into a single gate, angles
  are normalized to ``(-pi, pi]`` and rotations whose angle is at most
  ``1e-12`` in magnitude are dropped (near-``pi`` ``rx``/``ry`` rotations
  are kept as rotations, never rewritten to ``x``);
* gates that share no qubits are reordered by (lowest qubit, highest
  qubit, gate name, parameter text) while gates that share a qubit keep
  their dependency order;
* measurements are emitted sorted by classical bit, then qubit index, and
  the canonical registers are named ``q`` and ``c``.

Simplification and reordering are repeated until the gate sequence stops
changing, so circuits that differ only in the written order of
independent gates produce byte-identical output.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .openqasm import Operation, Program

_ROTATION_KINDS = frozenset({"rx", "ry", "rz"})
_TWO_PI = 2.0 * math.pi

# Rotations whose normalized angle is at most this magnitude are dropped.
_ANGLE_EPSILON = 1e-12


def _normalize_angle(angle: float) -> float:
    """Map *angle* into the canonical interval ``(-pi, pi]``.

    The result is never negative zero.
    """
    reduced = math.fmod(angle, _TWO_PI)
    if reduced <= -math.pi:
        reduced += _TWO_PI
    elif reduced > math.pi:
        reduced -= _TWO_PI
    # Adding 0.0 turns a possible -0.0 into 0.0 and is a no-op otherwise.
    return reduced + 0.0


def _format_angle(angle: float) -> str:
    """Render *angle* with at most 17 significant digits, no negative zero."""
    if angle == 0.0:
        return "0"
    # repr() is the shortest decimal string that round-trips the double,
    # which never exceeds 17 significant digits and uses a lowercase 'e'.
    return repr(angle)


def _params_text(op: Operation) -> str:
    if op.kind in _ROTATION_KINDS:
        return _format_angle(op.params[0])
    return ""


def _sort_key(op: Operation) -> tuple[int, int, str, str]:
    return (min(op.targets), max(op.targets), op.kind, _params_text(op))


def _simplify_once(gates: list[Operation]) -> list[Operation]:
    """Run one cancellation/merging sweep over *gates* in dependency order.

    For each qubit a stack of the surviving gates touching it is kept, so a
    new gate is always compared against the most recent survivor on its
    qubits — gates in between that act on disjoint qubits are ignored,
    which is exactly the "virtually adjacent" rule.

    Merged rotations accumulate the raw angle sum here; normalization to
    ``(-pi, pi]`` and the 1e-12 deletion happen in a separate pass so the
    accumulated value stays as close as possible to the original angles.
    """
    out: list[Operation | None] = []
    stacks: dict[int, list[int]] = {}

    def push(op: Operation) -> None:
        index = len(out)
        out.append(op)
        for qubit in op.targets:
            stacks.setdefault(qubit, []).append(index)

    for op in gates:
        if op.kind in ("x", "h"):
            qubit = op.targets[0]
            stack = stacks.get(qubit)
            if stack:
                top = stack[-1]
                prev = out[top]
                if prev is not None and prev.kind == op.kind:
                    out[top] = None
                    stack.pop()
                    continue
            push(op)
        elif op.kind in _ROTATION_KINDS:
            qubit = op.targets[0]
            stack = stacks.get(qubit)
            if stack:
                top = stack[-1]
                prev = out[top]
                if prev is not None and prev.kind == op.kind:
                    merged = prev.params[0] + op.params[0]
                    if math.isfinite(merged):
                        out[top] = Operation(op.kind, (qubit,), (merged,))
                        continue
            push(op)
        elif op.kind == "cx":
            control, target = op.targets
            control_stack = stacks.get(control)
            target_stack = stacks.get(target)
            top = max(
                control_stack[-1] if control_stack else -1,
                target_stack[-1] if target_stack else -1,
            )
            if top >= 0:
                prev = out[top]
                if prev is not None and prev.kind == "cx" and prev.targets == op.targets:
                    # A cx sits on both qubits' stacks; being the most recent
                    # survivor on either means it tops both.
                    out[top] = None
                    control_stack.pop()
                    target_stack.pop()
                    continue
            push(op)
        else:
            push(op)
    return [op for op in out if op is not None]


def _normalize_rotations(gates: list[Operation]) -> list[Operation]:
    """Normalize rotation angles to ``(-pi, pi]`` and drop tiny rotations."""
    normalized: list[Operation] = []
    for op in gates:
        if op.kind in _ROTATION_KINDS:
            angle = _normalize_angle(op.params[0])
            if abs(angle) <= _ANGLE_EPSILON:
                continue
            normalized.append(Operation(op.kind, op.targets, (angle,)))
        else:
            normalized.append(op)
    return normalized


def _canonical_order(gates: list[Operation]) -> list[Operation]:
    """Stable canonical topological sort of *gates*.

    Gates that share a qubit keep their dependency order; mutually
    independent gates are emitted in ascending (lowest qubit, highest
    qubit, gate name, parameter text) order.
    """
    count = len(gates)
    successors: list[list[int]] = [[] for _ in range(count)]
    indegree = [0] * count
    last_on_qubit: dict[int, int] = {}
    for index, op in enumerate(gates):
        predecessors = {last_on_qubit[q] for q in op.targets if q in last_on_qubit}
        for predecessor in predecessors:
            successors[predecessor].append(index)
        indegree[index] = len(predecessors)
        for qubit in op.targets:
            last_on_qubit[qubit] = index

    heap = [(_sort_key(gates[i]), i) for i in range(count) if indegree[i] == 0]
    heapq.heapify(heap)
    ordered: list[Operation] = []
    while heap:
        _, index = heapq.heappop(heap)
        ordered.append(gates[index])
        for follower in successors[index]:
            indegree[follower] -= 1
            if indegree[follower] == 0:
                heapq.heappush(heap, (_sort_key(gates[follower]), follower))
    return ordered


def _canonicalize_gates(gates: list[Operation]) -> list[Operation]:
    """Simplify, normalize and reorder *gates* until the sequence is stable."""
    current = list(gates)
    while True:
        simplified = _canonical_order(_normalize_rotations(_simplify_once(current)))
        if simplified == current:
            return simplified
        current = simplified


def _render_qasm(
    num_qubits: int,
    num_clbits: int,
    gates: list[Operation],
    measurements: list[tuple[int, int]],
) -> str:
    """Render the canonical circuit, one statement per line, newline at end."""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_qubits}];",
        f"creg c[{num_clbits}];",
    ]
    for op in gates:
        if op.kind == "cx":
            lines.append(f"cx q[{op.targets[0]}],q[{op.targets[1]}];")
        elif op.kind in _ROTATION_KINDS:
            lines.append(f"{op.kind}({_format_angle(op.params[0])}) q[{op.targets[0]}];")
        else:
            lines.append(f"{op.kind} q[{op.targets[0]}];")
    for qubit, clbit in measurements:
        lines.append(f"measure q[{qubit}] -> c[{clbit}];")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class OptimizeResult:
    num_qubits: int
    num_clbits: int
    original_gate_count: int
    optimized_gate_count: int
    changed: bool
    qasm: str


def optimize_program(program: Program) -> OptimizeResult:
    """Simplify *program* into its canonical form."""
    gates = [op for op in program.operations if op.kind != "measure"]
    measurements = sorted(
        (
            (op.targets[0], op.targets[1])
            for op in program.operations
            if op.kind == "measure"
        ),
        key=lambda pair: (pair[1], pair[0]),
    )
    optimized = _canonicalize_gates(gates)
    return OptimizeResult(
        num_qubits=program.num_qubits,
        num_clbits=program.num_clbits,
        original_gate_count=len(gates),
        optimized_gate_count=len(optimized),
        changed=optimized != gates,
        qasm=_render_qasm(program.num_qubits, program.num_clbits, optimized, measurements),
    )
