"""The single authoritative definition of the supported gate set.

Gate names, categories, operand structure (control/target roles, symmetric
operands), parameter counts, single-qubit matrices, inverse relations,
rotation periods and OpenQASM statement rendering are defined exactly once,
in this module. The parser (:mod:`quantum_circuit.openqasm`), the Python
DSL (:mod:`quantum_circuit.circuit`), the state-vector simulator
(:mod:`quantum_circuit.simulator`), the density-matrix/noise path
(:mod:`quantum_circuit.noise`), the optimizer
(:mod:`quantum_circuit.optimizer`) and the resource estimator
(:mod:`quantum_circuit.estimation`) all derive their view of a gate from
this registry, so a gate's semantics cannot drift between execution paths.

This module is a leaf: it imports nothing from the rest of the package and
performs no I/O.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

# Gate categories (the ``category`` field of :class:`GateSpec`).
SINGLE = "single"  # one qubit, no parameters: x h y z s sdg t tdg
ROTATION = "rotation"  # one qubit, one angle: rx ry rz
CONTROLLED = "controlled"  # control + target, no parameters: cx cz
CONTROLLED_ROTATION = "controlled_rotation"  # one angle, control + target: crx cry crz
SWAP = "swap"  # two symmetric operands: swap

# Rotation periods: a single-qubit rotation by theta + 2*pi differs only by
# a global phase, but a controlled rotation by theta + 2*pi picks up a sign
# on the control-1 block alone, so its period is 4*pi.
ROTATION_PERIOD = 2.0 * math.pi
CONTROLLED_ROTATION_PERIOD = 4.0 * math.pi

_SQRT1_2 = 2.0**-0.5


@dataclass(frozen=True)
class GateSpec:
    """The immutable semantics of one gate kind.

    *base* is the single-qubit gate a controlled gate applies to its target
    (``None`` for uncontrolled gates and ``swap``). *inverse* is the gate
    kind that cancels this one when adjacent on the same qubits (the kind
    itself for self-inverse gates, ``None`` for rotations, whose inverse is
    the same gate with a negated angle). *symmetric* marks gates whose two
    operands may be exchanged without changing the operation.
    *operands_error* is the message text used when a two-operand gate is
    given the same qubit twice.
    """

    kind: str
    category: str
    base: str | None = None
    inverse: str | None = None
    symmetric: bool = False
    angle_period: float | None = None
    diagonal: bool = False
    operands_error: str = ""

    @property
    def num_qubits(self) -> int:
        """How many qubit operands the gate takes (1 or 2)."""
        return 1 if self.category in (SINGLE, ROTATION) else 2

    @property
    def num_params(self) -> int:
        """How many angle parameters the gate takes (0 or 1)."""
        return 1 if self.category in (ROTATION, CONTROLLED_ROTATION) else 0


_DISTINCT_CONTROL_TARGET = "control and target must be different qubits"
_DISTINCT_OPERANDS = "operands must be different qubits"

# The registry, in the canonical gate order. Every per-gate fact above is
# stated here and nowhere else.
_SPECS: tuple[GateSpec, ...] = (
    GateSpec("x", SINGLE, inverse="x"),
    GateSpec("h", SINGLE, inverse="h"),
    GateSpec("y", SINGLE, inverse="y"),
    GateSpec("z", SINGLE, inverse="z", diagonal=True),
    GateSpec("s", SINGLE, inverse="sdg", diagonal=True),
    GateSpec("sdg", SINGLE, inverse="s", diagonal=True),
    GateSpec("t", SINGLE, inverse="tdg", diagonal=True),
    GateSpec("tdg", SINGLE, inverse="t", diagonal=True),
    GateSpec("cx", CONTROLLED, base="x", inverse="cx",
             operands_error=_DISTINCT_CONTROL_TARGET),
    GateSpec("cz", CONTROLLED, base="z", inverse="cz",
             operands_error=_DISTINCT_CONTROL_TARGET),
    GateSpec("swap", SWAP, inverse="swap", symmetric=True,
             operands_error=_DISTINCT_OPERANDS),
    GateSpec("rx", ROTATION, angle_period=ROTATION_PERIOD),
    GateSpec("ry", ROTATION, angle_period=ROTATION_PERIOD),
    GateSpec("rz", ROTATION, angle_period=ROTATION_PERIOD),
    GateSpec("crx", CONTROLLED_ROTATION, base="rx",
             angle_period=CONTROLLED_ROTATION_PERIOD,
             operands_error=_DISTINCT_CONTROL_TARGET),
    GateSpec("cry", CONTROLLED_ROTATION, base="ry",
             angle_period=CONTROLLED_ROTATION_PERIOD,
             operands_error=_DISTINCT_CONTROL_TARGET),
    GateSpec("crz", CONTROLLED_ROTATION, base="rz",
             angle_period=CONTROLLED_ROTATION_PERIOD,
             operands_error=_DISTINCT_CONTROL_TARGET),
)

# kind -> spec, in canonical order.
GATES: dict[str, GateSpec] = {spec.kind: spec for spec in _SPECS}

# All supported gate kinds, in canonical order.
GATE_KINDS: tuple[str, ...] = tuple(spec.kind for spec in _SPECS)


def _kinds_of(*categories: str) -> tuple[str, ...]:
    return tuple(spec.kind for spec in _SPECS if spec.category in categories)


# Gate kinds per category, in canonical order.
SINGLE_QUBIT_GATES: tuple[str, ...] = _kinds_of(SINGLE)
ROTATION_GATES: tuple[str, ...] = _kinds_of(ROTATION)
CONTROLLED_GATES: tuple[str, ...] = _kinds_of(CONTROLLED)
CONTROLLED_ROTATION_GATES: tuple[str, ...] = _kinds_of(CONTROLLED_ROTATION)
TWO_QUBIT_GATES: tuple[str, ...] = _kinds_of(CONTROLLED, CONTROLLED_ROTATION, SWAP)


def rotation_matrix(kind: str, theta: float) -> tuple[complex, complex, complex, complex]:
    """The single-qubit matrix of an rx/ry/rz rotation by *theta* radians."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if kind == "rx":
        return (c, -1j * s, -1j * s, c)
    if kind == "ry":
        return (c, -s, s, c)
    if kind == "rz":
        return (cmath.exp(-0.5j * theta), 0j, 0j, cmath.exp(0.5j * theta))
    raise AssertionError(f"no rotation matrix for gate {kind!r}")


def single_qubit_matrix(
    kind: str, params: tuple[float, ...] = ()
) -> tuple[complex, complex, complex, complex]:
    """The 2x2 matrix (row-major ``(g00, g01, g10, g11)``) of a single-qubit
    or rotation gate.

    This is the one matrix definition shared by the state-vector kernels,
    the density-matrix ``U rho U†`` evolution and the unitary construction,
    so all three paths apply the same transformation.
    """
    if kind == "x":
        return (0j, 1 + 0j, 1 + 0j, 0j)
    if kind == "h":
        return (_SQRT1_2, _SQRT1_2, _SQRT1_2, -_SQRT1_2)
    if kind == "y":
        return (0j, -1j, 1j, 0j)
    if kind == "z":
        return (1 + 0j, 0j, 0j, -1 + 0j)
    if kind == "s":
        return (1 + 0j, 0j, 0j, 1j)
    if kind == "sdg":
        return (1 + 0j, 0j, 0j, -1j)
    if kind == "t":
        return (1 + 0j, 0j, 0j, cmath.exp(0.25j * math.pi))
    if kind == "tdg":
        return (1 + 0j, 0j, 0j, cmath.exp(-0.25j * math.pi))
    if kind in ROTATION_GATES:
        return rotation_matrix(kind, params[0])
    raise AssertionError(f"no single-qubit matrix for gate {kind!r}")


def gate_matrix(
    kind: str, params: tuple[float, ...] = ()
) -> tuple[complex, complex, complex, complex]:
    """The 2x2 matrix a gate applies to its (target) qubit.

    For controlled gates this is the matrix of the gate's *base* kind; for
    single-qubit and rotation gates it is their own matrix. ``swap`` has no
    single-qubit matrix and raises :class:`AssertionError`.
    """
    spec = GATES[kind]
    base = spec.base if spec.base is not None else kind
    return single_qubit_matrix(base, params)


def format_angle(angle: float) -> str:
    """Render an angle deterministically.

    Uses the shortest decimal representation that round-trips to the same
    float (at most 17 significant digits), which Python writes as plain
    decimal or lowercase scientific notation. Negative zero is written as
    ``0``.
    """
    if angle == 0.0:
        return "0"
    return repr(angle)


def render_statement(kind: str, qubits: tuple[int, ...], params: tuple[float, ...] = ()) -> str:
    """Render one gate as an OpenQASM statement (register name ``q``).

    Operand order is exactly *qubits*: control before target for directed
    gates, the two operands in given order for ``swap``. Angles are rendered
    with :func:`format_angle`.
    """
    spec = GATES[kind]
    if spec.category in (CONTROLLED, SWAP):
        return f"{kind} q[{qubits[0]}],q[{qubits[1]}];"
    if spec.category == ROTATION:
        return f"{kind}({format_angle(params[0])}) q[{qubits[0]}];"
    if spec.category == CONTROLLED_ROTATION:
        return f"{kind}({format_angle(params[0])}) q[{qubits[0]}],q[{qubits[1]}];"
    return f"{kind} q[{qubits[0]}];"
