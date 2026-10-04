"""Single authoritative source of quantum-gate semantics.

Every other module that deals with gates reads the definitions it needs
from here instead of maintaining its own name lists or matrices:

* the OpenQASM parser (:mod:`quantum_circuit.openqasm`) and the Python DSL
  (:mod:`quantum_circuit.circuit`) learn gate names, operand counts,
  parameter rules and operand orientation from :data:`GATE_SPECS` and build
  immutable :class:`Operation` values only through this module;
* the state-vector simulator (:mod:`quantum_circuit.simulator`), the
  density-matrix path (:mod:`quantum_circuit.noise`) and the unitary mode
  all take their 2x2 unitary expressions from :func:`rotation_matrix` /
  :func:`single_qubit_unitary` / :func:`controlled_target_unitary`, so a
  gate has one mathematical definition on every path;
* the optimizer (:mod:`quantum_circuit.optimizer`) uses the same registry
  for inverse pairs, self-inverse gates, the swap symmetry and the
  ``2*pi``/``4*pi`` rotation periods;
* resource estimation (:mod:`quantum_circuit.estimation`) uses the fixed
  gate-count orders below.

This module is a dependency leaf: it imports only the standard library and
never imports the parser, simulator or command-line layers. Nothing here is
exported from the package root; gate semantics remain an internal contract.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# Operand shapes.
SINGLE_QUBIT = "single"
CONTROLLED = "controlled"
SWAP = "swap"

# Rotation periodicity (see :class:`GateSpec.rotation_period`).
ROTATION_PERIOD = 2.0 * math.pi
CONTROLLED_ROTATION_PERIOD = 4.0 * math.pi

# Same-qubit commutation classes (see :class:`GateSpec.commute_class`).
# ``x``/``rx`` and ``y``/``ry`` each share an axis and therefore commute;
# every computational-basis-diagonal gate belongs to the diagonal class,
# which is derived from the unitary rather than named on each spec.
COMMUTE_CLASS_PAULI_X = "pauli-x"
COMMUTE_CLASS_PAULI_Y = "pauli-y"
COMMUTE_CLASS_DIAGONAL = "diagonal"

_SQRT1_2 = 2.0**-0.5


@dataclass(frozen=True)
class GateSpec:
    """Immutable semantic definition of one supported gate.

    Attributes:
        name: canonical OpenQASM gate name.
        style: :data:`SINGLE_QUBIT` (one operand), :data:`CONTROLLED`
            (``(control, target)`` operands, the gate acts on *target* when
            *control* is 1) or :data:`SWAP` (two unordered operands).
        parameterized: whether the gate takes exactly one angle parameter.
        axis: rotation axis ``"x"``/``"y"``/``"z"`` for the rotation gates,
            otherwise ``None``.
        unitary: the constant 2x2 unitary ``(g00, g01, g10, g11)`` of a
            non-parameterized single-qubit gate, or the unitary embedded on
            the target of a non-parameterized controlled gate (``cx``/``cz``).
        phase_factor: for a diagonal single-qubit gate, the eigenvalue by
            which the ``|1>`` amplitude is multiplied; otherwise ``None``.
        inverse_name: name of the inverse gate when a distinct one exists.
        self_inverse: whether two adjacent applications on the same qubits
            cancel.
        symmetric: whether the two operands are unordered (only ``swap``).
        commute_class: optional same-qubit commutation group. Gates sharing
            the label commute whenever they share a qubit (``x``/``rx`` and
            ``y``/``ry``). Computational-basis-diagonal gates carry the
            shared diagonal label via :attr:`is_diagonal` rather than here.
    """

    name: str
    style: str
    parameterized: bool = False
    axis: str | None = None
    unitary: tuple[complex, complex, complex, complex] | None = None
    phase_factor: complex | None = None
    inverse_name: str | None = None
    self_inverse: bool = False
    symmetric: bool = False
    commute_class: str | None = None

    @property
    def num_operands(self) -> int:
        """Number of qubit operands a valid application must carry."""
        return 1 if self.style == SINGLE_QUBIT else 2

    @property
    def is_controlled(self) -> bool:
        """True for directed gates whose first operand is the control."""
        return self.style == CONTROLLED

    @property
    def is_diagonal(self) -> bool:
        """True for gates diagonal in the computational basis.

        Such gates share one commutation group: any two of them commute on
        every shared qubit. Single-qubit diagonal gates are the phase gates
        and the z rotation; a controlled gate is diagonal when its embedded
        target unitary is diagonal (``cz``) or its axis is ``z`` (``crz``),
        while ``cx``/``crx``/``cry`` flip the target and are not.
        """
        if self.style == SINGLE_QUBIT:
            return self.phase_factor is not None or self.axis == "z"
        if self.style == CONTROLLED:
            if self.axis is not None:
                return self.axis == "z"
            assert self.unitary is not None
            return self.unitary[1] == 0j and self.unitary[2] == 0j
        return False  # swap permutes basis states

    @property
    def rotation_period(self) -> float | None:
        """Angle period of a rotation gate, or ``None`` for fixed gates.

        Single-qubit rotations are ``2*pi`` periodic; controlled rotations
        are only ``4*pi`` periodic because a ``2*pi`` shift changes the
        control-1 block alone and is not a global phase of the controlled
        unitary.
        """
        if self.axis is None:
            return None
        return (
            CONTROLLED_ROTATION_PERIOD
            if self.style == CONTROLLED
            else ROTATION_PERIOD
        )

    def target_unitary(
        self, params: tuple[float, ...] = ()
    ) -> tuple[complex, complex, complex, complex]:
        """The 2x2 unitary acting on the gate's target qubit.

        For a single-qubit gate this is its full unitary; for a controlled
        gate it is the unitary embedded on the target when the control is 1.
        Parameterized rotations evaluate their (single) angle from *params*.
        """
        if self.axis is not None:
            return rotation_matrix(self.axis, params[0])
        assert self.unitary is not None
        return self.unitary

    def operation(
        self,
        targets: tuple[int, ...],
        params: tuple[float, ...] = (),
    ) -> "Operation":
        """Build an immutable application of this gate."""
        return Operation(self.name, tuple(targets), tuple(params))


@dataclass(frozen=True)
class Operation:
    """A validated gate application or measurement.

    ``kind`` is a canonical gate name or ``"measure"``; ``targets`` holds
    the qubit indices in the gate's operand order (for ``measure`` the
    clbit index is appended); ``params`` holds the gate angle in radians for
    the six rotation gates and is empty otherwise. Instances are frozen and
    are constructed only through :class:`GateSpec` or
    :func:`measurement_operation`.
    """

    kind: str
    targets: tuple[int, ...]
    params: tuple[float, ...] = ()

    @property
    def spec(self) -> GateSpec:
        """The semantic definition of a gate operation (never a measure)."""
        return GATE_SPECS[self.kind]

    def is_gate(self) -> bool:
        return self.kind != "measure"


# 2x2 unitary constants of the fixed single-qubit gates, in row-major
# (g00, g01, g10, g11) order. These are the one definition shared by the
# density-matrix evolution and the controlled-gate embedding.
_UNITARY_X = (0j, 1 + 0j, 1 + 0j, 0j)
_UNITARY_H = (_SQRT1_2, _SQRT1_2, _SQRT1_2, -_SQRT1_2)
_UNITARY_Y = (0j, -1j, 1j, 0j)
_UNITARY_Z = (1 + 0j, 0j, 0j, -1 + 0j)
_UNITARY_S = (1 + 0j, 0j, 0j, 1j)
_UNITARY_SDG = (1 + 0j, 0j, 0j, -1j)
_UNITARY_T = (1 + 0j, 0j, 0j, cmath.exp(0.25j * math.pi))
_UNITARY_TDG = (1 + 0j, 0j, 0j, cmath.exp(-0.25j * math.pi))


def _phase_factor(matrix: tuple[complex, complex, complex, complex]) -> complex:
    """The ``|1>`` eigenvalue of a diagonal single-qubit unitary."""
    return matrix[3]


_SPECS: tuple[GateSpec, ...] = (
    # Fixed single-qubit gates.
    GateSpec(
        "x", SINGLE_QUBIT, unitary=_UNITARY_X, self_inverse=True,
        commute_class=COMMUTE_CLASS_PAULI_X,
    ),
    GateSpec("h", SINGLE_QUBIT, unitary=_UNITARY_H, self_inverse=True),
    GateSpec(
        "y", SINGLE_QUBIT, unitary=_UNITARY_Y, self_inverse=True,
        commute_class=COMMUTE_CLASS_PAULI_Y,
    ),
    GateSpec(
        "z", SINGLE_QUBIT, unitary=_UNITARY_Z,
        phase_factor=_phase_factor(_UNITARY_Z), self_inverse=True,
    ),
    GateSpec(
        "s", SINGLE_QUBIT, unitary=_UNITARY_S,
        phase_factor=_phase_factor(_UNITARY_S), inverse_name="sdg",
    ),
    GateSpec(
        "sdg", SINGLE_QUBIT, unitary=_UNITARY_SDG,
        phase_factor=_phase_factor(_UNITARY_SDG), inverse_name="s",
    ),
    GateSpec(
        "t", SINGLE_QUBIT, unitary=_UNITARY_T,
        phase_factor=_phase_factor(_UNITARY_T), inverse_name="tdg",
    ),
    GateSpec(
        "tdg", SINGLE_QUBIT, unitary=_UNITARY_TDG,
        phase_factor=_phase_factor(_UNITARY_TDG), inverse_name="t",
    ),
    # Single-qubit angle rotations. Only the single-qubit x/y rotations
    # join the Pauli commutation groups; crx/cry are deliberately excluded
    # (the optimizer recognizes only the explicitly allowed relationships).
    GateSpec(
        "rx", SINGLE_QUBIT, parameterized=True, axis="x",
        commute_class=COMMUTE_CLASS_PAULI_X,
    ),
    GateSpec(
        "ry", SINGLE_QUBIT, parameterized=True, axis="y",
        commute_class=COMMUTE_CLASS_PAULI_Y,
    ),
    GateSpec("rz", SINGLE_QUBIT, parameterized=True, axis="z"),
    # Directed controlled gates (operands are control then target).
    GateSpec("cx", CONTROLLED, unitary=_UNITARY_X, self_inverse=True),
    GateSpec("cz", CONTROLLED, unitary=_UNITARY_Z, self_inverse=True),
    # Controlled angle rotations.
    GateSpec("crx", CONTROLLED, parameterized=True, axis="x"),
    GateSpec("cry", CONTROLLED, parameterized=True, axis="y"),
    GateSpec("crz", CONTROLLED, parameterized=True, axis="z"),
    # Symmetric two-qubit swap.
    GateSpec("swap", SWAP, self_inverse=True, symmetric=True),
)

GATE_SPECS: Mapping[str, GateSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)
GATE_NAMES = frozenset(GATE_SPECS)

# Gate categories derived from the registry, so the membership tables the
# parser and optimizer use can never drift away from the definitions.
SINGLE_QUBIT_GATES: Mapping[str, GateSpec] = MappingProxyType(
    {name: spec for name, spec in GATE_SPECS.items() if spec.style == SINGLE_QUBIT}
)
ROTATION_GATES = ("rx", "ry", "rz")
CONTROLLED_GATES: Mapping[str, GateSpec] = MappingProxyType(
    {name: spec for name, spec in GATE_SPECS.items() if spec.style == CONTROLLED}
)
CONTROLLED_ROTATION_GATES = ("crx", "cry", "crz")
SWAP_GATES = frozenset(
    name for name, spec in GATE_SPECS.items() if spec.style == SWAP
)

# Fixed gates that cancel with an identical neighbour on the same qubits,
# and explicit inverse pairs (both directions), all derived from the specs.
SELF_INVERSE_GATES = frozenset(
    name for name, spec in GATE_SPECS.items() if spec.self_inverse
)
INVERSE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    pair
    for name, spec in sorted(GATE_SPECS.items())
    if spec.inverse_name is not None
    for pair in ((name, spec.inverse_name),)
)

# Diagonal single-qubit gates diag(1, phase_factor): the state-vector
# kernel applies the |1> factor directly.
PHASE_FACTORS: Mapping[str, complex] = MappingProxyType(
    {name: spec.phase_factor for name, spec in GATE_SPECS.items() if spec.phase_factor is not None}
)

# Same-qubit commutation groups, all derived from the specs. Two gates in
# one group commute on every qubit they share, so a canonicalizer may move
# them past each other; every other shared-qubit pair keeps its dependency
# order. The diagonal group is z/s/sdg/t/tdg/rz plus the computational-basis
# diagonal two-qubit gates cz/crz.
COMMUTING_X_GATES = frozenset(
    name for name, spec in GATE_SPECS.items()
    if spec.commute_class == COMMUTE_CLASS_PAULI_X
)
COMMUTING_Y_GATES = frozenset(
    name for name, spec in GATE_SPECS.items()
    if spec.commute_class == COMMUTE_CLASS_PAULI_Y
)
DIAGONAL_GATES = frozenset(name for name, spec in GATE_SPECS.items() if spec.is_diagonal)

# Fixed gate-count order for the three estimation schema versions. The
# schema_version 1 mapping keeps its historical six entries; version 2 adds
# the controlled gates; version 3 adds the remaining fixed gates and swap.
GATE_ORDER_V1 = ("x", "h", "cx", "rx", "ry", "rz")
GATE_ORDER_V2 = ("x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz")
GATE_ORDER_V3 = GATE_ORDER_V2 + ("y", "z", "s", "sdg", "t", "tdg", "swap")
SCHEMA_V2_GATES = frozenset(GATE_ORDER_V2) - frozenset(GATE_ORDER_V1)
SCHEMA_V3_GATES = frozenset(GATE_ORDER_V3) - frozenset(GATE_ORDER_V2)

MEASURE = "measure"


def gate_spec(name: str) -> GateSpec:
    """Return the immutable spec of *name* (a canonical gate name)."""
    return GATE_SPECS[name]


def commute_group(name: str) -> str | None:
    """Return the same-qubit commutation group label of gate *name*.

    Gates that share a label commute on every shared qubit: the explicit
    Pauli x/y groups and the derived diagonal group. ``None`` means the gate
    may not cross another gate that shares a qubit with it.
    """
    spec = GATE_SPECS[name]
    if spec.commute_class is not None:
        return spec.commute_class
    if spec.is_diagonal:
        return COMMUTE_CLASS_DIAGONAL
    return None


def gate_operation(
    name: str, targets: tuple[int, ...], params: tuple[float, ...] = ()
) -> Operation:
    """Build an immutable gate application from a canonical gate name."""
    return GATE_SPECS[name].operation(targets, params)


def measurement_operation(qubit: int, clbit: int) -> Operation:
    """Build an immutable ``measure qubit -> clbit`` operation."""
    return Operation(MEASURE, (qubit, clbit))


def rotation_matrix(axis: str, theta: float) -> tuple[complex, complex, complex, complex]:
    """The single-qubit matrix of an rx/ry/rz rotation by *theta* radians.

    This is the one matrix expression shared by the state-vector controlled
    rotations, the density-matrix ``U rho U`` evolution and the unitary
    mode::

        Rx = [[cos t/2, -i sin t/2], [-i sin t/2, cos t/2]]
        Ry = [[cos t/2, -sin t/2], [sin t/2, cos t/2]]
        Rz = [[e^{-i t/2}, 0], [0, e^{i t/2}]]
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if axis == "x":
        return (c, -1j * s, -1j * s, c)
    if axis == "y":
        return (c, -s, s, c)
    if axis == "z":
        return (cmath.exp(-0.5j * theta), 0j, 0j, cmath.exp(0.5j * theta))
    raise AssertionError(f"unknown rotation axis {axis!r}")


def single_qubit_unitary(
    name: str, params: tuple[float, ...] = ()
) -> tuple[complex, complex, complex, complex]:
    """The 2x2 unitary of a single-qubit gate *name* (fixed or rotation)."""
    return GATE_SPECS[name].target_unitary(params)


def controlled_target_unitary(
    name: str, params: tuple[float, ...] = ()
) -> tuple[complex, complex, complex, complex]:
    """The 2x2 unitary embedded on the target of controlled gate *name*."""
    spec = GATE_SPECS[name]
    assert spec.style == CONTROLLED
    return spec.target_unitary(params)


def format_angle(angle: float) -> str:
    """Render an angle deterministically.

    The shortest decimal representation that round-trips to the same float
    (at most 17 significant digits), written as plain decimal or lowercase
    scientific notation. Negative zero is written as ``0``.
    """
    if angle == 0.0:
        return "0"
    return repr(angle)


def gate_qasm(
    name: str,
    qubits: tuple[int, ...],
    angle: float | None = None,
) -> str:
    """Render one gate application as an OpenQASM statement (registers q).

    The single rendering rule shared by the DSL serializer and the
    optimizer: two operands use comma-separated ``q[a],q[b]`` order with no
    space, parameterized gates render their angle in parentheses, and the
    statement ends with ``;``.
    """
    spec = GATE_SPECS[name]
    if spec.parameterized:
        assert angle is not None
        text = format_angle(angle)
        if spec.style == SINGLE_QUBIT:
            return f"{name}({text}) q[{qubits[0]}];"
        return f"{name}({text}) q[{qubits[0]}],q[{qubits[1]}];"
    if spec.style == SINGLE_QUBIT:
        return f"{name} q[{qubits[0]}];"
    return f"{name} q[{qubits[0]}],q[{qubits[1]}];"


__all__ = [
    "COMMUTING_X_GATES",
    "COMMUTING_Y_GATES",
    "CONTROLLED",
    "CONTROLLED_GATES",
    "CONTROLLED_ROTATION_GATES",
    "CONTROLLED_ROTATION_PERIOD",
    "COMMUTE_CLASS_DIAGONAL",
    "COMMUTE_CLASS_PAULI_X",
    "COMMUTE_CLASS_PAULI_Y",
    "DIAGONAL_GATES",
    "GATE_NAMES",
    "GATE_ORDER_V1",
    "GATE_ORDER_V2",
    "GATE_ORDER_V3",
    "GATE_SPECS",
    "GateSpec",
    "INVERSE_PAIRS",
    "MEASURE",
    "Operation",
    "PHASE_FACTORS",
    "ROTATION_GATES",
    "ROTATION_PERIOD",
    "SCHEMA_V2_GATES",
    "SCHEMA_V3_GATES",
    "SELF_INVERSE_GATES",
    "SINGLE_QUBIT",
    "SINGLE_QUBIT_GATES",
    "SWAP",
    "SWAP_GATES",
    "commute_group",
    "controlled_target_unitary",
    "format_angle",
    "gate_operation",
    "gate_qasm",
    "gate_spec",
    "measurement_operation",
    "rotation_matrix",
    "single_qubit_unitary",
]
