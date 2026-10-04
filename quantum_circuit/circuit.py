"""Python DSL for building circuits over the OpenQASM 2.0 subset.

A :class:`Circuit` is created with a quantum and a classical register size
and extended by chaining gate and measurement methods, mirroring the
statements of the supported OpenQASM subset one to one::

    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)

Operations keep their insertion order and every chaining method returns the
same circuit object. The circuit renders to (:meth:`Circuit.to_qasm`) and
parses from (:meth:`Circuit.from_qasm`) the OpenQASM 2.0 subset text the
command line accepts, and simulates through the exact code paths of the
``simulate`` and ``probabilities`` commands (:meth:`Circuit.sample`,
:meth:`Circuit.probabilities`), so the returned dictionaries match the
commands' JSON objects field by field.

Validation mirrors the parser's semantic rules: type mismatches raise
:class:`TypeError`, while out-of-range indices, non-finite angles and
semantic conflicts (control equal to target, repeated measurement, a gate
after a measurement) raise :class:`ValueError`. A rejected operation never
changes the circuit.
"""

from __future__ import annotations

import math

from .core import SimulationError, run_probabilities, run_simulation
from .gates import (
    Operation,
    gate_operation,
    gate_qasm,
    measurement_operation,
)
from .noise import validate_noise_model
from .openqasm import Program, parse

# Register sizes accepted by the DSL match the parser's limit.
_MIN_REGISTER_SIZE = 1
_MAX_REGISTER_SIZE = 20

_DEFAULT_SHOTS = 1024
_DEFAULT_SEED = 0


def _check_register_size(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        )
    if not _MIN_REGISTER_SIZE <= value <= _MAX_REGISTER_SIZE:
        raise ValueError(
            f"{name} must be between {_MIN_REGISTER_SIZE} and {_MAX_REGISTER_SIZE}, got {value}"
        )
    return value


def _check_index(register: str, value: object, size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{register} index must be an integer, got {type(value).__name__}"
        )
    if not 0 <= value < size:
        raise ValueError(
            f"{register} index {value} out of range for register of size {size}"
        )
    return value


def _check_angle(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"angle must be an integer or float, got {type(value).__name__}"
        )
    angle = float(value)
    if not math.isfinite(angle):
        raise ValueError("angle must be a finite number")
    return angle


def _operation_line(op: Operation) -> str:
    """Render one operation as an OpenQASM statement (register names q/c)."""
    if op.kind == "measure":
        return f"measure q[{op.targets[0]}] -> c[{op.targets[1]}];"
    return gate_qasm(op.kind, op.targets, op.params[0] if op.params else None)


class Circuit:
    """A quantum circuit over the supported OpenQASM 2.0 subset.

    Created with the quantum and classical register sizes (integers from 1
    to 20, booleans rejected) and extended through the chaining gate and
    measurement methods, which return the circuit itself.
    """

    def __init__(self, num_qubits: int, num_clbits: int):
        self._num_qubits = _check_register_size("num_qubits", num_qubits)
        self._num_clbits = _check_register_size("num_clbits", num_clbits)
        self._operations: list[Operation] = []
        self._measured_qubits: set[int] = set()
        self._measured_clbits: set[int] = set()
        self._measurement_started = False

    # ------------------------------------------------------------ properties

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        return self._num_clbits

    @property
    def operations(self) -> tuple[Operation, ...]:
        """The appended operations, in insertion order."""
        return tuple(self._operations)

    def __repr__(self) -> str:
        return (
            f"Circuit(num_qubits={self._num_qubits}, num_clbits={self._num_clbits}, "
            f"operations={len(self._operations)})"
        )

    # ------------------------------------------------------------------ gates

    def _qubit_index(self, value: object) -> int:
        return _check_index("qubit", value, self._num_qubits)

    def _clbit_index(self, value: object) -> int:
        return _check_index("clbit", value, self._num_clbits)

    def _check_gate_allowed(self, kind: str) -> None:
        if self._measurement_started:
            raise ValueError(f"quantum gate {kind!r} cannot be added after a measurement")

    def _append_gate(self, op: Operation) -> Circuit:
        # All validation has already happened in the calling method, so a
        # rejected operation never reaches this point and never mutates the
        # circuit.
        self._operations.append(op)
        return self

    def _fixed_single_gate(self, name: str, qubit: object) -> Circuit:
        target = self._qubit_index(qubit)
        self._check_gate_allowed(name)
        return self._append_gate(gate_operation(name, (target,)))

    def _controlled_gate(self, name: str, control: object, target: object) -> Circuit:
        ctrl = self._qubit_index(control)
        tgt = self._qubit_index(target)
        if ctrl == tgt:
            raise ValueError(f"{name} control and target must be different qubits")
        self._check_gate_allowed(name)
        return self._append_gate(gate_operation(name, (ctrl, tgt)))

    def _swap_gate(self, first: object, second: object) -> Circuit:
        a = self._qubit_index(first)
        b = self._qubit_index(second)
        if a == b:
            raise ValueError("swap operands must be different qubits")
        self._check_gate_allowed("swap")
        return self._append_gate(gate_operation("swap", (a, b)))

    def _rotation_gate(self, name: str, angle: object, qubit: object) -> Circuit:
        theta = _check_angle(angle)
        target = self._qubit_index(qubit)
        self._check_gate_allowed(name)
        return self._append_gate(gate_operation(name, (target,), (theta,)))

    def _controlled_rotation_gate(
        self, name: str, angle: object, control: object, target: object
    ) -> Circuit:
        theta = _check_angle(angle)
        ctrl = self._qubit_index(control)
        tgt = self._qubit_index(target)
        if ctrl == tgt:
            raise ValueError(f"{name} control and target must be different qubits")
        self._check_gate_allowed(name)
        return self._append_gate(gate_operation(name, (ctrl, tgt), (theta,)))

    def x(self, qubit: int) -> Circuit:
        """Append an ``x`` gate on *qubit*."""
        return self._fixed_single_gate("x", qubit)

    def h(self, qubit: int) -> Circuit:
        """Append an ``h`` gate on *qubit*."""
        return self._fixed_single_gate("h", qubit)

    def y(self, qubit: int) -> Circuit:
        """Append a ``y`` gate on *qubit*."""
        return self._fixed_single_gate("y", qubit)

    def z(self, qubit: int) -> Circuit:
        """Append a ``z`` gate on *qubit*."""
        return self._fixed_single_gate("z", qubit)

    def s(self, qubit: int) -> Circuit:
        """Append an ``s`` gate on *qubit*."""
        return self._fixed_single_gate("s", qubit)

    def sdg(self, qubit: int) -> Circuit:
        """Append an ``sdg`` gate on *qubit*."""
        return self._fixed_single_gate("sdg", qubit)

    def t(self, qubit: int) -> Circuit:
        """Append a ``t`` gate on *qubit*."""
        return self._fixed_single_gate("t", qubit)

    def tdg(self, qubit: int) -> Circuit:
        """Append a ``tdg`` gate on *qubit*."""
        return self._fixed_single_gate("tdg", qubit)

    def cx(self, control: int, target: int) -> Circuit:
        """Append a ``cx`` gate from *control* to *target*."""
        return self._controlled_gate("cx", control, target)

    def cz(self, control: int, target: int) -> Circuit:
        """Append a ``cz`` gate from *control* to *target*."""
        return self._controlled_gate("cz", control, target)

    def swap(self, first: int, second: int) -> Circuit:
        """Append a ``swap`` gate exchanging *first* and *second*."""
        return self._swap_gate(first, second)

    def rx(self, angle: float, qubit: int) -> Circuit:
        """Append an ``rx(angle)`` gate on *qubit* (angle in radians)."""
        return self._rotation_gate("rx", angle, qubit)

    def ry(self, angle: float, qubit: int) -> Circuit:
        """Append an ``ry(angle)`` gate on *qubit* (angle in radians)."""
        return self._rotation_gate("ry", angle, qubit)

    def rz(self, angle: float, qubit: int) -> Circuit:
        """Append an ``rz(angle)`` gate on *qubit* (angle in radians)."""
        return self._rotation_gate("rz", angle, qubit)

    def crx(self, angle: float, control: int, target: int) -> Circuit:
        """Append a ``crx(angle)`` gate from *control* to *target*."""
        return self._controlled_rotation_gate("crx", angle, control, target)

    def cry(self, angle: float, control: int, target: int) -> Circuit:
        """Append a ``cry(angle)`` gate from *control* to *target*."""
        return self._controlled_rotation_gate("cry", angle, control, target)

    def crz(self, angle: float, control: int, target: int) -> Circuit:
        """Append a ``crz(angle)`` gate from *control* to *target*."""
        return self._controlled_rotation_gate("crz", angle, control, target)

    # ------------------------------------------------------------ measurement

    def measure(self, qubit: int, clbit: int) -> Circuit:
        """Append ``measure qubit -> clbit``.

        Each qubit and each clbit may take part in at most one measurement,
        and no quantum gate may be added once measurement has started.
        """
        q = self._qubit_index(qubit)
        c = self._clbit_index(clbit)
        if q in self._measured_qubits:
            raise ValueError(f"qubit {q} is measured more than once")
        if c in self._measured_clbits:
            raise ValueError(f"clbit {c} is the target of more than one measurement")
        self._measured_qubits.add(q)
        self._measured_clbits.add(c)
        self._measurement_started = True
        self._operations.append(measurement_operation(q, c))
        return self

    # --------------------------------------------------------------- OpenQASM

    def to_qasm(self) -> str:
        """Render the circuit as OpenQASM 2.0 subset text.

        Uses the fixed register names ``q`` and ``c``, emits the operations
        in insertion order with the same statement semantics as the parser,
        and ends with a trailing newline. Angles use the shortest decimal
        representation that round-trips to the same float (negative zero is
        written as ``0``).
        """
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self._num_qubits}];",
            f"creg c[{self._num_clbits}];",
        ]
        lines.extend(_operation_line(op) for op in self._operations)
        return "\n".join(lines) + "\n"

    @classmethod
    def from_qasm(cls, source: str) -> Circuit:
        """Build a circuit from OpenQASM 2.0 subset text.

        Parsing and validation are exactly those of the command line:
        :class:`~quantum_circuit.openqasm.ParseError` and
        :class:`~quantum_circuit.openqasm.ValidationError` propagate with
        their 1-based line/column positions intact.
        """
        if not isinstance(source, str):
            raise TypeError(f"source must be a string, got {type(source).__name__}")
        program = parse(source)
        circuit = cls(program.num_qubits, program.num_clbits)
        for op in program.operations:
            # The program is fully validated, so these invariants hold.
            if op.kind == "measure":
                circuit._measured_qubits.add(op.targets[0])
                circuit._measured_clbits.add(op.targets[1])
                circuit._measurement_started = True
            circuit._operations.append(op)
        return circuit

    # -------------------------------------------------------------- simulation

    def _to_program(self) -> Program:
        return Program(self._num_qubits, self._num_clbits, tuple(self._operations))

    @staticmethod
    def _prepare_noise_model(noise_model: object) -> dict[str, float] | None:
        if noise_model is None:
            return None
        return validate_noise_model(noise_model)

    def sample(self, shots: int = _DEFAULT_SHOTS, seed: int = _DEFAULT_SEED, noise_model=None) -> dict:
        """Sample the circuit, returning the ``simulate`` payload as a dict.

        The result equals the JSON object the ``simulate`` command produces
        for :meth:`to_qasm` with the same *shots*, *seed* and noise model
        (defaults 1024 and 0). *noise_model* is a mapping with the four
        optional channels; invalid models raise
        :class:`~quantum_circuit.noise.NoiseModelError`. Each call samples
        from a fresh RNG seeded by *seed*, so repeated calls with equal
        arguments return equal results and no random state is shared. The
        circuit itself is never modified.
        """
        if isinstance(shots, bool) or not isinstance(shots, int):
            raise TypeError(f"shots must be an integer, got {type(shots).__name__}")
        if shots <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
        model = self._prepare_noise_model(noise_model)
        program = self._to_program()
        try:
            return run_simulation(program, shots, seed, model)
        except SimulationError as exc:
            # The DSL reports the noisy size limit as a ValueError, matching
            # its other value-range errors rather than the CLI's exit code 3.
            raise ValueError(str(exc)) from None

    def probabilities(self, noise_model=None) -> dict:
        """Exact final-state measurement probabilities, as a dict.

        The result equals the JSON object the ``probabilities`` command
        produces for :meth:`to_qasm` with the same noise model. The circuit
        itself is never modified.
        """
        model = self._prepare_noise_model(noise_model)
        program = self._to_program()
        try:
            return run_probabilities(program, model)
        except SimulationError as exc:
            raise ValueError(str(exc)) from None


__all__ = ["Circuit"]
