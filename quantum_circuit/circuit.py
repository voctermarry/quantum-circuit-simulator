"""Python circuit DSL for the OpenQASM 2.0 subset.

:class:`Circuit` constructs exactly the circuits the existing parser,
simulator, noise and validation machinery already understand::

    from quantum_circuit import Circuit

    circuit = (
        Circuit(2, 2)
        .h(0)
        .cx(0, 1)
        .measure(0, 0)
        .measure(1, 1)
    )
    circuit.to_qasm()
    circuit.sample()
    circuit.probabilities()

Every gate/measurement appender validates its arguments before touching the
circuit (a failed append leaves the circuit unchanged) and returns the same
circuit object, so calls chain. Type problems raise :class:`TypeError`;
out-of-range indices, non-finite angles and semantic conflicts (gate after
measurement, duplicate measurement, control equal to target) raise
:class:`ValueError`. :meth:`Circuit.from_qasm` keeps using the existing
parser and propagates its :class:`ParseError`/:class:`ValidationError`
(with line and column) unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .noise import MAX_NOISE_QUBITS, NoiseModelError, simulate_density_matrix, validate_noise_model
from .openqasm import Operation, Program, parse
from .results import probability_payload, simulation_payload
from .simulator import simulate_state_vector

# Register sizes share the parser's bounds.
_MAX_QUBITS = 20

_SINGLE_QUBIT_GATES = ("x", "h")
_ROTATIONS = ("rx", "ry", "rz")
_CONTROLLED_GATES = ("cx", "cz")
_CONTROLLED_ROTATIONS = ("crx", "cry", "crz")


def _plain_int(value: object, what: str) -> int:
    """Return *value* as a non-boolean integer, or raise :class:`TypeError`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an integer, got {type(value).__name__}")
    return value


def _finite_angle(value: object, gate: str) -> float:
    """Validate one rotation angle: a finite non-boolean int or float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{gate} angle must be an integer or float, got {type(value).__name__}"
        )
    angle = float(value)
    if not math.isfinite(angle):
        raise ValueError(f"{gate} angle must be finite, got {value!r}")
    if angle == 0.0:
        # Normalize -0.0 so serialization and the stored angle are canonical.
        angle = 0.0
    return angle


def _format_angle(angle: float) -> str:
    """Render an angle as the shortest decimal that round-trips to the float.

    Matches the optimizer's canonical angle text; negative zero is ``0``.
    """
    if angle == 0.0:
        return "0"
    return repr(angle)


class Circuit:
    """A quantum circuit over fixed-size quantum and classical registers."""

    def __init__(self, num_qubits: int, num_clbits: int):
        nq = _plain_int(num_qubits, "num_qubits")
        nc = _plain_int(num_clbits, "num_clbits")
        if not 1 <= nq <= _MAX_QUBITS:
            raise ValueError(f"num_qubits must be between 1 and {_MAX_QUBITS}, got {nq}")
        if not 1 <= nc <= _MAX_QUBITS:
            raise ValueError(f"num_clbits must be between 1 and {_MAX_QUBITS}, got {nc}")
        self._num_qubits = nq
        self._num_clbits = nc
        self._operations: list[Operation] = []
        self._measured_qubits: set[int] = set()
        self._measured_clbits: set[int] = set()
        self._measurement_started = False

    # ------------------------------------------------------------- accessors

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        return self._num_clbits

    def _program(self) -> Program:
        """An immutable :class:`Program` snapshot for the simulation engines."""
        return Program(self._num_qubits, self._num_clbits, tuple(self._operations))

    # ------------------------------------------------------------ validation

    def _check_qubit(self, index: object, what: str) -> int:
        idx = _plain_int(index, what)
        if not 0 <= idx < self._num_qubits:
            raise ValueError(
                f"{what} {idx} out of range for quantum register of size {self._num_qubits}"
            )
        return idx

    def _check_clbit(self, index: object) -> int:
        idx = _plain_int(index, "classical bit index")
        if not 0 <= idx < self._num_clbits:
            raise ValueError(
                f"classical bit {idx} out of range for classical register of size "
                f"{self._num_clbits}"
            )
        return idx

    def _require_gates_allowed(self) -> None:
        if self._measurement_started:
            raise ValueError("no quantum gate can be added after a measurement")

    def _append(self, operation: Operation) -> None:
        # Appends only run after every argument has been validated, so a
        # failed call never mutates this circuit.
        self._operations.append(operation)

    # ------------------------------------------------------------ single gates

    def _single(self, kind: str, target: object) -> "Circuit":
        qubit = self._check_qubit(target, f"{kind} target")
        self._require_gates_allowed()
        self._append(Operation(kind, (qubit,)))
        return self

    def x(self, qubit: int) -> "Circuit":
        """Apply an ``x`` (Pauli-X) gate to *qubit*."""
        return self._single("x", qubit)

    def h(self, qubit: int) -> "Circuit":
        """Apply a Hadamard (``h``) gate to *qubit*."""
        return self._single("h", qubit)

    def _rotation(self, kind: str, target: object, angle: object) -> "Circuit":
        value = _finite_angle(angle, kind)
        qubit = self._check_qubit(target, f"{kind} target")
        self._require_gates_allowed()
        self._append(Operation(kind, (qubit,), (value,)))
        return self

    def rx(self, qubit: int, angle: float) -> "Circuit":
        """Apply an ``rx`` rotation by *angle* radians to *qubit*."""
        return self._rotation("rx", qubit, angle)

    def ry(self, qubit: int, angle: float) -> "Circuit":
        """Apply an ``ry`` rotation by *angle* radians to *qubit*."""
        return self._rotation("ry", qubit, angle)

    def rz(self, qubit: int, angle: float) -> "Circuit":
        """Apply an ``rz`` rotation by *angle* radians to *qubit*."""
        return self._rotation("rz", qubit, angle)

    # --------------------------------------------------------- controlled gates

    def _controlled(self, kind: str, control: object, target: object) -> "Circuit":
        ctrl = self._check_qubit(control, f"{kind} control")
        tgt = self._check_qubit(target, f"{kind} target")
        self._require_gates_allowed()
        if ctrl == tgt:
            raise ValueError(f"{kind} control and target must be different qubits, got {ctrl}")
        self._append(Operation(kind, (ctrl, tgt)))
        return self

    def cx(self, control: int, target: int) -> "Circuit":
        """Apply a controlled-X (``cx``) gate."""
        return self._controlled("cx", control, target)

    def cz(self, control: int, target: int) -> "Circuit":
        """Apply a controlled-Z (``cz``) gate."""
        return self._controlled("cz", control, target)

    def _controlled_rotation(
        self, kind: str, control: object, target: object, angle: object
    ) -> "Circuit":
        value = _finite_angle(angle, kind)
        ctrl = self._check_qubit(control, f"{kind} control")
        tgt = self._check_qubit(target, f"{kind} target")
        self._require_gates_allowed()
        if ctrl == tgt:
            raise ValueError(f"{kind} control and target must be different qubits, got {ctrl}")
        self._append(Operation(kind, (ctrl, tgt), (value,)))
        return self

    def crx(self, control: int, target: int, angle: float) -> "Circuit":
        """Apply a controlled ``rx`` rotation."""
        return self._controlled_rotation("crx", control, target, angle)

    def cry(self, control: int, target: int, angle: float) -> "Circuit":
        """Apply a controlled ``ry`` rotation."""
        return self._controlled_rotation("cry", control, target, angle)

    def crz(self, control: int, target: int, angle: float) -> "Circuit":
        """Apply a controlled ``rz`` rotation."""
        return self._controlled_rotation("crz", control, target, angle)

    # ------------------------------------------------------------- measurement

    def measure(self, qubit: int, clbit: int) -> "Circuit":
        """Measure *qubit* into classical bit *clbit*."""
        q = self._check_qubit(qubit, "measured qubit")
        c = self._check_clbit(clbit)
        if q in self._measured_qubits:
            raise ValueError(f"qubit {q} is measured more than once")
        if c in self._measured_clbits:
            raise ValueError(f"classical bit {c} is the target of more than one measurement")
        self._measured_qubits.add(q)
        self._measured_clbits.add(c)
        self._measurement_started = True
        self._append(Operation("measure", (q, c)))
        return self

    # ------------------------------------------------------------- OpenQASM I/O

    def _operation_line(self, op: Operation) -> str:
        if op.kind in _SINGLE_QUBIT_GATES:
            return f"{op.kind} q[{op.targets[0]}];"
        if op.kind in _CONTROLLED_GATES:
            return f"{op.kind} q[{op.targets[0]}],q[{op.targets[1]}];"
        if op.kind in _ROTATIONS:
            return f"{op.kind}({_format_angle(op.params[0])}) q[{op.targets[0]}];"
        if op.kind in _CONTROLLED_ROTATIONS:
            return (
                f"{op.kind}({_format_angle(op.params[0])}) "
                f"q[{op.targets[0]}],q[{op.targets[1]}];"
            )
        return f"measure q[{op.targets[0]}] -> c[{op.targets[1]}];"

    def to_qasm(self) -> str:
        """Serialize the circuit to the OpenQASM 2.0 subset text.

        Uses fixed ``q``/``c`` register names, keeps operations in insertion
        order, renders angles with a deterministic round-tripping decimal
        (negative zero as ``0``) and ends with a trailing newline.
        """
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self._num_qubits}];",
            f"creg c[{self._num_clbits}];",
        ]
        lines.extend(self._operation_line(op) for op in self._operations)
        return "\n".join(lines) + "\n"

    @classmethod
    def from_qasm(cls, source: str) -> "Circuit":
        """Create a circuit from OpenQASM subset text.

        Propagates the parser's :class:`ParseError` or
        :class:`ValidationError` unchanged (including line and column).
        """
        if not isinstance(source, str):
            raise TypeError("qasm source must be a string")
        program = parse(source)
        circuit = cls(program.num_qubits, program.num_clbits)
        # The parser has already validated every operation and their order;
        # copy its state verbatim so semantic rules are never reported twice.
        # Serialization canonicalizes negative zero when rendering angles.
        circuit._operations = list(program.operations)
        for op in program.operations:
            if op.kind == "measure":
                circuit._measured_qubits.add(op.targets[0])
                circuit._measured_clbits.add(op.targets[1])
                circuit._measurement_started = True
        return circuit

    # -------------------------------------------------------------- simulation

    def _basis_probabilities(self, noise_model: dict[str, float] | None) -> list[float]:
        program = self._program()
        if noise_model is None:
            state = simulate_state_vector(program)
            return [abs(amplitude) ** 2 for amplitude in state]
        if program.num_qubits > MAX_NOISE_QUBITS:
            raise ValueError(
                f"noise simulation supports at most {MAX_NOISE_QUBITS} qubits, "
                f"got {program.num_qubits}"
            )
        return simulate_density_matrix(program, noise_model)

    @staticmethod
    def _validate_model(noise_model: object) -> dict[str, float]:
        if not isinstance(noise_model, Mapping):
            raise NoiseModelError("noise model must be a mapping")
        return validate_noise_model(dict(noise_model))

    def sample(
        self,
        shots: int = 1024,
        seed: int = 0,
        noise_model: Mapping[str, float] | None = None,
    ) -> dict[str, object]:
        """Sample measurement outcomes, matching the CLI ``simulate`` object.

        Returns the same fields and values the ``simulate`` command writes
        as JSON (``schema_version``, ``shots``, ``seed``, ``num_qubits``,
        ``num_clbits``, optional ``noise_model`` and ``counts``). Sampling
        uses a fresh RNG per call, so identical arguments return equal
        results and calls do not share random state. The circuit is not
        modified.
        """
        shots_value = _plain_int(shots, "shots")
        if shots_value <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots_value}")
        seed_value = _plain_int(seed, "seed")
        model = None if noise_model is None else self._validate_model(noise_model)

        basis_probabilities = self._basis_probabilities(model)
        return simulation_payload(self._program(), basis_probabilities, shots_value, seed_value, model)

    def probabilities(
        self, noise_model: Mapping[str, float] | None = None
    ) -> dict[str, object]:
        """Exact final-state probabilities, matching the CLI ``probabilities``.

        Returns the same fields and values the ``probabilities`` command
        writes as JSON. No sampling happens, so the result is deterministic
        and the circuit is not modified.
        """
        model = None if noise_model is None else self._validate_model(noise_model)
        basis_probabilities = self._basis_probabilities(model)
        return probability_payload(self._program(), basis_probabilities, model)

    # ----------------------------------------------------------------- dunder

    def __repr__(self) -> str:
        return (
            f"Circuit(num_qubits={self._num_qubits}, num_clbits={self._num_clbits}, "
            f"operations={len(self._operations)})"
        )
