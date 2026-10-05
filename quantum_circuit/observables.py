"""Pauli-product observable expectation values.

Used by the ``expectation`` subcommand, which evaluates a list of Pauli
products against the final pre-measurement state. Each observable is a
tensor product of single-qubit Pauli operators; qubits not listed are
acted on by the identity, and an empty operator list is the global
identity. Expectations follow ``Tr(rho P)`` (the state-vector path uses
``<psi|P|psi>``), so they live in ``[-1, 1]``.

The observables document is UTF-8 JSON with the same deterministic,
strict structural style as the other JSON inputs: duplicate keys,
unknown keys, type substitutions (including booleans posing as
integers) and out-of-range counts are all rejected.

The domain rules are shared by both entry points: the ``expectation``
command validates a decoded JSON document through
:func:`parse_observables`, while :meth:`Circuit.expectation
<quantum_circuit.circuit.Circuit.expectation>` validates native Python
objects through :func:`validate_observables`. Both are thin carriers
around :func:`_validate_observables`, the single carrier-independent
source of the batch/operator/qubit/Pauli rules; only the container
predicates (strict JSON ``dict``/``list`` versus mappings and
lists/tuples), the error wording and the exception types differ.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

OBSERVABLES_SCHEMA_VERSION = 1
MAX_OBSERVABLES = 100

# Expectations this close to an endpoint are reported as the endpoint.
EXPECTATION_TOLERANCE = 1e-15

_OBSERVABLES_ROOT_KEYS = ("schema_version", "observables")
_OBSERVABLE_FIELDS = ("id", "operators")
_OPERATOR_FIELDS = ("qubit", "pauli")
_PAULIS = ("X", "Y", "Z")


class ObservableError(Exception):
    """The observables document is invalid (``observable_error``)."""


def _is_json_int(value: object) -> bool:
    # bool is a subclass of int but is not a JSON integer here.
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_constant(value: str) -> None:
    # json accepts NaN/Infinity by default; they are not valid JSON numbers.
    raise ObservableError(f"observables document contains non-JSON constant {value!r}")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ObservableError(f"duplicate key {key!r} in observables document")
        result[key] = value
    return result


class _ObservablesCarrier:
    """Carrier-specific containers, wording and exception classes.

    The carrier-independent traversal lives once in
    :func:`_validate_observables`; a carrier only decides what counts as
    a mapping or a sequence, how each failure is worded and whether a
    type failure surfaces as :class:`TypeError` (the Python DSL) or as
    the carrier-neutral :class:`ObservableError` (the JSON command).
    """

    type_exc: type[Exception] = TypeError
    value_exc: type[Exception] = ValueError
    object_phrase = "a mapping"
    sequence_phrase = "a list or tuple"
    field_noun = "field"
    batch_name = "observables"
    append_type_name = True

    def is_mapping(self, value: object) -> bool:
        return isinstance(value, Mapping)

    def is_sequence(self, value: object) -> bool:
        # Lists and tuples only: strings, booleans and arbitrary other
        # iterables/generators are not observable batches.
        return isinstance(value, (list, tuple))

    def is_int(self, value: object) -> bool:
        # bool is a subclass of int but is never accepted as a qubit here.
        return isinstance(value, int) and not isinstance(value, bool)

    def _type(self, message: str) -> Exception:
        return self.type_exc(message)

    def _value(self, message: str) -> Exception:
        return self.value_exc(message)

    def _container_error(self, label: str, phrase: str, value: object) -> Exception:
        message = f"{label} must be {phrase}"
        if self.append_type_name:
            message += f", got {type(value).__name__}"
        return self._type(message)

    def batch_not_sequence(self, value: object) -> Exception:
        return self._container_error(self.batch_name, self.sequence_phrase, value)

    def bad_count(self, count: int) -> Exception:
        return self._value(
            f"{self.batch_name} must contain between 1 and {MAX_OBSERVABLES} "
            f"entries, got {count}"
        )

    def item_not_mapping(self, where: str, value: object) -> Exception:
        return self._container_error(where, self.object_phrase, value)

    def unknown_field(self, where: str, key: object) -> Exception:
        return self._value(f"{where} has unknown {self.field_noun} {key!r}")

    def missing_field(self, where: str, name: str) -> Exception:
        return self._value(f"{where} is missing required {self.field_noun} {name!r}")

    def id_wrong_type(self, where: str, value: object) -> Exception:
        return self._type(f"{where} 'id' must be a string, got {type(value).__name__}")

    def id_empty(self, where: str) -> Exception:
        return self._value(f"{where} 'id' must be a non-empty string")

    def duplicate_id(self, value: str) -> Exception:
        return self._value(f"duplicate observable id {value!r}")

    def operators_not_sequence(self, where: str, value: object) -> Exception:
        return self._container_error(f"{where} 'operators'", self.sequence_phrase, value)

    def operator_not_mapping(self, op_where: str, value: object) -> Exception:
        return self._container_error(op_where, self.object_phrase, value)

    def qubit_wrong_type(self, op_where: str, value: object, num_qubits: int) -> Exception:
        return self._type(
            f"{op_where} 'qubit' must be an integer, got {type(value).__name__}"
        )

    def qubit_out_of_range(self, op_where: str, qubit: int, num_qubits: int) -> Exception:
        return self._value(
            f"{op_where} 'qubit' {qubit} out of range for register of size {num_qubits}"
        )

    def duplicate_qubit(self, op_where: str, qubit: int, where: str) -> Exception:
        return self._value(f"{op_where} repeats qubit {qubit} within {where}")

    def pauli_wrong_type(self, op_where: str, value: object) -> Exception:
        return self._type(
            f"{op_where} 'pauli' must be a string, got {type(value).__name__}"
        )

    def bad_pauli(self, op_where: str, value: object) -> Exception:
        return self._value(
            f"{op_where} 'pauli' must be one of 'X', 'Y', 'Z', got {value!r}"
        )


class _JsonObservablesCarrier(_ObservablesCarrier):
    """Strict JSON shapes: only ``dict``/``list``, every failure one error."""

    type_exc = ObservableError
    value_exc = ObservableError
    object_phrase = "a JSON object"
    sequence_phrase = "an array"
    field_noun = "key"
    batch_name = "'observables'"
    append_type_name = False

    def is_mapping(self, value: object) -> bool:
        return isinstance(value, dict)

    def is_sequence(self, value: object) -> bool:
        return isinstance(value, list)

    # The document historically reports type and content of 'id', 'qubit'
    # and 'pauli' with one combined message each.
    def id_wrong_type(self, where: str, value: object) -> Exception:
        return ObservableError(f"{where} 'id' must be a non-empty string")

    def id_empty(self, where: str) -> Exception:
        return ObservableError(f"{where} 'id' must be a non-empty string")

    def qubit_wrong_type(self, op_where: str, value: object, num_qubits: int) -> Exception:
        return ObservableError(
            f"{op_where} 'qubit' must be an integer between 0 and {num_qubits - 1}, "
            f"got {value!r}"
        )

    def qubit_out_of_range(self, op_where: str, qubit: int, num_qubits: int) -> Exception:
        return ObservableError(
            f"{op_where} 'qubit' must be an integer between 0 and {num_qubits - 1}, "
            f"got {qubit!r}"
        )

    def pauli_wrong_type(self, op_where: str, value: object) -> Exception:
        return ObservableError(
            f"{op_where} 'pauli' must be one of 'X', 'Y', 'Z', got {value!r}"
        )


_PYTHON_CARRIER = _ObservablesCarrier()
_JSON_CARRIER = _JsonObservablesCarrier()


def _validate_observables(
    carrier: _ObservablesCarrier,
    observables: object,
    num_qubits: int,
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Validate one observables batch, the single carrier-independent rule set.

    Returns the observables in input order as ``(id, operators)`` pairs,
    each operator a ``(qubit, pauli)`` tuple. The batch is checked in one
    fixed order — container, count, then per observable the unknown-field
    sweep, ``id`` (presence, type, emptiness, uniqueness) and
    ``operators`` (container, then per operator the unknown-field sweep,
    ``qubit`` presence/type/range/uniqueness and ``pauli``
    presence/type/value) — before anything is returned, so callers always
    fail before evolving any state. *carrier* supplies only the accepted
    container types, the wording and the exception classes.
    """
    if not carrier.is_sequence(observables):
        raise carrier.batch_not_sequence(observables)
    if not 1 <= len(observables) <= MAX_OBSERVABLES:
        raise carrier.bad_count(len(observables))

    parsed: list[tuple[str, list[tuple[int, str]]]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(observables):
        where = f"observables[{index}]"
        if not carrier.is_mapping(item):
            raise carrier.item_not_mapping(where, item)
        for key in item:
            if key not in _OBSERVABLE_FIELDS:
                raise carrier.unknown_field(where, key)

        if "id" not in item:
            raise carrier.missing_field(where, "id")
        observable_id = item["id"]
        if not isinstance(observable_id, str):
            raise carrier.id_wrong_type(where, observable_id)
        if not observable_id:
            raise carrier.id_empty(where)
        if observable_id in seen_ids:
            raise carrier.duplicate_id(observable_id)
        seen_ids.add(observable_id)

        if "operators" not in item:
            raise carrier.missing_field(where, "operators")
        operators = item["operators"]
        if not carrier.is_sequence(operators):
            raise carrier.operators_not_sequence(where, operators)

        parsed_operators: list[tuple[int, str]] = []
        seen_qubits: set[int] = set()
        for op_index, operator in enumerate(operators):
            op_where = f"{where}.operators[{op_index}]"
            if not carrier.is_mapping(operator):
                raise carrier.operator_not_mapping(op_where, operator)
            for key in operator:
                if key not in _OPERATOR_FIELDS:
                    raise carrier.unknown_field(op_where, key)

            if "qubit" not in operator:
                raise carrier.missing_field(op_where, "qubit")
            qubit = operator["qubit"]
            if not carrier.is_int(qubit):
                raise carrier.qubit_wrong_type(op_where, qubit, num_qubits)
            if not 0 <= qubit < num_qubits:
                raise carrier.qubit_out_of_range(op_where, qubit, num_qubits)
            if qubit in seen_qubits:
                raise carrier.duplicate_qubit(op_where, qubit, where)
            seen_qubits.add(qubit)

            if "pauli" not in operator:
                raise carrier.missing_field(op_where, "pauli")
            pauli = operator["pauli"]
            if not isinstance(pauli, str):
                raise carrier.pauli_wrong_type(op_where, pauli)
            if pauli not in _PAULIS:
                raise carrier.bad_pauli(op_where, pauli)

            parsed_operators.append((qubit, pauli))
        parsed.append((observable_id, parsed_operators))

    return parsed


def validate_observables(
    observables: object, num_qubits: int
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Validate native Python observables for ``Circuit.expectation``.

    Mirrors :func:`parse_observables` for the DSL's public input shapes:
    a list or tuple of mappings whose ``operators`` are lists or tuples
    of mappings. Type mismatches raise the built-in :class:`TypeError`;
    size, field, identifier, qubit and Pauli violations raise the
    built-in :class:`ValueError` -- the exception type is itself a
    carrier property, so this path shares the rule set without adopting
    the command's :class:`ObservableError`.
    """
    return _validate_observables(_PYTHON_CARRIER, observables, num_qubits)


def parse_observables(
    text: str, num_qubits: int
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Parse and validate an observables JSON document.

    Returns the observables in input order as ``(id, operators)`` pairs,
    where each operator is a ``(qubit, pauli)`` tuple. *num_qubits* is the
    parsed circuit's register size and bounds every operator qubit. Raises
    :class:`ObservableError` on any syntax, structural, type or range
    problem.

    The JSON-specific envelope (syntax, duplicate keys, non-standard
    constants, the root object, ``schema_version``) is checked here; the
    observable entries themselves go through the same
    :func:`_validate_observables` traversal as the Python DSL.
    """
    try:
        data = json.loads(
            text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ObservableError(f"observables document is not valid JSON: {exc}") from None

    if not isinstance(data, dict):
        raise ObservableError("observables document must be a JSON object")
    for key in data:
        if key not in _OBSERVABLES_ROOT_KEYS:
            raise ObservableError(f"unknown observables document key {key!r}")

    if "schema_version" not in data:
        raise ObservableError("observables document is missing required key 'schema_version'")
    schema_version = data["schema_version"]
    if not _is_json_int(schema_version) or schema_version != OBSERVABLES_SCHEMA_VERSION:
        raise ObservableError(
            f"observables schema_version must be {OBSERVABLES_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )

    if "observables" not in data:
        raise ObservableError("observables document is missing required key 'observables'")

    return _validate_observables(_JSON_CARRIER, data["observables"], num_qubits)


def snap_expectation(value: float) -> float | int:
    """Snap one expectation for deterministic, in-range output.

    Values no more than :data:`EXPECTATION_TOLERANCE` from zero become
    ``0.0``; values within that distance of -1 or 1 become the integers
    ``-1``/``1``. Other values are clamped to ``[-1, 1]`` so finite
    precision round-off can never report an out-of-range expectation.
    """
    value = min(1.0, max(-1.0, float(value)))
    if abs(value) <= EXPECTATION_TOLERANCE:
        return 0.0
    if abs(value - 1.0) <= EXPECTATION_TOLERANCE:
        return 1
    if abs(value + 1.0) <= EXPECTATION_TOLERANCE:
        return -1
    return value


def _phase_bits(operators: list[tuple[int, str]]) -> tuple[int, int, int]:
    """Decompose a Pauli product into (X-bits, Y-bits, Z-bits) masks."""
    x_bits = 0
    y_bits = 0
    z_bits = 0
    for qubit, pauli in operators:
        bit = 1 << qubit
        if pauli == "X":
            x_bits |= bit
        elif pauli == "Y":
            y_bits |= bit
        else:
            z_bits |= bit
    return x_bits, y_bits, z_bits


def _pauli_phase(index: int, parity_bits: int, y_factor: complex) -> complex:
    """Scalar phase of ``P|index>``.

    Z sites contribute ``(-1)**b`` and Y sites contribute
    ``i(-1)**b``; over *m* Y sites the shared ``i**m`` is *y_factor*,
    so the total phase is ``i**m`` times the parity sign over the Z and
    Y sites.
    """
    sign = -1.0 if (index & parity_bits).bit_count() & 1 else 1.0
    return sign * y_factor


# i**m for the Y-site count modulo 4.
_I_POWERS = (1 + 0j, 1j, -1 + 0j, -1j)


def state_vector_expectation(
    state: list[complex], operators: list[tuple[int, str]]
) -> float:
    """Return ``<psi|P|psi>`` for a Pauli product *P*.

    The state is renormalized once (gate evolution drifts the norm by a
    few ulps, mirroring the explicit normalization in the metrics path),
    so the global-identity observable is exactly 1.
    """
    norm = math.sqrt(math.fsum(abs(amplitude) ** 2 for amplitude in state))
    if norm == 0.0:
        return 0.0

    x_bits, y_bits, z_bits = _phase_bits(operators)
    flip_bits = x_bits | y_bits
    parity_bits = z_bits | y_bits

    terms: list[float] = []
    if flip_bits == 0:
        # P is diagonal (only Z/I): P|j> = (-1)^<z,j> |j>.
        for index, amplitude in enumerate(state):
            weight = abs(amplitude) ** 2
            terms.append(
                (-weight if (index & z_bits).bit_count() & 1 else weight) / (norm * norm)
            )
        return math.fsum(terms)

    y_factor = _I_POWERS[y_bits.bit_count() & 3]
    for index, amplitude in enumerate(state):
        phase = _pauli_phase(index, parity_bits, y_factor)
        value = (state[index ^ flip_bits].conjugate() * phase * amplitude) / (norm * norm)
        terms.append(value.real)
    return math.fsum(terms)


def density_matrix_expectation(
    rho: list[list[complex]], operators: list[tuple[int, str]]
) -> float:
    """Return ``Tr(rho P)`` for a Pauli product *P*.

    Uses ``Tr(rho P) = sum_j phase(j) rho[j, j xor F]`` where *F* is the
    bit mask of the X/Y sites; *rho* must already be normalized to unit
    trace.
    """
    x_bits, y_bits, z_bits = _phase_bits(operators)
    flip_bits = x_bits | y_bits
    parity_bits = z_bits | y_bits

    if flip_bits == 0:
        terms: list[float] = []
        for index, row in enumerate(rho):
            diagonal = row[index].real
            terms.append(
                -diagonal if (index & z_bits).bit_count() & 1 else diagonal
            )
        return math.fsum(terms)

    y_factor = _I_POWERS[y_bits.bit_count() & 3]
    terms = []
    for index, row in enumerate(rho):
        phase = _pauli_phase(index, parity_bits, y_factor)
        terms.append((row[index ^ flip_bits] * phase).real)
    return math.fsum(terms)
