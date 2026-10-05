"""Carrier-independent domain rules for Pauli-product observable batches.

Both the ``expectation`` command's strict JSON document and the
:meth:`~quantum_circuit.circuit.Circuit.expectation` DSL entry point accept
the same observable *content*, so the rules for that content live here
exactly once:

* a batch holds between 1 and :data:`MAX_OBSERVABLES` observables in
  input order;
* each observable has a non-empty string ``id``, unique within the batch,
  and ``operators`` which may be empty;
* each operator names a non-boolean integer ``qubit`` inside the circuit's
  register, unique within the observable, and a ``pauli`` of ``"X"``,
  ``"Y"`` or ``"Z"``.

Nothing here knows how either entry point reads its input. A
:class:`Carrier` describes only what a carrier can hold: which types count
as a sequence and a mapping, whether booleans may stand in for integers,
and the nouns used in diagnostics. The two carriers (:data:`JSON_CARRIER`
for the strict JSON document and :data:`DSL_CARRIER` for the Python DSL)
turn every structural/type mismatch into a ``"type"`` failure and every
size, identifier, range, duplication or enumeration violation into a
``"value"`` failure, so each entry point keeps its own exception boundary
(:class:`~quantum_circuit.observables.ObservableError` versus
:class:`TypeError`/:class:`ValueError`) while sharing the single set of
rules and their check order.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping

MAX_OBSERVABLES = 100

OBSERVABLE_FIELDS = ("id", "operators")
OPERATOR_FIELDS = ("qubit", "pauli")
PAULIS = ("X", "Y", "Z")

ParsedOperators = list[tuple[int, str]]
ParsedBatch = list[tuple[str, ParsedOperators]]


class ObservableRuleError(Exception):
    """One observable batch broke a shared domain rule.

    ``kind`` is ``"type"`` for carrier-shape and value-type mismatches and
    ``"value"`` for size, field, identifier, range, duplication and
    enumeration violations. Each entry point maps the two kinds onto its
    own exception types.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class Carrier:
    """The container/type conventions of one input carrier.

    *sequence_types* are the accepted batch/operator containers (exact
    instances, so a mapping that is also a sequence is never accepted by
    accident); *mapping_type* is the accepted observable/operator record
    type; *rejects_bool_as_int* controls whether ``bool`` masquerading as
    ``int`` is a type failure. The nouns decorate diagnostics:
    *mapping_noun* ("JSON object" vs "mapping"), *sequence_noun*
    ("array" vs "list or tuple") and *field_noun* ("key" vs "field").
    """

    __slots__ = (
        "sequence_types",
        "mapping_type",
        "rejects_bool_as_int",
        "mapping_noun",
        "sequence_noun",
        "field_noun",
    )

    def __init__(
        self,
        sequence_types: tuple[type, ...],
        mapping_type: type,
        rejects_bool_as_int: bool,
        mapping_noun: str,
        sequence_noun: str,
        field_noun: str,
    ):
        self.sequence_types = sequence_types
        self.mapping_type = mapping_type
        self.rejects_bool_as_int = rejects_bool_as_int
        self.mapping_noun = mapping_noun
        self.sequence_noun = sequence_noun
        self.field_noun = field_noun

    def is_sequence(self, value: object) -> bool:
        return isinstance(value, self.sequence_types)

    def is_mapping(self, value: object) -> bool:
        return isinstance(value, self.mapping_type)

    def is_int(self, value: object) -> bool:
        if not isinstance(value, int):
            return False
        return not (self.rejects_bool_as_int and isinstance(value, bool))


JSON_CARRIER = Carrier(
    sequence_types=(list,),
    mapping_type=dict,
    rejects_bool_as_int=True,
    mapping_noun="JSON object",
    sequence_noun="array",
    field_noun="key",
)

DSL_CARRIER = Carrier(
    sequence_types=(list, tuple),
    mapping_type=_Mapping,
    rejects_bool_as_int=True,
    mapping_noun="mapping",
    sequence_noun="list or tuple",
    field_noun="field",
)


def _type_error(message: str) -> ObservableRuleError:
    return ObservableRuleError("type", message)


def _value_error(message: str) -> ObservableRuleError:
    return ObservableRuleError("value", message)


def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def validate_operators(
    carrier: Carrier, operators: object, where: str, num_qubits: int
) -> ParsedOperators:
    """Validate one observable's ``operators`` value against shared rules."""
    if not carrier.is_sequence(operators):
        raise _type_error(
            f"{where} 'operators' must be {_article(carrier.sequence_noun)} "
            f"{carrier.sequence_noun}, got {type(operators).__name__}"
        )

    parsed: ParsedOperators = []
    seen_qubits: set[int] = set()
    for op_index, operator in enumerate(operators):
        op_where = f"{where}.operators[{op_index}]"
        if not carrier.is_mapping(operator):
            raise _type_error(
                f"{op_where} must be {_article(carrier.mapping_noun)} "
                f"{carrier.mapping_noun}, got {type(operator).__name__}"
            )
        for key in operator:
            if key not in OPERATOR_FIELDS:
                raise _value_error(
                    f"{op_where} has unknown {carrier.field_noun} {key!r}"
                )

        if "qubit" not in operator:
            raise _value_error(
                f"{op_where} is missing required {carrier.field_noun} 'qubit'"
            )
        qubit = operator["qubit"]
        if not carrier.is_int(qubit):
            raise _type_error(
                f"{op_where} 'qubit' must be an integer, "
                f"got {type(qubit).__name__}"
            )
        if not 0 <= qubit < num_qubits:
            raise _value_error(
                f"{op_where} 'qubit' {qubit} out of range for register of "
                f"size {num_qubits}"
            )
        if qubit in seen_qubits:
            raise _value_error(f"{op_where} repeats qubit {qubit} within {where}")
        seen_qubits.add(qubit)

        if "pauli" not in operator:
            raise _value_error(
                f"{op_where} is missing required {carrier.field_noun} 'pauli'"
            )
        pauli = operator["pauli"]
        if not isinstance(pauli, str):
            raise _type_error(
                f"{op_where} 'pauli' must be a string, got {type(pauli).__name__}"
            )
        if pauli not in PAULIS:
            raise _value_error(
                f"{op_where} 'pauli' must be one of 'X', 'Y', 'Z', got {pauli!r}"
            )

        parsed.append((qubit, pauli))
    return parsed


def validate_observables_batch(
    carrier: Carrier, observables: object, num_qubits: int
) -> ParsedBatch:
    """Validate one observables batch, returning ``(id, operators)`` pairs.

    The batch is checked in input order and the whole batch is validated
    before any state evolution happens. Raises :class:`ObservableRuleError`
    on the first problem; the caller's containers are never modified.
    """
    if not carrier.is_sequence(observables):
        raise _type_error(
            f"observables must be {_article(carrier.sequence_noun)} "
            f"{carrier.sequence_noun}, got {type(observables).__name__}"
        )
    if not 1 <= len(observables) <= MAX_OBSERVABLES:
        raise _value_error(
            f"observables must contain between 1 and {MAX_OBSERVABLES} entries, "
            f"got {len(observables)}"
        )

    parsed: ParsedBatch = []
    seen_ids: set[str] = set()
    for index, item in enumerate(observables):
        where = f"observables[{index}]"
        if not carrier.is_mapping(item):
            raise _type_error(
                f"{where} must be {_article(carrier.mapping_noun)} "
                f"{carrier.mapping_noun}, got {type(item).__name__}"
            )
        for key in item:
            if key not in OBSERVABLE_FIELDS:
                raise _value_error(
                    f"{where} has unknown {carrier.field_noun} {key!r}"
                )

        if "id" not in item:
            raise _value_error(
                f"{where} is missing required {carrier.field_noun} 'id'"
            )
        observable_id = item["id"]
        if not isinstance(observable_id, str):
            raise _type_error(
                f"{where} 'id' must be a string, got {type(observable_id).__name__}"
            )
        if not observable_id:
            raise _value_error(f"{where} 'id' must be a non-empty string")
        if observable_id in seen_ids:
            raise _value_error(f"duplicate observable id {observable_id!r}")
        seen_ids.add(observable_id)

        if "operators" not in item:
            raise _value_error(
                f"{where} is missing required {carrier.field_noun} 'operators'"
            )
        parsed.append(
            (
                observable_id,
                validate_operators(carrier, item["operators"], where, num_qubits),
            )
        )
    return parsed
