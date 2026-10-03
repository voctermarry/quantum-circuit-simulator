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
"""

from __future__ import annotations

import json
import math

OBSERVABLES_SCHEMA_VERSION = 1
MAX_OBSERVABLES = 100

# Expectations this close to an endpoint are reported as the endpoint.
EXPECTATION_TOLERANCE = 1e-15

_OBSERVABLES_ROOT_KEYS = ("schema_version", "observables")
_OBSERVABLE_KEYS = ("id", "operators")
_OPERATOR_KEYS = ("qubit", "pauli")
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


def parse_observables(
    text: str, num_qubits: int
) -> list[tuple[str, list[tuple[int, str]]]]:
    """Parse and validate an observables JSON document.

    Returns the observables in input order as ``(id, operators)`` pairs,
    where each operator is a ``(qubit, pauli)`` tuple. *num_qubits* is the
    parsed circuit's register size and bounds every operator qubit. Raises
    :class:`ObservableError` on any syntax, structural, type or range
    problem.
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
    observables = data["observables"]
    if not isinstance(observables, list):
        raise ObservableError("'observables' must be an array")
    if not 1 <= len(observables) <= MAX_OBSERVABLES:
        raise ObservableError(
            f"'observables' must contain between 1 and {MAX_OBSERVABLES} entries, "
            f"got {len(observables)}"
        )

    parsed: list[tuple[str, list[tuple[int, str]]]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(observables):
        where = f"observables[{index}]"
        if not isinstance(item, dict):
            raise ObservableError(f"{where} must be a JSON object")
        for key in item:
            if key not in _OBSERVABLE_KEYS:
                raise ObservableError(f"{where} has unknown key {key!r}")

        if "id" not in item:
            raise ObservableError(f"{where} is missing required key 'id'")
        observable_id = item["id"]
        if not isinstance(observable_id, str) or not observable_id:
            raise ObservableError(f"{where} 'id' must be a non-empty string")
        if observable_id in seen_ids:
            raise ObservableError(f"duplicate observable id {observable_id!r}")
        seen_ids.add(observable_id)

        if "operators" not in item:
            raise ObservableError(f"{where} is missing required key 'operators'")
        operators = item["operators"]
        if not isinstance(operators, list):
            raise ObservableError(f"{where} 'operators' must be an array")

        parsed_operators: list[tuple[int, str]] = []
        seen_qubits: set[int] = set()
        for op_index, operator in enumerate(operators):
            op_where = f"{where}.operators[{op_index}]"
            if not isinstance(operator, dict):
                raise ObservableError(f"{op_where} must be a JSON object")
            for key in operator:
                if key not in _OPERATOR_KEYS:
                    raise ObservableError(f"{op_where} has unknown key {key!r}")

            if "qubit" not in operator:
                raise ObservableError(f"{op_where} is missing required key 'qubit'")
            qubit = operator["qubit"]
            if not _is_json_int(qubit) or not 0 <= qubit < num_qubits:
                raise ObservableError(
                    f"{op_where} 'qubit' must be an integer between 0 and {num_qubits - 1}, "
                    f"got {qubit!r}"
                )
            if qubit in seen_qubits:
                raise ObservableError(f"{op_where} repeats qubit {qubit} within {where}")
            seen_qubits.add(qubit)

            if "pauli" not in operator:
                raise ObservableError(f"{op_where} is missing required key 'pauli'")
            pauli = operator["pauli"]
            if pauli not in _PAULIS:
                raise ObservableError(
                    f"{op_where} 'pauli' must be one of 'X', 'Y', 'Z', got {pauli!r}"
                )

            parsed_operators.append((qubit, pauli))
        parsed.append((observable_id, parsed_operators))

    return parsed


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
