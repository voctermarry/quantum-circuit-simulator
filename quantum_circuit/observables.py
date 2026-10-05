"""Pauli-product observable expectation values.

Used by the ``expectation`` subcommand, which evaluates a list of Pauli
products against the final pre-measurement state. Each observable is a
tensor product of single-qubit Pauli operators; qubits not listed are
acted on by the identity, and an empty operator list is the global
identity. Expectations follow ``Tr(rho P)`` (the state-vector path uses
``<psi|P|psi>``), so they live in ``[-1, 1]``.

This module owns the command's strict JSON document envelope: UTF-8 JSON
with duplicate keys, unknown keys, non-standard ``NaN``/``Infinity``
constants and booleans posing as integers all rejected, plus the required
``schema_version`` 1. The carrier-independent content rules (batch size,
ids and operator qubits/Paulis) live once in :mod:`observable_rules` and
are shared with the Python DSL; here they are merely adapted to the JSON
document's array/object conventions and exception boundary.
"""

from __future__ import annotations

import json
import math

from .observable_rules import (
    JSON_CARRIER,
    MAX_OBSERVABLES,  # re-exported: the limit is defined once with the rules
    ObservableRuleError,
    validate_observables_batch,
)

OBSERVABLES_SCHEMA_VERSION = 1

# Expectations this close to an endpoint are reported as the endpoint.
EXPECTATION_TOLERANCE = 1e-15

_OBSERVABLES_ROOT_KEYS = ("schema_version", "observables")


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

    Only the strict JSON envelope (syntax, the root object and
    ``schema_version``) is command-specific; the observables content is
    checked by the shared :func:`validate_observables_batch` rules.
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

    try:
        return validate_observables_batch(JSON_CARRIER, data["observables"], num_qubits)
    except ObservableRuleError as exc:
        raise ObservableError(exc.message) from None



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
