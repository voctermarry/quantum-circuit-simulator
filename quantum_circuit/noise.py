"""Density-matrix simulation with configurable noise channels.

Enabled via ``simulate --noise-model PATH``. The model is a JSON object with
up to four channel keys (probabilities in ``[0, 1]``):

- ``amplitude_damping``: |1> decays to |0> with probability gamma.
- ``phase_damping``: populations unchanged, off-diagonal entries scaled by
  ``1 - gamma``.
- ``bit_flip``: ``rho -> (1 - p) rho + p X rho X``.
- ``depolarizing``: ``rho -> (1 - p) rho + p I/2`` (the qubit is replaced by
  the maximally mixed state with probability ``p``).

After every quantum gate, each configured channel is applied to every qubit
the gate touched. Qubits of a two-qubit gate (``cx``, ``cz``, ``crx``,
``cry``, ``crz``) are processed in ascending index order; channels on one
qubit are applied in :data:`CHANNEL_ORDER` order.
"""

from __future__ import annotations

import cmath
import json
import math
from collections.abc import Mapping

from .openqasm import Program

# Fixed order in which channels are applied per qubit and echoed back.
CHANNEL_ORDER = ("amplitude_damping", "phase_damping", "bit_flip", "depolarizing")

# A density matrix holds 4**n entries, so noisy simulation is capped well
# below the state-vector register limit.
MAX_NOISE_QUBITS = 10

_SQRT1_2 = 2.0**-0.5


class NoiseModelError(Exception):
    """The noise model content is invalid (reported as ``noise_model_error``)."""


def _reject_constant(value: str) -> None:
    # json accepts NaN/Infinity by default; they are not valid JSON numbers.
    raise NoiseModelError(f"noise model contains non-JSON constant {value!r}")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NoiseModelError(f"duplicate key {key!r} in noise model")
        result[key] = value
    return result


def parse_noise_model(text: str) -> dict[str, float]:
    """Parse and validate noise-model JSON text.

    Returns the configured channels as a mapping in :data:`CHANNEL_ORDER`
    order with float probabilities, so models differing only in whitespace
    or key order produce identical output. Raises :class:`NoiseModelError`
    on any invalid content.
    """
    try:
        data = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise NoiseModelError(f"noise model is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise NoiseModelError("noise model must be a JSON object")
    if not data:
        raise NoiseModelError("noise model must configure at least one channel")
    for key, value in data.items():
        if key not in CHANNEL_ORDER:
            raise NoiseModelError(f"unknown noise channel {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NoiseModelError(f"noise channel {key!r} probability must be a JSON number")
        probability = float(value)
        if not math.isfinite(probability):
            raise NoiseModelError(f"noise channel {key!r} probability must be finite")
        if not 0.0 <= probability <= 1.0:
            raise NoiseModelError(f"noise channel {key!r} probability must be between 0 and 1")
    return {channel: float(data[channel]) for channel in CHANNEL_ORDER if channel in data}


def validate_noise_model(model: object) -> dict[str, float]:
    """Validate a noise model given directly as a mapping (the Python DSL).

    Applies the same channel and probability rules as
    :func:`parse_noise_model` and returns the same canonical mapping in
    :data:`CHANNEL_ORDER` order with float probabilities, so a model passed
    as a Python mapping behaves exactly like the equivalent JSON file.
    Raises :class:`NoiseModelError` on any invalid content.
    """
    if isinstance(model, bool) or not isinstance(model, Mapping):
        raise NoiseModelError("noise model must be a mapping of channel names to probabilities")
    if not model:
        raise NoiseModelError("noise model must configure at least one channel")
    for key, value in model.items():
        if not isinstance(key, str) or key not in CHANNEL_ORDER:
            raise NoiseModelError(f"unknown noise channel {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NoiseModelError(f"noise channel {key!r} probability must be a number")
        probability = float(value)
        if not math.isfinite(probability):
            raise NoiseModelError(f"noise channel {key!r} probability must be finite")
        if not 0.0 <= probability <= 1.0:
            raise NoiseModelError(f"noise channel {key!r} probability must be between 0 and 1")
    return {channel: float(model[channel]) for channel in CHANNEL_ORDER if channel in model}


def _basis_offsets(size: int, bit: int):
    """Yield indices below *size* whose ``bit`` is clear, block by block."""
    step = bit << 1
    for base in range(0, size, step):
        yield from range(base, base + bit)


def _gate_matrix(kind: str, params: tuple[float, ...]) -> tuple[complex, complex, complex, complex]:
    if kind == "x":
        return (0j, 1 + 0j, 1 + 0j, 0j)
    if kind == "h":
        return (_SQRT1_2, _SQRT1_2, _SQRT1_2, -_SQRT1_2)
    theta = params[0]
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if kind == "rx":
        return (c, -1j * s, -1j * s, c)
    if kind == "ry":
        return (c, -s, s, c)
    if kind == "rz":
        return (cmath.exp(-0.5j * theta), 0j, 0j, cmath.exp(0.5j * theta))
    raise AssertionError(f"no single-qubit matrix for gate {kind!r}")


def _apply_single_qubit(rho: list[list[complex]], qubit: int, size: int, matrix) -> None:
    """Apply ``rho -> U rho U†`` for a single-qubit unitary ``U``."""
    bit = 1 << qubit
    g00, g01, g10, g11 = matrix
    h00, h01 = g00.conjugate(), g01.conjugate()
    h10, h11 = g10.conjugate(), g11.conjugate()
    for i0 in _basis_offsets(size, bit):
        i1 = i0 | bit
        row0 = rho[i0]
        row1 = rho[i1]
        for j0 in _basis_offsets(size, bit):
            j1 = j0 | bit
            m00 = row0[j0]
            m01 = row0[j1]
            m10 = row1[j0]
            m11 = row1[j1]
            t00 = g00 * m00 + g01 * m10
            t01 = g00 * m01 + g01 * m11
            t10 = g10 * m00 + g11 * m10
            t11 = g10 * m01 + g11 * m11
            row0[j0] = t00 * h00 + t01 * h01
            row0[j1] = t00 * h10 + t01 * h11
            row1[j0] = t10 * h00 + t11 * h01
            row1[j1] = t10 * h10 + t11 * h11


def _apply_cx(rho: list[list[complex]], control: int, target: int, size: int) -> None:
    """Apply the cx basis permutation to both axes of the density matrix."""
    cbit = 1 << control
    tbit = 1 << target
    swapped = [i for i in range(size) if (i & cbit) and not (i & tbit)]
    for i in swapped:
        j = i | tbit
        rho[i], rho[j] = rho[j], rho[i]
    for i in swapped:
        j = i | tbit
        for row in rho:
            row[i], row[j] = row[j], row[i]


def _apply_controlled(
    rho: list[list[complex]], control: int, target: int, size: int, matrix
) -> None:
    """Apply ``rho -> U rho U†`` for a controlled single-qubit unitary.

    ``U`` acts as the single-qubit *matrix* on *target* when the *control*
    bit is 1 and as the identity otherwise. Done as a left pass (rows whose
    control bit is set) followed by a right pass (columns whose control bit
    is set), which also covers the cross blocks between the two sectors.
    """
    cbit = 1 << control
    tbit = 1 << target
    g00, g01, g10, g11 = matrix
    h00, h01 = g00.conjugate(), g01.conjugate()
    h10, h11 = g10.conjugate(), g11.conjugate()
    for i0 in _basis_offsets(size, tbit):
        if not (i0 & cbit):
            continue
        i1 = i0 | tbit
        row0 = rho[i0]
        row1 = rho[i1]
        for j in range(size):
            m0 = row0[j]
            m1 = row1[j]
            row0[j] = g00 * m0 + g01 * m1
            row1[j] = g10 * m0 + g11 * m1
    for j0 in _basis_offsets(size, tbit):
        if not (j0 & cbit):
            continue
        j1 = j0 | tbit
        for i in range(size):
            row = rho[i]
            m0 = row[j0]
            m1 = row[j1]
            row[j0] = m0 * h00 + m1 * h01
            row[j1] = m0 * h10 + m1 * h11


def _amplitude_damping(rho: list[list[complex]], qubit: int, size: int, gamma: float) -> None:
    bit = 1 << qubit
    scale = math.sqrt(1.0 - gamma)
    damped = 1.0 - gamma
    for i0 in _basis_offsets(size, bit):
        i1 = i0 | bit
        row0 = rho[i0]
        row1 = rho[i1]
        for j0 in _basis_offsets(size, bit):
            j1 = j0 | bit
            m11 = row1[j1]
            row0[j0] = row0[j0] + gamma * m11
            row0[j1] = scale * row0[j1]
            row1[j0] = scale * row1[j0]
            row1[j1] = damped * m11


def _phase_damping(rho: list[list[complex]], qubit: int, size: int, gamma: float) -> None:
    bit = 1 << qubit
    factor = 1.0 - gamma
    for i0 in _basis_offsets(size, bit):
        i1 = i0 | bit
        row0 = rho[i0]
        row1 = rho[i1]
        for j0 in _basis_offsets(size, bit):
            j1 = j0 | bit
            row0[j1] = factor * row0[j1]
            row1[j0] = factor * row1[j0]


def _bit_flip(rho: list[list[complex]], qubit: int, size: int, p: float) -> None:
    bit = 1 << qubit
    keep = 1.0 - p
    for i0 in _basis_offsets(size, bit):
        i1 = i0 | bit
        row0 = rho[i0]
        row1 = rho[i1]
        for j0 in _basis_offsets(size, bit):
            j1 = j0 | bit
            m00 = row0[j0]
            m01 = row0[j1]
            m10 = row1[j0]
            m11 = row1[j1]
            row0[j0] = keep * m00 + p * m11
            row0[j1] = keep * m01 + p * m10
            row1[j0] = keep * m10 + p * m01
            row1[j1] = keep * m11 + p * m00


def _depolarizing(rho: list[list[complex]], qubit: int, size: int, p: float) -> None:
    bit = 1 << qubit
    keep = 1.0 - p
    half = p * 0.5
    for i0 in _basis_offsets(size, bit):
        i1 = i0 | bit
        row0 = rho[i0]
        row1 = rho[i1]
        for j0 in _basis_offsets(size, bit):
            j1 = j0 | bit
            m00 = row0[j0]
            m11 = row1[j1]
            mixed = half * (m00 + m11)
            row0[j0] = keep * m00 + mixed
            row1[j1] = keep * m11 + mixed
            row0[j1] = keep * row0[j1]
            row1[j0] = keep * row1[j0]


_CHANNELS = {
    "amplitude_damping": _amplitude_damping,
    "phase_damping": _phase_damping,
    "bit_flip": _bit_flip,
    "depolarizing": _depolarizing,
}


def evolve_density_matrix(program: Program, noise: dict[str, float]) -> list[list[complex]]:
    """Evolve and return the full final density matrix (row major).

    This is the same gate/noise evolution as :func:`simulate_density_matrix`
    but returns the complete matrix rather than only its diagonal. Noise
    evolution is deterministic and consumes no randomness; measurements do
    not collapse the state, matching the state-vector path.
    """
    n = program.num_qubits
    size = 1 << n
    rho = [[0j] * size for _ in range(size)]
    rho[0][0] = 1 + 0j

    for op in program.operations:
        if op.kind == "measure":
            continue
        if op.kind == "cx":
            _apply_cx(rho, op.targets[0], op.targets[1], size)
        elif op.kind == "cz":
            _apply_controlled(
                rho, op.targets[0], op.targets[1], size, (1 + 0j, 0j, 0j, -1 + 0j)
            )
        elif op.kind in ("crx", "cry", "crz"):
            matrix = _gate_matrix(op.kind[1:], op.params)
            _apply_controlled(rho, op.targets[0], op.targets[1], size, matrix)
        else:
            _apply_single_qubit(rho, op.targets[0], size, _gate_matrix(op.kind, op.params))
        for qubit in sorted(set(op.targets)):
            for channel in CHANNEL_ORDER:
                probability = noise.get(channel)
                if probability:
                    _CHANNELS[channel](rho, qubit, size, probability)

    return rho


def simulate_density_matrix(program: Program, noise: dict[str, float]) -> list[float]:
    """Evolve the density matrix under all gates and noise channels.

    Returns the diagonal of the final density matrix (measurement
    probabilities per basis state). Measurements do not collapse the state,
    matching the state-vector path. Noise evolution is deterministic and
    consumes no randomness.
    """
    rho = evolve_density_matrix(program, noise)
    return [row[i].real for i, row in enumerate(rho)]
