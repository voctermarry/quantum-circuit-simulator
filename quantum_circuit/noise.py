"""Noise-model parsing and single-qubit noise channel definitions.

A noise model is a JSON object mapping channel names to probabilities, e.g.::

    {"amplitude_damping": 0.1, "bit_flip": 0.05}

Only the four channels listed in :data:`CHANNEL_ORDER` are allowed, at least
one must be present, and every value must be a finite JSON number in [0, 1].
Duplicate keys, unknown keys, non-object documents and empty objects are all
rejected with :class:`NoiseModelError`.
"""

from __future__ import annotations

import json
import math

# Channels are always applied (and echoed back) in this fixed order.
CHANNEL_ORDER = ("amplitude_damping", "phase_damping", "bit_flip", "depolarizing")


class NoiseModelError(Exception):
    """The noise model document is not acceptable (reported as ``noise_model_error``)."""


def parse_noise_model(text: str) -> dict[str, float]:
    """Parse and validate a noise model JSON document.

    Returns a dict keyed by channel name in :data:`CHANNEL_ORDER` order with
    float probabilities, containing exactly the channels that were given.
    Raises :class:`NoiseModelError` on any problem.
    """

    def reject_constant(value: str) -> None:
        # json accepts NaN/Infinity/-Infinity by default; the model requires
        # finite JSON numbers only.
        raise NoiseModelError(f"non-finite number {value!r} is not allowed in the noise model")

    def build_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise NoiseModelError(f"duplicate key {key!r} in the noise model")
            result[key] = value
        return result

    try:
        model = json.loads(text, object_pairs_hook=build_object, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise NoiseModelError(f"noise model is not valid JSON: {exc}") from None

    if not isinstance(model, dict):
        raise NoiseModelError("noise model must be a JSON object")
    if not model:
        raise NoiseModelError("noise model must configure at least one channel")
    for key in model:
        if key not in CHANNEL_ORDER:
            raise NoiseModelError(f"unknown noise channel {key!r}")

    normalized: dict[str, float] = {}
    for name in CHANNEL_ORDER:
        if name not in model:
            continue
        value = model[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NoiseModelError(f"probability for channel {name!r} must be a JSON number")
        if not 0 <= value <= 1:  # also rejects non-finite floats such as 1e999
            raise NoiseModelError(f"probability for channel {name!r} must be between 0 and 1")
        normalized[name] = float(value)
    return normalized


def channel_kraus(name: str, probability: float) -> list[tuple[tuple[complex, ...], ...]]:
    """Return the Kraus operators of channel *name* as 2x2 matrices.

    - ``amplitude_damping``: |1> decays to |0> with probability ``p``.
    - ``phase_damping``: populations unchanged, off-diagonals scaled by ``1 - p``.
    - ``bit_flip``: ``rho -> (1 - p) rho + p X rho X``.
    - ``depolarizing``: ``rho -> (1 - p) rho + p I/2``.
    """
    p = probability
    if name == "amplitude_damping":
        return [
            ((1, 0), (0, math.sqrt(1 - p))),
            ((0, math.sqrt(p)), (0, 0)),
        ]
    if name == "phase_damping":
        # K0 scales |1> by (1 - p); K1 restores the |1> population.
        return [
            ((1, 0), (0, 1 - p)),
            ((0, 0), (0, math.sqrt(p * (2 - p)))),
        ]
    if name == "bit_flip":
        return [
            ((math.sqrt(1 - p), 0), (0, math.sqrt(1 - p))),
            ((0, math.sqrt(p)), (math.sqrt(p), 0)),
        ]
    if name == "depolarizing":
        # rho -> (1 - p) rho + (p/2) Tr(rho) I, via K0 = sqrt(1-p) I and
        # K_ij = sqrt(p/2) |i><j|.
        keep = math.sqrt(1 - p)
        s = math.sqrt(p / 2)
        return [
            ((keep, 0), (0, keep)),
            ((s, 0), (0, 0)),
            ((0, s), (0, 0)),
            ((0, 0), (s, 0)),
            ((0, 0), (0, s)),
        ]
    raise ValueError(f"unknown noise channel {name!r}")
