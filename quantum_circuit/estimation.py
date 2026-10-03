"""Static size and feasibility estimation for parsed circuits.

Unlike :mod:`quantum_circuit.simulator`, estimation never evolves the
circuit, samples it, or allocates the exponentially sized state vector,
density matrix or unitary matrix. All figures are derived purely from the
validated :class:`~quantum_circuit.openqasm.Program`.
"""

from __future__ import annotations

from .openqasm import Program

# Gates reported in ``gate_counts``, in the fixed output order. The
# schema_version 1 payload (circuits using only the original gate set) keeps
# the historical six-entry mapping byte-identical; as soon as any of the
# controlled gates ``cz``/``crx``/``cry``/``crz`` appears the payload becomes
# schema_version 2 with all ten entries in this fixed order. Once any of the
# phase/swap gates ``y``/``z``/``s``/``sdg``/``t``/``tdg``/``swap`` appears
# the payload becomes schema_version 3, appending their counts after the ten.
_GATE_ORDER_V1 = ("x", "h", "cx", "rx", "ry", "rz")
_GATE_ORDER_V2 = ("x", "h", "cx", "cz", "rx", "ry", "rz", "crx", "cry", "crz")
_GATE_ORDER_V3 = _GATE_ORDER_V2 + ("y", "z", "s", "sdg", "t", "tdg", "swap")
_V2_GATES = frozenset(_GATE_ORDER_V2) - frozenset(_GATE_ORDER_V1)
_V3_GATES = frozenset(_GATE_ORDER_V3) - frozenset(_GATE_ORDER_V2)

# One complex amplitude/matrix element payload: two doubles (real, imag).
_COMPLEX_BYTES = 16

# mode -> (exponent base of entry_count, qubit limit)
_MODE_TABLE = {
    "state-vector": (2, 20),
    "density-matrix": (4, 10),
    "unitary": (4, 8),
}


def circuit_depth(program: Program) -> int:
    """Compute the circuit depth using an as-soon-as-possible schedule.

    Each gate occupies layer ``1 + max(layer of its qubits)`` and gates are
    processed in source order, so gates on disjoint qubits share a layer.
    Measurements do not contribute to the depth.
    """
    depths = [0] * program.num_qubits
    for op in program.operations:
        if op.kind == "measure":
            continue
        layer = max(depths[qubit] for qubit in op.targets) + 1
        for qubit in op.targets:
            depths[qubit] = layer
    return max(depths, default=0)


def estimate(program: Program, mode: str) -> dict[str, object]:
    """Return the deterministic estimate payload for *program* and *mode*."""
    base, qubit_limit = _MODE_TABLE[mode]

    schema_version = (
        3 if any(op.kind in _V3_GATES for op in program.operations)
        else 2 if any(op.kind in _V2_GATES for op in program.operations)
        else 1
    )
    gate_order = (
        _GATE_ORDER_V3 if schema_version == 3
        else _GATE_ORDER_V2 if schema_version == 2
        else _GATE_ORDER_V1
    )
    gate_counts = {name: 0 for name in gate_order}
    gate_count = 0
    measurement_count = 0
    for op in program.operations:
        if op.kind == "measure":
            measurement_count += 1
        else:
            gate_count += 1
            gate_counts[op.kind] += 1

    entry_count = base**program.num_qubits
    result = {
        "schema_version": schema_version,
        "mode": mode,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "gate_count": gate_count,
        "measurement_count": measurement_count,
        "gate_counts": gate_counts,
        "circuit_depth": circuit_depth(program),
        "entry_count": entry_count,
        "complex_payload_bytes": entry_count * _COMPLEX_BYTES,
        "supported": program.num_qubits <= qubit_limit,
        "qubit_limit": qubit_limit,
    }
    return result
