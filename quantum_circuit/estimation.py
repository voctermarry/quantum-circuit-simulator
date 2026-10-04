"""Static size and feasibility estimation for parsed circuits.

Unlike :mod:`quantum_circuit.simulator`, estimation never evolves the
circuit, samples it, or allocates the exponentially sized state vector,
density matrix or unitary matrix. All figures are derived purely from the
validated :class:`~quantum_circuit.openqasm.Program`.
"""

from __future__ import annotations

from .gates import (
    GATE_ORDER_V1 as _GATE_ORDER_V1,
    GATE_ORDER_V2 as _GATE_ORDER_V2,
    GATE_ORDER_V3 as _GATE_ORDER_V3,
    SCHEMA_V2_GATES as _V2_GATES,
    SCHEMA_V3_GATES as _V3_GATES,
)
from .openqasm import Program

# Gate-count orders for the three schema versions, fixed in the single gate
# registry. The schema_version 1 payload keeps the historical six-entry
# mapping byte-identical; as soon as any of the controlled gates
# ``cz``/``crx``/``cry``/``crz`` appears the payload becomes schema_version 2
# with all ten entries; any of the gates
# ``y``/``z``/``s``/``sdg``/``t``/``tdg``/``swap`` makes it schema_version 3
# with seventeen entries.

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

    if any(op.kind in _V3_GATES for op in program.operations):
        schema_version = 3
        gate_order = _GATE_ORDER_V3
    elif any(op.kind in _V2_GATES for op in program.operations):
        schema_version = 2
        gate_order = _GATE_ORDER_V2
    else:
        schema_version = 1
        gate_order = _GATE_ORDER_V1
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
