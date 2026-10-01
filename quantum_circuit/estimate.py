"""Static circuit size and feasibility estimation.

Analyses a parsed :class:`~quantum_circuit.openqasm.Program` without
evolving states, sampling outcomes, or allocating exponentially sized
state vectors or matrices.
"""

from __future__ import annotations

from .openqasm import Program

# Gate kinds reported by the estimate, in fixed output order.
GATE_KINDS = ("x", "h", "cx", "rx", "ry", "rz")

# Execution modes and the largest qubit count each can actually run.
# A state vector holds 2**n amplitudes; density matrices and unitaries
# hold 4**n complex entries.
MODE_QUBIT_LIMITS = {
    "state-vector": 20,
    "density-matrix": 10,
    "unitary": 8,
}

# Bytes per complex number payload (two float64 values).
COMPLEX_BYTES = 16


def circuit_depth(program: Program) -> int:
    """Layer count of the gate sequence, ignoring measurements.

    Operations are processed in source order; each gate lands one layer
    above the deepest layer already occupied by any of its qubits, so
    gates on disjoint qubits may share a layer.
    """
    layers = [0] * program.num_qubits
    depth = 0
    for op in program.operations:
        if op.kind == "measure":
            continue
        layer = max(layers[q] for q in op.targets) + 1
        for q in op.targets:
            layers[q] = layer
        if layer > depth:
            depth = layer
    return depth


def estimate(program: Program, mode: str) -> dict[str, object]:
    """Build the ordered estimate payload for *program* under *mode*.

    The estimate is always produced, even when the circuit exceeds the
    mode's qubit limit (``supported`` is then ``False``); nothing is
    simulated either way.
    """
    gate_counts: dict[str, int] = {kind: 0 for kind in GATE_KINDS}
    measurement_count = 0
    for op in program.operations:
        if op.kind == "measure":
            measurement_count += 1
        else:
            gate_counts[op.kind] += 1

    if mode == "state-vector":
        entry_count = 1 << program.num_qubits
    else:
        entry_count = 1 << (2 * program.num_qubits)
    qubit_limit = MODE_QUBIT_LIMITS[mode]

    return {
        "schema_version": 1,
        "mode": mode,
        "num_qubits": program.num_qubits,
        "num_clbits": program.num_clbits,
        "gate_count": sum(gate_counts.values()),
        "measurement_count": measurement_count,
        "gate_counts": gate_counts,
        "circuit_depth": circuit_depth(program),
        "entry_count": entry_count,
        "complex_payload_bytes": entry_count * COMPLEX_BYTES,
        "supported": program.num_qubits <= qubit_limit,
        "qubit_limit": qubit_limit,
    }
