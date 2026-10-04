"""Deterministic differential tests over the full seventeen-gate set.

A fixed battery of seeded generators builds small legal circuits (one to
four qubits) through the Python DSL, covering every supported gate,
adjacent inverse pairs, repeated rotations, permutations of mutually
disjoint gates and angles at or near the rotation periods. Each sample is
serialized with :meth:`Circuit.to_qasm` and rebuilt with
:meth:`Circuit.from_qasm`; the original and the rebuilt circuit must render
identical QASM and return identical ``probabilities``/``sample`` payloads.
Every rebuilt circuit is then executed on the noiseless state-vector path
and on the density-matrix path with a zero-probability noise model, where
the payloads must agree once the contract-mandated ``schema_version`` and
``noise_model`` fields are set aside. All data, traversal orders and
assertions are deterministic; every failure message carries the sample
seed and the reproducible QASM text.
"""

from __future__ import annotations

import math
import random

import pytest

from quantum_circuit import Circuit

_FIXED_GATES = ("x", "h", "y", "z", "s", "sdg", "t", "tdg")
_ROTATION_GATES = ("rx", "ry", "rz")
_CONTROLLED_GATES = ("cx", "cz")
_CONTROLLED_ROTATION_GATES = ("crx", "cry", "crz")
_ALL_GATES = frozenset(
    _FIXED_GATES
    + _ROTATION_GATES
    + _CONTROLLED_GATES
    + _CONTROLLED_ROTATION_GATES
    + ("swap",)
)

_ROTATION_PERIOD = 2.0 * math.pi

# Angles exactly at zero, at the single (2*pi) and controlled (4*pi)
# rotation periods and within 1e-9 of zero, exercised alongside
# pseudo-random angles drawn from the seeded generator.
_BOUNDARY_ANGLES = (
    0.0,
    1e-15,
    -1e-15,
    1e-12,
    -1e-12,
    math.pi / 4,
    math.pi / 2,
    math.pi,
    -math.pi,
    _ROTATION_PERIOD,
    -_ROTATION_PERIOD,
    2.0 * _ROTATION_PERIOD,
    -2.0 * _ROTATION_PERIOD,
    3.0 * math.pi,
)

# Measurement modes, rotated by sample seed.
_MODE_NONE = 0
_MODE_PARTIAL = 1
_MODE_PERMUTED = 2
_MODE_ALL = 3

_SEED_BASE = 20260401
SAMPLE_SEEDS = tuple(_SEED_BASE + offset for offset in range(64))

_SAMPLE_SHOTS = 384
_SAMPLE_SEED = 918273645

# The density-matrix path is exercised with zero-probability channels only,
# so its results must match the noiseless state-vector path.
_ZERO_NOISE_MODEL = {"amplitude_damping": 0.0, "bit_flip": 0.0}

# Fields that differ between the two simulation paths by contract.
_CONTRACT_ONLY_FIELDS = frozenset({"schema_version", "noise_model"})

# Bit-exact agreement is not meaningful across two different floating-point
# evolutions: probability values must agree to this deterministic round-off
# bound, while keys, counts and every integer field must match exactly.
_PROBABILITY_TOLERANCE = 1e-12


# --------------------------------------------------------------- generator


def _random_angle(rng: random.Random) -> float:
    roll = rng.randrange(3)
    if roll == 0:
        return rng.choice(_BOUNDARY_ANGLES)
    if roll == 1:
        return rng.uniform(-2.0 * _ROTATION_PERIOD, 2.0 * _ROTATION_PERIOD)
    return rng.uniform(-1e-9, 1e-9)


def _apply_gate(circuit: Circuit, name: str, targets: tuple, angle: float | None = None) -> None:
    method = getattr(circuit, name)
    if angle is None:
        method(*targets)
    else:
        method(angle, *targets)


def _distinct_pair(rng: random.Random, num_qubits: int) -> tuple[int, int]:
    first, second = rng.sample(range(num_qubits), 2)
    return first, second


def _random_single(rng: random.Random, qubit: int) -> tuple:
    name = rng.choice(_FIXED_GATES + _ROTATION_GATES)
    if name in _ROTATION_GATES:
        return (name, (qubit,), _random_angle(rng))
    return (name, (qubit,), None)


def _append_adjacent_inverse_pair(circuit, rng, num_qubits, features) -> None:
    features.add("adjacent_inverse")
    options = ["fixed", "rotation"]
    if num_qubits >= 2:
        options.append("controlled")
    choice = rng.choice(options)
    if choice == "fixed":
        first, second = rng.choice(
            (
                ("x", "x"),
                ("h", "h"),
                ("y", "y"),
                ("z", "z"),
                ("s", "sdg"),
                ("sdg", "s"),
                ("t", "tdg"),
                ("tdg", "t"),
            )
        )
        qubit = rng.randrange(num_qubits)
        _apply_gate(circuit, first, (qubit,))
        _apply_gate(circuit, second, (qubit,))
        return
    if choice == "controlled":
        # cx, cz and swap are self-inverse on identical operands.
        name = rng.choice(_CONTROLLED_GATES + ("swap",))
        pair = _distinct_pair(rng, num_qubits)
        _apply_gate(circuit, name, pair)
        _apply_gate(circuit, name, pair)
        return
    # A rotation immediately undone by its negated angle.
    if num_qubits >= 2 and rng.random() < 0.5:
        name = rng.choice(_CONTROLLED_ROTATION_GATES)
        targets = _distinct_pair(rng, num_qubits)
    else:
        name = rng.choice(_ROTATION_GATES)
        targets = (rng.randrange(num_qubits),)
    theta = _random_angle(rng)
    _apply_gate(circuit, name, targets, theta)
    _apply_gate(circuit, name, targets, -theta)


def _append_repeated_rotations(circuit, rng, num_qubits, features) -> None:
    features.add("repeated_rotation")
    if num_qubits >= 2 and rng.random() < 0.4:
        name = rng.choice(_CONTROLLED_ROTATION_GATES)
        targets = _distinct_pair(rng, num_qubits)
    else:
        name = rng.choice(_ROTATION_GATES)
        targets = (rng.randrange(num_qubits),)
    for _ in range(rng.randint(2, 4)):
        _apply_gate(circuit, name, targets, _random_angle(rng))


def _append_disjoint_permutation(circuit, rng, num_qubits, features) -> None:
    features.add("disjoint_permutation")
    gates = []
    if num_qubits >= 3 and rng.random() < 0.5:
        first, second, third = rng.sample(range(num_qubits), 3)
        two_qubit = rng.choice(_CONTROLLED_GATES + ("swap",) + _CONTROLLED_ROTATION_GATES)
        if two_qubit in _CONTROLLED_ROTATION_GATES:
            gates.append((two_qubit, (first, second), _random_angle(rng)))
        else:
            gates.append((two_qubit, (first, second), None))
        gates.append(_random_single(rng, third))
    else:
        for qubit in rng.sample(range(num_qubits), rng.randint(2, num_qubits)):
            gates.append(_random_single(rng, qubit))
    # Mutually disjoint gates commute; emit one seeded permutation.
    rng.shuffle(gates)
    for name, targets, angle in gates:
        _apply_gate(circuit, name, targets, angle)


def _append_random_gate(circuit, rng, num_qubits) -> None:
    pool = list(_FIXED_GATES) + list(_ROTATION_GATES)
    if num_qubits >= 2:
        pool += list(_CONTROLLED_GATES) + ["swap"] + list(_CONTROLLED_ROTATION_GATES)
    name = rng.choice(pool)
    if name in _FIXED_GATES:
        _apply_gate(circuit, name, (rng.randrange(num_qubits),))
    elif name in _ROTATION_GATES:
        _apply_gate(circuit, name, (rng.randrange(num_qubits),), _random_angle(rng))
    elif name in _CONTROLLED_ROTATION_GATES:
        _apply_gate(circuit, name, _distinct_pair(rng, num_qubits), _random_angle(rng))
    else:
        _apply_gate(circuit, name, _distinct_pair(rng, num_qubits))


def _append_gates(circuit, rng, num_qubits, features) -> None:
    for _ in range(rng.randint(5, 12)):
        roll = rng.randrange(8)
        if roll == 0:
            _append_adjacent_inverse_pair(circuit, rng, num_qubits, features)
        elif roll == 1:
            _append_repeated_rotations(circuit, rng, num_qubits, features)
        elif roll == 2 and num_qubits >= 2:
            _append_disjoint_permutation(circuit, rng, num_qubits, features)
        else:
            _append_random_gate(circuit, rng, num_qubits)


def _append_measurements(circuit, rng, mode, num_qubits, num_clbits) -> None:
    if mode == _MODE_NONE:
        return
    if mode == _MODE_ALL:
        # Every qubit onto the matching clbit (num_clbits >= num_qubits).
        for index in range(num_qubits):
            circuit.measure(index, index)
        return
    if mode == _MODE_PARTIAL:
        # A random subset of qubits onto random distinct clbits.
        count = rng.randint(1, num_qubits - 1) if num_qubits > 1 else 1
        qubits = rng.sample(range(num_qubits), count)
        clbits = rng.sample(range(num_clbits), min(count, num_clbits))
        for qubit, clbit in zip(qubits, clbits):
            circuit.measure(qubit, clbit)
        return
    # _MODE_PERMUTED: every qubit measured, classical bits rearranged so at
    # least one qubit lands on a clbit other than its own index.
    clbits = list(range(num_clbits))
    rng.shuffle(clbits)
    targets = clbits[:num_qubits]
    if all(qubit == clbit for qubit, clbit in enumerate(targets)):
        if num_qubits >= 2:
            targets = targets[1:] + targets[:1]
        else:
            targets = [(targets[0] + 1) % num_clbits]
    for qubit, clbit in enumerate(targets):
        circuit.measure(qubit, clbit)


def _generate_sample(seed: int) -> tuple[Circuit, int, set]:
    """Build one deterministic sample circuit from *seed*.

    Returns ``(circuit, mode, features)`` where *mode* is the measurement
    mode (rotating with the seed) and *features* records which structured
    gate blocks were emitted. Only legal circuits are produced: controls
    differ from targets, measurements never repeat a qubit or a clbit and
    no gate is appended after a measurement.
    """
    rng = random.Random(seed)
    mode = seed % 4
    num_qubits = rng.randint(1, 4)
    if mode == _MODE_PERMUTED:
        num_clbits = rng.randint(max(num_qubits, 2), 4)
    elif mode == _MODE_ALL:
        num_clbits = rng.randint(num_qubits, 4)
    else:
        num_clbits = rng.randint(1, 4)
    circuit = Circuit(num_qubits, num_clbits)
    features: set = set()
    _append_gates(circuit, rng, num_qubits, features)
    _append_measurements(circuit, rng, mode, num_qubits, num_clbits)
    return circuit, mode, features


# ----------------------------------------------------------------- helpers


def _context(seed: int, qasm: str) -> str:
    return f"sample seed={seed}\n--- reproducible QASM ---\n{qasm}"


def _without_contract_fields(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in _CONTRACT_ONLY_FIELDS}


def _check_classical_bit_layout(circuit: Circuit, outcomes: dict, context: str) -> None:
    width = circuit.num_clbits
    measured_clbits = {op.targets[1] for op in circuit.operations if op.kind == "measure"}
    for key in outcomes:
        assert len(key) == width, context
        for clbit in range(width):
            if clbit not in measured_clbits:
                # Highest clbit index first: clbit c sits at position width-1-c.
                assert key[width - 1 - clbit] == "0", context


# ------------------------------------------------------------------- tests


def test_generated_corpus_is_deterministic_and_covers_the_full_gate_set():
    kinds: set = set()
    modes: set = set()
    features: set = set()
    angles: list = []
    for seed in SAMPLE_SEEDS:
        circuit, mode, sample_features = _generate_sample(seed)
        again, _, _ = _generate_sample(seed)
        assert again.to_qasm() == circuit.to_qasm()
        modes.add(mode)
        features.update(sample_features)
        for op in circuit.operations:
            if op.kind == "measure":
                continue
            kinds.add(op.kind)
            angles.extend(op.params)
    assert kinds == _ALL_GATES
    assert modes == {_MODE_NONE, _MODE_PARTIAL, _MODE_PERMUTED, _MODE_ALL}
    assert {"adjacent_inverse", "repeated_rotation", "disjoint_permutation"} <= features
    assert 0.0 in angles
    assert _ROTATION_PERIOD in angles
    assert -_ROTATION_PERIOD in angles
    assert 2.0 * _ROTATION_PERIOD in angles
    assert any(angle != 0.0 and abs(angle) <= 1e-12 for angle in angles)


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_qasm_roundtrip_preserves_text_operations_and_results(seed):
    circuit, _mode, _features = _generate_sample(seed)
    qasm = circuit.to_qasm()
    context = _context(seed, qasm)

    rebuilt = Circuit.from_qasm(qasm)
    assert rebuilt.operations == circuit.operations, context
    assert rebuilt.to_qasm() == qasm, context
    # A second serialize/parse cycle is stable as well.
    assert Circuit.from_qasm(rebuilt.to_qasm()).to_qasm() == qasm, context

    # The full return objects of both simulation entry points are equal.
    assert rebuilt.probabilities() == circuit.probabilities(), context
    original_sample = circuit.sample(shots=_SAMPLE_SHOTS, seed=_SAMPLE_SEED)
    rebuilt_sample = rebuilt.sample(shots=_SAMPLE_SHOTS, seed=_SAMPLE_SEED)
    assert rebuilt_sample == original_sample, context


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_zero_noise_density_path_agrees_with_state_vector_path(seed):
    circuit, _mode, _features = _generate_sample(seed)
    qasm = circuit.to_qasm()
    context = _context(seed, qasm)
    rebuilt = Circuit.from_qasm(qasm)

    clean_probabilities = rebuilt.probabilities()
    dense_probabilities = rebuilt.probabilities(noise_model=_ZERO_NOISE_MODEL)
    clean_sample = rebuilt.sample(shots=_SAMPLE_SHOTS, seed=_SAMPLE_SEED)
    dense_sample = rebuilt.sample(
        shots=_SAMPLE_SHOTS, seed=_SAMPLE_SEED, noise_model=_ZERO_NOISE_MODEL
    )

    # Payload shapes and field order follow the schema contract.
    assert list(clean_probabilities) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "probabilities",
    ], context
    assert list(dense_probabilities) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "noise_model",
        "probabilities",
    ], context
    assert list(clean_sample) == [
        "schema_version",
        "shots",
        "seed",
        "num_qubits",
        "num_clbits",
        "counts",
    ], context
    assert list(dense_sample) == [
        "schema_version",
        "shots",
        "seed",
        "num_qubits",
        "num_clbits",
        "noise_model",
        "counts",
    ], context
    assert clean_probabilities["schema_version"] == 1, context
    assert dense_probabilities["schema_version"] == 2, context
    assert clean_sample["schema_version"] == 1, context
    assert dense_sample["schema_version"] == 2, context
    assert dense_probabilities["noise_model"] == _ZERO_NOISE_MODEL, context
    assert dense_sample["noise_model"] == _ZERO_NOISE_MODEL, context

    # Ignoring the contract-only fields, the sample payloads (including the
    # exact fixed-seed counts) must be equal, and so must the integer
    # fields of the probabilities payloads.
    assert _without_contract_fields(clean_sample) == _without_contract_fields(dense_sample), context
    for field in ("num_qubits", "num_clbits"):
        assert clean_probabilities[field] == dense_probabilities[field], context

    # Probability keys (set and order) match exactly; values agree to the
    # deterministic round-off bound.
    clean = clean_probabilities["probabilities"]
    dense = dense_probabilities["probabilities"]
    assert list(clean) == sorted(clean), context
    assert list(dense) == sorted(dense), context
    assert list(clean) == list(dense), context
    for key in clean:
        difference = abs(clean[key] - dense[key])
        assert difference <= _PROBABILITY_TOLERANCE, (
            f"probability mismatch for outcome {key!r}: "
            f"{clean[key]!r} vs {dense[key]!r}\n{context}"
        )
    assert abs(math.fsum(clean.values()) - 1.0) <= 1e-12, context
    assert abs(math.fsum(dense.values()) - 1.0) <= 1e-12, context

    # Count keys are ordered, cover every shot and match exactly (checked
    # above via the stripped payloads); the classical bit layout holds on
    # both paths.
    for counts in (clean_sample["counts"], dense_sample["counts"]):
        assert list(counts) == sorted(counts), context
        assert sum(counts.values()) == _SAMPLE_SHOTS, context
    _check_classical_bit_layout(rebuilt, clean, context)
    _check_classical_bit_layout(rebuilt, dense, context)
    _check_classical_bit_layout(rebuilt, clean_sample["counts"], context)
    _check_classical_bit_layout(rebuilt, dense_sample["counts"], context)

    # With no measurement the only outcome is the all-zero string.
    if not any(op.kind == "measure" for op in rebuilt.operations):
        all_zero = "0" * rebuilt.num_clbits
        assert clean == {all_zero: 1.0}, context
        assert dense == {all_zero: 1.0}, context
        assert clean_sample["counts"] == {all_zero: _SAMPLE_SHOTS}, context
        assert dense_sample["counts"] == {all_zero: _SAMPLE_SHOTS}, context
