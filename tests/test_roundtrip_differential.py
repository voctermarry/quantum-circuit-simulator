"""Deterministic differential round-trip tests over the full gate set.

A fixed-seed generator builds small legal circuits (one to four qubits)
through the Python DSL, covering all seventeen supported gates (``x h y z s
sdg t tdg cx cz swap rx ry rz crx cry crz``). The corpus deliberately
contains adjacent inverse-gate pairs, repeated rotations on one qubit,
mutually disjoint gates emitted in different orders, and finite angles at
near-zero magnitudes and rotation-period boundaries. The generator only
produces legal input: two-qubit gates always have distinct control and
target, no qubit or clbit is measured twice, and no gate follows a
measurement. Measurement modes rotate deterministically over none, partial,
full with a shuffled classical-bit assignment, and full identity.

For every sample the test proves, with fully deterministic data and
assertions:

1. **Round-trip stability** -- ``Circuit.to_qasm`` text re-parsed by
   ``Circuit.from_qasm`` serializes back to the identical text, and the
   original and rebuilt circuits return equal ``probabilities`` objects and
   equal ``sample`` objects for the same shots and seed.
2. **Cross-path consistency** -- the rebuilt circuit is also executed on the
   density-matrix path with a noise model whose channels all have
   probability zero. The two return objects differ by contract only in
   ``schema_version`` and the echoed ``noise_model``; after setting those
   two fields aside, the exact probabilities and the fixed-seed counts must
   agree. Probability *values* are compared with a fixed ``1e-12`` absolute
   tolerance: the density-matrix evolution orders its floating-point
   operations differently from the state-vector kernel, so mathematically
   identical amplitudes can differ in the last bit (the package documents
   the same 1e-12 scale for probability sums). This is not a statistical
   tolerance: counts, key sets, key order, classical bit widths and every
   round-trip object are compared exactly, and no count difference is ever
   tolerated.
3. **Payload shape** -- probability and count keys are fixed-width,
   lexicographically ordered bit strings; classical bits never targeted by a
   measurement stay zero in every key; with no measurement the only outcome
   is the all-zero string with probability 1 and all shots.

Every assertion message carries the sample seed and the exact QASM text, so
any failure can be reproduced directly. Nothing here touches the command
line, public entry points or any existing success/failure semantics.
"""

from __future__ import annotations

import math
import random

import pytest

from quantum_circuit import Circuit

# 17 fixed seeds, one per gate kind: slot 0 of every case forces
# ALL_GATES[seed], so the corpus covers every supported gate regardless of
# random draws. The four cases of a seed rotate the measurement mode.
SEEDS = tuple(range(17))
CASES_PER_SEED = 4
MIN_QUBITS, MAX_QUBITS = 1, 4

SINGLE_FIXED = ("x", "h", "y", "z", "s", "sdg", "t", "tdg")
ROTATIONS = ("rx", "ry", "rz")
TWO_QUBIT = ("cx", "cz", "swap")
CONTROLLED_ROTATIONS = ("crx", "cry", "crz")
ALL_GATES = SINGLE_FIXED + ROTATIONS + TWO_QUBIT + CONTROLLED_ROTATIONS

# Finite angles: ordinary values, near-zero magnitudes and the single/controlled
# rotation period boundaries (2*pi and 4*pi). Forced rotations cycle through
# the whole palette (see _build_circuit), so every entry appears in the corpus.
ANGLES = (
    0.3,
    -0.7,
    1.1,
    math.pi / 2,
    -math.pi / 3,
    math.pi,
    -math.pi,
    3 * math.pi / 2,
    2 * math.pi,
    -2 * math.pi,
    4 * math.pi,
    1e-13,
    -1e-13,
    1e-12,
    0.0,
)

# A density-matrix path that must behave exactly like the noiseless path:
# every configured channel has probability zero.
ZERO_NOISE_MODEL = {"amplitude_damping": 0.0, "phase_damping": 0.0}

# Fields the two paths' return objects differ in by contract.
_CONTRACT_FIELDS = ("schema_version", "noise_model")

# Fixed absolute tolerance for cross-path probability *values* only (see the
# module docstring); counts and everything structural stay exact.
CROSS_PATH_TOLERANCE = 1e-12

MEASURE_MODES = ("none", "partial", "permuted", "full")


# ------------------------------------------------------------- case generation


def _case_rng(seed: int, case: int) -> random.Random:
    # String seeding is deterministic across runs and independent of
    # PYTHONHASHSEED (unlike seeding from a hashed tuple).
    return random.Random(f"roundtrip-differential:{seed}:{case}")


def _distinct_qubits(rng: random.Random, n: int) -> tuple[int, int]:
    """Two distinct qubit indices in ``range(n)`` (n must be at least 2)."""
    a = rng.randrange(n)
    b = rng.randrange(n - 1)
    if b >= a:
        b += 1
    return a, b


def _add_gate(
    circuit: Circuit,
    rng: random.Random,
    kind: str,
    n: int,
    angle: float | None = None,
) -> None:
    """Append one legal application of *kind* via the DSL."""
    if kind in SINGLE_FIXED:
        getattr(circuit, kind)(rng.randrange(n))
    elif kind in ROTATIONS:
        getattr(circuit, kind)(rng.choice(ANGLES) if angle is None else angle, rng.randrange(n))
    elif kind in TWO_QUBIT:
        control, target = _distinct_qubits(rng, n)
        getattr(circuit, kind)(control, target)
    else:
        control, target = _distinct_qubits(rng, n)
        getattr(circuit, kind)(rng.choice(ANGLES) if angle is None else angle, control, target)


def _add_inverse_pair(circuit: Circuit, rng: random.Random, n: int) -> None:
    """Append two adjacent gates that are each other's inverse."""
    style = rng.randrange(3)
    if style == 0:
        # Self-inverse fixed gate, or an explicit s/sdg, t/tdg pair.
        pool = ["x", "h", "y", "z", "s", "t"]
        if n >= 2:
            pool += ["cx", "cz", "swap"]
        first = rng.choice(pool)
        second = {"s": "sdg", "t": "tdg"}.get(first, first)
        if first in TWO_QUBIT:
            control, target = _distinct_qubits(rng, n)
            getattr(circuit, first)(control, target)
            getattr(circuit, second)(control, target)
        else:
            qubit = rng.randrange(n)
            getattr(circuit, first)(qubit)
            getattr(circuit, second)(qubit)
    elif style == 1:
        # A rotation immediately undone by its negated angle.
        kind = rng.choice(ROTATIONS)
        qubit = rng.randrange(n)
        theta = rng.choice(ANGLES)
        getattr(circuit, kind)(theta, qubit)
        getattr(circuit, kind)(-theta, qubit)
    elif n >= 2:
        kind = rng.choice(CONTROLLED_ROTATIONS)
        control, target = _distinct_qubits(rng, n)
        theta = rng.choice(ANGLES)
        getattr(circuit, kind)(theta, control, target)
        getattr(circuit, kind)(-theta, control, target)
    else:
        first = rng.choice(("x", "h", "y", "z"))
        qubit = rng.randrange(n)
        getattr(circuit, first)(qubit)
        getattr(circuit, first)(qubit)


def _add_repeated_rotations(circuit: Circuit, rng: random.Random, n: int) -> None:
    """Append the same rotation kind on the same qubit several times."""
    kind = rng.choice(ROTATIONS)
    qubit = rng.randrange(n)
    for _ in range(rng.randint(2, 4)):
        getattr(circuit, kind)(rng.choice(ANGLES), qubit)


def _disjoint_gate_set(rng: random.Random, n: int) -> list[tuple]:
    """Resolved gates on pairwise disjoint qubits: ``(kind, qubits, angle)``."""
    free = list(range(n))
    rng.shuffle(free)
    ops: list[tuple] = []
    if n >= 3 and rng.random() < 0.5:
        control, target = free.pop(), free.pop()
        kind = rng.choice(TWO_QUBIT + CONTROLLED_ROTATIONS)
        angle = rng.choice(ANGLES) if kind in CONTROLLED_ROTATIONS else None
        ops.append((kind, (control, target), angle))
    while free:
        qubit = free.pop()
        kind = rng.choice(SINGLE_FIXED + ROTATIONS)
        angle = rng.choice(ANGLES) if kind in ROTATIONS else None
        ops.append((kind, (qubit,), angle))
    return ops


def _apply_resolved(circuit: Circuit, ops: list[tuple]) -> None:
    for kind, qubits, angle in ops:
        if angle is None:
            getattr(circuit, kind)(*qubits)
        else:
            getattr(circuit, kind)(angle, *qubits)


def _add_measurements(circuit: Circuit, rng: random.Random, n: int, mode: int) -> None:
    """Append the measurement section for one of the four rotated modes."""
    if mode == 0:  # no measurement at all
        return
    if mode == 1:  # partial: a subset of qubits onto a shuffled clbit subset
        count = rng.randint(1, n)
        qubits = rng.sample(range(n), count)
        clbits = rng.sample(range(n), count)
    elif mode == 2:  # full, with a shuffled classical-bit assignment
        qubits = list(range(n))
        clbits = list(range(n))
        rng.shuffle(clbits)
    else:  # full identity mapping
        qubits = list(range(n))
        clbits = list(range(n))
    for qubit, clbit in zip(qubits, clbits):
        circuit.measure(qubit, clbit)


def _build_circuit(seed: int, case: int) -> Circuit:
    """Generate one complete, legal circuit from its fixed seed and case."""
    rng = _case_rng(seed, case)
    forced = ALL_GATES[seed % len(ALL_GATES)]
    n = (
        rng.randint(2, MAX_QUBITS)
        if forced in TWO_QUBIT + CONTROLLED_ROTATIONS
        else rng.randint(MIN_QUBITS, MAX_QUBITS)
    )
    circuit = Circuit(n, n)

    # Slot 0: the forced gate. Forced rotations cycle the whole angle palette,
    # so near-zero and period-boundary angles are guaranteed to appear.
    forced_angle = ANGLES[(seed * CASES_PER_SEED + case) % len(ANGLES)]
    _add_gate(circuit, rng, forced, n, angle=forced_angle)

    for _ in range(rng.randint(2, 5)):
        motif = rng.randrange(5)
        if motif == 0:
            pool = ALL_GATES if n >= 2 else SINGLE_FIXED + ROTATIONS
            _add_gate(circuit, rng, rng.choice(pool), n)
        elif motif == 1:
            _add_inverse_pair(circuit, rng, n)
        elif motif == 2:
            _add_repeated_rotations(circuit, rng, n)
        elif motif == 3 and n >= 2:
            ops = _disjoint_gate_set(rng, n)
            rng.shuffle(ops)  # disjoint gates emitted in a random order
            _apply_resolved(circuit, ops)
        else:
            pool = ALL_GATES if n >= 2 else SINGLE_FIXED + ROTATIONS
            _add_gate(circuit, rng, rng.choice(pool), n)

    _add_measurements(circuit, rng, n, case % len(MEASURE_MODES))
    return circuit


def _corpus() -> list[tuple[int, int, Circuit]]:
    return [
        (seed, case, _build_circuit(seed, case))
        for seed in SEEDS
        for case in range(CASES_PER_SEED)
    ]


def _sample_params(seed: int, case: int) -> tuple[int, int]:
    """Deterministic shots and sampling seed for one corpus sample."""
    index = seed * CASES_PER_SEED + case
    return 64 + 7 * index, 1_000_000 + index


# --------------------------------------------------------------- case checking


def _check_payload_shape(
    circuit: Circuit,
    probabilities_payload: dict,
    sample_payload: dict,
    shots: int,
    where: str,
) -> None:
    """Key order, classical width, unwritten clbits and no-measurement shape."""
    width = circuit.num_clbits
    probabilities = probabilities_payload["probabilities"]
    counts = sample_payload["counts"]
    measurements = [op for op in circuit.operations if op.kind == "measure"]
    written = {op.targets[1] for op in measurements}
    unwritten = [clbit for clbit in range(width) if clbit not in written]

    assert list(probabilities) == sorted(probabilities), (
        f"probability keys are not in lexicographic order\n{where}"
    )
    assert list(counts) == sorted(counts), (
        f"count keys are not in lexicographic order\n{where}"
    )
    assert sum(counts.values()) == shots, f"counts do not add up to shots\n{where}"
    for key in list(probabilities) + list(counts):
        assert len(key) == width and set(key) <= {"0", "1"}, (
            f"key {key!r} is not a {width}-bit classical string\n{where}"
        )
        for clbit in unwritten:
            # Keys list the highest clbit index first.
            assert key[width - 1 - clbit] == "0", (
                f"unwritten clbit {clbit} is not zero in key {key!r}\n{where}"
            )
    if not measurements:
        assert probabilities == {"0" * width: 1.0}, (
            f"no-measurement probabilities are not the all-zero outcome\n{where}"
        )
        assert counts == {"0" * width: shots}, (
            f"no-measurement counts are not the all-zero outcome\n{where}"
        )
    # The documented contract: reported probabilities sum to within 1e-12 of 1.
    assert abs(math.fsum(probabilities.values()) - 1.0) <= 1e-12, (
        f"reported probabilities do not sum to 1\n{where}"
    )


def _check_circuit(circuit: Circuit, shots: int, sample_seed: int, label: str) -> None:
    """Run every differential assertion for one generated circuit."""
    qasm = circuit.to_qasm()
    where = f"{label} shots={shots} sample_seed={sample_seed}\n--- qasm ---\n{qasm}"

    rebuilt = Circuit.from_qasm(qasm)
    assert rebuilt.to_qasm() == qasm, f"re-parsed circuit serializes differently\n{where}"
    assert rebuilt.operations == circuit.operations, (
        f"re-parsed operations differ from the DSL circuit\n{where}"
    )

    original_operations = circuit.operations
    rebuilt_operations = rebuilt.operations

    # Same simulation path: original and rebuilt circuits agree exactly.
    assert circuit.probabilities() == rebuilt.probabilities(), (
        f"probabilities differ between DSL circuit and re-parsed circuit\n{where}"
    )
    assert circuit.sample(shots=shots, seed=sample_seed) == rebuilt.sample(
        shots=shots, seed=sample_seed
    ), f"sampled counts differ between DSL circuit and re-parsed circuit\n{where}"

    # Cross-path: noiseless state vector vs zero-probability density matrix.
    clean = rebuilt.probabilities()
    noisy = rebuilt.probabilities(noise_model=ZERO_NOISE_MODEL)
    assert clean["schema_version"] == 1 and "noise_model" not in clean, (
        f"noiseless payload does not have the schema_version 1 shape\n{where}"
    )
    assert noisy["schema_version"] == 2 and noisy["noise_model"] == ZERO_NOISE_MODEL, (
        f"noisy payload does not echo the zero-probability noise model\n{where}"
    )
    assert list(clean) == ["schema_version", "num_qubits", "num_clbits", "probabilities"], (
        f"noiseless payload field order changed\n{where}"
    )
    assert list(noisy) == [
        "schema_version",
        "num_qubits",
        "num_clbits",
        "noise_model",
        "probabilities",
    ], f"noisy payload field order changed\n{where}"
    for field in ("num_qubits", "num_clbits"):
        assert noisy[field] == clean[field], f"{field} differs across paths\n{where}"

    clean_probabilities = clean["probabilities"]
    noisy_probabilities = noisy["probabilities"]
    assert list(noisy_probabilities) == list(clean_probabilities), (
        f"probability keys differ across paths\n{where}"
    )
    for key in clean_probabilities:
        difference = abs(clean_probabilities[key] - noisy_probabilities[key])
        assert difference <= CROSS_PATH_TOLERANCE, (
            f"probability of {key!r} differs across paths by {difference!r}\n{where}"
        )

    clean_run = rebuilt.sample(shots=shots, seed=sample_seed)
    noisy_run = rebuilt.sample(shots=shots, seed=sample_seed, noise_model=ZERO_NOISE_MODEL)

    def strip_contract_fields(payload: dict) -> dict:
        return {key: value for key, value in payload.items() if key not in _CONTRACT_FIELDS}

    assert strip_contract_fields(noisy_run) == strip_contract_fields(clean_run), (
        f"fixed-seed counts differ across paths (no tolerance applied)\n{where}"
    )

    _check_payload_shape(rebuilt, clean, clean_run, shots, where)

    # Simulation never mutates the circuits it runs on.
    assert circuit.operations == original_operations, f"DSL circuit was mutated\n{where}"
    assert rebuilt.operations == rebuilt_operations, f"rebuilt circuit was mutated\n{where}"


# ----------------------------------------------------------------- corpus tests


@pytest.mark.parametrize(
    "seed,case",
    [
        pytest.param(seed, case, id=f"seed{seed:02d}-case{case}")
        for seed in SEEDS
        for case in range(CASES_PER_SEED)
    ],
)
def test_generated_sample_is_consistent_across_paths(seed, case):
    circuit = _build_circuit(seed, case)
    shots, sample_seed = _sample_params(seed, case)
    _check_circuit(circuit, shots, sample_seed, f"seed={seed} case={case}")


def test_corpus_covers_every_supported_gate():
    kinds = {op.kind for _seed, _case, circuit in _corpus() for op in circuit.operations}
    assert set(ALL_GATES) <= kinds


def test_corpus_covers_near_zero_and_period_boundary_angles():
    angles = {
        op.params[0]
        for _seed, _case, circuit in _corpus()
        for op in circuit.operations
        if op.params
    }
    assert 1e-13 in angles
    assert 1e-12 in angles
    assert 2 * math.pi in angles
    assert 4 * math.pi in angles


def test_corpus_rotates_through_all_measurement_modes():
    none = partial = permuted = full = 0
    for _seed, _case, circuit in _corpus():
        measures = [op for op in circuit.operations if op.kind == "measure"]
        if not measures:
            none += 1
        elif len(measures) < circuit.num_qubits:
            partial += 1
        elif any(op.targets[0] != op.targets[1] for op in measures):
            permuted += 1
        else:
            full += 1
    assert none and partial and permuted and full


def test_corpus_contains_inverse_pairs_and_repeated_rotations():
    inverses = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
    self_inverse = {"x", "h", "y", "z", "cx", "cz", "swap"}
    found_inverse = found_repeated = False
    for _seed, _case, circuit in _corpus():
        gates = [op for op in circuit.operations if op.kind != "measure"]
        for left, right in zip(gates, gates[1:]):
            if left.targets != right.targets:
                continue
            if left.kind in self_inverse and right.kind == left.kind:
                found_inverse = True
            if inverses.get(left.kind) == right.kind:
                found_inverse = True
            if left.kind in ROTATIONS + CONTROLLED_ROTATIONS and right.kind == left.kind:
                found_repeated = True
                if left.params[0] == -right.params[0]:
                    found_inverse = True
    assert found_inverse, "corpus has no adjacent inverse-gate pair"
    assert found_repeated, "corpus has no repeated rotation on one qubit"


def test_generator_only_produces_legal_layouts():
    for seed, case, circuit in _corpus():
        operations = circuit.operations
        kinds = [op.kind for op in operations]
        first_measure = kinds.index("measure") if "measure" in kinds else len(kinds)
        assert all(kind != "measure" for kind in kinds[:first_measure]), (seed, case)
        assert all(kind == "measure" for kind in kinds[first_measure:]), (seed, case)
        measured_qubits = [op.targets[0] for op in operations if op.kind == "measure"]
        measured_clbits = [op.targets[1] for op in operations if op.kind == "measure"]
        assert len(measured_qubits) == len(set(measured_qubits)), (seed, case)
        assert len(measured_clbits) == len(set(measured_clbits)), (seed, case)
        for op in operations:
            if op.kind in TWO_QUBIT + CONTROLLED_ROTATIONS:
                assert op.targets[0] != op.targets[1], (seed, case)


def test_same_seeds_reproduce_the_identical_corpus():
    first = [circuit.to_qasm() for _seed, _case, circuit in _corpus()]
    second = [circuit.to_qasm() for _seed, _case, circuit in _corpus()]
    assert first == second


# --------------------------------------- disjoint-gate permutation round trips


@pytest.mark.parametrize("seed", range(8))
def test_disjoint_gate_permutations_each_round_trip(seed):
    rng = random.Random(f"disjoint-permutation:{seed}")
    n = rng.randint(2, MAX_QUBITS)
    ops = _disjoint_gate_set(rng, n)
    assert len(ops) >= 2

    shuffled_a = ops[:]
    rng.shuffle(shuffled_a)
    shuffled_b = ops[:]
    rng.shuffle(shuffled_b)
    orders: list[list[tuple]] = []
    for order in (ops, list(reversed(ops)), shuffled_a, shuffled_b):
        if order not in orders:
            orders.append(order)
    assert len(orders) >= 2

    # The same disjoint gate multiset in different emission orders: each
    # permutation is an independent legal sample and must satisfy the full
    # round-trip and cross-path checks.
    for index, order in enumerate(orders):
        circuit = Circuit(n, n)
        _apply_resolved(circuit, order)
        for qubit in range(n):
            circuit.measure(qubit, qubit)
        _check_circuit(
            circuit,
            shots=64 + seed,
            sample_seed=2_000_000 + seed,
            label=f"permutation-seed={seed} order={index}",
        )
