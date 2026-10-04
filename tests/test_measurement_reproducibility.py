"""End-to-end measurement-sampling reproducibility and distribution tests.

These tests exercise measurement only through the project's public circuit
construction and execution entry points (the :class:`~quantum_circuit.Circuit`
DSL, :meth:`Circuit.from_qasm` and :meth:`Circuit.sample`), so the DSL,
OpenQASM parser, simulator, noise channels and result export semantics are
all covered on their real code paths.

The random seed is treated as an end-to-end observable contract:

* equal circuit/backend/shots/noise/seed arguments produce the complete
  result payload (and exportable statistics derived from it) item by item,
  even when unrelated random work happens in between;
* state-vector and density-matrix sampling are each checked for their own
  repeat stability, but the two algorithms are never assumed to consume
  random numbers in the same order, so their sampled counts are never
  compared against each other;
* without an explicit seed only statistical properties are asserted.

Every frequency threshold is computed at test run time from the theoretical
outcome probability, the shot count and a fixed significance level using the
exact binomial distribution -- no sampled count is ever hard coded.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from quantum_circuit import Circuit

# Shot count and fixed per-tail significance level for every statistical
# assertion. Bounds are exact binomial tails, so with ALPHA this small the
# whole suite has a negligible false-failure probability regardless of run
# order while keeping expected intervals tight (half-width ~3% at p = 1/2).
SHOTS = 8000
ALPHA = 1e-7

# Seeds reused across the reproducibility scenarios.
SEEDS = (0, 1, 42, 2024)


# --------------------------------------------------------------------- helpers


def _binomial_bounds(shots: int, probability: float, alpha: float) -> tuple[int, int]:
    """Inclusive count interval ``(lo, hi)`` for ``Binomial(shots, p)``.

    The exact binomial tails satisfy ``P(X <= lo - 1) <= alpha/2`` and
    ``P(X >= hi + 1) <= alpha/2``. Probabilities are seeded from the
    distribution mode and expanded by the binomial recurrence, so the
    computation stays exact (no underflow, no normal approximation) using
    only the standard library.
    """
    if probability == 0.0:
        return 0, 0
    if probability == 1.0:
        return shots, shots

    mode = min(shots, max(0, math.floor((shots + 1) * probability)))
    pmf = [0.0] * (shots + 1)
    log_mode = (
        math.lgamma(shots + 1)
        - math.lgamma(mode + 1)
        - math.lgamma(shots - mode + 1)
        + mode * math.log(probability)
        + (shots - mode) * math.log(1.0 - probability)
    )
    pmf[mode] = math.exp(log_mode)
    for k in range(mode - 1, -1, -1):
        pmf[k] = pmf[k + 1] * (k + 1) / (shots - k) * (1.0 - probability) / probability
    for k in range(mode + 1, shots + 1):
        pmf[k] = pmf[k - 1] * (shots - k + 1) / k * probability / (1.0 - probability)

    tail = alpha / 2.0
    cumulative = 0.0
    lo = 0
    for k in range(shots + 1):
        cumulative += pmf[k]
        if cumulative > tail:
            lo = k
            break

    cumulative = 0.0
    hi = shots
    for k in range(shots, -1, -1):
        cumulative += pmf[k]
        if cumulative > tail:
            hi = k
            break
    return lo, hi


def _failure_detail(scenario, seed, counts, key, probability, lo, hi, actual):
    return (
        f"scenario={scenario!r} seed={seed} shots={SHOTS}: outcome {key!r} "
        f"count {actual} lies outside the binomial interval [{lo}, {hi}] for "
        f"p={probability} (alpha={ALPHA:g}); observed counts={counts}"
    )


def assert_matches_distribution(
    payload: dict,
    expected: dict[str, float],
    scenario: str,
    seed: int | str,
) -> None:
    """Assert one sampled payload follows the theoretical outcome distribution.

    Checks the classical width on every key, the total sample count, the
    exact absence of every theoretical-zero bit string, and a per-outcome
    exact-binomial interval computed from *expected*. Failure messages name
    the scenario, seed, expected interval and the full observed counts.
    """
    counts = payload["counts"]
    shots = payload["shots"]
    width = payload["num_clbits"]
    assert shots == SHOTS, f"scenario={scenario!r}: payload shots {shots} != {SHOTS}"

    for key in counts:
        assert len(key) == width, (
            f"scenario={scenario!r}: classical key {key!r} has width "
            f"{len(key)}, expected {width}"
        )
        assert set(key) <= {"0", "1"}

    observed_total = sum(counts.values())
    assert observed_total == shots, (
        f"scenario={scenario!r} seed={seed}: counts sum to {observed_total}, "
        f"expected {shots}; counts={counts}"
    )

    # Outcomes with theoretical probability zero must never appear.
    support = set(expected)
    for key in counts:
        assert key in support, (
            f"scenario={scenario!r} seed={seed}: forbidden (zero-probability) "
            f"bit string {key!r} observed; counts={counts}"
        )

    for key, probability in expected.items():
        actual = counts.get(key, 0)
        lo, hi = _binomial_bounds(shots, probability, ALPHA)
        assert lo <= actual <= hi, _failure_detail(
            scenario, seed, counts, key, probability, lo, hi, actual
        )


def exported_statistics(payload: dict) -> dict[str, float]:
    """The exportable per-outcome statistics derivable from a sample payload."""
    shots = payload["shots"]
    return {key: count / shots for key, count in payload["counts"].items()}


def canonical_json(payload: dict) -> str:
    """Stable exported JSON text, mirroring the exportable result document."""
    return json.dumps(payload, sort_keys=True)


# ------------------------------------------------------------- test circuits


def ground_state() -> Circuit:
    return Circuit(2, 2).measure(0, 0).measure(1, 1)


def x_flipped() -> Circuit:
    return Circuit(2, 2).x(0).measure(0, 0).measure(1, 1)


def remapped_flip() -> Circuit:
    # q[0] -> c[1] in a 3-bit register: "010", unwritten clbits stay 0.
    return Circuit(2, 3).x(0).measure(0, 1)


def unwritten_clbits() -> Circuit:
    return Circuit(2, 3).x(0).measure(0, 0)


def hadamard_state() -> Circuit:
    return Circuit(1, 1).h(0).measure(0, 0)


def bell_state() -> Circuit:
    return Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)


def bell_state_partial_mapping() -> Circuit:
    # Entangle q[0] with q[2], read them into c[2] and c[0].
    return Circuit(3, 3).h(0).cx(0, 2).measure(0, 2).measure(2, 0)


def controlled_z_phase_state() -> Circuit:
    # x q[1]; h q[0]; cz q[0],q[1] -> (|01> - |11>)/sqrt(2). q[1] is the
    # high-order classical bit, so the observed strings are "10" and "11".
    return Circuit(2, 2).x(1).h(0).cz(0, 1).measure(0, 0).measure(1, 1)


def controlled_rx_pi_state() -> Circuit:
    # h q[0]; crx(pi) q[0],q[1] -> (|00> - i|11>)/sqrt(2).
    return Circuit(2, 2).h(0).crx(math.pi, 0, 1).measure(0, 0).measure(1, 1)


def ry_angle_state() -> Circuit:
    # ry(pi/3) on |0>: P(1) = sin^2(pi/6) = 1/4.
    return Circuit(1, 1).ry(math.pi / 3, 0).measure(0, 0)


def rx_pi_state() -> Circuit:
    # rx(pi) maps |0> to -i|1>: deterministic.
    return Circuit(1, 1).rx(math.pi, 0).measure(0, 0)


def noisy_x_state() -> Circuit:
    return Circuit(1, 1).x(0).measure(0, 0)


# Deterministic ideal scenarios: builder -> the single expected bit string.
DETERMINISTIC_IDEAL = [
    ("ground state", ground_state, "00"),
    ("x flip", x_flipped, "01"),
    ("rx(pi) flip", rx_pi_state, "1"),
    ("remapped q->c flip", remapped_flip, "010"),
    ("unwritten clbits zero", unwritten_clbits, "001"),
]

# Statistical ideal scenarios: builder -> theoretical classical distribution.
STATISTICAL_IDEAL = [
    ("hadamard", hadamard_state, {"0": 0.5, "1": 0.5}),
    ("bell", bell_state, {"00": 0.5, "11": 0.5}),
    ("bell partial mapping", bell_state_partial_mapping, {"000": 0.5, "101": 0.5}),
    ("controlled-z phase", controlled_z_phase_state, {"10": 0.5, "11": 0.5}),
    ("controlled-rx(pi)", controlled_rx_pi_state, {"00": 0.5, "11": 0.5}),
    ("ry(pi/3) biased", ry_angle_state, {"0": 0.75, "1": 0.25}),
]

# Every builder participates in OpenQASM round trips, proving each circuit
# is expressible in and parseable from the supported subset.
ALL_BUILDERS = [
    ground_state,
    x_flipped,
    remapped_flip,
    unwritten_clbits,
    hadamard_state,
    bell_state,
    bell_state_partial_mapping,
    controlled_z_phase_state,
    controlled_rx_pi_state,
    ry_angle_state,
    rx_pi_state,
]


# ------------------------------------------------- deterministic ideal circuits


@pytest.mark.parametrize("scenario,builder,expected", DETERMINISTIC_IDEAL)
def test_deterministic_circuits_fall_on_unique_outcome(scenario, builder, expected):
    shots = 37
    payload = builder().sample(shots=shots, seed=7)
    assert payload["shots"] == shots
    assert payload["num_qubits"] == builder().num_qubits
    assert payload["num_clbits"] == len(expected)
    assert payload["counts"] == {expected: shots}, (
        f"scenario={scenario!r}: expected unique outcome {expected!r} for all "
        f"{shots} shots, got {payload['counts']}"
    )


def test_classical_width_and_qubit_to_clbit_mapping():
    # Identity-order mapping on the Bell circuit: keys are two bits wide.
    payload = bell_state().sample(shots=50, seed=3)
    assert payload["num_clbits"] == 2
    assert set(payload["counts"]) <= {"00", "11"}
    assert all(len(key) == 2 for key in payload["counts"])

    # Crossed mapping: q[0] -> c[2], q[2] -> c[0] yields "000"/"101".
    mapped = bell_state_partial_mapping().sample(shots=50, seed=3)
    assert mapped["num_clbits"] == 3
    assert sum(mapped["counts"].values()) == 50
    assert set(mapped["counts"]) <= {"000", "101"}
    assert all(len(key) == 3 for key in mapped["counts"])


# ------------------------------------------------- statistical ideal circuits


@pytest.mark.parametrize("scenario,builder,expected", STATISTICAL_IDEAL)
def test_superposition_frequencies_match_theoretical_distribution(
    scenario, builder, expected
):
    payload = builder().sample(shots=SHOTS, seed=2026)
    assert_matches_distribution(payload, expected, scenario, 2026)


def test_bell_state_explicitly_forbids_zero_probability_strings():
    for seed in SEEDS:
        counts = bell_state().sample(shots=SHOTS, seed=seed)["counts"]
        for forbidden in ("01", "10"):
            assert forbidden not in counts, (
                f"scenario='bell' seed={seed}: theoretically impossible bit "
                f"string {forbidden!r} observed; counts={counts}"
            )
        assert set(counts) == {"00", "11"}


# --------------------------------------------- OpenQASM round trip as an entry


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_circuits_round_trip_through_public_qasm_entry_with_identical_payload(builder):
    circuit = builder()
    reparsed = Circuit.from_qasm(circuit.to_qasm())
    direct = circuit.sample(shots=256, seed=123)
    via_parser = reparsed.sample(shots=256, seed=123)
    assert via_parser == direct


# ------------------------------------------------------- seed as contract: SV


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "scenario,builder",
    [("hadamard", hadamard_state), ("bell", bell_state), ("ry(pi/3)", ry_angle_state)],
)
def test_seeded_statevector_payloads_are_itemwise_reproducible(seed, scenario, builder):
    circuit = builder()
    first = circuit.sample(shots=SHOTS, seed=seed)
    second = circuit.sample(shots=SHOTS, seed=seed)
    # Complete counts dictionary and every exportable payload field.
    assert second == first, (
        f"scenario={scenario!r} seed={seed}: state-vector payloads differ: "
        f"{first} != {second}"
    )
    assert exported_statistics(second) == exported_statistics(first)
    assert canonical_json(second) == canonical_json(first)


@pytest.mark.parametrize("seed", SEEDS)
def test_seed_reproducibility_survives_unrelated_random_work(seed):
    circuit = bell_state()
    first = circuit.sample(shots=SHOTS, seed=seed)

    # Consume global-module randomness and run other seeded simulations
    # (different circuits, seeds and backends) in between.
    random.shuffle([random.random() for _ in range(500)])
    hadamard_state().sample(shots=1000, seed=seed + 1)
    noisy_x_state().sample(
        shots=1000, seed=seed + 2, noise_model={"bit_flip": 0.3}
    )
    bell_state().sample(shots=500, seed=999, noise_model={"depolarizing": 0.1})
    random.getrandbits(256)

    again = circuit.sample(shots=SHOTS, seed=seed)
    assert again == first, (
        f"scenario='bell' seed={seed}: payload changed after unrelated random "
        f"tasks: {first['counts']} != {again['counts']}"
    )
    assert exported_statistics(again) == exported_statistics(first)
    assert canonical_json(again) == canonical_json(first)


def test_unseeded_run_is_checked_only_for_statistical_properties():
    # No explicit seed: only statistical invariants are required. Two random
    # outcomes are neither required to differ nor required to agree.
    payload = bell_state().sample(shots=SHOTS)
    assert_matches_distribution(payload, {"00": 0.5, "11": 0.5}, "bell unseeded", "default")
    assert set(payload["counts"]) == {"00", "11"}

    again = bell_state().sample(shots=SHOTS)
    assert sum(again["counts"].values()) == SHOTS
    assert set(again["counts"]) == {"00", "11"}


# ------------------------------------------------------- seed as contract: DM


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "scenario,model",
    [
        ("x + zero bit flip", {"bit_flip": 0.0}),
        ("x + bit flip 0.3", {"bit_flip": 0.3}),
        ("bell + depolarizing 0.1", {"depolarizing": 0.1}),
        ("bell + decoherence", {"amplitude_damping": 0.2, "phase_damping": 0.1}),
    ],
)
def test_seeded_density_matrix_payloads_are_itemwise_reproducible(seed, scenario, model):
    builder = noisy_x_state if scenario.startswith("x") else bell_state
    circuit = builder()
    first = circuit.sample(shots=SHOTS, seed=seed, noise_model=model)
    second = circuit.sample(shots=SHOTS, seed=seed, noise_model=model)
    assert first["schema_version"] == 2
    assert first["noise_model"] == second["noise_model"] == model
    assert second == first, (
        f"scenario={scenario!r} seed={seed}: density-matrix payloads differ: "
        f"{first} != {second}"
    )
    assert exported_statistics(second) == exported_statistics(first)
    assert canonical_json(second) == canonical_json(first)


def test_density_matrix_reproducibility_survives_unrelated_random_work():
    model = {"bit_flip": 0.3}
    circuit = noisy_x_state()
    first = circuit.sample(shots=SHOTS, seed=55, noise_model=model)
    random.sample(range(10_000), 2000)
    bell_state().sample(shots=2000, seed=55)
    noisy_x_state().sample(shots=500, seed=56, noise_model={"depolarizing": 0.4})
    again = circuit.sample(shots=SHOTS, seed=55, noise_model=model)
    assert again == first
    assert canonical_json(again) == canonical_json(first)


def test_statevector_and_density_matrix_are_each_stable_not_cross_compared():
    # Both backends must be individually repeatable. The test deliberately
    # never asserts equality (or inequality) between the two backends, since
    # the two sampling algorithms must not be assumed to draw RNG values in
    # the same order -- even at equal shots and seed.
    seed, shots = 7, 2000

    ideal_a = hadamard_state().sample(shots=shots, seed=seed)
    ideal_b = hadamard_state().sample(shots=shots, seed=seed)
    assert ideal_a == ideal_b

    zero_noise = {"bit_flip": 0.0}
    density_a = hadamard_state().sample(shots=shots, seed=seed, noise_model=zero_noise)
    density_b = hadamard_state().sample(shots=shots, seed=seed, noise_model=zero_noise)
    assert density_a == density_b

    # Each backend still shows the 50/50 physics; no cross-backend count
    # comparison is made anywhere in this test.
    for payload, label in ((ideal_a, "state-vector"), (density_a, "density-matrix")):
        counts = payload["counts"]
        assert sum(counts.values()) == shots
        assert set(counts) == {"0", "1"}, label


# --------------------------------------------------------------- noise physics


def test_zero_noise_degrades_to_ideal_statistics():
    # A zero-probability channel still selects the density-matrix backend but
    # must reproduce the ideal distribution statistically.
    payload = hadamard_state().sample(
        shots=SHOTS, seed=17, noise_model={"bit_flip": 0.0, "depolarizing": 0.0}
    )
    assert payload["schema_version"] == 2
    assert_matches_distribution(payload, {"0": 0.5, "1": 0.5}, "h + zero noise", 17)


@pytest.mark.parametrize(
    "scenario,model,expected",
    [
        # |1> under bit flip p: P(0) = p, P(1) = 1 - p.
        ("x + bit_flip", {"bit_flip": 0.3}, {"0": 0.3, "1": 0.7}),
        # |1> under depolarizing p: P(0) = p/2, P(1) = 1 - p/2.
        ("x + depolarizing", {"depolarizing": 0.3}, {"0": 0.15, "1": 0.85}),
        # |1> under amplitude damping gamma: P(0) = gamma, P(1) = 1 - gamma.
        ("x + amplitude_damping", {"amplitude_damping": 0.3}, {"0": 0.3, "1": 0.7}),
        # Phase damping preserves populations: H stays an even split.
        ("h + phase_damping", {"phase_damping": 0.9}, {"0": 0.5, "1": 0.5}),
    ],
)
def test_single_qubit_noise_follows_closed_form_distribution(
    scenario, model, expected
):
    builder = hadamard_state if scenario.startswith("h") else noisy_x_state
    payload = builder().sample(shots=SHOTS, seed=31, noise_model=model)
    assert payload["noise_model"] == model
    assert_matches_distribution(payload, expected, scenario, 31)


@pytest.mark.parametrize(
    "scenario,model",
    [
        # Full bit flip after x sends |1> deterministically back to |0>.
        ("x + bit_flip=1", {"bit_flip": 1.0}),
        # Full amplitude damping after x deterministically decays |1> to |0>.
        ("x + amplitude_damping=1", {"amplitude_damping": 1.0}),
    ],
)
def test_probability_boundary_noise_is_deterministic(scenario, model):
    shots = 64
    payload = noisy_x_state().sample(shots=shots, seed=4, noise_model=model)
    assert payload["schema_version"] == 2
    assert payload["noise_model"] == model
    assert payload["counts"] == {"0": shots}, (
        f"scenario={scenario!r}: boundary noise must give the unique outcome "
        f"'0' for all {shots} shots, got {payload['counts']}"
    )
