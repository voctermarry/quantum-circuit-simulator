"""End-to-end tests for measurement-sampling reproducibility.

These tests treat the random seed as an observable contract of the public
construction and execution entry points (:class:`~quantum_circuit.Circuit`
and the sampling functions in :mod:`quantum_circuit.simulator`):

- identical circuit/shots/seed/noise configuration reproduces the exact
  counts dictionary, even after unrelated random work has run in between;
- ideal deterministic circuits collapse to a single outcome;
- superposition, Bell and parameterized/controlled-gate circuits match
  their closed-form distributions within a confidence interval derived
  from the theoretical probability, the shot count and a fixed
  significance level (never a hard-coded count);
- noisy single-qubit circuits reproduce their closed-form distributions,
  with zero-noise models degenerating to the ideal statistics and
  boundary probabilities producing deterministic outcomes.

All statistical checks use a two-sided normal-approximation confidence
interval at significance level ``_ALPHA``; failure messages name the
circuit scenario, the seed, the expected interval and the observed count.
"""

from __future__ import annotations

import json
import math
import random
import statistics

from quantum_circuit import Circuit
from quantum_circuit.noise import simulate_density_matrix
from quantum_circuit.openqasm import parse
from quantum_circuit.simulator import (
    sample_counts,
    sample_counts_from_probabilities,
    simulate_state_vector,
)


def _program(circuit: Circuit):
    """Parse the circuit's own OpenQASM rendering (the CLI's input path)."""
    return parse(circuit.to_qasm())

# Fixed significance level for every statistical frequency check. With
# shots in the thousands the normal approximation is accurate and the
# per-check false-failure probability is 0.1%.
_ALPHA = 0.001

_SHOTS = 4000


# ------------------------------------------------------------------- helpers


def _confidence_interval(probability: float, shots: int, alpha: float = _ALPHA):
    """Two-sided normal-approximation count interval for one outcome."""
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    center = probability * shots
    margin = z * math.sqrt(shots * probability * (1.0 - probability))
    return center - margin, center + margin


def _assert_frequency(counts, key, probability, shots, *, scenario, seed):
    """Assert the observed count of *key* lies in the confidence interval."""
    lo, hi = _confidence_interval(probability, shots)
    observed = counts.get(key, 0)
    assert lo <= observed <= hi, (
        f"{scenario}: outcome {key!r} count {observed} outside the "
        f"{1.0 - _ALPHA:.1%} confidence interval [{lo:.1f}, {hi:.1f}] "
        f"(theoretical p={probability}, shots={shots}, seed={seed})"
    )


def _assert_forbidden(counts, keys, *, scenario, seed):
    """Assert outcomes with theoretical probability zero never appear."""
    for key in keys:
        observed = counts.get(key, 0)
        assert observed == 0, (
            f"{scenario}: outcome {key!r} has theoretical probability 0 but "
            f"was observed {observed} times (seed={seed})"
        )


def _assert_wellformed_payload(result, shots, num_qubits, num_clbits, *, scenario):
    """Verify classical width, field echo and total sample count."""
    assert result["shots"] == shots, f"{scenario}: echoed shots differ"
    assert result["num_qubits"] == num_qubits, f"{scenario}: num_qubits differs"
    assert result["num_clbits"] == num_clbits, f"{scenario}: num_clbits differs"
    counts = result["counts"]
    total = sum(counts.values())
    assert total == shots, (
        f"{scenario}: counts sum to {total}, expected {shots} samples"
    )
    for key in counts:
        assert len(key) == num_clbits and set(key) <= {"0", "1"}, (
            f"{scenario}: counts key {key!r} is not a {num_clbits}-bit string"
        )
    assert list(counts) == sorted(counts), (
        f"{scenario}: counts keys are not in canonical sorted order"
    )


# ------------------------------------------------------- deterministic circuits


def test_ground_state_collapses_to_all_zeros():
    scenario = "ground state, 2 qubits, both measured"
    circuit = Circuit(2, 2).measure(0, 0).measure(1, 1)
    result = circuit.sample(shots=256, seed=7)
    _assert_wellformed_payload(result, 256, 2, 2, scenario=scenario)
    assert result["counts"] == {"00": 256}, (
        f"{scenario}: expected the unique outcome '00', got {result['counts']}"
    )


def test_pauli_x_collapses_to_single_excited_outcome():
    scenario = "x on q[0], 2 qubits"
    circuit = Circuit(2, 2).x(0).measure(0, 0).measure(1, 1)
    result = circuit.sample(shots=128, seed=3)
    _assert_wellformed_payload(result, 128, 2, 2, scenario=scenario)
    # q[0] is the least significant bit, read into c[0] (rightmost character).
    assert result["counts"] == {"01": 128}, (
        f"{scenario}: expected the unique outcome '01', got {result['counts']}"
    )


def test_rx_pi_is_deterministic_not():
    scenario = "rx(pi) on q[0]"
    circuit = Circuit(1, 1).rx(math.pi, 0).measure(0, 0)
    result = circuit.sample(shots=64, seed=11)
    _assert_wellformed_payload(result, 64, 1, 1, scenario=scenario)
    assert result["counts"] == {"1": 64}, (
        f"{scenario}: expected the unique outcome '1', got {result['counts']}"
    )


def test_controlled_gates_on_computational_basis_are_deterministic():
    scenario = "x q[0]; cx q[0],q[1]; x q[1]... cz"
    # x q[0]; cx q[0],q[1] prepares |11>; a following cz only adds a phase,
    # so the measured outcome stays deterministically '11'.
    circuit = Circuit(2, 2).x(0).cx(0, 1).cz(0, 1).measure(0, 0).measure(1, 1)
    result = circuit.sample(shots=96, seed=5)
    _assert_wellformed_payload(result, 96, 2, 2, scenario=scenario)
    assert result["counts"] == {"11": 96}, (
        f"{scenario}: expected the unique outcome '11', got {result['counts']}"
    )


# ------------------------------------------- classical width and qubit mapping


def test_partial_measurement_maps_qubit_to_requested_clbit():
    scenario = "x q[0] measured into c[1] of a 3-bit register"
    circuit = Circuit(2, 3).x(0).measure(0, 1)
    result = circuit.sample(shots=32, seed=1)
    _assert_wellformed_payload(result, 32, 2, 3, scenario=scenario)
    # c[1] set, unwritten c[2] and c[0] stay 0; keys read c[2]c[1]c[0].
    assert result["counts"] == {"010": 32}, (
        f"{scenario}: expected the unique outcome '010', got {result['counts']}"
    )


def test_superposition_measured_into_high_clbit():
    scenario = "h q[0] measured into c[2] of a 3-bit register"
    seed = 21
    circuit = Circuit(1, 3).h(0).measure(0, 2)
    result = circuit.sample(shots=_SHOTS, seed=seed)
    _assert_wellformed_payload(result, _SHOTS, 1, 3, scenario=scenario)
    counts = result["counts"]
    _assert_forbidden(
        counts, {"001", "010", "011", "101", "110", "111"}, scenario=scenario, seed=seed
    )
    _assert_frequency(counts, "000", 0.5, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "100", 0.5, _SHOTS, scenario=scenario, seed=seed)


# ------------------------------------------- ideal superposition and Bell state


def test_hadamard_superposition_frequencies():
    scenario = "h q[0], single qubit"
    seed = 101
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    result = circuit.sample(shots=_SHOTS, seed=seed)
    _assert_wellformed_payload(result, _SHOTS, 1, 1, scenario=scenario)
    counts = result["counts"]
    _assert_frequency(counts, "0", 0.5, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "1", 0.5, _SHOTS, scenario=scenario, seed=seed)


def test_bell_state_frequencies_and_forbidden_outcomes():
    scenario = "Bell state h q[0]; cx q[0],q[1]"
    seed = 202
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    result = circuit.sample(shots=_SHOTS, seed=seed)
    _assert_wellformed_payload(result, _SHOTS, 2, 2, scenario=scenario)
    counts = result["counts"]
    # The anti-correlated bit strings have theoretical probability zero.
    _assert_forbidden(counts, {"01", "10"}, scenario=scenario, seed=seed)
    _assert_frequency(counts, "00", 0.5, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "11", 0.5, _SHOTS, scenario=scenario, seed=seed)


# --------------------------------- controlled and parameterized gate statistics


def test_cry_with_control_set_matches_closed_form():
    scenario = "x q[0]; cry(pi/3) q[0],q[1]"
    seed = 303
    circuit = Circuit(2, 2).x(0).cry(math.pi / 3, 0, 1).measure(0, 0).measure(1, 1)
    result = circuit.sample(shots=_SHOTS, seed=seed)
    _assert_wellformed_payload(result, _SHOTS, 2, 2, scenario=scenario)
    counts = result["counts"]
    # Control q[0] is |1>, so cry(pi/3) acts fully on the target:
    # P(target=1) = sin^2(pi/6) = 1/4. c[0] is always 1.
    p_target_one = math.sin(math.pi / 6) ** 2
    _assert_forbidden(counts, {"00", "10"}, scenario=scenario, seed=seed)
    _assert_frequency(counts, "11", p_target_one, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(
        counts, "01", 1.0 - p_target_one, _SHOTS, scenario=scenario, seed=seed
    )


def test_cz_on_double_superposition_is_uniform():
    scenario = "h q[0]; h q[1]; cz q[0],q[1]"
    seed = 404
    circuit = Circuit(2, 2).h(0).h(1).cz(0, 1).measure(0, 0).measure(1, 1)
    result = circuit.sample(shots=_SHOTS, seed=seed)
    _assert_wellformed_payload(result, _SHOTS, 2, 2, scenario=scenario)
    counts = result["counts"]
    # cz only changes phases, so all four outcomes keep probability 1/4.
    for key in ("00", "01", "10", "11"):
        _assert_frequency(counts, key, 0.25, _SHOTS, scenario=scenario, seed=seed)


# ------------------------------------------------------------- the seed contract


def test_same_seed_reproduces_full_payload_item_by_item():
    scenario = "Bell state, seed reproducibility"
    seed = 1234
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    first = circuit.sample(shots=2000, seed=seed)
    second = circuit.sample(shots=2000, seed=seed)
    assert first == second, f"{scenario}: equal arguments produced different payloads"
    assert first["counts"] == second["counts"], (
        f"{scenario}: counts dictionaries differ (seed={seed})"
    )
    # The exportable statistical content is identical item by item.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True), (
        f"{scenario}: serialized payloads differ (seed={seed})"
    )


def test_seed_reproducibility_survives_interleaved_random_work():
    scenario = "Bell state, seed reproducibility after other random work"
    seed = 555
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    reference = circuit.sample(shots=2000, seed=seed)

    # Consume randomness elsewhere: the process-global RNG, other seeds on
    # the same circuit, and other circuits with their own noise models.
    random.seed(999)
    for _ in range(1000):
        random.random()
    circuit.sample(shots=2000, seed=seed + 1)
    Circuit(1, 1).h(0).measure(0, 0).sample(shots=500, seed=seed)
    Circuit(1, 1).x(0).measure(0, 0).sample(
        shots=500, seed=seed, noise_model={"bit_flip": 0.3}
    )

    rerun = circuit.sample(shots=2000, seed=seed)
    assert rerun == reference, (
        f"{scenario}: rerunning with seed={seed} after unrelated random "
        f"work changed the result"
    )


def test_parsed_roundtrip_preserves_seeded_counts():
    scenario = "Bell state via to_qasm/from_qasm round trip"
    seed = 777
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    parsed = Circuit.from_qasm(circuit.to_qasm())
    direct = circuit.sample(shots=1500, seed=seed)
    via_parse = parsed.sample(shots=1500, seed=seed)
    assert via_parse["counts"] == direct["counts"], (
        f"{scenario}: parsed circuit sampled differently (seed={seed})"
    )


def test_default_arguments_are_checked_statistically_only():
    # No seed is supplied: only the statistical contract is verified. The
    # two runs are deliberately NOT required to differ from each other.
    scenario = "h q[0], default arguments"
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    for result in (circuit.sample(shots=_SHOTS), circuit.sample(shots=_SHOTS)):
        _assert_wellformed_payload(result, _SHOTS, 1, 1, scenario=scenario)
        counts = result["counts"]
        _assert_frequency(
            counts, "0", 0.5, _SHOTS, scenario=scenario, seed=result["seed"]
        )
        _assert_frequency(
            counts, "1", 0.5, _SHOTS, scenario=scenario, seed=result["seed"]
        )


# ------------------------------------- state-vector and density-matrix entries


def test_state_vector_sampling_entry_is_reproducible():
    scenario = "state-vector entry sample_counts"
    seed = 88
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    program = _program(circuit)
    state = simulate_state_vector(program)
    first = sample_counts(program, state, 1000, seed)
    second = sample_counts(program, state, 1000, seed)
    assert first == second, f"{scenario}: repeated draws differ (seed={seed})"
    assert sum(first.values()) == 1000
    _assert_forbidden(first, {"01", "10"}, scenario=scenario, seed=seed)


def test_density_matrix_sampling_entry_is_reproducible():
    scenario = "density-matrix entry sample_counts_from_probabilities"
    seed = 89
    noise = {"bit_flip": 0.1}
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    program = _program(circuit)
    probabilities = simulate_density_matrix(program, noise)
    first = sample_counts_from_probabilities(program, probabilities, 1000, seed)
    second = sample_counts_from_probabilities(program, probabilities, 1000, seed)
    assert first == second, f"{scenario}: repeated draws differ (seed={seed})"
    assert sum(first.values()) == 1000
    # Each entry is checked on its own only: the two sampling algorithms
    # are not required to consume random numbers in the same order, so no
    # cross-entry equality is asserted.


# ----------------------------------------------------------------- noisy circuits


def test_zero_noise_degenerates_to_ideal_statistics():
    scenario = "h q[0] with zero-probability noise channels"
    seed = 606
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    result = circuit.sample(
        shots=_SHOTS, seed=seed, noise_model={"bit_flip": 0.0, "depolarizing": 0.0}
    )
    _assert_wellformed_payload(result, _SHOTS, 1, 1, scenario=scenario)
    assert result["noise_model"] == {"bit_flip": 0.0, "depolarizing": 0.0}
    counts = result["counts"]
    _assert_frequency(counts, "0", 0.5, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "1", 0.5, _SHOTS, scenario=scenario, seed=seed)


def test_bit_flip_probability_one_is_deterministic():
    scenario = "x q[0] with bit_flip p=1"
    seed = 707
    circuit = Circuit(1, 1).x(0).measure(0, 0)
    result = circuit.sample(shots=200, seed=seed, noise_model={"bit_flip": 1.0})
    _assert_wellformed_payload(result, 200, 1, 1, scenario=scenario)
    # x prepares |1> and the certain bit flip returns it to |0>.
    assert result["counts"] == {"0": 200}, (
        f"{scenario}: expected the unique outcome '0', got {result['counts']} "
        f"(seed={seed})"
    )


def test_amplitude_damping_probability_one_is_deterministic():
    scenario = "h q[0] with amplitude_damping gamma=1"
    seed = 708
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    result = circuit.sample(shots=200, seed=seed, noise_model={"amplitude_damping": 1.0})
    _assert_wellformed_payload(result, 200, 1, 1, scenario=scenario)
    # Certain decay maps any population to |0>.
    assert result["counts"] == {"0": 200}, (
        f"{scenario}: expected the unique outcome '0', got {result['counts']} "
        f"(seed={seed})"
    )


def test_bit_flip_closed_form_distribution():
    scenario = "x q[0] with bit_flip p=0.25"
    seed = 808
    p_flip = 0.25
    circuit = Circuit(1, 1).x(0).measure(0, 0)
    result = circuit.sample(shots=_SHOTS, seed=seed, noise_model={"bit_flip": p_flip})
    _assert_wellformed_payload(result, _SHOTS, 1, 1, scenario=scenario)
    counts = result["counts"]
    # Closed form: P(1) = 1 - p, P(0) = p.
    _assert_frequency(counts, "1", 1.0 - p_flip, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "0", p_flip, _SHOTS, scenario=scenario, seed=seed)


def test_depolarizing_closed_form_distribution():
    scenario = "ry(pi/3) q[0] with depolarizing p=0.2"
    seed = 809
    p_depol = 0.2
    circuit = Circuit(1, 1).ry(math.pi / 3, 0).measure(0, 0)
    result = circuit.sample(
        shots=_SHOTS, seed=seed, noise_model={"depolarizing": p_depol}
    )
    _assert_wellformed_payload(result, _SHOTS, 1, 1, scenario=scenario)
    counts = result["counts"]
    # Closed form: P(1) = (1 - p) * sin^2(pi/6) + p/2 = 0.8*0.25 + 0.1 = 0.3.
    p_one = (1.0 - p_depol) * math.sin(math.pi / 6) ** 2 + p_depol / 2.0
    _assert_frequency(counts, "1", p_one, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "0", 1.0 - p_one, _SHOTS, scenario=scenario, seed=seed)


def test_full_depolarizing_gives_uniform_statistics():
    scenario = "h q[0] with depolarizing p=1"
    seed = 810
    circuit = Circuit(1, 1).h(0).measure(0, 0)
    result = circuit.sample(shots=_SHOTS, seed=seed, noise_model={"depolarizing": 1.0})
    _assert_wellformed_payload(result, _SHOTS, 1, 1, scenario=scenario)
    counts = result["counts"]
    # The maximally mixed state yields P(0) = P(1) = 1/2.
    _assert_frequency(counts, "0", 0.5, _SHOTS, scenario=scenario, seed=seed)
    _assert_frequency(counts, "1", 0.5, _SHOTS, scenario=scenario, seed=seed)


def test_noisy_sampling_seed_reproducibility():
    scenario = "Bell state with bit_flip+depolarizing, seed reproducibility"
    seed = 909
    noise_model = {"bit_flip": 0.1, "depolarizing": 0.05}
    circuit = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    reference = circuit.sample(shots=2000, seed=seed, noise_model=noise_model)

    # Unrelated random work must not disturb the seeded rerun.
    random.seed(31337)
    for _ in range(500):
        random.random()
    circuit.sample(shots=2000, seed=seed + 1, noise_model=noise_model)
    circuit.sample(shots=2000, seed=seed, noise_model={"bit_flip": 0.4})

    rerun = circuit.sample(shots=2000, seed=seed, noise_model=noise_model)
    assert rerun == reference, (
        f"{scenario}: rerunning with seed={seed} and identical noise model "
        f"changed the result"
    )
    assert rerun["counts"] == reference["counts"]
    assert json.dumps(rerun, sort_keys=True) == json.dumps(reference, sort_keys=True)
