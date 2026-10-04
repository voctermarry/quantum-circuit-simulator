"""Pure validation for the JSON document inputs (samples, manifests, baselines).

The command layer only reads bytes and decodes UTF-8; every syntactic,
structural, type and range rule for the auxiliary JSON documents lives
here, operating on text or already-parsed values. Nothing in this module
touches the file system, standard input or the standard streams, so the
rules can be exercised independently of the command entry point.

The three document families share strict-JSON conventions (duplicate keys
and ``NaN``/``Infinity`` constants are rejected, booleans never stand in
for integers); :func:`load_json` centralizes them and each family supplies
its own structural validator.
"""

from __future__ import annotations

import json
import math

from .noise import MAX_NOISE_QUBITS

BATCH_SCHEMA_VERSION = 1
MAX_BATCH_JOBS = 100

_VERIFY_SAMPLES_ROOT_KEYS = ("schema_version", "counts")
_BATCH_ROOT_KEYS = ("schema_version", "jobs")
_BATCH_JOB_KEYS = ("id", "source", "shots", "seed", "noise_model")

_BASELINE_ROOT_KEYS = ("schema_version", "job_count", "succeeded", "failed", "results")
_BASELINE_RESULT_KEYS = ("id", "status", "output", "error")
_OUTPUT_V1_KEYS = ("schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts")
_OUTPUT_V2_KEYS = (
    "schema_version",
    "shots",
    "seed",
    "num_qubits",
    "num_clbits",
    "noise_model",
    "counts",
)
_NOISE_CHANNELS = ("amplitude_damping", "phase_damping", "bit_flip", "depolarizing")
_ERROR_KEYS = ("error", "message", "line", "column")
_RECONCILE_ERROR_NAMES = (
    "io_error",
    "parse_error",
    "validation_error",
    "noise_model_error",
    "simulation_error",
)
# Task errors that carry a source position; every other batch-simulate task
# error is exactly ``{"error", "message"}``.
_POSITIONED_ERROR_NAMES = ("parse_error", "validation_error")


class DocumentError(Exception):
    """A document failed a syntax, structural, type or range rule.

    The command layer maps the same exception to family-specific error
    names (``sample_input_error``, ``batch_input_error``,
    ``reconcile_input_error``); the message text is already final.
    """


def is_json_int(value: object) -> bool:
    """True for JSON integers (``bool`` is a subclass of int but not one)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _pairs_no_dupes(pairs: list[tuple[str, object]], noun: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DocumentError(f"duplicate key {key!r} in {noun}")
        result[key] = value
    return result


def _reject_constant(noun: str):
    def reject(value: str) -> None:
        # json accepts NaN/Infinity by default; they are not valid JSON numbers.
        raise DocumentError(f"{noun} contains non-JSON constant {value!r}")

    return reject


def load_json(text: str, noun: str) -> object:
    """Parse strict JSON for the document family named *noun*.

    *noun* is the exact phrase used in diagnostics ("samples document",
    "batch manifest", "baseline"). Duplicate keys and non-JSON constants
    raise :class:`DocumentError`; plain syntax errors propagate as
    :class:`json.JSONDecodeError` so callers can apply the family's own
    "is not valid JSON" prefix.
    """
    return json.loads(
        text,
        object_pairs_hook=lambda pairs: _pairs_no_dupes(pairs, noun),
        parse_constant=_reject_constant(noun),
    )


# --------------------------------------------------------------- samples doc


def parse_samples_document(text: str, num_clbits: int) -> dict[str, int]:
    """Validate the samples document, returning its counts mapping.

    Every structural, type and width rule is checked here, before any
    state is evolved. Raises :class:`DocumentError` on the first problem.
    """
    try:
        data = load_json(text, "samples document")
    except json.JSONDecodeError as exc:
        raise DocumentError(f"samples document is not valid JSON: {exc}") from None

    if not isinstance(data, dict):
        raise DocumentError("samples document must be a JSON object")
    for key in data:
        if key not in _VERIFY_SAMPLES_ROOT_KEYS:
            raise DocumentError(f"unknown samples key {key!r}")

    if "schema_version" not in data:
        raise DocumentError("samples document is missing required key 'schema_version'")
    schema_version = data["schema_version"]
    if not is_json_int(schema_version) or schema_version != 1:
        raise DocumentError(f"samples schema_version must be 1, got {schema_version!r}")

    if "counts" not in data:
        raise DocumentError("samples document is missing required key 'counts'")
    counts = data["counts"]
    if not isinstance(counts, dict) or not counts:
        raise DocumentError("samples 'counts' must be a non-empty JSON object")
    for key, value in counts.items():
        if len(key) != num_clbits or any(bit not in "01" for bit in key):
            raise DocumentError(
                f"samples 'counts' key {key!r} is not a {num_clbits}-bit measurement string"
            )
        if not is_json_int(value) or value <= 0:
            raise DocumentError("samples 'counts' values must be positive integers")
    return counts


# -------------------------------------------------------------- batch manifest


def validate_batch_manifest(data: object) -> list[dict[str, object]]:
    """Validate the parsed manifest, returning the raw jobs in list order.

    Every structural, type and range rule is checked here, before any task
    file is read. Raises :class:`DocumentError` on the first problem.
    """
    if not isinstance(data, dict):
        raise DocumentError("batch manifest must be a JSON object")

    for key in data:
        if key not in _BATCH_ROOT_KEYS:
            raise DocumentError(f"unknown batch manifest key {key!r}")

    if "schema_version" not in data:
        raise DocumentError("batch manifest is missing required key 'schema_version'")
    schema_version = data["schema_version"]
    if not is_json_int(schema_version) or schema_version != BATCH_SCHEMA_VERSION:
        raise DocumentError(
            f"batch manifest schema_version must be {BATCH_SCHEMA_VERSION}, got {schema_version!r}"
        )

    if "jobs" not in data:
        raise DocumentError("batch manifest is missing required key 'jobs'")
    jobs = data["jobs"]
    if not isinstance(jobs, list):
        raise DocumentError("batch manifest 'jobs' must be an array")
    if not 1 <= len(jobs) <= MAX_BATCH_JOBS:
        raise DocumentError(
            f"batch manifest 'jobs' must contain between 1 and {MAX_BATCH_JOBS} jobs, got {len(jobs)}"
        )

    seen_ids: set[str] = set()
    for index, job in enumerate(jobs):
        where = f"job at jobs[{index}]"
        if not isinstance(job, dict):
            raise DocumentError(f"{where} must be a JSON object")
        for key in job:
            if key not in _BATCH_JOB_KEYS:
                raise DocumentError(f"{where} has unknown key {key!r}")

        if "id" not in job:
            raise DocumentError(f"{where} is missing required key 'id'")
        job_id = job["id"]
        if not isinstance(job_id, str) or not job_id:
            raise DocumentError(f"{where} 'id' must be a non-empty string")
        if job_id in seen_ids:
            raise DocumentError(f"duplicate job id {job_id!r}")
        seen_ids.add(job_id)

        if "source" not in job:
            raise DocumentError(f"{where} is missing required key 'source'")
        source = job["source"]
        if not isinstance(source, str):
            raise DocumentError(f"{where} 'source' must be a string")
        if source == "-":
            raise DocumentError(f"{where} 'source' must not be '-'; stdin is reserved for the manifest")

        if "shots" in job:
            shots = job["shots"]
            if not is_json_int(shots) or shots <= 0:
                raise DocumentError(f"{where} 'shots' must be a positive integer")

        if "seed" in job and not is_json_int(job["seed"]):
            raise DocumentError(f"{where} 'seed' must be an integer")

        if "noise_model" in job:
            noise_model = job["noise_model"]
            if not isinstance(noise_model, str):
                raise DocumentError(f"{where} 'noise_model' must be a string path")
            if noise_model == "-":
                raise DocumentError(
                    f"{where} 'noise_model' must not be '-'; stdin is reserved for the manifest"
                )

    return jobs


# ------------------------------------------------------------------ baselines


def _require_int(value: object, description: str) -> int:
    if not is_json_int(value):
        raise DocumentError(f"{description} must be an integer")
    return value


def check_finite_numbers(value: object, description: str) -> None:
    """Reject non-finite floats anywhere in an already-parsed JSON payload.

    Integers, strings, booleans and ``None`` are unproblematic; floats are
    only reachable through explicit decimal literals (constants such as
    ``NaN`` are refused at parse time) but must still be finite.
    """
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DocumentError(f"{description} is not a finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_finite_numbers(item, f"{description}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite_numbers(item, f"{description}[{index}]")


def _validate_baseline_output(output: object, where: str) -> dict[str, object]:
    if not isinstance(output, dict):
        raise DocumentError(f"{where} must be a JSON object")
    for key in output:
        if key not in _OUTPUT_V2_KEYS:
            raise DocumentError(f"{where} has unknown key {key!r}")

    for required in _OUTPUT_V1_KEYS:
        if required not in output:
            raise DocumentError(f"{where} is missing required key {required!r}")

    schema_version = _require_int(output["schema_version"], f"{where} 'schema_version'")
    shots = _require_int(output["shots"], f"{where} 'shots'")
    _require_int(output["seed"], f"{where} 'seed'")
    num_qubits = _require_int(output["num_qubits"], f"{where} 'num_qubits'")
    num_clbits = _require_int(output["num_clbits"], f"{where} 'num_clbits'")

    if schema_version not in (1, 2):
        raise DocumentError(f"{where} 'schema_version' must be 1 or 2")
    if shots <= 0:
        raise DocumentError(f"{where} 'shots' must be a positive integer")
    if not 1 <= num_qubits <= 20:
        raise DocumentError(f"{where} 'num_qubits' must be between 1 and 20")
    if not 1 <= num_clbits <= 20:
        raise DocumentError(f"{where} 'num_clbits' must be between 1 and 20")

    counts = output["counts"]
    if not isinstance(counts, dict) or not counts:
        raise DocumentError(f"{where} 'counts' must be a non-empty JSON object")
    total = 0
    for key, value in counts.items():
        if not isinstance(key, str) or not key:
            raise DocumentError(f"{where} 'counts' keys must be non-empty strings")
        if not is_json_int(value) or value <= 0:
            raise DocumentError(f"{where} 'counts' values must be positive integers")
        if len(key) != num_clbits or any(bit not in "01" for bit in key):
            raise DocumentError(
                f"{where} 'counts' key {key!r} is not a {num_clbits}-bit measurement string"
            )
        total += value
    if total != shots:
        raise DocumentError(f"{where} 'counts' must sum to shots ({shots}), got {total}")

    if schema_version == 2:
        if num_qubits > MAX_NOISE_QUBITS:
            raise DocumentError(
                f"{where} noisy output must have at most {MAX_NOISE_QUBITS} qubits"
            )
        if "noise_model" not in output:
            raise DocumentError(f"{where} is missing required key 'noise_model'")
        noise_model = output["noise_model"]
        if not isinstance(noise_model, dict) or not noise_model:
            raise DocumentError(f"{where} 'noise_model' must be a non-empty JSON object")
        for channel, probability in noise_model.items():
            if channel not in _NOISE_CHANNELS:
                raise DocumentError(f"{where} has unknown noise channel {channel!r}")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise DocumentError(f"{where} noise channel {channel!r} must be a number")
            if not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
                raise DocumentError(
                    f"{where} noise channel {channel!r} probability must be between 0 and 1"
                )
    elif "noise_model" in output:
        raise DocumentError(f"{where} schema_version 1 output must not contain 'noise_model'")

    return output


def _validate_baseline_error(error: object, where: str) -> dict[str, object]:
    if not isinstance(error, dict):
        raise DocumentError(f"{where} must be a JSON object")
    for key in error:
        if key not in _ERROR_KEYS:
            raise DocumentError(f"{where} has unknown key {key!r}")

    name = error.get("error")
    if not isinstance(name, str) or name not in _RECONCILE_ERROR_NAMES:
        raise DocumentError(f"{where} 'error' must name a batch-simulate task error")
    if "message" not in error or not isinstance(error["message"], str):
        raise DocumentError(f"{where} is missing a string 'message'")

    if name in _POSITIONED_ERROR_NAMES:
        for position in ("line", "column"):
            if position not in error:
                raise DocumentError(f"{where} is missing required key {position!r}")
            value = error[position]
            if not is_json_int(value) or value < 1:
                raise DocumentError(f"{where} {position!r} must be a positive integer")
    elif "line" in error or "column" in error:
        raise DocumentError(f"{where} {name!r} errors must not carry a source position")

    return error


def validate_baseline(data: object, jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Validate the batch-result *data* against the manifest *jobs*.

    Self-consistency and correspondence (count, ids, order) are checked
    here, before any task is re-run. Raises :class:`DocumentError`.
    """
    if not isinstance(data, dict):
        raise DocumentError("baseline must be a JSON object")
    for key in data:
        if key not in _BASELINE_ROOT_KEYS:
            raise DocumentError(f"unknown baseline key {key!r}")
    if any(key not in data for key in _BASELINE_ROOT_KEYS):
        raise DocumentError("baseline is missing one or more required root keys")

    schema_version = _require_int(data["schema_version"], "baseline 'schema_version'")
    if schema_version != BATCH_SCHEMA_VERSION:
        raise DocumentError(
            f"baseline schema_version must be {BATCH_SCHEMA_VERSION}, got {schema_version!r}"
        )

    job_count = _require_int(data["job_count"], "baseline 'job_count'")
    succeeded = _require_int(data["succeeded"], "baseline 'succeeded'")
    failed = _require_int(data["failed"], "baseline 'failed'")
    if job_count < 0 or succeeded < 0 or failed < 0:
        raise DocumentError("baseline counters must be non-negative")
    if succeeded + failed != job_count:
        raise DocumentError("baseline 'succeeded' + 'failed' must equal 'job_count'")
    if job_count != len(jobs):
        raise DocumentError(
            f"baseline job_count {job_count} does not match manifest job count {len(jobs)}"
        )

    results = data["results"]
    if not isinstance(results, list):
        raise DocumentError("baseline 'results' must be an array")
    if len(results) != job_count:
        raise DocumentError("baseline 'results' length must equal 'job_count'")

    seen: set[str] = set()
    actual_succeeded = 0
    for index, entry in enumerate(results):
        where = f"baseline result[{index}]"
        if not isinstance(entry, dict):
            raise DocumentError(f"{where} must be a JSON object")
        for key in entry:
            if key not in _BASELINE_RESULT_KEYS:
                raise DocumentError(f"{where} has unknown key {key!r}")

        if "id" not in entry:
            raise DocumentError(f"{where} is missing required key 'id'")
        job_id = entry["id"]
        if not isinstance(job_id, str) or not job_id:
            raise DocumentError(f"{where} 'id' must be a non-empty string")
        if job_id in seen:
            raise DocumentError(f"duplicate baseline result id {job_id!r}")
        seen.add(job_id)
        if job_id != jobs[index]["id"]:
            raise DocumentError(
                f"{where} id {job_id!r} does not match manifest id {jobs[index]['id']!r}"
            )

        if "status" not in entry:
            raise DocumentError(f"{where} is missing required key 'status'")
        status = entry["status"]
        if status == "succeeded":
            if "output" not in entry:
                raise DocumentError(f"{where} succeeded result is missing 'output'")
            if "error" in entry:
                raise DocumentError(f"{where} succeeded result must not contain 'error'")
            _validate_baseline_output(entry["output"], f"{where} output")
            actual_succeeded += 1
        elif status == "failed":
            if "error" not in entry:
                raise DocumentError(f"{where} failed result is missing 'error'")
            if "output" in entry:
                raise DocumentError(f"{where} failed result must not contain 'output'")
            _validate_baseline_error(entry["error"], f"{where} error")
        else:
            raise DocumentError(f"{where} 'status' must be 'succeeded' or 'failed'")

        required_keys = (
            {"id", "status", "output"} if status == "succeeded" else {"id", "status", "error"}
        )
        if set(entry) != required_keys:
            raise DocumentError(f"{where} has unexpected keys for status {status!r}")

    if actual_succeeded != succeeded or len(results) - actual_succeeded != failed:
        raise DocumentError("baseline counters do not match result statuses")

    return results


def json_equal(expected: object, actual: object) -> bool:
    """Compare parsed JSON ignoring object key order and whitespace.

    Array order and scalar values (including int-vs-float distinctions and
    numeric precision) are significant.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(json_equal(expected[key], actual[key]) for key in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            json_equal(left, right) for left, right in zip(expected, actual)
        )
    return expected == actual and type(expected) is type(actual)
