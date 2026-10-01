"""End-to-end tests for the batch-simulate subcommand."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent

BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)
X_GATE = "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"


@pytest.fixture
def make_files(tmp_path):
    def _make(files: dict[str, str]):
        paths = {}
        for name, content in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            paths[name] = str(path)
        return paths

    return _make


@pytest.fixture
def basic_files(make_files):
    return make_files(
        {
            "bell.qasm": HEADER + BELL,
            "x.qasm": HEADER + X_GATE,
            "noise.json": json.dumps({"bit_flip": 0.1}),
        }
    )


def write_manifest(tmp_path, manifest) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def run_batch(manifest_arg, capsys):
    rc = cli.main(["batch-simulate", manifest_arg])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# ------------------------------------------------------------------- successes


def test_all_success_summary_and_field_order(tmp_path, basic_files, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [
            {"id": "bell", "source": "bell.qasm", "shots": 50, "seed": 7},
            {"id": "noisy", "source": "bell.qasm", "noise_model": "noise.json", "shots": 40, "seed": 3},
        ],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 0
    assert err == ""
    assert out.count("\n") == 1
    data = json.loads(out)
    assert list(data) == ["schema_version", "job_count", "succeeded", "failed", "results"]
    assert (data["schema_version"], data["job_count"]) == (1, 2)
    assert (data["succeeded"], data["failed"]) == (2, 0)

    first, second = data["results"]
    assert list(first) == ["id", "status", "output"]
    assert first["id"] == "bell" and first["status"] == "succeeded"
    assert list(first["output"]) == [
        "schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts"
    ]
    assert first["output"]["shots"] == 50
    assert sum(first["output"]["counts"].values()) == 50

    assert list(second) == ["id", "status", "output"]
    output = second["output"]
    assert list(output) == [
        "schema_version", "shots", "seed", "num_qubits", "num_clbits", "noise_model", "counts"
    ]
    assert output["schema_version"] == 2
    assert output["noise_model"] == {"bit_flip": 0.1}


def test_job_defaults_match_simulate(tmp_path, basic_files, capsys):
    manifest = {"schema_version": 1, "jobs": [{"id": "x", "source": "x.qasm"}]}
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 0
    output = json.loads(out)["results"][0]["output"]
    assert (output["shots"], output["seed"]) == (1024, 0)
    assert output["counts"] == {"1": 1024}


def test_output_identical_to_standalone_simulate(tmp_path, basic_files, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [{"id": "bell", "source": "bell.qasm", "shots": 77, "seed": -123}],
    }
    assert cli.main(["batch-simulate", write_manifest(tmp_path, manifest)]) == 0
    batch_out = json.loads(capsys.readouterr().out)["results"][0]["output"]

    assert cli.main(["simulate", str(tmp_path / "bell.qasm"), "--shots", "77", "--seed", "-123"]) == 0
    solo_out = json.loads(capsys.readouterr().out)
    assert batch_out == solo_out


def test_results_keep_manifest_order_and_ids(tmp_path, basic_files, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [
            {"id": "z", "source": "x.qasm", "shots": 4, "seed": 1},
            {"id": "a", "source": "bell.qasm", "shots": 4, "seed": 2},
            {"id": "m", "source": "bell.qasm", "noise_model": "noise.json", "shots": 4, "seed": 3},
        ],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 0
    assert [entry["id"] for entry in json.loads(out)["results"]] == ["z", "a", "m"]


def test_repeated_runs_byte_identical(tmp_path, basic_files, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [
            {"id": "bell", "source": "bell.qasm", "shots": 128, "seed": 123},
            {"id": "noisy", "source": "bell.qasm", "noise_model": "noise.json", "shots": 64, "seed": 9},
        ],
    }
    path = write_manifest(tmp_path, manifest)
    outputs = set()
    for _ in range(3):
        assert cli.main(["batch-simulate", path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_seed_is_independent_of_other_jobs(tmp_path, basic_files, capsys):
    def target_output(jobs):
        path = write_manifest(tmp_path, {"schema_version": 1, "jobs": jobs})
        assert cli.main(["batch-simulate", path]) == 0
        results = json.loads(capsys.readouterr().out)["results"]
        return next(entry["output"] for entry in results if entry["id"] == "target")

    target = {"id": "target", "source": "bell.qasm", "shots": 64, "seed": 42}
    alone = target_output([target])
    with_companions = target_output(
        [
            {"id": "other1", "source": "x.qasm", "shots": 5, "seed": 999},
            {"id": "other2", "source": "bell.qasm", "noise_model": "noise.json", "shots": 7, "seed": -5},
            target,
        ]
    )
    assert alone == with_companions


# ------------------------------------------------------------ per-job failures


def test_job_failures_are_embedded_and_do_not_stop_batch(tmp_path, make_files, capsys):
    make_files(
        {
            "ok.qasm": HEADER + X_GATE,
            "bad_syntax.qasm": HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n",
        }
    )
    manifest = {
        "schema_version": 1,
        "jobs": [
            {"id": "missing", "source": "nope.qasm"},
            {"id": "parse", "source": "bad_syntax.qasm"},
            {"id": "ok", "source": "ok.qasm", "shots": 8, "seed": 1},
        ],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 3
    assert err == ""
    data = json.loads(out)
    assert (data["succeeded"], data["failed"], data["job_count"]) == (1, 2, 3)

    missing, parse_fail, ok = data["results"]
    assert list(missing) == ["id", "status", "error"]
    assert missing["status"] == "failed" and missing["error"]["error"] == "io_error"

    assert parse_fail["status"] == "failed"
    assert parse_fail["error"]["error"] == "parse_error"
    assert parse_fail["error"]["line"] >= 1 and parse_fail["error"]["column"] >= 1

    assert ok["status"] == "succeeded"
    assert ok["output"]["counts"] == {"1": 8}


def test_noise_simulation_error_is_job_failure(tmp_path, make_files, capsys):
    body = (
        "qreg q[11];\ncreg c[11];\n"
        + "".join(f"x q[{i}];\n" for i in range(11))
        + "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(11))
    )
    make_files({"big.qasm": HEADER + body, "noise.json": json.dumps({"bit_flip": 0.2})})
    manifest = {
        "schema_version": 1,
        "jobs": [{"id": "big", "source": "big.qasm", "noise_model": "noise.json"}],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 3
    assert err == ""
    error = json.loads(out)["results"][0]["error"]
    assert error["error"] == "simulation_error"
    assert "10" in error["message"]


def test_noise_model_error_is_job_failure(tmp_path, make_files, capsys):
    make_files({"x.qasm": HEADER + X_GATE, "bad_noise.json": "not json"})
    manifest = {
        "schema_version": 1,
        "jobs": [{"id": "n", "source": "x.qasm", "noise_model": "bad_noise.json"}],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 3
    assert json.loads(out)["results"][0]["error"]["error"] == "noise_model_error"


def test_validation_error_is_job_failure(tmp_path, make_files, capsys):
    make_files({"bad.qasm": HEADER + "qreg q[1];\nqreg q2[1];\n"})
    manifest = {
        "schema_version": 1,
        "jobs": [{"id": "v", "source": "bad.qasm"}],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 3
    error = json.loads(out)["results"][0]["error"]
    assert error["error"] == "validation_error"
    assert error["line"] >= 1


# ------------------------------------------------------- manifest input errors


INVALID_MANIFESTS = [
    ("unknown root key", {"schema_version": 1, "jobs": [], "extra": 1}),
    ("wrong version", {"schema_version": 2, "jobs": []}),
    ("version float", {"schema_version": 1.0, "jobs": []}),
    ("version bool", {"schema_version": True, "jobs": []}),
    ("version string", {"schema_version": "1", "jobs": []}),
    ("missing version", {"jobs": []}),
    ("missing jobs", {"schema_version": 1}),
    ("jobs object", {"schema_version": 1, "jobs": {}}),
    ("jobs string", {"schema_version": 1, "jobs": "x"}),
    ("empty jobs", {"schema_version": 1, "jobs": []}),
    ("job not object", {"schema_version": 1, "jobs": [5]}),
    ("job unknown key", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "boom": 1}]}),
    ("missing id", {"schema_version": 1, "jobs": [{"source": "x"}]}),
    ("missing source", {"schema_version": 1, "jobs": [{"id": "a"}]}),
    ("id empty", {"schema_version": 1, "jobs": [{"id": "", "source": "x"}]}),
    ("id number", {"schema_version": 1, "jobs": [{"id": 1, "source": "x"}]}),
    ("id null", {"schema_version": 1, "jobs": [{"id": None, "source": "x"}]}),
    ("duplicate id", {"schema_version": 1, "jobs": [{"id": "a", "source": "x"}, {"id": "a", "source": "y"}]}),
    ("source number", {"schema_version": 1, "jobs": [{"id": "a", "source": 3}]}),
    ("source dash", {"schema_version": 1, "jobs": [{"id": "a", "source": "-"}]}),
    ("shots zero", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "shots": 0}]}),
    ("shots negative", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "shots": -2}]}),
    ("shots float", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "shots": 1.5}]}),
    ("shots bool", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "shots": True}]}),
    ("shots string", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "shots": "10"}]}),
    ("seed float", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "seed": 1.0}]}),
    ("seed bool", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "seed": False}]}),
    ("seed string", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "seed": "0"}]}),
    ("noise model number", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "noise_model": 1}]}),
    ("noise model dash", {"schema_version": 1, "jobs": [{"id": "a", "source": "x", "noise_model": "-"}]}),
    ("root array", [{"schema_version": 1}]),
    ("root null", None),
    ("root number", 42),
]


@pytest.mark.parametrize("label,manifest", INVALID_MANIFESTS)
def test_invalid_manifest_is_batch_input_error(tmp_path, label, manifest, capsys):
    path = write_manifest(tmp_path, manifest)
    rc, out, err = run_batch(path, capsys)
    assert rc == 2, label
    assert out == "", label
    assert err.count("\n") == 1, label
    payload = json.loads(err)
    assert payload["error"] == "batch_input_error", label
    assert isinstance(payload["message"], str) and payload["message"], label


def test_too_many_jobs_rejected(tmp_path, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [{"id": str(i), "source": "x"} for i in range(101)],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "batch_input_error"


def test_exactly_one_hundred_jobs_accepted(tmp_path, basic_files, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [{"id": f"j{i}", "source": "x.qasm", "shots": 1, "seed": i} for i in range(100)],
    }
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 0 and err == ""
    data = json.loads(out)
    assert data["job_count"] == 100 and data["succeeded"] == 100


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version": 1, "jobs": [',
        "{not json",
        '{"schema_version":1,"jobs":[{"id":"a","source":"x","shots":NaN}]}',
        '{"schema_version":1,"jobs":[{"id":"a","source":"x","shots":Infinity}]}',
        '{"schema_version": 1, "schema_version": 1, "jobs": []}',
        '{"schema_version":1,"jobs":[{"id":"a","id":"b","source":"x"}]}',
    ],
)
def test_malformed_or_duplicate_json_is_batch_input_error(tmp_path, text, capsys):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    rc, out, err = run_batch(str(path), capsys)
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "batch_input_error"


def test_missing_manifest_file_is_io_error(tmp_path, capsys):
    rc, out, err = run_batch(str(tmp_path / "nope.json"), capsys)
    assert rc == 1 and out == ""
    assert err.count("\n") == 1
    assert json.loads(err)["error"] == "io_error"


def test_invalid_utf8_manifest_is_io_error(tmp_path, capsys):
    path = tmp_path / "badutf8.json"
    path.write_bytes(b'{"schema_version":1,\xff\xfe')
    rc, out, err = run_batch(str(path), capsys)
    assert rc == 1 and out == ""
    assert json.loads(err)["error"] == "io_error"


def test_invalid_manifest_does_not_touch_task_files(tmp_path, make_files, capsys):
    # The referenced task file does not exist, but validation fails first.
    manifest = {"schema_version": 1, "jobs": [{"id": "a", "source": "whatever.qasm", "shots": 0}]}
    rc, out, err = run_batch(write_manifest(tmp_path, manifest), capsys)
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "batch_input_error"


# ------------------------------------------------------------- path resolution


def test_relative_paths_resolve_against_manifest_directory(tmp_path, make_files, monkeypatch, capsys):
    paths = make_files(
        {
            "sub/x.qasm": HEADER + X_GATE,
            "sub/noise.json": json.dumps({"bit_flip": 0.3}),
            "sub/manifest.json": "",
        }
    )
    manifest = {
        "schema_version": 1,
        "jobs": [
            {"id": "x", "source": "x.qasm", "shots": 6, "seed": 1},
            {"id": "n", "source": "x.qasm", "noise_model": "noise.json", "shots": 6, "seed": 1},
        ],
    }
    Path(paths["sub/manifest.json"]).write_text(json.dumps(manifest), encoding="utf-8")
    # Running from an unrelated cwd must still resolve against the manifest dir.
    monkeypatch.chdir(tmp_path)
    rc, out, err = run_batch(paths["sub/manifest.json"], capsys)
    assert rc == 0, err
    data = json.loads(out)
    assert data["failed"] == 0
    assert data["results"][0]["output"]["counts"] == {"1": 6}
    assert data["results"][1]["output"]["noise_model"] == {"bit_flip": 0.3}


def test_stdin_manifest_resolves_against_cwd(tmp_path, basic_files, monkeypatch, capsys):
    import io

    manifest = {
        "schema_version": 1,
        "jobs": [{"id": "x", "source": "x.qasm", "shots": 5, "seed": 1}],
    }
    raw = json.dumps(manifest).encode("utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))
    rc, out, err = run_batch("-", capsys)
    assert rc == 0 and err == ""
    assert json.loads(out)["results"][0]["output"]["counts"] == {"1": 5}


def test_absolute_task_paths_supported(tmp_path, basic_files, capsys):
    manifest = {
        "schema_version": 1,
        "jobs": [
            {"id": "a", "source": str(tmp_path / "bell.qasm"), "shots": 4, "seed": 1},
            {"id": "b", "source": str(tmp_path / "bell.qasm"), "noise_model": str(tmp_path / "noise.json"),
             "shots": 4, "seed": 1},
        ],
    }
    # Put the manifest in a different directory than the task files.
    other = tmp_path / "other"
    other.mkdir()
    manifest_path = other / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc, out, err = run_batch(str(manifest_path), capsys)
    assert rc == 0 and json.loads(out)["failed"] == 0


# --------------------------------------------------------- real-process checks


def _run_process(*args: str, stdin: str | None = None, cwd: str | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "quantum_circuit.cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=env,
        cwd=cwd,
    )


def test_process_success_single_line_json(tmp_path):
    (tmp_path / "x.qasm").write_text(HEADER + X_GATE, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "jobs": [{"id": "x", "source": "x.qasm", "shots": 8, "seed": 2}]}),
        encoding="utf-8",
    )
    result = _run_process("batch-simulate", str(manifest))
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1 and result.stderr == ""
    assert json.loads(result.stdout)["results"][0]["output"]["shots"] == 8


def test_process_invalid_manifest_exit_2(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema_version": 1, "jobs": []}', encoding="utf-8")
    result = _run_process("batch-simulate", str(manifest))
    assert result.returncode == 2 and result.stdout == ""
    assert json.loads(result.stderr)["error"] == "batch_input_error"


def test_process_job_failure_exit_3(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "jobs": [{"id": "x", "source": "missing.qasm"}]}),
        encoding="utf-8",
    )
    result = _run_process("batch-simulate", str(manifest))
    assert result.returncode == 3 and result.stderr == ""
    data = json.loads(result.stdout)
    assert data["failed"] == 1
    assert data["results"][0]["error"]["error"] == "io_error"


def test_process_stdin_manifest(tmp_path):
    manifest = json.dumps(
        {"schema_version": 1, "jobs": [{"id": "x", "source": "x.qasm", "shots": 3, "seed": 1}]}
    )
    (tmp_path / "x.qasm").write_text(HEADER + X_GATE, encoding="utf-8")
    result = _run_process("batch-simulate", "-", stdin=manifest, cwd=str(tmp_path))
    assert result.returncode == 0
    assert json.loads(result.stdout)["results"][0]["output"]["counts"] == {"1": 3}
