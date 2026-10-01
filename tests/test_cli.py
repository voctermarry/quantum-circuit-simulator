"""End-to-end tests for the command line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_circuit import __version__, cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def write_qasm(tmp_path):
    def _write(body: str, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def write_bytes(tmp_path):
    def _write(data: bytes, name: str = "circuit.qasm"):
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)

    return _write


# ------------------------------------------------------------- legacy behavior


def test_version_subcommand_prints_version(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_no_subcommand_prints_help(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "simulate" in out


def test_help_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_simulate_help_flag(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["simulate", "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--shots" in out and "--seed" in out


# --------------------------------------------------------------- success output


def test_success_json_shape_and_field_order(write_qasm):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    assert cli.main(["simulate", path, "--shots", "50", "--seed", "7"]) == 0


def test_success_payload(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    rc = cli.main(["simulate", path, "--shots", "50", "--seed", "7"])
    captured = capsys.readouterr()
    assert rc == 0
    raw = captured.out
    assert raw.endswith("\n")
    assert raw.count("\n") == 1

    data = json.loads(raw)
    assert data["schema_version"] == 1
    assert (data["shots"], data["seed"]) == (50, 7)
    assert (data["num_qubits"], data["num_clbits"]) == (2, 2)
    assert sum(data["counts"].values()) == 50

    # Field order is part of the contract.
    assert list(data) == ["schema_version", "shots", "seed", "num_qubits", "num_clbits", "counts"]
    counts = data["counts"]
    assert list(counts) == sorted(counts)


def test_defaults_are_1024_shots_and_seed_zero(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["simulate", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["shots"] == 1024 and data["seed"] == 0
    assert data["counts"] == {"0": 1024}


def test_repeated_runs_are_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[3];\ncreg c[3];\nh q[0];\nh q[1];\ncx q[1],q[2];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["simulate", path, "--shots", "128", "--seed", "123"]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_negative_seed_accepted(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n")
    assert cli.main(["simulate", path, "--seed", "-999999"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["seed"] == -999999


def test_stdin_source(monkeypatch, capsys):
    import io

    source = (HEADER + "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n").encode()
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(source)))
    rc = cli.main(["simulate", "-"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["counts"] == {"1": 1024}


# -------------------------------------------------------------------- io errors


def test_missing_file_is_io_error(capsys):
    rc = cli.main(["simulate", "/nonexistent/path/circuit.qasm"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert captured.err.count("\n") == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_unreadable_file_is_io_error(tmp_path, capsys):
    path = tmp_path / "secret.qasm"
    path.write_text(HEADER + "qreg q[1];\n")
    path.chmod(0o000)
    try:
        rc = cli.main(["simulate", str(path)])
    finally:
        path.chmod(0o644)
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["error"] == "io_error"


def test_invalid_utf8_is_io_error(write_bytes, capsys):
    path = write_bytes(HEADER.encode() + b"\xff\xfe bad")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


# ------------------------------------------------------------- parse / validation


def test_parse_error_exit_code_and_payload(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nx q[0]\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert isinstance(payload["line"], int) and payload["line"] >= 1
    assert isinstance(payload["column"], int) and payload["column"] >= 1


def test_validation_error_exit_code_and_payload(write_qasm, capsys):
    path = write_qasm("qreg q[1];\nqreg q2[1];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


# ------------------------------------------------------------- argument errors


def test_shots_must_be_positive(capsys):
    for bad in ["0", "-1", "abc", "1.5"]:
        with pytest.raises(SystemExit) as info:
            cli.main(["simulate", "-", "--shots", bad])
        assert info.value.code == 2
        assert capsys.readouterr().out == ""


def test_seed_must_be_integer(capsys):
    for bad in ["abc", "1.5", "0x1"]:
        with pytest.raises(SystemExit) as info:
            cli.main(["simulate", "-", "--seed", bad])
        assert info.value.code == 2
        assert capsys.readouterr().out == ""


def test_argument_error_does_not_open_source(capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["simulate", "/nonexistent/never-read.qasm", "--shots", "0"])
    assert info.value.code == 2
    captured = capsys.readouterr()
    assert "io_error" not in captured.err


# ----------------------------------------------------------- real-process checks


def _run_process(*args: str, stdin: str | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "quantum_circuit.cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=env,
    )


def test_process_bad_argument_exit_code():
    result = _run_process("simulate", "-", "--shots", "x")
    assert result.returncode == 2
    assert result.stdout == ""


def test_process_io_error_exit_code():
    result = _run_process("simulate", "/nonexistent/circuit.qasm")
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "io_error"


def test_process_stdin_success_is_single_json_line():
    source = HEADER + "qreg q[1];\ncreg c[1];\nh q[0];\nmeasure q[0] -> c[0];\n"
    result = _run_process("simulate", "-", "--shots", "8", "--seed", "2", stdin=source)
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["shots"] == 8
    assert sum(data["counts"].values()) == 8


# ------------------------------------------------------- parameterized gates


def test_parameterized_gates_end_to_end(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nrx(pi) q[0];\nry(pi/2) q[1];\nrz(-pi/4) q[0];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    rc = cli.main(["simulate", path, "--shots", "64", "--seed", "5"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert sum(data["counts"].values()) == 64
    # rx(pi) flips q[0] deterministically, so c[0] is always 1.
    assert all(key.endswith("1") for key in data["counts"])


def test_parameterized_gates_output_is_byte_identical(write_qasm, capsys):
    path = write_qasm(
        "qreg q[2];\ncreg c[2];\nrx(0.3+pi/4) q[0];\nry(1e-1*2) q[1];\n"
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
    )
    outputs = set()
    for _ in range(3):
        assert cli.main(["simulate", path, "--shots", "256", "--seed", "77"]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_parameterized_gate_parse_error(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nrx() q[0];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "parse_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_parameterized_gate_validation_error(write_qasm, capsys):
    path = write_qasm("qreg q[1];\ncreg c[1];\nrx(1/0) q[0];\n")
    rc = cli.main(["simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "validation_error"
    assert payload["line"] >= 1 and payload["column"] >= 1


def test_process_parameterized_gate_from_stdin():
    source = HEADER + "qreg q[1];\ncreg c[1];\nrx(pi) q[0];\nmeasure q[0] -> c[0];\n"
    result = _run_process("simulate", "-", "--shots", "16", "--seed", "3", stdin=source)
    assert result.returncode == 0
    assert result.stdout.count("\n") == 1
    assert json.loads(result.stdout)["counts"] == {"1": 16}


# -------------------------------------------------------------- batch-simulate


BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)
X_CIRCUIT = "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"


@pytest.fixture
def batch_env(tmp_path):
    def _qasm(body: str = X_CIRCUIT, name: str = "a.qasm") -> str:
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    def _model(payload: object, name: str = "model.json") -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _manifest(payload: object, name: str = "jobs.json") -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    return tmp_path, _qasm, _model, _manifest


def test_batch_success_mixed_noise_and_order(batch_env, capsys):
    tmp_path, qasm, model, manifest = batch_env
    a = qasm(X_CIRCUIT, "a.qasm")
    bell = qasm(BELL, "bell.qasm")
    noise = model({"bit_flip": 0.5})
    path = manifest(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "a", "source": "a.qasm", "shots": 16, "seed": 3},
                {"id": "bell", "source": "bell.qasm"},
                {"id": "noisy", "source": "a.qasm", "noise_model": "model.json", "shots": 8, "seed": 1},
            ],
        }
    )
    assert cli.main(["batch-simulate", path]) == 0
    raw = capsys.readouterr().out
    assert raw.count("\n") == 1
    data = json.loads(raw)
    assert list(data) == ["schema_version", "job_count", "succeeded", "failed", "results"]
    assert (data["schema_version"], data["job_count"]) == (1, 3)
    assert (data["succeeded"], data["failed"]) == (3, 0)

    results = data["results"]
    assert [r["id"] for r in results] == ["a", "bell", "noisy"]
    assert all(list(r) == ["id", "status", "output"] for r in results)
    assert all(r["status"] == "succeeded" for r in results)

    assert results[0]["output"] == {
        "schema_version": 1,
        "shots": 16,
        "seed": 3,
        "num_qubits": 1,
        "num_clbits": 1,
        "counts": {"1": 16},
    }
    assert results[1]["output"]["shots"] == 1024  # default
    assert results[1]["output"]["seed"] == 0
    noisy_output = results[2]["output"]
    assert noisy_output["schema_version"] == 2
    assert noisy_output["noise_model"] == {"bit_flip": 0.5}
    assert sum(noisy_output["counts"].values()) == 8


def test_batch_output_matches_standalone_simulate(batch_env, capsys):
    tmp_path, qasm, model, manifest = batch_env
    qasm(BELL, "bell.qasm")
    model({"depolarizing": 0.25}, "noise.json")
    jobs = [
        {"id": "clean", "source": "bell.qasm", "shots": 77, "seed": -12},
        {"id": "noisy", "source": "bell.qasm", "shots": 33, "seed": 9, "noise_model": "noise.json"},
    ]
    path = manifest({"schema_version": 1, "jobs": jobs})

    standalone = {}
    for job in jobs:
        assert cli.main(
            [
                "simulate",
                str(tmp_path / job["source"]),
                "--shots",
                str(job["shots"]),
                "--seed",
                str(job["seed"]),
                *(["--noise-model", str(tmp_path / job["noise_model"])] if "noise_model" in job else []),
            ]
        ) == 0
        standalone[job["id"]] = json.loads(capsys.readouterr().out)

    assert cli.main(["batch-simulate", path]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    for result in results:
        assert result["output"] == standalone[result["id"]]


def test_batch_job_failure_is_embedded_and_continues(batch_env, capsys):
    tmp_path, qasm, model, manifest = batch_env
    bad_parse = tmp_path / "bad.qasm"
    bad_parse.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    model({"bit_flip": 2}, "badmodel.json")
    qasm(X_CIRCUIT, "ok.qasm")
    path = manifest(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "parse", "source": "bad.qasm"},
                {"id": "missing", "source": "nope.qasm"},
                {"id": "badnoise", "source": "ok.qasm", "noise_model": "badmodel.json"},
                {"id": "ok", "source": "ok.qasm", "shots": 4, "seed": 1},
            ],
        }
    )
    rc = cli.main(["batch-simulate", path])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.err == ""
    data = json.loads(captured.out)
    assert (data["succeeded"], data["failed"]) == (1, 3)
    results = data["results"]
    assert all(list(r) == ["id", "status", "error"] for r in results[:3])
    assert results[0]["error"]["error"] == "parse_error"
    assert results[0]["error"]["line"] >= 1 and results[0]["error"]["column"] >= 1
    assert results[1]["error"]["error"] == "io_error"
    assert results[2]["error"]["error"] == "noise_model_error"
    assert results[3]["status"] == "succeeded"
    assert results[3]["output"]["counts"] == {"1": 4}


def test_batch_relative_paths_resolve_from_manifest_directory(batch_env, capsys, monkeypatch):
    tmp_path, qasm, model, _manifest = batch_env
    sub = tmp_path / "nested"
    sub.mkdir()
    qasm(X_CIRCUIT, "nested/c.qasm")
    model({"phase_damping": 0.1}, "nested/m.json")
    manifest_path = sub / "jobs.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "jobs": [{"id": "j", "source": "c.qasm", "shots": 3, "noise_model": "m.json"}],
            }
        ),
        encoding="utf-8",
    )
    # Run from another cwd; the manifest directory must anchor the paths.
    monkeypatch.chdir(tmp_path)
    assert cli.main(["batch-simulate", str(manifest_path)]) == 0
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["status"] == "succeeded"
    assert result["output"]["schema_version"] == 2


def test_batch_stdin_manifest_uses_cwd(batch_env, capsys, monkeypatch):
    import io

    tmp_path, qasm, model, _manifest = batch_env
    qasm(X_CIRCUIT, "cwd.qasm")
    monkeypatch.chdir(tmp_path)
    payload = json.dumps({"schema_version": 1, "jobs": [{"id": "j", "source": "cwd.qasm", "shots": 2}]})
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(payload.encode())))
    assert cli.main(["batch-simulate", "-"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["results"][0]["status"] == "succeeded"


def test_batch_byte_identical_and_independent_of_other_jobs(batch_env, capsys):
    tmp_path, qasm, model, manifest = batch_env
    qasm(BELL, "bell.qasm")
    manifest_one = manifest(
        {"schema_version": 1, "jobs": [{"id": "B", "source": "bell.qasm", "shots": 30, "seed": 7}]},
        "one.json",
    )
    manifest_many = manifest(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "X", "source": "bell.qasm", "shots": 999, "seed": 12345},
                {"id": "B", "source": "bell.qasm", "shots": 30, "seed": 7},
                {"id": "Y", "source": "bell.qasm"},
            ],
        },
        "many.json",
    )

    outputs = set()
    for _ in range(2):
        assert cli.main(["batch-simulate", manifest_one]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1

    assert cli.main(["batch-simulate", manifest_one]) == 0
    one_output = json.loads(capsys.readouterr().out)["results"][0]["output"]
    assert cli.main(["batch-simulate", manifest_many]) == 0
    many = json.loads(capsys.readouterr().out)["results"]
    b_output = next(result["output"] for result in many if result["id"] == "B")
    assert b_output == one_output


def test_batch_noisy_too_many_qubits_is_simulation_error(batch_env, capsys):
    tmp_path, qasm, model, manifest = batch_env
    body = (
        "qreg q[11];\ncreg c[11];\n"
        + "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(11))
    )
    qasm(body, "big.qasm")
    model({"bit_flip": 0.1}, "m.json")
    path = manifest(
        {"schema_version": 1, "jobs": [{"id": "big", "source": "big.qasm", "noise_model": "m.json"}]}
    )
    assert cli.main(["batch-simulate", path]) == 3
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["status"] == "failed"
    assert result["error"]["error"] == "simulation_error"


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1},  # missing jobs
        {"jobs": []},  # missing schema_version
        {"schema_version": 0, "jobs": []},
        {"schema_version": "1", "jobs": []},
        {"schema_version": 1, "jobs": {}},
        {"schema_version": 1, "jobs": []},  # too few
        {"schema_version": 1, "jobs": [], "extra": 1},  # unknown root key
        [1, 2],  # root not object
    ],
)
def test_batch_invalid_manifest_structure(batch_env, capsys, payload):
    tmp_path, _qasm, _model, manifest = batch_env
    path = manifest(payload)
    rc = cli.main(["batch-simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "batch_input_error"
    assert captured.err.count("\n") == 1


def test_batch_too_many_jobs(batch_env, capsys):
    tmp_path, _qasm, _model, manifest = batch_env
    jobs = [{"id": str(i), "source": "a.qasm"} for i in range(101)]
    rc = cli.main(["batch-simulate", manifest({"schema_version": 1, "jobs": jobs})])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "batch_input_error"


@pytest.mark.parametrize(
    "job",
    [
        {"source": "a.qasm"},  # missing id
        {"id": "", "source": "a.qasm"},  # empty id
        {"id": 1, "source": "a.qasm"},  # id not string
        {"id": "a"},  # missing source
        {"id": "a", "source": 3},  # source not string
        {"id": "a", "source": "-"},  # stdin reserved
        {"id": "a", "source": "a.qasm", "shots": 0},
        {"id": "a", "source": "a.qasm", "shots": -2},
        {"id": "a", "source": "a.qasm", "shots": 1.5},
        {"id": "a", "source": "a.qasm", "shots": True},
        {"id": "a", "source": "a.qasm", "seed": "x"},
        {"id": "a", "source": "a.qasm", "seed": 1.0},
        {"id": "a", "source": "a.qasm", "noise_model": 4},
        {"id": "a", "source": "a.qasm", "noise_model": "-"},
        {"id": "a", "source": "a.qasm", "bogus": 1},
    ],
)
def test_batch_invalid_job_fields(batch_env, capsys, job):
    tmp_path, _qasm, _model, manifest = batch_env
    path = manifest({"schema_version": 1, "jobs": [job]})
    rc = cli.main(["batch-simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "batch_input_error"


def test_batch_duplicate_id_rejected(batch_env, capsys):
    tmp_path, _qasm, _model, manifest = batch_env
    path = manifest(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "same", "source": "a.qasm"},
                {"id": "other", "source": "a.qasm"},
                {"id": "same", "source": "a.qasm"},
            ],
        }
    )
    rc = cli.main(["batch-simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert "same" in json.loads(captured.err)["message"]


def test_batch_duplicate_json_key_rejected(batch_env, capsys):
    tmp_path, _qasm, _model, _manifest = batch_env
    path = tmp_path / "dup.json"
    path.write_text(
        '{"schema_version": 1, "schema_version": 1, "jobs": '
        '[{"id": "a", "source": "a.qasm"}]}',
        encoding="utf-8",
    )
    rc = cli.main(["batch-simulate", str(path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "batch_input_error"


def test_batch_duplicate_job_json_key_rejected(batch_env, capsys):
    tmp_path, _qasm, _model, _manifest = batch_env
    path = tmp_path / "dupjob.json"
    path.write_text(
        '{"schema_version": 1, "jobs": [{"id": "a", "id": "b", "source": "a.qasm"}]}',
        encoding="utf-8",
    )
    rc = cli.main(["batch-simulate", str(path)])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "batch_input_error"


def test_batch_syntax_error_is_batch_input_error(batch_env, capsys):
    tmp_path, _qasm, _model, _manifest = batch_env
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 1, "jobs": [', encoding="utf-8")
    rc = cli.main(["batch-simulate", str(path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "batch_input_error"


def test_batch_json_constants_rejected(batch_env, capsys):
    tmp_path, _qasm, _model, _manifest = batch_env
    path = tmp_path / "const.json"
    path.write_text('{"schema_version": 1, "jobs": [{"id": "a", "shots": NaN}]}', encoding="utf-8")
    # The source is also missing; the structural check happens before reads.
    rc = cli.main(["batch-simulate", str(path)])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "batch_input_error"


def test_batch_validates_before_reading_any_task_file(batch_env, capsys):
    tmp_path, _qasm, _model, manifest = batch_env
    # Neither referenced file exists; an invalid manifest must still report
    # batch_input_error rather than an embedded io_error.
    path = manifest(
        {"schema_version": 1, "jobs": [{"id": "a", "source": "missing1.qasm"}, {"id": "a", "source": "x"}]}
    )
    rc = cli.main(["batch-simulate", path])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "batch_input_error"


def test_batch_missing_manifest_is_io_error(capsys):
    rc = cli.main(["batch-simulate", "/nonexistent/manifest.json"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_batch_invalid_utf8_manifest_is_io_error(tmp_path, capsys):
    path = tmp_path / "utf8.json"
    path.write_bytes(b'{"schema_version": 1, "jobs": []}\xff')
    rc = cli.main(["batch-simulate", str(path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "io_error"


def test_process_batch_end_to_end(tmp_path):
    circuit = tmp_path / "c.qasm"
    circuit.write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "jobs": [{"id": "j", "source": "c.qasm", "shots": 6}]}),
        encoding="utf-8",
    )
    result = _run_process("batch-simulate", str(manifest))
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["results"][0]["output"]["counts"] == {"1": 6}


def test_process_batch_input_error_exit_code(tmp_path):
    manifest = tmp_path / "jobs.json"
    manifest.write_text('{"schema_version": 1, "jobs": []}', encoding="utf-8")
    result = _run_process("batch-simulate", str(manifest))
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "batch_input_error"


# ---------------------------------------------------------------- reconcile


@pytest.fixture
def reconcile_env(batch_env, capsys):
    tmp_path, qasm, model, manifest = batch_env

    def _baseline(payload: object, name: str = "baseline.json") -> str:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _baseline_text(text: str, name: str = "baseline.json") -> str:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _run_batch(path: str) -> dict:
        assert cli.main(["batch-simulate", path]) in (0, 3)
        return json.loads(capsys.readouterr().out)

    return tmp_path, qasm, model, manifest, _baseline, _baseline_text, _run_batch


def _jobs_payload(jobs):
    return {"schema_version": 1, "jobs": jobs}


def test_reconcile_all_identical_mixed_noise(reconcile_env, capsys):
    tmp_path, qasm, model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    qasm(BELL, "bell.qasm")
    model({"bit_flip": 0.5}, "model.json")
    path = manifest(
        _jobs_payload(
            [
                {"id": "a", "source": "a.qasm", "shots": 16, "seed": 3},
                {"id": "bell", "source": "bell.qasm"},
                {"id": "noisy", "source": "a.qasm", "noise_model": "model.json", "shots": 8, "seed": 1},
            ]
        )
    )
    base_path = baseline(run_batch(path))

    rc = cli.main(["reconcile", path, base_path])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    data = json.loads(captured.out)
    assert list(data) == ["schema_version", "job_count", "matched", "mismatched", "consistent", "results"]
    assert (data["schema_version"], data["job_count"]) == (1, 3)
    assert (data["matched"], data["mismatched"], data["consistent"]) == (3, 0, True)
    assert [r["id"] for r in data["results"]] == ["a", "bell", "noisy"]
    assert all(list(r) == ["id", "consistent", "reason"] for r in data["results"])
    assert all(r["consistent"] is True and r["reason"] == "identical" for r in data["results"])


def test_reconcile_identical_failures_exit_zero(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    bad = tmp_path / "bad.qasm"
    bad.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    qasm(X_CIRCUIT, "ok.qasm")
    path = manifest(
        _jobs_payload(
            [
                {"id": "parse", "source": "bad.qasm"},
                {"id": "ok", "source": "ok.qasm", "shots": 4, "seed": 1},
            ]
        )
    )
    base_path = baseline(run_batch(path))

    rc = cli.main(["reconcile", path, baseline(json.load(open(base_path)))])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    data = json.loads(captured.out)
    assert (data["matched"], data["mismatched"], data["consistent"]) == (2, 0, True)


def test_reconcile_output_mismatch_report_and_exit_code(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 10, "seed": 1}]))
    payload = run_batch(path)
    expected_output = json.loads(json.dumps(payload["results"][0]["output"]))
    payload["results"][0]["output"]["counts"] = {"1": 9, "0": 1}

    rc = cli.main(["reconcile", path, baseline(payload)])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.err == ""
    data = json.loads(captured.out)
    assert (data["matched"], data["mismatched"], data["consistent"]) == (0, 1, False)
    entry = data["results"][0]
    assert list(entry) == ["id", "consistent", "reason", "expected", "actual"]
    assert entry["id"] == "a"
    assert entry["consistent"] is False
    assert entry["reason"] == "output_mismatch"
    # The complete baseline and current task results are preserved.
    assert list(entry["expected"]) == ["id", "status", "output"]
    assert entry["expected"]["id"] == "a"
    assert entry["expected"]["output"]["counts"] == {"1": 9, "0": 1}
    assert list(entry["actual"]) == ["id", "status", "output"]
    assert entry["actual"]["output"] == expected_output


def test_reconcile_error_mismatch(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    bad = tmp_path / "bad.qasm"
    bad.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    qasm(X_CIRCUIT, "ok.qasm")
    path = manifest(_jobs_payload([{"id": "bad", "source": "bad.qasm"}, {"id": "ok", "source": "ok.qasm"}]))
    payload = run_batch(path)
    payload["results"][0]["error"]["message"] = "something changed"

    rc = cli.main(["reconcile", path, baseline(payload)])
    captured = capsys.readouterr()
    assert rc == 3
    data = json.loads(captured.out)
    assert (data["matched"], data["mismatched"]) == (1, 1)
    bad_entry, ok_entry = data["results"]
    assert bad_entry["reason"] == "error_mismatch"
    assert bad_entry["expected"]["status"] == "failed"
    assert bad_entry["expected"]["error"]["message"] == "something changed"
    assert bad_entry["actual"]["status"] == "failed"
    assert bad_entry["actual"]["error"]["error"] == "parse_error"
    assert ok_entry["reason"] == "identical"


def test_reconcile_status_mismatch_both_directions(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    bad = tmp_path / "bad.qasm"
    bad.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    qasm(X_CIRCUIT, "ok.qasm")
    path = manifest(_jobs_payload([{"id": "bad", "source": "bad.qasm"}]))
    failed_payload = run_batch(path)

    # Baseline claims success, the re-run actually fails.
    succeeded_payload = {
        "schema_version": 1,
        "job_count": 1,
        "succeeded": 1,
        "failed": 0,
        "results": [
            {
                "id": "bad",
                "status": "succeeded",
                "output": {
                    "schema_version": 1,
                    "shots": 1024,
                    "seed": 0,
                    "num_qubits": 1,
                    "num_clbits": 1,
                    "counts": {"0": 1024},
                },
            }
        ],
    }
    rc = cli.main(["reconcile", path, baseline(succeeded_payload, "succeeded.json")])
    data = json.loads(capsys.readouterr().out)
    assert rc == 3
    entry = data["results"][0]
    assert entry["reason"] == "status_mismatch"
    assert entry["expected"]["status"] == "succeeded"
    assert entry["actual"]["status"] == "failed"
    assert entry["actual"]["error"]["error"] == "parse_error"

    # Baseline claims failure; fix the circuit so the re-run succeeds.
    bad.write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    rc = cli.main(["reconcile", path, baseline(failed_payload, "failed.json")])
    data = json.loads(capsys.readouterr().out)
    assert rc == 3
    entry = data["results"][0]
    assert entry["reason"] == "status_mismatch"
    assert entry["expected"]["status"] == "failed"
    assert entry["actual"]["status"] == "succeeded"
    assert entry["actual"]["output"]["counts"] == {"1": 1024}


def test_reconcile_task_file_disappearing_is_status_mismatch(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 3, "seed": 1}]))
    base_path = baseline(run_batch(path))
    (tmp_path / "a.qasm").unlink()

    rc = cli.main(["reconcile", path, base_path])
    captured = capsys.readouterr()
    assert rc == 3
    assert captured.err == ""
    entry = json.loads(captured.out)["results"][0]
    assert entry["reason"] == "status_mismatch"
    assert entry["actual"]["error"]["error"] == "io_error"


def test_reconcile_ignores_key_order_and_whitespace(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, _baseline, baseline_text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 6, "seed": 2}]))
    payload = run_batch(path)
    # Re-serialise with indentation and sorted (different) key order.
    base_path = baseline_text(json.dumps(payload, indent=4, sort_keys=True))

    rc = cli.main(["reconcile", path, base_path])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["consistent"] is True


def test_reconcile_array_order_and_number_type_are_significant():
    assert cli._json_equal({"a": [1, 2]}, {"a": [1, 2]}) is True
    assert cli._json_equal({"a": [1, 2]}, {"a": [2, 1]}) is False
    assert cli._json_equal({"a": 1}, {"a": 1}) is True
    assert cli._json_equal({"a": 1}, {"a": 1.0}) is False
    assert cli._json_equal({"a": 1}, {"a": True}) is False
    assert cli._json_equal({"x": 1, "y": 2}, {"y": 2, "x": 1}) is True


def test_reconcile_byte_identical_repeated_runs(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(BELL, "bell.qasm")
    path = manifest(_jobs_payload([{"id": "b", "source": "bell.qasm", "shots": 30, "seed": 7}]))
    base_path = baseline(run_batch(path))

    outputs = set()
    for _ in range(3):
        assert cli.main(["reconcile", path, base_path]) == 0
        outputs.add(capsys.readouterr().out)
    assert len(outputs) == 1


def test_reconcile_relative_paths_anchor_at_manifest_directory(reconcile_env, capsys, monkeypatch):
    tmp_path, qasm, model, _manifest, baseline, _text, run_batch = reconcile_env
    sub = tmp_path / "nested"
    sub.mkdir()
    qasm(X_CIRCUIT, "nested/c.qasm")
    model({"depolarizing": 0.2}, "nested/m.json")
    manifest_path = sub / "jobs.json"
    manifest_path.write_text(
        json.dumps(_jobs_payload([{"id": "j", "source": "c.qasm", "shots": 5, "noise_model": "m.json"}])),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    batch_output = run_batch(str(manifest_path))
    base_path = sub / "baseline.json"
    base_path.write_text(json.dumps(batch_output), encoding="utf-8")

    assert cli.main(["reconcile", str(manifest_path), str(base_path)]) == 0
    assert capsys.readouterr().err == ""


def test_reconcile_stdin_inputs(reconcile_env, capsys, monkeypatch):
    import io

    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "cwd.qasm")
    monkeypatch.chdir(tmp_path)
    jobs_text = json.dumps(_jobs_payload([{"id": "j", "source": "cwd.qasm", "shots": 4, "seed": 1}]))
    manifest_path = manifest(json.loads(jobs_text))
    batch_output = run_batch(manifest_path)

    # Baseline from stdin.
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(batch_output).encode())))
    assert cli.main(["reconcile", manifest_path, "-"]) == 0
    assert capsys.readouterr().err == ""

    # Manifest from stdin.
    base_path = baseline(batch_output)
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(jobs_text.encode())))
    assert cli.main(["reconcile", "-", base_path]) == 0
    assert capsys.readouterr().err == ""


def test_reconcile_both_stdin_rejected_without_reading(monkeypatch, capsys):
    class _ExplodingStdin:
        class buffer:
            @staticmethod
            def read():
                raise AssertionError("standard input must not be read")

    monkeypatch.setattr(cli.sys, "stdin", _ExplodingStdin())
    rc = cli.main(["reconcile", "-", "-"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "reconciliation_error"
    assert "input" not in payload
    assert captured.err.count("\n") == 1


@pytest.mark.parametrize("side", ["manifest", "baseline"])
def test_reconcile_missing_file_is_io_error_with_input(reconcile_env, capsys, side):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 2}]))
    base_path = baseline(run_batch(path))
    args = ["reconcile", path, base_path]
    args[1 if side == "manifest" else 2] = "/nonexistent/missing.json"

    rc = cli.main(args)
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == side


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
@pytest.mark.parametrize("side", ["manifest", "baseline"])
def test_reconcile_unreadable_file_is_io_error_with_input(reconcile_env, capsys, side):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 2}]))
    base_path = baseline(run_batch(path))
    target = tmp_path / ("secret_manifest.json" if side == "manifest" else "secret_baseline.json")
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o000)
    try:
        args = ["reconcile", path, base_path]
        args[1 if side == "manifest" else 2] = str(target)
        rc = cli.main(args)
    finally:
        target.chmod(0o644)
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.err)["input"] == side


@pytest.mark.parametrize("side", ["manifest", "baseline"])
def test_reconcile_invalid_utf8_is_io_error_with_input(reconcile_env, capsys, side):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 2}]))
    base_path = baseline(run_batch(path))
    bad = tmp_path / ("utf8_manifest.json" if side == "manifest" else "utf8_baseline.json")
    bad.write_bytes(b'{"schema_version": 1}\xff')
    args = ["reconcile", path, base_path]
    args[1 if side == "manifest" else 2] = str(bad)

    rc = cli.main(args)
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"] == "io_error"
    assert payload["input"] == side


def test_reconcile_manifest_io_error_reported_before_baseline(reconcile_env, capsys):
    _tmp, _q, _m, _manifest, _b, _t, _run = reconcile_env
    rc = cli.main(["reconcile", "/no/such/manifest.json", "/no/such/baseline.json"])
    assert rc == 1
    assert json.loads(capsys.readouterr().err)["input"] == "manifest"


def test_reconcile_invalid_manifest_is_reconcile_input_error(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    good_jobs = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 2}]), "good.json")
    base_path = baseline(run_batch(good_jobs))
    bad_manifest = manifest({"schema_version": 1, "jobs": []}, "empty.json")

    rc = cli.main(["reconcile", bad_manifest, base_path])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "reconcile_input_error"


def test_reconcile_manifest_duplicate_id_rejected(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    good_jobs = manifest(_jobs_payload([{"id": "a", "source": "a.qasm"}]), "good.json")
    base_path = baseline(run_batch(good_jobs))
    bad_manifest = manifest(
        _jobs_payload(
            [
                {"id": "a", "source": "a.qasm"},
                {"id": "a", "source": "a.qasm"},
            ]
        ),
        "dup.json",
    )
    rc = cli.main(["reconcile", bad_manifest, base_path])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"


def _valid_baseline():
    return {
        "schema_version": 1,
        "job_count": 1,
        "succeeded": 1,
        "failed": 0,
        "results": [
            {
                "id": "a",
                "status": "succeeded",
                "output": {
                    "schema_version": 1,
                    "shots": 4,
                    "seed": 0,
                    "num_qubits": 1,
                    "num_clbits": 1,
                    "counts": {"1": 4},
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("schema_version"),
        lambda d: d.update(schema_version=2),
        lambda d: d.update(schema_version="1"),
        lambda d: d.update(extra=1),
        lambda d: d.update(job_count=2),
        lambda d: d.update(succeeded=0, failed=1),
        lambda d: d.update(results="x"),
        lambda d: d.update(results=[]),
        lambda d: d["results"].append(dict(d["results"][0])),
        lambda d: d["results"][0].update(status="bogus"),
        lambda d: d["results"][0].update(id="b"),
        lambda d: d["results"][0].update(id=""),
        lambda d: d["results"][0].update(id=1),
        lambda d: d["results"][0].update(error={"error": "io_error", "message": "x"}),
        lambda d: d["results"][0].pop("output"),
        lambda d: d["results"][0]["output"].pop("shots"),
        lambda d: d["results"][0]["output"].update(extra=1),
        lambda d: d["results"][0]["output"].update(shots=0),
        lambda d: d["results"][0]["output"].update(shots=4.0),
        lambda d: d["results"][0]["output"].update(num_qubits=0),
        lambda d: d["results"][0]["output"].update(num_qubits=21),
        lambda d: d["results"][0]["output"]["counts"].update({"02": 4}),
        lambda d: d["results"][0]["output"]["counts"].update({"1": 3, "0": 2}),
        lambda d: d["results"][0]["output"]["counts"].clear(),
    ],
)
def test_reconcile_invalid_baseline_structure(reconcile_env, capsys, mutate):
    tmp_path, qasm, _model, manifest, baseline, _text, _run = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 4}]))
    payload = _valid_baseline()
    mutate(payload)

    rc = cli.main(["reconcile", path, baseline(payload)])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "reconcile_input_error"
    assert captured.err.count("\n") == 1


def test_reconcile_failed_baseline_error_validation(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, _run = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm"}]))

    def _payload():
        return {
            "schema_version": 1,
            "job_count": 1,
            "succeeded": 0,
            "failed": 1,
            "results": [{"id": "a", "status": "failed", "error": error}],
        }

    for error in (
        "io_error",
        {"error": "unknown_error", "message": "x"},
        {"error": "io_error"},
        {"error": "io_error", "message": 1},
        {"error": "io_error", "message": "x", "line": 1},
        {"error": "parse_error", "message": "x"},
        {"error": "parse_error", "message": "x", "line": 1, "column": 0},
        {"error": "parse_error", "message": "x", "line": 1, "column": 1, "input": "left"},
    ):
        payload = _payload()
        if isinstance(error, str):
            payload["results"][0]["error"] = error
        rc = cli.main(["reconcile", path, baseline(payload)])
        assert rc == 2
        assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"


def test_reconcile_noisy_output_validation(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, _run = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 4}]))
    good_noise_output = {
        "schema_version": 2,
        "shots": 4,
        "seed": 0,
        "num_qubits": 1,
        "num_clbits": 1,
        "noise_model": {"bit_flip": 0.5},
        "counts": {"1": 4},
    }

    bad = _valid_baseline()
    bad["results"][0]["output"] = dict(good_noise_output, noise_model={"nope": 0.5})
    assert cli.main(["reconcile", path, baseline(bad, "b1.json")]) == 2
    capsys.readouterr()

    bad = _valid_baseline()
    bad["results"][0]["output"] = dict(good_noise_output, noise_model={"bit_flip": 2})
    assert cli.main(["reconcile", path, baseline(bad, "b2.json")]) == 2
    capsys.readouterr()

    # Schema v1 output must not carry a noise model.
    bad = _valid_baseline()
    bad["results"][0]["output"]["noise_model"] = {"bit_flip": 0.5}
    assert cli.main(["reconcile", path, baseline(bad, "b3.json")]) == 2
    capsys.readouterr()

    # Noisy results above 10 qubits cannot be produced by batch-simulate.
    big = _valid_baseline()
    big["results"][0]["output"] = dict(
        good_noise_output, num_qubits=11, num_clbits=11, counts={"00000000000": 4}
    )
    assert cli.main(["reconcile", path, baseline(big, "b4.json")]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"


def test_reconcile_job_count_mismatch(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, _run = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    qasm(X_CIRCUIT, "b.qasm")
    path = manifest(
        _jobs_payload(
            [
                {"id": "a", "source": "a.qasm", "shots": 4},
                {"id": "b", "source": "b.qasm", "shots": 4},
            ]
        )
    )
    payload = _valid_baseline()  # claims a single job "a"
    rc = cli.main(["reconcile", path, baseline(payload)])
    assert rc == 2
    assert "job_count" in json.loads(capsys.readouterr().err)["message"]


def test_reconcile_duplicate_baseline_id(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, _run = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(
        _jobs_payload(
            [
                {"id": "a", "source": "a.qasm", "shots": 4},
                {"id": "b", "source": "a.qasm", "shots": 4},
            ]
        )
    )
    payload = _valid_baseline()
    payload.update(job_count=2, succeeded=2, failed=0)
    payload["results"].append(json.loads(json.dumps(payload["results"][0])))
    payload["results"][1]["id"] = "a"

    rc = cli.main(["reconcile", path, baseline(payload)])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"


def test_reconcile_baseline_json_syntax_and_constants(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, _baseline, baseline_text, run_batch = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 4}]))

    rc = cli.main(["reconcile", path, baseline_text('{"schema_version": 1, "results": [')])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"

    raw = json.dumps(_valid_baseline()).replace('"shots": 4', '"shots": NaN')
    rc = cli.main(["reconcile", path, baseline_text(raw, "nan.json")])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"

    raw = json.dumps(_valid_baseline()).replace('"shots": 4', '"shots": 1e999')
    rc = cli.main(["reconcile", path, baseline_text(raw, "inf.json")])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"


def test_reconcile_duplicate_baseline_json_key(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, _baseline, baseline_text, _run = reconcile_env
    qasm(X_CIRCUIT, "a.qasm")
    path = manifest(_jobs_payload([{"id": "a", "source": "a.qasm", "shots": 4}]))
    raw = json.dumps(_valid_baseline()).replace(
        '"job_count": 1', '"job_count": 1, "job_count": 1', 1
    )
    rc = cli.main(["reconcile", path, baseline_text(raw)])
    assert rc == 2
    assert json.loads(capsys.readouterr().err)["error"] == "reconcile_input_error"


def test_reconcile_validates_before_reading_task_files(reconcile_env, capsys):
    tmp_path, qasm, _model, manifest, baseline, _text, _run = reconcile_env
    # The referenced task files do not exist; an invalid baseline must still
    # surface reconcile_input_error, never an embedded task failure.
    path = manifest(_jobs_payload([{"id": "a", "source": "missing.qasm"}]))
    rc = cli.main(["reconcile", path, baseline({"schema_version": 1})])
    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.err)["error"] == "reconcile_input_error"


def test_process_reconcile_end_to_end(tmp_path):
    (tmp_path / "c.qasm").write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(_jobs_payload([{"id": "j", "source": "c.qasm", "shots": 6}])),
        encoding="utf-8",
    )
    batch = _run_process("batch-simulate", str(manifest))
    assert batch.returncode == 0
    baseline = tmp_path / "baseline.json"
    baseline.write_text(batch.stdout, encoding="utf-8")

    result = _run_process("reconcile", str(manifest), str(baseline))
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    data = json.loads(result.stdout)
    assert data["consistent"] is True and data["matched"] == 1


def test_process_reconcile_mismatch_exit_code(tmp_path):
    (tmp_path / "c.qasm").write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(_jobs_payload([{"id": "j", "source": "c.qasm", "shots": 6}])),
        encoding="utf-8",
    )
    batch = _run_process("batch-simulate", str(manifest))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(batch.stdout.replace('"1": 6', '"1": 5, "0": 1'), encoding="utf-8")

    result = _run_process("reconcile", str(manifest), str(baseline))
    assert result.returncode == 3
    assert result.stderr == ""
    assert json.loads(result.stdout)["results"][0]["reason"] == "output_mismatch"


def test_process_reconcile_both_stdin_exit_code():
    result = _run_process("reconcile", "-", "-")
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "reconciliation_error"


def test_process_reconcile_missing_baseline_exit_code(tmp_path):
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(_jobs_payload([{"id": "a", "source": "x.qasm"}])), encoding="utf-8")
    result = _run_process("reconcile", str(manifest), "/no/such/baseline.json")
    assert result.returncode == 1
    payload = json.loads(result.stderr)
    assert payload["error"] == "io_error" and payload["input"] == "baseline"
