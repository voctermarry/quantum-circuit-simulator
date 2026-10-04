"""Tests for the optional ``--output PATH`` result export on every subcommand.

The export is byte-identical to what the command would otherwise print to
stdout (including the trailing newline), stdout stays empty, and original
exit codes are preserved. Failures of the command itself never touch the
target; export failures are a single ``output_error`` line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
REPO_ROOT = Path(__file__).resolve().parent.parent

X_CIRCUIT = "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\n"
BELL = (
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


# --------------------------------------------------------------------- helpers


@pytest.fixture
def env(tmp_path):
    def _qasm(body: str = X_CIRCUIT, name: str = "a.qasm") -> str:
        path = tmp_path / name
        path.write_text(HEADER + body, encoding="utf-8")
        return str(path)

    def _file(text: str | bytes, name: str) -> str:
        path = tmp_path / name
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8")
        return str(path)

    def _json(payload: object, name: str) -> str:
        return _file(json.dumps(payload), name)

    return tmp_path, _qasm, _file, _json


def _run_main(argv):
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _run_process(*args: str, stdin: str | None = None):
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "quantum_circuit.cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        env=environ,
    )


# ---------------------------------------------------------- byte identity basics


@pytest.mark.parametrize(
    "argv_factory",
    [
        lambda a: ["simulate", a, "--shots", "50", "--seed", "7"],
        lambda a: ["probabilities", a],
        lambda a: ["optimize", a],
        lambda a: ["estimate", a],
        lambda a: ["equivalent", a, a],
        lambda a: ["state-metrics", a, a],
    ],
)
def test_exported_bytes_equal_stdout_for_single_circuit_commands(tmp_path, env, argv_factory):
    _tmp, qasm, _file, _json = env
    source = qasm(BELL, "c.qasm")
    argv = argv_factory(source)

    rc_stdout, stdout_text, stderr_text = _run_main(argv)
    assert rc_stdout == 0
    assert stderr_text == ""

    target = str(tmp_path / "out.json")
    rc, out, err = _run_main([*argv, "--output", target])
    assert rc == 0
    assert out == ""
    assert err == ""
    assert Path(target).read_bytes().decode("utf-8") == stdout_text
    assert stdout_text.endswith("\n")
    assert stdout_text.count("\n") == 1


def test_simulate_export_is_byte_identical(env):
    _tmp, qasm, _file, _json = env
    source = qasm(BELL, "c.qasm")
    argv = ["simulate", source, "--shots", "50", "--seed", "7"]
    _rc, stdout_text, _err = _run_main(argv)
    target = str(Path(source).parent / "r.json")
    rc, out, err = _run_main([*argv, "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert Path(target).read_text(encoding="utf-8") == stdout_text


def test_same_input_exported_to_two_paths_is_identical(env):
    _tmp, qasm, _file, _json = env
    source = qasm(BELL, "c.qasm")
    argv = ["simulate", source, "--shots", "33", "--seed", "9"]
    p1 = str(Path(source).parent / "one.json")
    p2 = str(Path(source).parent / "two.json")
    assert _run_main([*argv, "--output", p1])[0] == 0
    assert _run_main([*argv, "--output", p2])[0] == 0
    assert Path(p1).read_bytes() == Path(p2).read_bytes()


def test_export_overwrites_existing_regular_file(env):
    _tmp, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    target = _file("STALE-CONTENT-THAT-MUST-GO-AWAY", "out.json")
    rc, out, err = _run_main(["simulate", source, "--shots", "4", "--seed", "1", "--output", target])
    assert (rc, out, err) == (0, "", "")
    data = json.loads(Path(target).read_text())
    assert data["counts"] == {"1": 4}


def test_relative_output_resolves_from_cwd(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    monkeypatch.chdir(tmp_path)
    rc, out, err = _run_main(
        ["simulate", str(Path(source).name), "--shots", "3", "--seed", "1", "--output", "rel.json"]
    )
    assert (rc, out, err) == (0, "", "")
    assert (tmp_path / "rel.json").is_file()


def test_export_from_stdin_input(env, monkeypatch):
    import io

    tmp_path, _qasm, _file, _json = env
    target = str(tmp_path / "stdin_out.json")
    monkeypatch.setattr(
        cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + X_CIRCUIT).encode()))
    )
    rc, out, err = _run_main(["simulate", "-", "--shots", "4", "--seed", "1", "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert json.loads(Path(target).read_text())["counts"] == {"1": 4}


# --------------------------------------------------------------- batch / reconcile


@pytest.fixture
def manifest_env(env):
    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    qasm(BELL, "bell.qasm")
    jobs = [
        {"id": "a", "source": "a.qasm", "shots": 16, "seed": 3},
        {"id": "bell", "source": "bell.qasm", "shots": 8, "seed": 2},
    ]
    manifest = _json({"schema_version": 1, "jobs": jobs}, "jobs.json")
    return tmp_path, qasm, _file, _json, manifest


def test_batch_export_matches_stdout(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    _rc, stdout_text, _err = _run_main(["batch-simulate", manifest])
    target = str(tmp_path / "batch_out.json")
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert Path(target).read_text() == stdout_text


def test_batch_task_failures_still_export_full_report_with_exit_3(manifest_env):
    tmp_path, _qasm, _file, _json, _manifest = manifest_env
    manifest = _json(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "ok", "source": "a.qasm", "shots": 4, "seed": 1},
                {"id": "missing", "source": "nope.qasm"},
            ],
        },
        "mixed.json",
    )
    _rc, stdout_text, _err = _run_main(["batch-simulate", manifest])
    assert _rc == 3
    target = str(tmp_path / "mixed_out.json")
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", target])
    assert rc == 3
    assert out == ""
    assert err == ""
    assert Path(target).read_text() == stdout_text
    data = json.loads(Path(target).read_text())
    assert (data["succeeded"], data["failed"]) == (1, 1)


def test_reconcile_export_matches_stdout(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    baseline = _file(batch_stdout, "baseline.json")
    _rc, stdout_text, _err = _run_main(["reconcile", manifest, baseline])
    target = str(tmp_path / "reconcile_out.json")
    rc, out, err = _run_main(["reconcile", manifest, baseline, "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert Path(target).read_text() == stdout_text


def test_reconcile_mismatch_exports_full_report_with_exit_3(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    payload = json.loads(batch_stdout)
    # Corrupt one expected output to force an output_mismatch.
    payload["results"][0]["output"]["counts"] = {"0": 16}
    baseline = _json(payload, "bad_baseline.json")

    _rc, stdout_text, _err = _run_main(["reconcile", manifest, baseline])
    assert _rc == 3
    target = str(tmp_path / "reconcile_diff.json")
    rc, out, err = _run_main(["reconcile", manifest, baseline, "--output", target])
    assert rc == 3
    assert out == ""
    assert err == ""
    assert Path(target).read_text() == stdout_text
    report = json.loads(Path(target).read_text())
    assert report["consistent"] is False


# ------------------------------------------------------------- conflict guard


def test_conflict_output_equal_circuit_source_is_exit_2(env):
    _tmp, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(["simulate", source, "--output", source])
    assert rc == 2
    assert out == ""
    payload = json.loads(err)
    assert payload["error"] == "output_error"
    assert Path(source).read_bytes() == backup


def test_conflict_output_equal_noise_model_is_exit_2(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    model = _json({"bit_flip": 0.3}, "model.json")
    backup = Path(model).read_bytes()
    rc, out, err = _run_main(
        ["probabilities", source, "--noise-model", model, "--output", model]
    )
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert Path(model).read_bytes() == backup


@pytest.mark.parametrize("side", ["left", "right"])
def test_conflict_equivalent_output_is_either_input(env, side):
    tmp_path, qasm, _file, _json = env
    left = qasm(X_CIRCUIT, "l.qasm")
    right = qasm(X_CIRCUIT, "r.qasm")
    target = left if side == "left" else right
    rc, out, err = _run_main(["equivalent", left, right, "--output", target])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"


@pytest.mark.parametrize("command", ["equivalent", "state-metrics"])
def test_conflict_two_side_commands(command, env):
    tmp_path, qasm, _file, _json = env
    left = qasm(X_CIRCUIT, "l.qasm")
    right = qasm(X_CIRCUIT, "r.qasm")
    rc, _out, err = _run_main([command, left, right, "--output", right])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_batch_manifest_itself(env):
    tmp_path, qasm, _file, _json = env
    manifest = _json(
        {"schema_version": 1, "jobs": [{"id": "a", "source": "a.qasm", "shots": 3, "seed": 1}]},
        "jobs.json",
    )
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", manifest])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"


def test_conflict_batch_referenced_circuit(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    manifest = _json(
        {"schema_version": 1, "jobs": [{"id": "a", "source": "a.qasm", "shots": 3, "seed": 1}]},
        "jobs.json",
    )
    rc, _out, err = _run_main(["batch-simulate", manifest, "--output", source])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_batch_referenced_noise_model(env):
    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    model = _json({"bit_flip": 0.2}, "model.json")
    manifest = _json(
        {"schema_version": 1, "jobs": [{"id": "a", "source": "a.qasm", "noise_model": "model.json"}]},
        "jobs.json",
    )
    rc, _out, err = _run_main(["batch-simulate", manifest, "--output", model])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_covers_missing_referenced_manifest_file(env):
    # A failed (missing) task still names an input; exporting the report onto
    # that path is refused even though the file does not exist.
    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    manifest = _json(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "a", "source": "a.qasm", "shots": 3, "seed": 1},
                {"id": "ghost", "source": "missing.qasm"},
            ],
        },
        "jobs.json",
    )
    target = str(tmp_path / "missing.qasm")
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", target])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert not Path(target).exists()


def test_conflict_reconcile_baseline_and_manifest(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    baseline = _file(batch_stdout, "baseline.json")

    rc, _out, err = _run_main(["reconcile", manifest, baseline, "--output", baseline])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"

    rc, _out, err = _run_main(["reconcile", manifest, baseline, "--output", manifest])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_reconcile_referenced_task_file(manifest_env):
    tmp_path, qasm, _file, _json, manifest = manifest_env
    source = str(tmp_path / "a.qasm")
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    baseline = _file(batch_stdout, "baseline.json")
    rc, _out, err = _run_main(["reconcile", manifest, baseline, "--output", source])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_normalizes_path_spelling(manifest_env, monkeypatch):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    target = str(tmp_path / "a.qasm")
    monkeypatch.chdir(tmp_path)
    # Output spelled "./a.json"; manifest references "a.qasm": same file.
    rc, _out, err = _run_main(
        ["batch-simulate", "jobs.json", "--output", "./a.qasm"]
    )
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"
    assert Path(target).exists()  # not truncated


# ----------------------------------------------------- original failures are inert


def test_original_parse_error_does_not_create_target(env):
    tmp_path, _qasm, _file, _json = env
    bad = tmp_path / "bad.qasm"
    bad.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    target = tmp_path / "out.json"
    target.write_text("KEEP-ME", encoding="utf-8")
    rc, out, err = _run_main(["simulate", str(bad), "--output", str(target)])
    assert rc == 2
    assert out == ""
    assert json.loads(err)["error"] == "parse_error"
    assert target.read_text() == "KEEP-ME"


def test_original_io_error_does_not_create_target(env):
    target = env[0] / "out.json"
    target.write_text("KEEP-ME", encoding="utf-8")
    rc, out, err = _run_main(
        ["simulate", str(env[0] / "missing.qasm"), "--output", str(target)]
    )
    assert rc == 1 and out == ""
    assert json.loads(err)["error"] == "io_error"
    assert target.read_text() == "KEEP-ME"


def test_batch_input_error_does_not_create_target(env):
    tmp_path, _qasm, _file, _json = env
    manifest = _json({"schema_version": 1, "jobs": []}, "bad.json")
    target = tmp_path / "out.json"
    target.write_text("KEEP-ME", encoding="utf-8")
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", str(target)])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "batch_input_error"
    assert target.read_text() == "KEEP-ME"


def test_reconcile_input_error_does_not_create_target(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    bad_baseline = _json({"schema_version": 1}, "broken.json")
    target = tmp_path / "out.json"
    target.write_text("KEEP-ME", encoding="utf-8")
    rc, out, err = _run_main(["reconcile", manifest, bad_baseline, "--output", str(target)])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "reconcile_input_error"
    assert target.read_text() == "KEEP-ME"


def test_probabilities_size_error_takes_precedence_and_leaves_target(env):
    tmp_path, qasm, _file, _json = env
    body = "qreg q[11];\ncreg c[11];\n" + "".join(
        f"measure q[{i}] -> c[{i}];\n" for i in range(11)
    )
    source = qasm(body, "big.qasm")
    model = _json({"bit_flip": 0.1}, "model.json")
    target = tmp_path / "out.json"
    target.write_text("KEEP-ME", encoding="utf-8")
    rc, out, err = _run_main(
        ["probabilities", source, "--noise-model", model, "--output", str(target)]
    )
    assert rc == 3 and out == ""
    assert json.loads(err)["error"] == "simulation_error"
    assert target.read_text() == "KEEP-ME"


# ------------------------------------------------------------- export errors


def test_output_dash_rejected_with_exit_1(env):
    _tmp, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    rc, out, err = _run_main(["simulate", source, "--output", "-"])
    assert rc == 1 and out == ""
    payload = json.loads(err)
    assert payload["error"] == "output_error"
    assert err.count("\n") == 1


def test_output_directory_rejected_with_exit_1(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    directory = tmp_path / "adir"
    directory.mkdir()
    rc, out, err = _run_main(["simulate", source, "--output", str(directory)])
    assert rc == 1 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert directory.is_dir()


def test_output_missing_parent_is_exit_1(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    target = tmp_path / "no_such_dir" / "out.json"
    rc, out, err = _run_main(["simulate", source, "--output", str(target)])
    assert rc == 1 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert not target.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_output_permission_failure_preserves_existing_file(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    target = ro_dir / "out.json"
    target.write_text("ORIGINAL", encoding="utf-8")
    ro_dir.chmod(0o555)
    try:
        rc, out, err = _run_main(["simulate", source, "--output", str(target)])
    finally:
        ro_dir.chmod(0o755)
    assert rc == 1 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert target.read_text() == "ORIGINAL"
    # No half-written temporary file is left behind.
    assert not list(ro_dir.glob(".qcs-output-*"))


def test_failed_export_writes_nothing_to_stdout(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    rc, out, _err = _run_main(["simulate", source, "--output", "-"])
    assert rc == 1
    assert out == ""


# ----------------------------------------------------------- help and processes


@pytest.mark.parametrize(
    "command",
    [
        "simulate",
        "probabilities",
        "expectation",
        "equivalent",
        "optimize",
        "estimate",
        "state-metrics",
        "batch-simulate",
        "reconcile",
    ],
)
def test_help_lists_output_option(command, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main([command, "--help"])
    assert info.value.code == 0
    assert "--output" in capsys.readouterr().out


def test_version_command_unaffected(capsys):
    # version is not one of the eight exporting commands and behaves as before.
    assert cli.main(["version"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip()


def test_process_export_end_to_end(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    target = tmp_path / "proc_out.json"
    result = _run_process(
        "simulate", source, "--shots", "6", "--seed", "1", "--output", str(target)
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    data = json.loads(target.read_text())
    assert data["counts"] == {"1": 6}


def test_process_output_dash_exit_code(env):
    source = env[1](X_CIRCUIT)
    result = _run_process("simulate", source, "--output", "-")
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"] == "output_error"


# --------------------------------------------- file-identity conflict aliases
#
# The guard must recognize the *same file* under every path alias the
# platform allows -- relative "."/".." spellings, redundant separators,
# symbolic links (including chains and linked directories), Windows
# junctions (same stat(2) identity as symlinks) and hard links -- while
# never rejecting two genuinely distinct files, however similar.


def _conflict(argv):
    rc, out, err = _run_main(argv)
    return rc, out, err


def test_conflict_relative_dot_alias_of_absolute_input(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    monkeypatch.chdir(tmp_path)
    backup = Path(source).read_bytes()
    rc, out, err = _conflict(["simulate", source, "--output", "./a.qasm"])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert err.count("\n") == 1
    assert Path(source).read_bytes() == backup


def test_conflict_dotdot_alias_from_nested_cwd(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    rc, _out, err = _conflict(
        ["simulate", "../a.qasm", "--output", str(tmp_path / "nested/../a.qasm")]
    )
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"
    assert Path(source).read_bytes() != ""  # untouched


def test_conflict_redundant_separator_spellings(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    alias = str(tmp_path) + "//./a.qasm"
    rc, _out, err = _conflict(["simulate", alias, "--output", source])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_symlink_that_targets_the_input(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    link = tmp_path / "alias.qasm"
    os.symlink(source, link)
    backup = Path(source).read_bytes()
    rc, out, err = _conflict(["simulate", source, "--output", str(link)])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    # The symlink itself is not replaced and its referent is untouched.
    assert link.is_symlink()
    assert Path(source).read_bytes() == backup


def test_conflict_input_reached_through_symlink(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    link = tmp_path / "alias.qasm"
    os.symlink(source, link)
    rc, _out, err = _conflict(["simulate", str(link), "--output", source])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"
    assert link.is_symlink()


def test_conflict_symlink_chain(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    os.symlink(source, tmp_path / "one.qasm")
    os.symlink(tmp_path / "one.qasm", tmp_path / "two.qasm")
    rc, _out, err = _conflict(["simulate", str(tmp_path / "two.qasm"), "--output", source])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_path_through_symlinked_directory(env):
    tmp_path, qasm, _file, _json = env
    real = tmp_path / "real"
    real.mkdir()
    source = real / "a.qasm"
    source.write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    os.symlink(real, tmp_path / "linked")
    rc, _out, err = _conflict(
        ["simulate", str(tmp_path / "linked" / "a.qasm"), "--output", str(source)]
    )
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_hard_link(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    hard = tmp_path / "hard.qasm"
    try:
        os.link(source, hard)
    except OSError:
        pytest.skip("hard links are not supported on this volume")
    backup = Path(source).read_bytes()
    rc, out, err = _conflict(["simulate", str(hard), "--output", source])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert Path(source).read_bytes() == backup
    assert Path(hard).read_bytes() == backup


def test_distinct_files_with_identical_content_are_not_conflicts(env):
    tmp_path, qasm, _file, _json = env
    first = qasm(X_CIRCUIT, "a.qasm")
    second = qasm(X_CIRCUIT, "b.qasm")
    rc, out, err = _conflict(
        ["simulate", first, "--shots", "4", "--seed", "1", "--output", second]
    )
    assert (rc, out, err) == (0, "", "")
    # The distinct target is overwritten with the result, as for any target.
    assert json.loads(Path(second).read_text())["counts"] == {"1": 4}


def test_similar_filenames_are_not_conflicts(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    target = qasm(X_CIRCUIT, "ab.qasm")
    rc, out, err = _conflict(
        ["simulate", source, "--shots", "2", "--seed", "1", "--output", target]
    )
    assert (rc, out, err) == (0, "", "")


def test_export_through_symlinked_directory_to_a_new_file(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    real = tmp_path / "outdir"
    real.mkdir()
    os.symlink(real, tmp_path / "linkdir")
    target = tmp_path / "linkdir" / "result.json"
    rc, out, err = _conflict(
        ["simulate", source, "--shots", "3", "--seed", "1", "--output", str(target)]
    )
    assert (rc, out, err) == (0, "", "")
    assert (real / "result.json").is_file()


OBSERVABLES = {
    "schema_version": 1,
    "observables": [{"id": "z", "operators": [{"qubit": 0, "pauli": "Z"}]}],
}
SAMPLES = {"schema_version": 1, "counts": {"0": 3, "1": 1}}


@pytest.mark.parametrize(
    "command,document",
    [
        ("probabilities", {"bit_flip": 0.2}),
        ("expectation", OBSERVABLES),
        ("verify-samples", SAMPLES),
    ],
)
def test_conflict_auxiliary_input_reached_via_dot_alias(env, monkeypatch, command, document):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    _json(document, "doc.json")
    monkeypatch.chdir(tmp_path)
    extra = ["--noise-model", "doc.json"] if command == "probabilities" else []
    argv = [command, source, *extra]
    if command != "probabilities":
        argv.append("doc.json")
    rc, out, err = _conflict([*argv, "--output", "./doc.json"])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert json.loads(Path(tmp_path / "doc.json").read_text())  # valid JSON, untouched


def test_conflict_state_metrics_noise_model_alias(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    left = qasm(X_CIRCUIT, "l.qasm")
    right = qasm(X_CIRCUIT, "r.qasm")
    _json({"bit_flip": 0.2}, "model.json")
    monkeypatch.chdir(tmp_path)
    rc, out, err = _conflict(
        [
            "state-metrics", left, right,
            "--right-noise-model", "model.json",
            "--output", "./model.json",
        ]
    )
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"


def test_conflict_reconcile_baseline_alias(manifest_env, monkeypatch):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    _file(batch_stdout, "baseline.json")
    monkeypatch.chdir(tmp_path)
    rc, out, err = _conflict(
        ["reconcile", "jobs.json", "./baseline.json", "--output", "baseline.json"]
    )
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"


def test_stdin_input_does_not_participate_in_identity_guard(env, monkeypatch):
    import io

    tmp_path, _qasm, _file, _json = env
    target = str(tmp_path / "from_stdin.json")
    monkeypatch.setattr(
        cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + X_CIRCUIT).encode()))
    )
    rc, out, err = _conflict(["simulate", "-", "--shots", "4", "--seed", "1", "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert json.loads(Path(target).read_text())["counts"] == {"1": 4}


def test_conflict_still_covers_file_input_when_another_input_is_stdin(env, monkeypatch):
    import io

    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    samples = _json(SAMPLES, "samples.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + X_CIRCUIT).encode()))
    )
    rc, out, err = _conflict(
        ["verify-samples", "-", "samples.json", "--output", "./samples.json"]
    )
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert json.loads(Path(samples).read_text())["counts"] == SAMPLES["counts"]


# --------------------------------------------------------- case folding behavior


def _volume_folds_case(directory: Path) -> bool:
    probe = directory / ".qcs-fold-probe"
    try:
        probe.write_text("x", encoding="utf-8")
        return (directory / ".QCS-FOLD-PROBE").exists()
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


@pytest.fixture
def case_sensitive_dir(tmp_path):
    if _volume_folds_case(tmp_path):
        pytest.skip("the test temp volume folds letter case")
    return tmp_path


def test_case_only_spelling_is_distinct_on_case_sensitive_volume(case_sensitive_dir):
    tmp = case_sensitive_dir
    source = tmp / "a.qasm"
    source.write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    target = tmp / "A.QASM"
    rc, out, err = _conflict(
        ["simulate", str(source), "--shots", "2", "--seed", "1", "--output", str(target)]
    )
    assert (rc, out, err) == (0, "", "")
    # Two genuinely different files coexist after the export.
    assert source.is_file() and target.is_file()
    assert source.read_text().startswith("OPENQASM")
    assert json.loads(target.read_text())["counts"] == {"1": 2}


def _find_case_folding_dir():
    roots = [tempfile.gettempdir(), "/mnt/c", "/mnt/d", "/Volumes"]
    for root in roots:
        if not os.path.isdir(root) or not os.access(root, os.W_OK):
            continue
        try:
            directory = Path(tempfile.mkdtemp(prefix="qcs-ciprobe-", dir=root))
        except OSError:
            continue
        if _volume_folds_case(directory):
            return directory
        import shutil

        shutil.rmtree(directory, ignore_errors=True)
    return None


@pytest.fixture
def case_insensitive_dir():
    directory = _find_case_folding_dir()
    if directory is None:
        pytest.skip("no writable case-insensitive volume is available")
    yield directory
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def _write_qasm(path: Path, body: str = X_CIRCUIT) -> str:
    path.write_text(HEADER + body, encoding="utf-8")
    return str(path)


def test_case_only_alias_conflicts_on_case_insensitive_volume(case_insensitive_dir):
    d = case_insensitive_dir
    source = Path(_write_qasm(d / "a.qasm"))
    rc, out, err = _conflict(["simulate", str(source), "--output", str(d / "A.QASM")])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    # The folded spelling must not have been created as a separate file.
    assert source.read_text().startswith("OPENQASM")
    assert not (d / "A.QASM").exists() or source.stat().st_ino == (d / "A.QASM").stat().st_ino


def test_case_only_input_alias_conflicts_on_case_insensitive_volume(case_insensitive_dir):
    d = case_insensitive_dir
    source = Path(_write_qasm(d / "a.qasm"))
    backup = source.read_bytes()
    # The input is spelled with different case than the name on disk; the
    # output uses the on-disk spelling. Both resolve to the same inode.
    rc, _out, err = _conflict(
        ["simulate", str(d / "A.qasm"), "--output", str(d / "a.qasm")]
    )
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"
    assert source.read_bytes() == backup


def test_distinct_files_still_distinct_on_case_insensitive_volume(case_insensitive_dir):
    d = case_insensitive_dir
    first = Path(_write_qasm(d / "a.qasm"))
    second = Path(_write_qasm(d / "b.qasm"))
    rc, out, err = _conflict(
        ["simulate", str(first), "--shots", "2", "--seed", "1", "--output", str(second)]
    )
    assert (rc, out, err) == (0, "", "")
    assert json.loads(second.read_text())["counts"] == {"1": 2}


def test_missing_declared_ref_case_alias_conflicts_on_insensitive_volume(
    case_insensitive_dir,
):
    d = case_insensitive_dir
    _write_qasm(d / "a.qasm")
    manifest = d / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "jobs": [
                    {"id": "ok", "source": "a.qasm", "shots": 2, "seed": 1},
                    {"id": "ghost", "source": "missing.qasm"},
                ],
            }
        ),
        encoding="utf-8",
    )
    target = d / "MISSING.QASM"
    rc, out, err = _conflict(["batch-simulate", str(manifest), "--output", str(target)])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert not target.exists()


# ------------------------------------ manifest references: aliases and absences


def test_conflict_existing_manifest_reference_through_symlink(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    manifest = _json(
        {"schema_version": 1, "jobs": [{"id": "a", "source": "a.qasm", "shots": 2, "seed": 1}]},
        "jobs.json",
    )
    link = tmp_path / "jobs-link.json"
    os.symlink(manifest, link)
    # The manifest reached through the link still declares the same a.qasm.
    rc, _out, err = _conflict(["batch-simulate", str(link), "--output", source])
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"


def test_conflict_missing_reference_with_dotdot_normalization(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    nested = tmp_path / "nested"
    nested.mkdir()
    manifest = _json(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "a", "source": "a.qasm", "shots": 2, "seed": 1},
                {"id": "ghost", "source": "nested/../ghost.qasm"},
            ],
        },
        "jobs.json",
    )
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "ghost.qasm"
    rc, out, err = _conflict(["batch-simulate", manifest, "--output", "./ghost.qasm"])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert not target.exists()


def test_conflict_missing_reference_declared_by_stdin_manifest_uses_cwd(env, monkeypatch):
    import io

    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    manifest_text = json.dumps(
        {"schema_version": 1, "jobs": [{"id": "ghost", "source": "ghost.qasm"}]}
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(manifest_text.encode())))
    target = tmp_path / "ghost.qasm"
    rc, out, err = _conflict(["batch-simulate", "-", "--output", "./ghost.qasm"])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "output_error"
    assert not target.exists()


def test_missing_reference_with_distinct_similar_target_still_exports(env):
    tmp_path, qasm, _file, _json = env
    qasm(X_CIRCUIT, "a.qasm")
    manifest = _json(
        {"schema_version": 1, "jobs": [{"id": "ghost", "source": "ghost.qasm"}]},
        "jobs.json",
    )
    target = tmp_path / "ghost2.qasm"
    rc, out, err = _conflict(["batch-simulate", manifest, "--output", str(target)])
    # No conflict: the failing task is embedded and the report exports with 3.
    assert rc == 3 and out == "" and err == ""
    report = json.loads(target.read_text())
    assert report["failed"] == 1


def test_missing_reference_declared_through_symlinked_manifest_dir(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    real = tmp_path / "real"
    real.mkdir()
    qasm_path = real / "a.qasm"
    qasm_path.write_text(HEADER + X_CIRCUIT, encoding="utf-8")
    (real / "jobs.json").write_text(
        json.dumps(
            {"schema_version": 1, "jobs": [{"id": "ghost", "source": "ghost.qasm"}]}
        ),
        encoding="utf-8",
    )
    os.symlink(real, tmp_path / "alias")
    monkeypatch.chdir(tmp_path)
    # The declared path follows the same manifest-dir alias the open() call
    # would use, so the normalized lexical spellings must match.
    target_alias = "alias/ghost.qasm"
    rc, _out, err = _conflict(
        ["batch-simulate", "alias/jobs.json", "--output", target_alias]
    )
    assert rc == 2
    assert json.loads(err)["error"] == "output_error"
    assert not (real / "ghost.qasm").exists()


def test_existing_guard_error_precedence_unchanged_with_aliased_conflict(env):
    # A bad circuit reports parse_error (exit 2) even though the same
    # invocation also names the source as the output via an alias.
    tmp_path, _qasm, _file, _json = env
    bad = tmp_path / "bad.qasm"
    bad.write_text(HEADER + "qreg q[1];\ncreg c[1];\nx q[0]\n", encoding="utf-8")
    rc, out, err = _run_main(["simulate", str(bad), "--output", "./bad.qasm"])
    assert rc == 2 and out == ""
    assert json.loads(err)["error"] == "parse_error"


# ------------------------------------------------------------- unit-level rules


def test_registered_existing_input_compared_by_identity(tmp_path):
    from quantum_circuit.delivery import RunContext, output_conflict

    source = tmp_path / "a.qasm"
    source.write_text("input", encoding="utf-8")
    link = tmp_path / "a-link.qasm"
    os.symlink(source, link)

    ctx = RunContext(str(link))
    ctx.register_input(str(source))
    failure = output_conflict(ctx)
    assert failure is not None
    assert (failure.error, failure.exit_code) == ("output_error", 2)


def test_declared_nonexistent_input_compared_lexically(tmp_path):
    from quantum_circuit.delivery import RunContext, output_conflict

    ctx = RunContext(str(tmp_path / "nested/../ghost.qasm"))
    ctx.register_declared("ghost.qasm", str(tmp_path))
    failure = output_conflict(ctx)
    assert failure is not None and failure.error == "output_error"

    ctx = RunContext(str(tmp_path / "other.qasm"))
    ctx.register_declared("ghost.qasm", str(tmp_path))
    assert output_conflict(ctx) is None


def test_stdin_is_never_registered(tmp_path):
    from quantum_circuit.delivery import RunContext, output_conflict

    ctx = RunContext(str(tmp_path / "out.json"))
    ctx.register_input("-")
    ctx.register_declared("-")
    assert output_conflict(ctx) is None

