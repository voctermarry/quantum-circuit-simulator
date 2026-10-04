"""File-identity based ``--output`` conflict protection.

The guard must recognize every alias of the same underlying file --
``.``/``..`` spellings, symbolic links, hard links, Windows directory
junctions and (on case-insensitive file systems) case-only respellings --
while never refusing two genuinely different files, however similar their
names or identical their contents. Manifest-declared paths that do not
exist are still compared lexically against the manifest directory / cwd
baseline. On conflict the command prints nothing to stdout, emits one
``output_error`` JSON line on stderr, exits 2 and touches no file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quantum_circuit import cli

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

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


def _assert_conflict(rc, out, err):
    assert rc == 2
    assert out == ""
    assert err.count("\n") == 1
    assert json.loads(err)["error"] == "output_error"


def _make_symlink(link: Path, target: str) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable here: {exc}")


def _make_hardlink(link: Path, target: str) -> None:
    try:
        os.link(target, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hard links unavailable here: {exc}")


def _case_insensitive_fs(tmp_path: Path) -> bool:
    probe = tmp_path / "CaSePrObE.tmp"
    probe.write_text("x", encoding="utf-8")
    try:
        return (tmp_path / "caseprobe.TMP").exists()
    finally:
        probe.unlink()


# ---------------------------------------------------------- lexical aliases


def test_dot_dot_spelling_of_same_file_conflicts(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    subdir = tmp_path / "sub"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(
        ["simulate", "../a.qasm", "--shots", "4", "--seed", "1", "--output", "../a.qasm"]
    )
    _assert_conflict(rc, out, err)
    assert Path(source).read_bytes() == backup


def test_dot_dot_output_alias_against_plain_input(env, monkeypatch):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    subdir = tmp_path / "sub"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(
        ["optimize", str(Path(source).resolve()), "--output", "../a.qasm"]
    )
    _assert_conflict(rc, out, err)
    assert Path(source).read_bytes() == backup


def test_redundant_dot_segments_in_output_conflict(env):
    _tmp, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    alias = str(Path(source).parent / "." / "sub" / ".." / "a.qasm")
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(["probabilities", source, "--output", alias])
    _assert_conflict(rc, out, err)
    assert Path(source).read_bytes() == backup


# ------------------------------------------------------------- link aliases


def test_symlink_output_alias_conflicts(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    link = tmp_path / "alias.qasm"
    _make_symlink(link, source)
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(["simulate", source, "--shots", "4", "--seed", "1", "--output", str(link)])
    _assert_conflict(rc, out, err)
    assert link.is_symlink()  # the link itself is not replaced
    assert Path(source).read_bytes() == backup


def test_symlinked_input_matches_real_output_path(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    link = tmp_path / "alias.qasm"
    _make_symlink(link, source)
    backup = Path(source).read_bytes()
    # The input is read through the symlink; the output names the real file.
    rc, out, err = _run_main(["optimize", str(link), "--output", source])
    _assert_conflict(rc, out, err)
    assert Path(source).read_bytes() == backup


def test_hardlink_output_alias_conflicts(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    link = tmp_path / "hard.qasm"
    _make_hardlink(link, source)
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(["probabilities", source, "--output", str(link)])
    _assert_conflict(rc, out, err)
    assert Path(source).read_bytes() == backup
    assert Path(source).stat().st_ino == link.stat().st_ino


def test_equivalent_right_side_symlink_alias_conflicts(env):
    tmp_path, qasm, _file, _json = env
    left = qasm(X_CIRCUIT, "l.qasm")
    right = qasm(BELL, "r.qasm")
    link = tmp_path / "r_alias.qasm"
    _make_symlink(link, right)
    rc, out, err = _run_main(["equivalent", left, right, "--output", str(link)])
    _assert_conflict(rc, out, err)
    assert link.is_symlink()


def test_state_metrics_noise_model_hardlink_alias_conflicts(env):
    tmp_path, qasm, _file, _json = env
    left = qasm(X_CIRCUIT, "l.qasm")
    right = qasm(X_CIRCUIT, "r.qasm")
    model = _json({"bit_flip": 0.1}, "model.json")
    link = tmp_path / "model_alias.json"
    _make_hardlink(link, model)
    backup = Path(model).read_bytes()
    rc, out, err = _run_main(
        [
            "state-metrics",
            left,
            right,
            "--left-noise-model",
            model,
            "--output",
            str(link),
        ]
    )
    _assert_conflict(rc, out, err)
    assert Path(model).read_bytes() == backup


# ------------------------------------------------------- case-only spellings


def test_case_only_difference_conflicts_on_case_insensitive_fs(env):
    tmp_path, qasm, _file, _json = env
    if not _case_insensitive_fs(tmp_path):
        pytest.skip("case-sensitive file system")
    source = qasm(X_CIRCUIT, "a.qasm")
    alias = str(tmp_path / "A.QASM")
    backup = Path(source).read_bytes()
    rc, out, err = _run_main(["simulate", source, "--shots", "4", "--seed", "1", "--output", alias])
    _assert_conflict(rc, out, err)
    assert Path(source).read_bytes() == backup


def test_case_only_difference_allowed_on_case_sensitive_fs(env):
    tmp_path, qasm, _file, _json = env
    if _case_insensitive_fs(tmp_path):
        pytest.skip("case-insensitive file system")
    source = qasm(X_CIRCUIT, "a.qasm")
    target = str(tmp_path / "A.QASM")
    rc, out, err = _run_main(["simulate", source, "--shots", "4", "--seed", "1", "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert json.loads(Path(target).read_text())["counts"] == {"1": 4}
    # The original input is untouched and the two files are distinct.
    assert Path(source).read_text().endswith(X_CIRCUIT)


# ------------------------------------------------------- non-conflict controls


def test_identical_content_different_file_is_allowed(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    twin = qasm(X_CIRCUIT, "b.qasm")  # same bytes, different file
    assert Path(source).stat().st_ino != Path(twin).stat().st_ino or os.name == "nt"
    rc, out, err = _run_main(["simulate", source, "--shots", "4", "--seed", "1", "--output", twin])
    assert (rc, out, err) == (0, "", "")
    assert json.loads(Path(twin).read_text())["counts"] == {"1": 4}


def test_similar_name_is_allowed(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    for name in ("a.qasm.json", "a.qasm.bak", "a.qasm2"):
        target = str(tmp_path / name)
        rc, out, err = _run_main(["probabilities", source, "--output", target])
        assert (rc, out, err) == (0, "", "")
        assert Path(target).is_file()


def test_unrelated_existing_file_is_overwritten(env):
    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT)
    target = _file("STALE", "out.json")
    rc, out, err = _run_main(["simulate", source, "--shots", "4", "--seed", "1", "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert json.loads(Path(target).read_text())["counts"] == {"1": 4}


# ----------------------------------------------- batch / reconcile references


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


def test_batch_missing_reference_dot_alias_conflicts(manifest_env, monkeypatch):
    # The declared task path does not exist; a ``./``-spelled output that
    # lexically resolves to it is still refused and nothing is created.
    tmp_path, _qasm, _file, _json, _manifest = manifest_env
    manifest = _json(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "a", "source": "a.qasm", "shots": 3, "seed": 1},
                {"id": "ghost", "source": "missing.qasm"},
            ],
        },
        "ghost.json",
    )
    monkeypatch.chdir(tmp_path)
    rc, out, err = _run_main(["batch-simulate", "ghost.json", "--output", "./missing.qasm"])
    _assert_conflict(rc, out, err)
    assert not (tmp_path / "missing.qasm").exists()


def test_batch_missing_reference_dotdot_alias_conflicts(manifest_env, monkeypatch):
    tmp_path, _qasm, _file, _json, _manifest = manifest_env
    manifest = _json(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "a", "source": "a.qasm", "shots": 3, "seed": 1},
                {"id": "ghost", "source": "missing.qasm"},
            ],
        },
        "ghost.json",
    )
    subdir = tmp_path / "sub"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    rc, out, err = _run_main(
        ["batch-simulate", "../ghost.json", "--output", "../sub/../missing.qasm"]
    )
    _assert_conflict(rc, out, err)
    assert not (tmp_path / "missing.qasm").exists()


def test_batch_symlink_to_referenced_circuit_conflicts(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    link = tmp_path / "alias.json"
    _make_symlink(link, str(tmp_path / "a.qasm"))
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", str(link)])
    _assert_conflict(rc, out, err)
    assert link.is_symlink()


def test_batch_symlink_to_manifest_conflicts(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    link = tmp_path / "manifest_alias.json"
    _make_symlink(link, manifest)
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", str(link)])
    _assert_conflict(rc, out, err)
    assert link.is_symlink()


def test_batch_manifest_in_symlinked_directory(manifest_env):
    # The manifest lives behind a symlinked directory; the output names the
    # referenced circuit through its real path.
    tmp_path, _qasm, _file, _json, _manifest = manifest_env
    link_dir = tmp_path / "linked"
    _make_symlink(link_dir, str(tmp_path))
    manifest_via_link = str(link_dir / "jobs.json")
    rc, out, err = _run_main(
        ["batch-simulate", manifest_via_link, "--output", str(tmp_path / "a.qasm")]
    )
    _assert_conflict(rc, out, err)


def test_reconcile_hardlink_to_baseline_conflicts(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    baseline = _file(batch_stdout, "baseline.json")
    link = tmp_path / "baseline_alias.json"
    _make_hardlink(link, baseline)
    backup = Path(baseline).read_bytes()
    rc, out, err = _run_main(["reconcile", manifest, baseline, "--output", str(link)])
    _assert_conflict(rc, out, err)
    assert Path(baseline).read_bytes() == backup


def test_reconcile_missing_reference_dot_alias_conflicts(manifest_env, monkeypatch):
    tmp_path, _qasm, _file, _json, _manifest = manifest_env
    manifest = _json(
        {
            "schema_version": 1,
            "jobs": [
                {"id": "a", "source": "a.qasm", "shots": 3, "seed": 1},
                {"id": "ghost", "source": "missing.qasm"},
            ],
        },
        "ghost.json",
    )
    _rc, batch_stdout, _err = _run_main(["batch-simulate", manifest])
    baseline = _file(batch_stdout, "baseline.json")
    monkeypatch.chdir(tmp_path)
    rc, out, err = _run_main(
        ["reconcile", "ghost.json", baseline, "--output", "./sub/../missing.qasm"]
    )
    _assert_conflict(rc, out, err)
    assert not (tmp_path / "missing.qasm").exists()


def test_batch_unrelated_output_still_exports(manifest_env):
    tmp_path, _qasm, _file, _json, manifest = manifest_env
    target = str(tmp_path / "report.json")
    rc, out, err = _run_main(["batch-simulate", manifest, "--output", target])
    assert (rc, out, err) == (0, "", "")
    report = json.loads(Path(target).read_text())
    assert report["succeeded"] == 2


# -------------------------------------------------------------- stdin inputs


def test_stdin_input_does_not_participate_in_identity_check(env, monkeypatch):
    import io

    tmp_path, _qasm, _file, _json = env
    target = str(tmp_path / "out.json")
    monkeypatch.setattr(
        cli.sys, "stdin", io.TextIOWrapper(io.BytesIO((HEADER + X_CIRCUIT).encode()))
    )
    rc, out, err = _run_main(["simulate", "-", "--shots", "4", "--seed", "1", "--output", target])
    assert (rc, out, err) == (0, "", "")
    assert json.loads(Path(target).read_text())["counts"] == {"1": 4}


def test_stdin_manifest_references_still_guarded(env, monkeypatch):
    import io

    tmp_path, qasm, _file, _json = env
    source = qasm(X_CIRCUIT, "a.qasm")
    manifest_text = json.dumps(
        {"schema_version": 1, "jobs": [{"id": "a", "source": "a.qasm", "shots": 3, "seed": 1}]}
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(manifest_text.encode()))
    )
    # A manifest read from stdin resolves task paths against the cwd; the
    # output naming that same file (via a ./ alias) is refused.
    rc, out, err = _run_main(["batch-simulate", "-", "--output", "./a.qasm"])
    _assert_conflict(rc, out, err)
    assert Path(source).read_text().endswith(X_CIRCUIT)
