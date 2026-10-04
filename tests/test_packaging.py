"""Release-packaging tests for the in-tree PEP 517 build backend.

These tests exercise ``_build/backend.py`` directly: reproducible sdist/wheel
builds, wheel-from-sdist equivalence, archive content rules and the
fail-closed validation of the source tree.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = REPO_ROOT / "_build" / "backend.py"

spec = importlib.util.spec_from_file_location("release_backend", BACKEND_PATH)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)

SOURCE_EPOCH = "1700000000"
TREE_ITEMS = ("pyproject.toml", "MANIFEST.in", "README.md", "quantum_circuit", "_build")


def _copy_tree(dst: Path) -> Path:
    dst.mkdir(parents=True)
    for item in TREE_ITEMS:
        src = REPO_ROOT / item
        target = dst / item
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, target)
    return dst


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    """A clean source snapshot the backend and setuptools operate on."""
    tree = _copy_tree(tmp_path / "src")
    monkeypatch.setattr(backend, "PROJECT_ROOT", tree)
    monkeypatch.chdir(tree)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", SOURCE_EPOCH)
    return tree


def _build_sdist_and_wheel(tree: Path, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir()
    sdist = outdir / backend.build_sdist(str(outdir))
    wheel = outdir / backend.build_wheel(str(outdir))
    return sdist, wheel


# ---------------------------------------------------------------------------
# reproducible artifacts
# ---------------------------------------------------------------------------


def test_repeated_builds_are_byte_identical(source_tree):
    sdist_a, wheel_a = _build_sdist_and_wheel(source_tree, source_tree / "out_a")
    sdist_b, wheel_b = _build_sdist_and_wheel(source_tree, source_tree / "out_b")

    assert sdist_a.name == sdist_b.name
    assert wheel_a.name == wheel_b.name
    assert wheel_a.name.endswith("-py3-none-any.whl")
    assert sdist_a.read_bytes() == sdist_b.read_bytes()
    assert wheel_a.read_bytes() == wheel_b.read_bytes()


def test_wheel_rebuilt_from_sdist_matches_direct_wheel(source_tree, tmp_path, monkeypatch):
    sdist, direct_wheel = _build_sdist_and_wheel(source_tree, source_tree / "out")

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(sdist, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    (sdist_root,) = [p for p in extracted.iterdir() if p.is_dir()]

    monkeypatch.setattr(backend, "PROJECT_ROOT", sdist_root)
    monkeypatch.chdir(sdist_root)
    outdir = sdist_root / "out"
    outdir.mkdir()
    rebuilt = outdir / backend.build_wheel(str(outdir))

    assert rebuilt.name == direct_wheel.name
    assert rebuilt.read_bytes() == direct_wheel.read_bytes()


def test_artifact_contents(source_tree):
    sdist, wheel = _build_sdist_and_wheel(source_tree, source_tree / "out")

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()
    assert not any(name.split("/")[0] != sdist_names[0].split("/")[0] for name in sdist_names)
    prefix = sdist_names[0].split("/")[0]
    relative = {name[len(prefix) + 1:] for name in sdist_names if name != prefix}
    for required in (
        "README.md",
        "pyproject.toml",
        "MANIFEST.in",
        "PKG-INFO",
        "_build/backend.py",
        "quantum_circuit/__init__.py",
        "quantum_circuit/cli.py",
    ):
        assert required in relative
    assert not any("tests" in Path(name).parts for name in sdist_names)
    assert not any("__pycache__" in name for name in sdist_names)

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        metadata = archive.read(
            next(n for n in wheel_names if n.endswith(".dist-info/METADATA"))
        ).decode("utf-8")
        entry_points = archive.read(
            next(n for n in wheel_names if n.endswith(".dist-info/entry_points.txt"))
        ).decode("utf-8")

    facts = backend._validate_project()
    dist_info = f"{backend._escape(facts['name'])}-{backend._escape(facts['version'])}.dist-info"
    tops = {name.split("/", 1)[0] for name in wheel_names}
    assert tops == {"quantum_circuit", dist_info}
    assert not any("tests" in Path(name).parts for name in wheel_names)
    assert not any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in wheel_names)
    assert "Name: quantum-circuit-simulator\n" in metadata
    assert f"Version: {facts['version']}\n" in metadata
    assert "Requires-Python: >=3.11\n" in metadata
    assert "quantum-circuit-simulator = quantum_circuit.cli:main" in entry_points


# ---------------------------------------------------------------------------
# fail-closed source tree validation
# ---------------------------------------------------------------------------


def _assert_validation_fails(source_tree, outdir_name="out"):
    outdir = source_tree / outdir_name
    outdir.mkdir()
    with pytest.raises(backend.BuildValidationError):
        backend.build_sdist(str(outdir))
    assert list(outdir.iterdir()) == []
    with pytest.raises(backend.BuildValidationError):
        backend.build_wheel(str(outdir))
    assert list(outdir.iterdir()) == []


def test_missing_readme_fails(source_tree):
    (source_tree / "README.md").unlink()
    _assert_validation_fails(source_tree)


def test_missing_package_directory_fails(source_tree):
    shutil.rmtree(source_tree / "quantum_circuit")
    _assert_validation_fails(source_tree)


def test_missing_entry_module_fails(source_tree):
    (source_tree / "quantum_circuit" / "cli.py").unlink()
    _assert_validation_fails(source_tree)


def test_version_mismatch_fails(source_tree):
    init = source_tree / "quantum_circuit" / "__init__.py"
    init.write_text(init.read_text(encoding="utf-8").replace('__version__ = "0.1.0"', '__version__ = "9.9.9"'), encoding="utf-8")
    _assert_validation_fails(source_tree)


def test_non_utf8_metadata_fails(source_tree):
    pyproject = source_tree / "pyproject.toml"
    pyproject.write_bytes(pyproject.read_bytes() + b"\xff\xfe")
    _assert_validation_fails(source_tree)


def test_invalid_source_date_epoch_fails(source_tree, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-a-number")
    _assert_validation_fails(source_tree)


def test_member_name_validation():
    check = backend._check_member_names
    for bad in (
        ["pkg/a.py", "pkg/a.py"],            # duplicate member
        ["pkg/../escape", "pkg/a.py"],       # escapes the project directory
        ["/absolute/path", "pkg/a.py"],
        ["C:/drive/path", "pkg/a.py"],
        ["pkg\\backslash", "pkg/a.py"],
        ["pkg/__pycache__/x.pyc", "pkg/a.py"],
        ["tests/test_x.py", "pkg/a.py"],
        ["pkg/leftover.bak", "pkg/a.py"],
    ):
        with pytest.raises(backend.BuildValidationError):
            check(bad, "test-archive")
    check(["pkg/a.py", "pkg/b.py", "pkg/c/"], "test-archive")


# ---------------------------------------------------------------------------
# repository invariants
# ---------------------------------------------------------------------------


def test_metadata_version_matches_package_version():
    import quantum_circuit

    facts = backend._validate_project()
    assert facts["version"] == quantum_circuit.__version__
    assert facts["name"] == "quantum-circuit-simulator"
