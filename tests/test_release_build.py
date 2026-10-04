"""Release packaging tests: reproducible artifacts, install behavior, and
build-time validation failures.

Builds run in subprocesses against a disposable copy of the project so the
source tree under test stays clean.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_FILES = ["pyproject.toml", "README.md", "MANIFEST.in", "_release_build.py"]
EPOCH = "1700000000"
SDIST_NAME = "quantum-circuit-simulator-0.1.0.tar.gz"
WHEEL_NAME = "quantum_circuit_simulator-0.1.0-py3-none-any.whl"


def make_project_copy(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_FILES:
        shutil.copy2(PROJECT_ROOT / name, dest / name)
    shutil.copytree(PROJECT_ROOT / "quantum_circuit", dest / "quantum_circuit")
    return dest


def run_build(project: Path, snippet: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, SOURCE_DATE_EPOCH=EPOCH)
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    if expect_ok:
        assert result.returncode == 0, result.stderr
    return result


def build_all(project: Path) -> Path:
    run_build(project, "import _release_build as b; b.build_sdist('dist'); b.build_wheel('dist')")
    return project / "dist"


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    return make_project_copy(tmp_path / "proj")


def test_reproducible_sdist_and_wheel(tmp_path: Path) -> None:
    first = build_all(make_project_copy(tmp_path / "a"))
    second = build_all(make_project_copy(tmp_path / "b"))
    for name in (SDIST_NAME, WHEEL_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_wheel_from_sdist_matches_direct_wheel(project: Path, tmp_path: Path) -> None:
    dist = build_all(project)
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with tarfile.open(dist / SDIST_NAME) as archive:
        archive.extractall(unpacked, filter="data")
    inner = unpacked / SDIST_NAME[: -len(".tar.gz")]
    run_build(inner, "import _release_build as b; b.build_wheel('dist')")
    assert (inner / "dist" / WHEEL_NAME).read_bytes() == (dist / WHEEL_NAME).read_bytes()


def test_wheel_contents(project: Path) -> None:
    dist = build_all(project)
    with zipfile.ZipFile(dist / WHEEL_NAME) as archive:
        names = archive.namelist()
        assert not any("tests" in n or "__pycache__" in n or n.endswith(".pyc") for n in names)
        assert "quantum_circuit/__init__.py" in names
        assert "quantum_circuit/cli.py" in names
        assert not any(n.startswith("/") or ".." in n.split("/") for n in names)
        dist_info = "quantum_circuit_simulator-0.1.0.dist-info"
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        assert "Name: quantum-circuit-simulator" in metadata.splitlines()
        assert "Version: 0.1.0" in metadata.splitlines()
        assert "量子线路仿真与验证平台" in metadata  # README carried in metadata
        wheel_meta = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
        assert "Tag: py3-none-any" in wheel_meta.splitlines()
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        assert "quantum-circuit-simulator = quantum_circuit.cli:main" in entry_points.splitlines()
        host_path = os.fsencode(str(project))
        assert not any(host_path in archive.read(n) for n in names)


def test_sdist_contents(project: Path) -> None:
    dist = build_all(project)
    with tarfile.open(dist / SDIST_NAME) as archive:
        names = archive.getnames()
    top = "quantum-circuit-simulator-0.1.0"
    for rel in ("pyproject.toml", "README.md", "PKG-INFO", "_release_build.py"):
        assert f"{top}/{rel}" in names
    package_modules = {f"{top}/quantum_circuit/{p.name}" for p in (PROJECT_ROOT / "quantum_circuit").glob("*.py")}
    assert package_modules <= set(names)
    assert not any("tests" in n.split("/") or "__pycache__" in n for n in names)
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda p: (p / "README.md").unlink(), "README.md is missing"),
        (lambda p: shutil.rmtree(p / "quantum_circuit"), "package directory"),
        (lambda p: (p / "quantum_circuit" / "cli.py").unlink(), "entry module"),
        (
            lambda p: (p / "quantum_circuit" / "__init__.py").write_text(
                (p / "quantum_circuit" / "__init__.py").read_text().replace(
                    '__version__ = "0.1.0"', '__version__ = "9.9.9"'
                )
            ),
            "version mismatch",
        ),
        (
            lambda p: (p / "README.md").write_bytes(
                (p / "README.md").read_bytes() + b"\xff\xfe"
            ),
            "not valid UTF-8",
        ),
    ],
    ids=["missing-readme", "missing-package", "missing-entry", "version-mismatch", "non-utf8-readme"],
)
def test_build_failures_leave_no_artifacts(project: Path, mutate, message: str) -> None:
    mutate(project)
    result = run_build(
        project,
        "import _release_build as b; b.build_sdist('dist'); b.build_wheel('dist')",
        expect_ok=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
    dist = project / "dist"
    leftovers = list(dist.glob("*.whl")) + list(dist.glob("*.tar.gz")) if dist.exists() else []
    assert leftovers == []


def test_installed_wheel_behaves_like_source(project: Path, tmp_path: Path) -> None:
    dist = build_all(project)
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / "bin" / "python"
    pip_install = [str(python), "-m", "pip", "install", "--quiet", "--no-index"]
    subprocess.run(pip_install + [str(dist / WHEEL_NAME)], check=True)

    def run_outside_source(*args: str, **kwargs) -> subprocess.CompletedProcess:
        # cwd is a neutral directory so the source tree is not on sys.path
        return subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, **kwargs)

    version = run_outside_source(str(env_dir / "bin" / "quantum-circuit-simulator"), "version")
    assert version.stdout.strip() == "0.1.0"
    help_result = run_outside_source(str(env_dir / "bin" / "quantum-circuit-simulator"), "--help")
    for subcommand in (
        "simulate", "probabilities", "expectation", "verify-samples", "batch-simulate",
        "reconcile", "equivalent", "optimize", "state-metrics", "estimate",
    ):
        assert subcommand in help_result.stdout
    surface = run_outside_source(
        str(python),
        "-c",
        "import quantum_circuit as q; print(q.__version__); print(sorted(q.__all__))",
    )
    assert surface.stdout.splitlines()[0] == "0.1.0"
    assert "Circuit" in surface.stdout
    bell = run_outside_source(
        str(python),
        "-c",
        "from quantum_circuit import Circuit; import json;"
        "c = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1);"
        "print(json.dumps(c.sample(shots=1024, seed=0)))",
    )
    expected = json.dumps(
        json.loads(bell.stdout)
    )
    baseline = subprocess.run(
        [sys.executable, "-c",
         "from quantum_circuit import Circuit; import json;"
         "c = Circuit(2, 2).h(0).cx(0, 1).measure(0, 0).measure(1, 1);"
         "print(json.dumps(c.sample(shots=1024, seed=0)))"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert expected == json.dumps(json.loads(baseline.stdout))
    assert json.loads(bell.stdout)["counts"] == {"00": 517, "11": 507}

    subprocess.run(
        [str(python), "-m", "pip", "uninstall", "-y", "quantum-circuit-simulator"],
        check=True,
        capture_output=True,
    )
    assert not (env_dir / "bin" / "quantum-circuit-simulator").exists()
    site_packages = next((env_dir / "lib").glob("python*/site-packages"))
    assert not (site_packages / "quantum_circuit").exists()
    assert not list(site_packages.glob("quantum_circuit*"))
