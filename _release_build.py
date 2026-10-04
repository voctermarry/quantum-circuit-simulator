"""In-tree PEP 517 build backend with release validation and normalization.

Wraps ``setuptools.build_meta`` and enforces the release invariants of
quantum-circuit-simulator:

* the source tree must be complete (``README.md``, the ``quantum_circuit``
  package directory and the ``quantum_circuit.cli:main`` entry module) and
  the static ``project.version`` in ``pyproject.toml`` must match
  ``quantum_circuit.__version__``;
* produced archives must be well formed: UTF-8 metadata, no duplicate
  members, no absolute or escaping paths, no tests, caches or temporary
  files, and no host absolute paths embedded in the payload;
* both artifacts are normalized so that identical sources, Python minor
  version and ``SOURCE_DATE_EPOCH`` produce byte-identical files.

Any violation raises :class:`ReleaseBuildError` and the offending artifact
is removed from the target directory, so a failed build never leaves an
installable artifact behind.
"""

from __future__ import annotations

import ast
import gzip
import io
import os
import re
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from setuptools.build_meta import *  # noqa: F401,F403  (re-export PEP 517 hooks)
from setuptools import build_meta as _setuptools

__all__ = [
    "ReleaseBuildError",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "build_sdist",
    "build_wheel",
    "build_editable",
]

PROJECT_ROOT = Path(__file__).resolve().parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
README_PATH = PROJECT_ROOT / "README.md"
PACKAGE_DIR = PROJECT_ROOT / "quantum_circuit"
ENTRY_MODULE_PATH = PACKAGE_DIR / "cli.py"

PROJECT_NAME = "quantum-circuit-simulator"
NORMALIZED_NAME = "quantum_circuit_simulator"
ENTRY_POINT_NAME = "quantum-circuit-simulator"
ENTRY_POINT_TARGET = "quantum_circuit.cli:main"
WHEEL_TAG = "py3-none-any"

_FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "test",
    "tests",
}
_FORBIDDEN_FILE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".tmp", ".bak", ".swp", "~")
_FORBIDDEN_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


class ReleaseBuildError(RuntimeError):
    """The release inputs or the produced artifacts failed validation."""


# ---------------------------------------------------------------------------
# source tree validation
# ---------------------------------------------------------------------------


def _read_pyproject() -> dict:
    if not PYPROJECT_PATH.is_file():
        raise ReleaseBuildError("pyproject.toml is missing")
    try:
        with PYPROJECT_PATH.open("rb") as handle:
            return tomllib.load(handle)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ReleaseBuildError(f"pyproject.toml is not valid UTF-8 TOML: {exc}") from exc


def _read_package_version() -> str:
    init_path = PACKAGE_DIR / "__init__.py"
    if not init_path.is_file():
        raise ReleaseBuildError("quantum_circuit/__init__.py is missing")
    try:
        tree = ast.parse(init_path.read_bytes(), filename=str(init_path))
    except (SyntaxError, ValueError) as exc:
        raise ReleaseBuildError(f"quantum_circuit/__init__.py does not parse: {exc}") from exc
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    break
                if isinstance(value, str) and value.strip():
                    return value
    raise ReleaseBuildError(
        "quantum_circuit.__version__ is not assigned a literal string"
    )


def _validate_entry_module() -> None:
    if not ENTRY_MODULE_PATH.is_file():
        raise ReleaseBuildError("entry module quantum_circuit/cli.py is missing")
    try:
        tree = ast.parse(ENTRY_MODULE_PATH.read_bytes(), filename=str(ENTRY_MODULE_PATH))
    except (SyntaxError, ValueError) as exc:
        raise ReleaseBuildError(
            f"entry module quantum_circuit/cli.py does not parse: {exc}"
        ) from exc
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
        for node in tree.body
    ):
        raise ReleaseBuildError(
            "entry module quantum_circuit/cli.py does not define main()"
        )


def _validate_source_tree() -> str:
    """Validate the build inputs and return the declared project version."""
    pyproject = _read_pyproject()
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ReleaseBuildError("pyproject.toml is missing the [project] table")
    name = project.get("name")
    if name != PROJECT_NAME:
        raise ReleaseBuildError(f"pyproject.toml: unexpected project name {name!r}")
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ReleaseBuildError("pyproject.toml: [project] must declare a static version")
    if not README_PATH.is_file():
        raise ReleaseBuildError("README.md is missing")
    try:
        README_PATH.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseBuildError(f"README.md is not valid UTF-8: {exc}") from exc
    if not PACKAGE_DIR.is_dir():
        raise ReleaseBuildError("package directory quantum_circuit/ is missing")
    package_version = _read_package_version()
    if package_version != version:
        raise ReleaseBuildError(
            "version mismatch: pyproject.toml declares "
            f"{version!r} but quantum_circuit.__version__ is {package_version!r}"
        )
    _validate_entry_module()
    scripts = project.get("scripts")
    if not isinstance(scripts, dict) or scripts.get(ENTRY_POINT_NAME) != ENTRY_POINT_TARGET:
        raise ReleaseBuildError(
            "pyproject.toml: [project.scripts] must map "
            f"{ENTRY_POINT_NAME!r} to {ENTRY_POINT_TARGET!r}"
        )
    return version


# ---------------------------------------------------------------------------
# archive validation
# ---------------------------------------------------------------------------


def _check_member_names(names, kind: str) -> None:
    seen = set()
    for name in names:
        if not name:
            raise ReleaseBuildError(f"{kind} contains an empty member name")
        if name in seen:
            raise ReleaseBuildError(f"{kind} contains duplicate member {name!r}")
        seen.add(name)
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ReleaseBuildError(
                f"{kind} member name is not valid UTF-8: {name!r}"
            ) from exc
        if (
            name.startswith("/")
            or "\\" in name
            or re.match(r"^[A-Za-z]:", name)
            or ".." in PurePosixPath(name).parts
        ):
            raise ReleaseBuildError(
                f"{kind} member {name!r} is not a safe relative path inside the project"
            )


def _check_forbidden_members(names, kind: str) -> None:
    for name in names:
        parts = PurePosixPath(name).parts
        for part in parts:
            if part in _FORBIDDEN_DIR_NAMES:
                raise ReleaseBuildError(
                    f"{kind} member {name!r} contains forbidden directory {part!r}"
                )
        base = parts[-1] if parts else ""
        if base in _FORBIDDEN_FILE_NAMES or base.endswith(_FORBIDDEN_FILE_SUFFIXES):
            raise ReleaseBuildError(
                f"{kind} member {name!r} is a cache or temporary file"
            )


def _check_no_host_paths(payload: dict, kind: str) -> None:
    root = os.fsencode(str(PROJECT_ROOT))
    if len(root) < 5:
        return
    for name, data in payload.items():
        if root in data:
            raise ReleaseBuildError(
                f"{kind} member {name!r} embeds the build host path "
                f"{str(PROJECT_ROOT)!r}"
            )


def _decode_utf8(data: bytes, what: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseBuildError(f"{what} is not valid UTF-8: {exc}") from exc


def _validate_sdist(path: Path, version: str) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            _check_member_names([m.name for m in members], "sdist")
            _check_forbidden_members([m.name for m in members], "sdist")
            payload = {}
            for member in members:
                if member.isfile():
                    payload[member.name] = archive.extractfile(member).read()
                elif not member.isdir():
                    raise ReleaseBuildError(
                        f"sdist member {member.name!r} has an unsupported entry type"
                    )
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ReleaseBuildError(f"sdist {path.name} is not a valid tar archive: {exc}") from exc
    tops = {name.split("/", 1)[0] for name in payload}
    if len(tops) != 1:
        raise ReleaseBuildError("sdist must contain exactly one top-level directory")
    top = tops.pop()
    if top not in {f"{PROJECT_NAME}-{version}", f"{NORMALIZED_NAME}-{version}"}:
        raise ReleaseBuildError(
            f"sdist top-level directory {top!r} does not match the project name/version"
        )
    for rel in ("pyproject.toml", "README.md", "PKG-INFO", "_release_build.py"):
        if f"{top}/{rel}" not in payload:
            raise ReleaseBuildError(f"sdist is missing {rel}")
    pkg_info = _decode_utf8(payload[f"{top}/PKG-INFO"], "sdist PKG-INFO").splitlines()
    if f"Name: {PROJECT_NAME}" not in pkg_info or f"Version: {version}" not in pkg_info:
        raise ReleaseBuildError("sdist PKG-INFO name/version does not match the project")
    if payload[f"{top}/README.md"] != README_PATH.read_bytes():
        raise ReleaseBuildError("sdist README.md does not match the source tree")
    for source in sorted(PACKAGE_DIR.glob("*.py")):
        member = f"{top}/quantum_circuit/{source.name}"
        if payload.get(member) != source.read_bytes():
            raise ReleaseBuildError(
                f"sdist member quantum_circuit/{source.name} does not match the source tree"
            )
    _check_no_host_paths(payload, "sdist")


def _validate_wheel(path: Path, version: str) -> None:
    expected_name = f"{NORMALIZED_NAME}-{version}-{WHEEL_TAG}.whl"
    if path.name != expected_name:
        raise ReleaseBuildError(
            f"wheel filename {path.name!r} does not match expected {expected_name!r}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            _check_member_names([info.filename for info in infos], "wheel")
            _check_forbidden_members([info.filename for info in infos], "wheel")
            for info in infos:
                if not info.flag_bits & 0x800:
                    try:
                        info.filename.encode("ascii")
                    except UnicodeEncodeError as exc:
                        raise ReleaseBuildError(
                            f"wheel member {info.filename!r} is not stored as UTF-8"
                        ) from exc
            payload = {
                info.filename: archive.read(info) for info in infos if not info.is_dir()
            }
    except zipfile.BadZipFile as exc:
        raise ReleaseBuildError(f"wheel {path.name} is not a valid zip archive: {exc}") from exc
    dist_info = f"{NORMALIZED_NAME}-{version}.dist-info"
    for name in payload:
        if not name.startswith(("quantum_circuit/", dist_info + "/")):
            raise ReleaseBuildError(f"wheel contains unexpected member {name!r}")
    for suffix in ("METADATA", "WHEEL", "RECORD", "entry_points.txt"):
        if f"{dist_info}/{suffix}" not in payload:
            raise ReleaseBuildError(f"wheel is missing {dist_info}/{suffix}")
    metadata = _decode_utf8(payload[f"{dist_info}/METADATA"], "wheel METADATA").splitlines()
    if f"Name: {PROJECT_NAME}" not in metadata or f"Version: {version}" not in metadata:
        raise ReleaseBuildError("wheel METADATA name/version does not match the project")
    wheel_meta = _decode_utf8(payload[f"{dist_info}/WHEEL"], "wheel WHEEL").splitlines()
    if f"Tag: {WHEEL_TAG}" not in wheel_meta:
        raise ReleaseBuildError(f"wheel WHEEL tag is not {WHEEL_TAG}")
    entry_points = _decode_utf8(
        payload[f"{dist_info}/entry_points.txt"], "wheel entry_points.txt"
    ).splitlines()
    if f"{ENTRY_POINT_NAME} = {ENTRY_POINT_TARGET}" not in entry_points:
        raise ReleaseBuildError("wheel entry_points.txt is missing the console script")
    for source in sorted(PACKAGE_DIR.glob("*.py")):
        member = f"quantum_circuit/{source.name}"
        if payload.get(member) != source.read_bytes():
            raise ReleaseBuildError(
                f"wheel member {member!r} does not match the source tree"
            )
    _check_no_host_paths(payload, "wheel")


# ---------------------------------------------------------------------------
# deterministic normalization
# ---------------------------------------------------------------------------


def _source_date_epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None or raw == "":
        return 0
    try:
        epoch = int(raw)
    except ValueError as exc:
        raise ReleaseBuildError(f"SOURCE_DATE_EPOCH {raw!r} is not an integer") from exc
    if epoch < 0:
        raise ReleaseBuildError("SOURCE_DATE_EPOCH must not be negative")
    return epoch


def _normalize_sdist(path: Path) -> None:
    epoch = _source_date_epoch()
    with tarfile.open(path, "r:gz") as source:
        entries = []
        for member in source.getmembers():
            data = source.extractfile(member).read() if member.isfile() else None
            entries.append((member, data))
    entries.sort(key=lambda item: item[0].name)
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=epoch
    ) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as dest:
            for member, data in entries:
                member.mtime = epoch
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                member.pax_headers = {}
                if member.isdir():
                    member.mode = 0o755
                elif member.isfile():
                    member.mode = 0o755 if member.mode & 0o111 else 0o644
                dest.addfile(member, io.BytesIO(data) if data is not None else None)
    path.write_bytes(buffer.getvalue())


def _normalize_wheel(path: Path) -> None:
    epoch = _source_date_epoch()
    stamp = time.gmtime(min(epoch, 0xFFFFFFFF))[:6]
    if stamp[0] < 1980:
        stamp = (1980, 1, 1, 0, 0, 0)
    elif stamp[0] > 2107:
        stamp = (2107, 12, 31, 23, 59, 58)
    with zipfile.ZipFile(path) as source:
        entries = [
            (info.filename, source.read(info))
            for info in source.infolist()
            if not info.is_dir()
        ]
    entries.sort(key=lambda item: item[0])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as dest:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            dest.writestr(info, data)
    path.write_bytes(buffer.getvalue())


# ---------------------------------------------------------------------------
# PEP 517 hooks
# ---------------------------------------------------------------------------


def _run_build(directory, build, finalize):
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    before = {entry.name for entry in target_dir.iterdir()}
    artifact = None
    try:
        version = _validate_source_tree()
        filename = build()
        artifact = target_dir / filename
        if not artifact.is_file():
            raise ReleaseBuildError(f"build did not produce {filename}")
        finalize(artifact, version)
        return filename
    except Exception:
        if artifact is not None:
            artifact.unlink(missing_ok=True)
        for name in {entry.name for entry in target_dir.iterdir()} - before:
            candidate = target_dir / name
            if candidate.is_file() and (
                candidate.suffix == ".whl" or candidate.name.endswith(".tar.gz")
            ):
                candidate.unlink(missing_ok=True)
        raise


def build_sdist(sdist_directory, config_settings=None):
    return _run_build(
        sdist_directory,
        lambda: _setuptools.build_sdist(sdist_directory, config_settings),
        lambda artifact, version: (
            _normalize_sdist(artifact),
            _validate_sdist(artifact, version),
        ),
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _run_build(
        wheel_directory,
        lambda: _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory),
        lambda artifact, version: (
            _normalize_wheel(artifact),
            _validate_wheel(artifact, version),
        ),
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    # Editable installs intentionally reference the source tree, so only the
    # source-tree invariants are enforced; archive normalization is skipped.
    _validate_source_tree()
    return _setuptools.build_editable(wheel_directory, config_settings, metadata_directory)
