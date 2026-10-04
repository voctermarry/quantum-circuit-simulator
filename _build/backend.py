"""In-tree PEP 517 build backend for quantum-circuit-simulator.

Wraps :mod:`setuptools.build_meta` with the release guarantees documented in
README.md:

* the source tree is validated before anything is built — README, package
  directory, console-entry module, pyproject/version consistency and UTF-8
  metadata;
* the produced sdist/wheel is validated — no archive members escaping the
  project root, no duplicate members, UTF-8 metadata, and no tests, caches,
  temporary files or local absolute paths inside the wheel;
* both artifacts are normalised so that identical source, Python minor
  version and ``SOURCE_DATE_EPOCH`` produce byte-identical files, and a
  wheel rebuilt from the sdist matches a directly built wheel;
* any failure removes the half-built artifact from the output directory and
  aborts the build with a non-zero exit status.

Every other hook (including PEP 660 editable installs) is delegated to
setuptools unchanged via the module-level ``__getattr__``.
"""

from __future__ import annotations

import ast
import gzip
import io
import os
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path, PurePosixPath

from setuptools import build_meta as _setuptools_backend

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Timestamp used inside archives when SOURCE_DATE_EPOCH is not provided.
_FALLBACK_EPOCH = 0
#: ZIP timestamps cannot predate 1980-01-01.
_ZIP_EPOCH_FLOOR = 315532800

_VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']\s*(?:#.*)?$""", re.MULTILINE)
#: Wheel filename escaping (PEP 427): runs of characters outside [a-zA-Z0-9_.]
#: collapse to a single underscore; dots are preserved.
_ESCAPE_RE = re.compile(r"[^\w\d.]+", re.UNICODE)
_TEMP_SUFFIXES = ("~", ".tmp", ".temp", ".bak", ".swp", ".pyc", ".pyo", ".orig", ".rej")
_CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


class BuildValidationError(RuntimeError):
    """The source tree or a built artifact failed a release check."""


def _fail(message: str) -> None:
    raise BuildValidationError(message)


# ---------------------------------------------------------------------------
# source tree validation
# ---------------------------------------------------------------------------


def _epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None:
        return _FALLBACK_EPOCH
    try:
        value = int(raw, 10)
    except ValueError:
        _fail(f"SOURCE_DATE_EPOCH is not a decimal integer: {raw!r}")
    if value < 0:
        _fail(f"SOURCE_DATE_EPOCH must not be negative: {value}")
    return value


def _read_utf8(path: Path, what: str) -> str:
    if not path.is_file():
        _fail(f"missing {what}: {path.relative_to(PROJECT_ROOT)}")
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{what} is not valid UTF-8: {path.relative_to(PROJECT_ROOT)}")


def _escape(value: str) -> str:
    return _ESCAPE_RE.sub("_", value)


def _entry_point_defined(source: str, attr: str, module: str) -> bool:
    top = attr.split(".", 1)[0]
    tree = ast.parse(source, filename=module)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == top:
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == top for t in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == top:
                return True
    return False


def _validate_project() -> dict:
    """Validate the source tree; return the checked release facts."""
    import tomllib

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    pyproject_text = _read_utf8(pyproject_path, "project metadata (pyproject.toml)")
    try:
        data = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError as exc:
        _fail(f"project metadata (pyproject.toml) is not valid TOML: {exc}")

    project = data.get("project")
    if not isinstance(project, dict):
        _fail("pyproject.toml has no [project] table")

    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        _fail("project metadata declares no usable project name")

    # The package directory (or directories) that must ship in the artifacts.
    tool = data.get("tool") or {}
    setuptools_cfg = tool.get("setuptools") or {}
    packages = setuptools_cfg.get("packages")
    if packages is None:
        packages = ["quantum_circuit"]
    if not isinstance(packages, list) or not packages or not all(
        isinstance(p, str) and p for p in packages
    ):
        _fail("tool.setuptools.packages must be a non-empty list of package names")

    for package in packages:
        package_dir = PROJECT_ROOT.joinpath(*package.split("."))
        if not package_dir.is_dir():
            _fail(f"missing package directory: {package.replace('.', '/')}")
        init = package_dir / "__init__.py"
        _read_utf8(init, f"package initialiser ({package}.__init__)")

    # Version consistency: the static metadata version must match the
    # package's __version__ exactly.
    root_package = packages[0]
    init_text = _read_utf8(
        PROJECT_ROOT.joinpath(*root_package.split("."), "__init__.py"),
        f"package initialiser ({root_package}.__init__)",
    )
    match = _VERSION_RE.search(init_text)
    if match is None:
        _fail(f"{root_package}.__init__ defines no __version__")
    package_version = match.group(1)

    meta_version = project.get("version")
    dynamic = project.get("dynamic") or []
    if meta_version is None:
        if "version" not in dynamic:
            _fail("project metadata declares no version")
    elif meta_version != package_version:
        _fail(
            f"project metadata version {meta_version!r} does not match "
            f"{root_package}.__version__ {package_version!r}"
        )

    # README (declared via project.readme, defaulting to README.md).
    readme = project.get("readme")
    if isinstance(readme, str):
        readme_name = readme
    elif isinstance(readme, dict) and isinstance(readme.get("file"), str):
        readme_name = readme["file"]
    else:
        readme_name = "README.md"
    _read_utf8(PROJECT_ROOT / readme_name, "README")

    # Console entry points: the referenced module file must exist and define
    # the advertised attribute.
    scripts = project.get("scripts") or {}
    if not isinstance(scripts, dict):
        _fail("project.scripts must be a table")
    for script, target in scripts.items():
        if not isinstance(target, str):
            _fail(f"entry point {script!r} must be a string")
        module, _, attr = target.partition(":")
        module_path = PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
        source = _read_utf8(module_path, f"entry point module ({module})")
        if attr and not _entry_point_defined(source, attr, module):
            _fail(f"entry point {script!r}: {module} defines no {attr.split('.', 1)[0]!r}")

    return {
        "name": name,
        "version": package_version,
        "packages": packages,
        "readme": readme_name,
        "scripts": scripts,
    }


# ---------------------------------------------------------------------------
# artifact validation
# ---------------------------------------------------------------------------


def _check_member_names(names, kind: str) -> None:
    seen = set()
    for name in names:
        if not name or name != name.strip():
            _fail(f"{kind} contains an invalid member name: {name!r}")
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            _fail(f"{kind} contains a non-UTF-8 member name: {name!r}")
        pure = PurePosixPath(name)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in name
            or (len(name) > 1 and name[1] == ":")
        ):
            _fail(f"{kind} member escapes the project directory: {name!r}")
        if name in seen:
            _fail(f"{kind} contains a duplicate member: {name!r}")
        seen.add(name)
        parts = PurePosixPath(name).parts
        if _CACHE_PARTS.intersection(parts):
            _fail(f"{kind} contains a cache member: {name!r}")
        if "tests" in parts:
            _fail(f"{kind} contains test files: {name!r}")
        if name.endswith(_TEMP_SUFFIXES):
            _fail(f"{kind} contains a temporary or compiled file: {name!r}")


def _decode_utf8(data: bytes, what: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{what} is not valid UTF-8")


def _check_metadata_text(text: str, facts: dict, what: str) -> None:
    found = {}
    for line in text.splitlines():
        if not line.strip():
            break
        key, _, value = line.partition(":")
        if key.strip() in ("Name", "Version"):
            found[key.strip()] = value.strip()
    if found.get("Name") != facts["name"]:
        _fail(f"{what} Name {found.get('Name')!r} != {facts['name']!r}")
    if found.get("Version") != facts["version"]:
        _fail(f"{what} Version {found.get('Version')!r} != {facts['version']!r}")


def _expected_package_files(facts: dict) -> set:
    expected = set()
    for package in facts["packages"]:
        package_dir = PROJECT_ROOT.joinpath(*package.split("."))
        for path in sorted(package_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            expected.add(path.relative_to(PROJECT_ROOT).as_posix())
    return expected


def _check_sdist(path: Path, facts: dict) -> None:
    try:
        archive = tarfile.open(path, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        _fail(f"sdist is not a readable gzip tarball: {exc}")
    with archive:
        members = archive.getmembers()
        for member in members:
            if not (member.isfile() or member.isdir()):
                _fail(f"sdist contains an unsupported member type: {member.name!r}")
        names = [member.name for member in members]
        _check_member_names(names, "sdist")
        prefixes = {name.split("/", 1)[0] for name in names}
        if len(prefixes) != 1:
            _fail("sdist members do not share a single top-level directory")
        prefix = prefixes.pop()
        contained = set()
        pkg_info = None
        for member in members:
            if not member.isfile():
                continue
            relative = member.name[len(prefix) + 1 :] if member.name.startswith(prefix + "/") else member.name
            contained.add(relative)
            if relative == "PKG-INFO":
                pkg_info = _decode_utf8(archive.extractfile(member).read(), "sdist PKG-INFO")
    if pkg_info is None:
        _fail("sdist contains no PKG-INFO metadata")
    _check_metadata_text(pkg_info, facts, "sdist PKG-INFO")

    expected = {"pyproject.toml", facts["readme"], "MANIFEST.in", "_build/backend.py"}
    expected |= _expected_package_files(facts)
    missing = sorted(expected - contained)
    if missing:
        _fail(f"sdist is missing required files: {', '.join(missing)}")


def _check_wheel(path: Path, facts: dict) -> None:
    stem = f"{_escape(facts['name'])}-{_escape(facts['version'])}"
    expected_name = f"{stem}-py3-none-any.whl"
    if path.name != expected_name:
        _fail(f"wheel file name {path.name!r} != expected {expected_name!r}")
    dist_info = f"{stem}.dist-info"
    allowed_top = {dist_info}
    for package in facts["packages"]:
        allowed_top.add(package.split(".")[0])

    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        _fail(f"wheel is not a readable zip archive: {exc}")
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        _check_member_names(names, "wheel")
        metadata = entry_points = record = None
        root_marker = str(PROJECT_ROOT)
        for info in infos:
            name = info.filename
            top = name.split("/", 1)[0]
            if top not in allowed_top:
                _fail(f"wheel contains an unexpected top-level member: {name!r}")
            if info.is_dir():
                continue
            data = archive.read(info)
            if name == f"{dist_info}/METADATA":
                metadata = _decode_utf8(data, "wheel METADATA")
            elif name == f"{dist_info}/entry_points.txt":
                entry_points = _decode_utf8(data, "wheel entry_points.txt")
            elif name == f"{dist_info}/RECORD":
                record = _decode_utf8(data, "wheel RECORD")
            if name.startswith(dist_info + "/"):
                text = _decode_utf8(data, f"wheel metadata {name}")
                if root_marker in text:
                    _fail(f"wheel metadata {name} contains a local absolute path")
    if metadata is None:
        _fail("wheel contains no dist-info METADATA")
    if record is None:
        _fail("wheel contains no dist-info RECORD")
    _check_metadata_text(metadata, facts, "wheel METADATA")

    if facts["scripts"]:
        if entry_points is None:
            _fail("wheel declares console scripts but has no entry_points.txt")
        for script, target in facts["scripts"].items():
            if f"{script} = {target}" not in entry_points:
                _fail(f"wheel entry_points.txt is missing {script!r}")

    contained = {name for name in names if not name.endswith("/")}
    missing = sorted(_expected_package_files(facts) - contained)
    if missing:
        _fail(f"wheel is missing package files: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# artifact normalisation (reproducibility)
# ---------------------------------------------------------------------------


def _normalise_sdist(path: Path, epoch: int) -> None:
    with tarfile.open(path, "r:gz") as source:
        members = [
            (member, source.extractfile(member).read() if member.isfile() else None)
            for member in source.getmembers()
        ]
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=epoch) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as out:
            for member, data in sorted(members, key=lambda item: item[0].name):
                info = tarfile.TarInfo(member.name)
                info.mtime = epoch
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                if member.isdir():
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    out.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.mode = 0o755 if member.mode & 0o111 else 0o644
                    info.size = len(data)
                    out.addfile(info, io.BytesIO(data))
    path.write_bytes(buffer.getvalue())


def _normalise_wheel(path: Path, epoch: int) -> None:
    stamp = time.gmtime(max(epoch, _ZIP_EPOCH_FLOOR))[:6]
    with zipfile.ZipFile(path) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for info, data in entries:
            rewritten = zipfile.ZipInfo(info.filename, date_time=stamp)
            rewritten.compress_type = zipfile.ZIP_DEFLATED
            rewritten.create_system = 3
            rewritten.external_attr = (0o755 if info.is_dir() else 0o644) << 16
            out.writestr(rewritten, data)
    path.write_bytes(buffer.getvalue())


# ---------------------------------------------------------------------------
# PEP 517 hooks
# ---------------------------------------------------------------------------


def _remove_new(directory: str, before: set) -> None:
    """Delete anything the failed hook created in the output directory."""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name in before:
            continue
        target = os.path.join(directory, name)
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                os.unlink(target)
            except OSError:
                pass


def _reset_distutils_state() -> None:
    """Clear process-global distutils caches between hook calls.

    ``distutils.dir_util.mkpath`` remembers every directory it created in a
    module-level cache.  ``bdist_wheel`` deletes its staging tree at the end
    of a build, so a second in-process hook call would otherwise trust the
    stale cache and copy files into directories that no longer exist.
    Frontends that run each hook in a fresh subprocess never see this; the
    reset keeps in-process callers (and the test-suite) safe.
    """
    seen = set()
    for module_name in ("setuptools._distutils.dir_util", "distutils.dir_util"):
        try:
            module = __import__(module_name, fromlist=["_path_created"])
        except ImportError:
            continue
        if id(module) in seen:
            continue
        seen.add(id(module))
        cache = getattr(module, "_path_created", None)
        if cache is not None:
            cache.clear()


def _build_artifact(kind, directory, produce, check, normalise):
    """Run a build hook with validation, checking and cleanup-on-failure."""
    facts = _validate_project()
    epoch = _epoch()
    _reset_distutils_state()
    before = set(os.listdir(directory)) if os.path.isdir(directory) else set()
    try:
        name = produce()
    except BaseException:
        _remove_new(directory, before)
        raise
    artifact = Path(directory, name)
    try:
        check(artifact, facts)
        normalise(artifact, epoch)
    except BaseException:
        try:
            artifact.unlink()
        except OSError:
            pass
        _remove_new(directory, before)
        raise
    return name


def build_sdist(sdist_directory, config_settings=None):
    return _build_artifact(
        "sdist",
        sdist_directory,
        lambda: _setuptools_backend.build_sdist(sdist_directory, config_settings),
        _check_sdist,
        _normalise_sdist,
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_artifact(
        "wheel",
        wheel_directory,
        lambda: _setuptools_backend.build_wheel(
            wheel_directory, config_settings, metadata_directory
        ),
        _check_wheel,
        _normalise_wheel,
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    # Editable installs are a development convenience, not a release
    # artifact: validate the source tree and clean up on failure, but skip
    # the release-artifact checks (editable wheels legitimately reference
    # the source tree by absolute path).
    _validate_project()
    _reset_distutils_state()
    before = set(os.listdir(wheel_directory)) if os.path.isdir(wheel_directory) else set()
    try:
        return _setuptools_backend.build_editable(
            wheel_directory, config_settings, metadata_directory
        )
    except BaseException:
        _remove_new(wheel_directory, before)
        raise


def __getattr__(name):
    # Delegate every other PEP 517/660 hook (get_requires_for_build_*,
    # prepare_metadata_for_build_*, ...) to setuptools unchanged.
    return getattr(_setuptools_backend, name)
