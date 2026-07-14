from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


_MISSING_LIBRARY_RE = re.compile(r"^\s*(\S+)\s+=>\s+not found\s*$", re.MULTILINE)

_RPM_PACKAGES = {
    "libGL.so.1": "mesa-libGL",
    "libX11-xcb.so.1": "libX11-xcb",
    "libX11.so.6": "libX11",
    "libxcb-cursor.so.0": "xcb-util-cursor",
    "libxcb-icccm.so.4": "xcb-util-wm",
    "libxcb-image.so.0": "xcb-util-image",
    "libxcb-keysyms.so.1": "xcb-util-keysyms",
    "libxcb-render-util.so.0": "xcb-util-renderutil",
    "libxcb-util.so.1": "xcb-util",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11",
    "libxkbcommon.so.0": "libxkbcommon",
}

_DEB_PACKAGES = {
    "libGL.so.1": "libgl1",
    "libX11-xcb.so.1": "libx11-xcb1",
    "libX11.so.6": "libx11-6",
    "libxcb-cursor.so.0": "libxcb-cursor0",
    "libxcb-icccm.so.4": "libxcb-icccm4",
    "libxcb-image.so.0": "libxcb-image0",
    "libxcb-keysyms.so.1": "libxcb-keysyms1",
    "libxcb-render-util.so.0": "libxcb-render-util0",
    "libxcb-util.so.1": "libxcb-util1",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
    "libxkbcommon.so.0": "libxkbcommon0",
}


def _uses_xcb(environ: Mapping[str, str]) -> bool:
    requested = environ.get("QT_QPA_PLATFORM", "").strip().lower()
    return not requested or requested.partition(":")[0] == "xcb"


def _plugin_candidates(environ: Mapping[str, str]) -> list[Path]:
    candidates: list[Path] = []
    for variable in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        for entry in environ.get(variable, "").split(os.pathsep):
            if not entry:
                continue
            root = Path(entry).expanduser()
            candidates.extend((root / "libqxcb.so", root / "platforms" / "libqxcb.so"))

    try:
        spec = importlib.util.find_spec("PyQt6")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        roots = [Path(entry) for entry in (spec.submodule_search_locations or ())]
        if not roots and spec.origin:
            roots.append(Path(spec.origin).parent)
        candidates.extend(
            root / "Qt6" / "plugins" / "platforms" / "libqxcb.so" for root in roots
        )
    return candidates


def _find_xcb_plugin(environ: Mapping[str, str]) -> Path | None:
    for candidate in _plugin_candidates(environ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _parse_missing_libraries(output: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_MISSING_LIBRARY_RE.findall(output)))


def _missing_shared_libraries(
    plugin: Path, environ: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    ldd = shutil.which("ldd")
    if ldd is None:
        return ()
    environment = dict(os.environ if environ is None else environ)
    environment["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            [ldd, str(plugin)],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    return _parse_missing_libraries(f"{result.stdout}\n{result.stderr}")


def _os_release() -> Mapping[str, str]:
    try:
        return platform.freedesktop_os_release()
    except OSError:
        return {}


def _distribution_ids(release: Mapping[str, str]) -> set[str]:
    identifiers = {release.get("ID", "").lower()}
    identifiers.update(value.lower() for value in release.get("ID_LIKE", "").split())
    identifiers.discard("")
    return identifiers


def _mapped_packages(
    missing: Sequence[str], package_map: Mapping[str, str]
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(package_map[name] for name in missing if name in package_map))


def _installation_hint(missing: Sequence[str], release: Mapping[str, str]) -> list[str]:
    identifiers = _distribution_ids(release)
    version = release.get("VERSION_ID", "")
    if identifiers & {"centos", "rhel", "rocky", "almalinux", "fedora"}:
        packages = _mapped_packages(missing, _RPM_PACKAGES)
        lines: list[str] = []
        distribution = release.get("ID", "").lower()
        if "xcb-util-cursor" in packages and version.startswith("8") and distribution != "fedora":
            lines.append("  sudo dnf install epel-release")
        if packages:
            lines.append(f"  sudo dnf install {' '.join(packages)}")
        for library in missing:
            if library not in _RPM_PACKAGES:
                lines.append(f"  dnf provides '*/{library}'")
        return lines

    if identifiers & {"debian", "ubuntu"}:
        packages = _mapped_packages(missing, _DEB_PACKAGES)
        lines = [f"  sudo apt install {' '.join(packages)}"] if packages else []
        for library in missing:
            if library not in _DEB_PACKAGES:
                lines.append(f"  apt-file search '/{library}'")
        return lines

    if "libxcb-cursor.so.0" in missing:
        return [
            "  Install xcb-util-cursor (RPM systems) or libxcb-cursor0 (Debian systems)."
        ]
    return [f"  Install the system package that provides {library}." for library in missing]


def linux_qt_platform_issue(
    *,
    environ: Mapping[str, str] | None = None,
    host_platform: str | None = None,
) -> str | None:
    environment = os.environ if environ is None else environ
    current_platform = sys.platform if host_platform is None else host_platform
    if not current_platform.startswith("linux") or not _uses_xcb(environment):
        return None

    plugin = _find_xcb_plugin(environment)
    if plugin is None:
        return None
    missing = _missing_shared_libraries(plugin, environment)
    if not missing:
        return None

    lines = [
        "MOview cannot start the Qt X11 (xcb) platform plugin.",
        "Missing system libraries:",
        *(f"  - {library}" for library in missing),
        "",
        "Install the required operating-system packages:",
        *_installation_hint(missing, _os_release()),
        "",
        "These are native Linux libraries and cannot be installed by pip.",
        "After installing them, rerun the same moview command.",
    ]
    return "\n".join(lines)
