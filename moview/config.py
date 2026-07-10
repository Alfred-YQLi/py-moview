from __future__ import annotations

import configparser
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_ISOVALUE, DEFAULT_PREFETCH_WORKERS
from .grid import DEFAULT_GRID_CHUNK_POINTS, MIN_GRID_SIZE


CONFIG_FILENAME = "moview.ini"
MIB = 1024**2

SURFACE_STYLE_VALUES = frozenset(("glass", "solid", "wireframe", "solid_edges"))
ATOM_STYLE_VALUES = frozenset(("ball_stick", "space_filling", "licorice"))
ATOM_LABEL_VALUES = frozenset(("off", "number", "element", "number_element"))
ATOM_LABEL_PLACEMENT_VALUES = frozenset(("attached", "floating"))

DEFAULT_SURFACE_STYLE = "glass"
DEFAULT_ATOM_STYLE = "ball_stick"
DEFAULT_ATOM_LABEL_MODE = "off"
DEFAULT_ATOM_LABEL_PLACEMENT = "attached"


class ConfigError(ValueError):
    """Raised when a MOview configuration file is invalid."""


@dataclass(frozen=True)
class ColorPreset:
    name: str
    rgb: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name != self.name.strip()
            or len(self.name) > 40
            or any(ord(char) < 32 for char in self.name)
        ):
            raise ConfigError(f"Invalid color name: {self.name!r}")
        if len(self.rgb) != 3 or any(
            not isinstance(channel, int) or channel < 0 or channel > 255
            for channel in self.rgb
        ):
            raise ConfigError(f"Invalid RGB value for {self.name!r}: {self.rgb!r}")

    @property
    def hex_color(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)


DEFAULT_COLOR_PRESETS = (
    ColorPreset("Red", (239, 59, 44)),
    ColorPreset("Blue", (37, 99, 235)),
    ColorPreset("Violet", (139, 92, 246)),
    ColorPreset("Cyan", (6, 182, 212)),
    ColorPreset("Orange", (249, 115, 22)),
    ColorPreset("Green", (34, 197, 94)),
    ColorPreset("Magenta", (236, 72, 153)),
    ColorPreset("Gold", (234, 179, 8)),
    ColorPreset("Teal", (20, 184, 166)),
)


@dataclass(frozen=True)
class ResourceSettings:
    basis_workers: int = DEFAULT_PREFETCH_WORKERS
    background_jobs: int = 1
    basis_cache_mib: int = 768
    max_basis_cache_entry_mib: int = 640
    render_cache_mib: int = 512
    render_cache_entries: int = 240
    prefetch_field_budget_mib: int = 192
    max_prefetch_orbitals: int = 48
    grid_chunk_points: int = DEFAULT_GRID_CHUNK_POINTS
    surface_face_limit: int = 160_000

    @property
    def basis_cache_bytes(self) -> int:
        return self.basis_cache_mib * MIB

    @property
    def max_basis_cache_entry_bytes(self) -> int:
        return self.max_basis_cache_entry_mib * MIB

    @property
    def render_cache_bytes(self) -> int:
        return self.render_cache_mib * MIB

    @property
    def prefetch_field_budget_bytes(self) -> int:
        return self.prefetch_field_budget_mib * MIB


@dataclass(frozen=True)
class RenderSettings:
    grid: int = 81
    margin_bohr: float = 4.0
    isovalue: float = DEFAULT_ISOVALUE
    surface_style: str = DEFAULT_SURFACE_STYLE
    atom_style: str = DEFAULT_ATOM_STYLE
    atom_scale: float = 1.0
    label_mode: str = DEFAULT_ATOM_LABEL_MODE
    label_placement: str = DEFAULT_ATOM_LABEL_PLACEMENT
    label_size: int = 12
    positive_color: str = "Red"
    negative_color: str = "Blue"


@dataclass(frozen=True)
class ViewSettings:
    zoom: float = 1.0
    sync_views: bool = True
    show_axes: bool = True
    auto_render: bool = True


@dataclass(frozen=True)
class AppConfig:
    path: Path | None = None
    resources: ResourceSettings = ResourceSettings()
    render: RenderSettings = RenderSettings()
    view: ViewSettings = ViewSettings()
    colors: tuple[ColorPreset, ...] = DEFAULT_COLOR_PRESETS

    def color(self, name: str) -> ColorPreset:
        requested = name.casefold()
        for preset in self.colors:
            if preset.name.casefold() == requested:
                return preset
        raise ConfigError(f"Unknown color preset: {name!r}")


DEFAULT_CONFIG = AppConfig()

_RESOURCE_KEYS = frozenset(
    (
        "basis_workers",
        "background_jobs",
        "basis_cache_mib",
        "max_basis_cache_entry_mib",
        "render_cache_mib",
        "render_cache_entries",
        "prefetch_field_budget_mib",
        "max_prefetch_orbitals",
        "grid_chunk_points",
        "surface_face_limit",
    )
)
_RENDER_KEYS = frozenset(
    (
        "grid",
        "margin_bohr",
        "isovalue",
        "surface_style",
        "atom_style",
        "atom_scale",
        "label_mode",
        "label_placement",
        "label_size",
        "positive_color",
        "negative_color",
    )
)
_VIEW_KEYS = frozenset(("zoom", "sync_views", "show_axes", "auto_render"))
_SECTIONS = frozenset(("resources", "render", "view", "colors"))


def _specified_path(raw_path: str | Path, cwd: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def discover_config_path(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Path | None:
    env = os.environ if environ is None else environ
    working_dir = Path.cwd() if cwd is None else Path(cwd)

    if explicit is not None:
        path = _specified_path(explicit, working_dir)
        if not path.is_file():
            raise ConfigError(f"Configuration file does not exist: {path}")
        return path

    env_path = env.get("MOVIEW_CONFIG", "").strip()
    if env_path:
        path = _specified_path(env_path, working_dir)
        if not path.is_file():
            raise ConfigError(f"MOVIEW_CONFIG does not point to a file: {path}")
        return path

    home = Path(env.get("HOME", str(Path.home()))).expanduser()
    xdg_root = Path(env.get("XDG_CONFIG_HOME", home / ".config")).expanduser()
    candidates = [working_dir / CONFIG_FILENAME]
    if sys.platform == "darwin":
        candidates.append(home / "Library" / "Application Support" / "MOview" / "config.ini")
    candidates.extend(
        (
            xdg_root / "moview" / "config.ini",
            Path(__file__).resolve().parent.parent / CONFIG_FILENAME,
        )
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _section_names(parser: configparser.ConfigParser, path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for raw_name in parser.sections():
        name = raw_name.casefold()
        if name not in _SECTIONS:
            raise ConfigError(f"{path}: unknown section [{raw_name}]")
        if name in names:
            raise ConfigError(f"{path}: duplicate section [{raw_name}]")
        names[name] = raw_name
    return names


def _section_values(
    parser: configparser.ConfigParser,
    section_names: Mapping[str, str],
    section: str,
    allowed: frozenset[str],
    path: Path,
) -> dict[str, str]:
    raw_section = section_names.get(section)
    if raw_section is None:
        return {}
    values: dict[str, str] = {}
    for raw_key, raw_value in parser.items(raw_section, raw=True):
        key = raw_key.casefold()
        if key not in allowed:
            raise ConfigError(f"{path}: unknown setting [{raw_section}] {raw_key}")
        if key in values:
            raise ConfigError(f"{path}: duplicate setting [{raw_section}] {raw_key}")
        values[key] = raw_value.strip()
    return values


def _parse_int(
    values: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    section: str,
    path: Path,
) -> int:
    raw = values.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{path}: [{section}] {key} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ConfigError(
            f"{path}: [{section}] {key} must be between {minimum} and {maximum}"
        )
    return value


def _parse_float(
    values: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
    section: str,
    path: Path,
) -> float:
    raw = values.get(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{path}: [{section}] {key} must be numeric") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ConfigError(
            f"{path}: [{section}] {key} must be finite and between {minimum:g} and {maximum:g}"
        )
    return value


def _parse_bool(
    values: Mapping[str, str],
    key: str,
    default: bool,
    *,
    section: str,
    path: Path,
) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    normalized = raw.casefold()
    if normalized not in configparser.ConfigParser.BOOLEAN_STATES:
        raise ConfigError(f"{path}: [{section}] {key} must be true or false")
    return bool(configparser.ConfigParser.BOOLEAN_STATES[normalized])


def _parse_choice(
    values: Mapping[str, str],
    key: str,
    default: str,
    choices: frozenset[str],
    *,
    section: str,
    path: Path,
) -> str:
    value = values.get(key, default).casefold()
    if value not in choices:
        options = ", ".join(sorted(choices))
        raise ConfigError(f"{path}: [{section}] {key} must be one of: {options}")
    return value


def _parse_rgb(raw: str, name: str, path: Path) -> tuple[int, int, int]:
    parts = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    if len(parts) != 3:
        raise ConfigError(f"{path}: color {name!r} must use R, G, B")
    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ConfigError(f"{path}: color {name!r} channels must be integers") from exc
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ConfigError(f"{path}: color {name!r} channels must be between 0 and 255")
    return rgb


def _parse_colors(
    parser: configparser.ConfigParser,
    section_names: Mapping[str, str],
    path: Path,
) -> tuple[ColorPreset, ...]:
    colors = list(DEFAULT_COLOR_PRESETS)
    raw_section = section_names.get("colors")
    if raw_section is None:
        return tuple(colors)
    for raw_name, raw_value in parser.items(raw_section, raw=True):
        name = raw_name.strip()
        if not name or len(name) > 40 or any(ord(char) < 32 for char in name):
            raise ConfigError(f"{path}: invalid color name {raw_name!r}")
        preset = ColorPreset(name, _parse_rgb(raw_value, name, path))
        match = next(
            (index for index, current in enumerate(colors) if current.name.casefold() == name.casefold()),
            None,
        )
        if match is None:
            colors.append(preset)
        else:
            colors[match] = preset
    return tuple(colors)


def _canonical_color_name(name: str, colors: tuple[ColorPreset, ...], path: Path) -> str:
    requested = name.strip().casefold()
    for preset in colors:
        if preset.name.casefold() == requested:
            return preset.name
    options = ", ".join(preset.name for preset in colors)
    raise ConfigError(f"{path}: unknown color {name!r}; available colors: {options}")


def load_config(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> AppConfig:
    path = discover_config_path(explicit, environ=environ, cwd=cwd)
    if path is None:
        return DEFAULT_CONFIG

    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        strict=True,
    )
    parser.optionxform = str
    try:
        with path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
        raise ConfigError(f"Failed to read configuration {path}: {exc}") from exc

    sections = _section_names(parser, path)
    resource_values = _section_values(
        parser,
        sections,
        "resources",
        _RESOURCE_KEYS,
        path,
    )
    render_values = _section_values(parser, sections, "render", _RENDER_KEYS, path)
    view_values = _section_values(parser, sections, "view", _VIEW_KEYS, path)
    colors = _parse_colors(parser, sections, path)

    resources = ResourceSettings(
        basis_workers=_parse_int(
            resource_values,
            "basis_workers",
            DEFAULT_CONFIG.resources.basis_workers,
            minimum=1,
            maximum=256,
            section="resources",
            path=path,
        ),
        background_jobs=_parse_int(
            resource_values,
            "background_jobs",
            DEFAULT_CONFIG.resources.background_jobs,
            minimum=1,
            maximum=8,
            section="resources",
            path=path,
        ),
        basis_cache_mib=_parse_int(
            resource_values,
            "basis_cache_mib",
            DEFAULT_CONFIG.resources.basis_cache_mib,
            minimum=16,
            maximum=1_048_576,
            section="resources",
            path=path,
        ),
        max_basis_cache_entry_mib=_parse_int(
            resource_values,
            "max_basis_cache_entry_mib",
            DEFAULT_CONFIG.resources.max_basis_cache_entry_mib,
            minimum=16,
            maximum=1_048_576,
            section="resources",
            path=path,
        ),
        render_cache_mib=_parse_int(
            resource_values,
            "render_cache_mib",
            DEFAULT_CONFIG.resources.render_cache_mib,
            minimum=16,
            maximum=1_048_576,
            section="resources",
            path=path,
        ),
        render_cache_entries=_parse_int(
            resource_values,
            "render_cache_entries",
            DEFAULT_CONFIG.resources.render_cache_entries,
            minimum=1,
            maximum=100_000,
            section="resources",
            path=path,
        ),
        prefetch_field_budget_mib=_parse_int(
            resource_values,
            "prefetch_field_budget_mib",
            DEFAULT_CONFIG.resources.prefetch_field_budget_mib,
            minimum=16,
            maximum=1_048_576,
            section="resources",
            path=path,
        ),
        max_prefetch_orbitals=_parse_int(
            resource_values,
            "max_prefetch_orbitals",
            DEFAULT_CONFIG.resources.max_prefetch_orbitals,
            minimum=1,
            maximum=100_000,
            section="resources",
            path=path,
        ),
        grid_chunk_points=_parse_int(
            resource_values,
            "grid_chunk_points",
            DEFAULT_CONFIG.resources.grid_chunk_points,
            minimum=1_024,
            maximum=16_777_216,
            section="resources",
            path=path,
        ),
        surface_face_limit=_parse_int(
            resource_values,
            "surface_face_limit",
            DEFAULT_CONFIG.resources.surface_face_limit,
            minimum=1_000,
            maximum=100_000_000,
            section="resources",
            path=path,
        ),
    )
    if resources.max_basis_cache_entry_mib > resources.basis_cache_mib:
        raise ConfigError(
            f"{path}: max_basis_cache_entry_mib cannot exceed basis_cache_mib"
        )

    render = RenderSettings(
        grid=_parse_int(
            render_values,
            "grid",
            DEFAULT_CONFIG.render.grid,
            minimum=MIN_GRID_SIZE,
            maximum=2_147_483_647,
            section="render",
            path=path,
        ),
        margin_bohr=_parse_float(
            render_values,
            "margin_bohr",
            DEFAULT_CONFIG.render.margin_bohr,
            minimum=0.0,
            maximum=30.0,
            section="render",
            path=path,
        ),
        isovalue=_parse_float(
            render_values,
            "isovalue",
            DEFAULT_CONFIG.render.isovalue,
            minimum=1.0e-8,
            maximum=1.0e8,
            section="render",
            path=path,
        ),
        surface_style=_parse_choice(
            render_values,
            "surface_style",
            DEFAULT_CONFIG.render.surface_style,
            SURFACE_STYLE_VALUES,
            section="render",
            path=path,
        ),
        atom_style=_parse_choice(
            render_values,
            "atom_style",
            DEFAULT_CONFIG.render.atom_style,
            ATOM_STYLE_VALUES,
            section="render",
            path=path,
        ),
        atom_scale=_parse_float(
            render_values,
            "atom_scale",
            DEFAULT_CONFIG.render.atom_scale,
            minimum=0.35,
            maximum=2.2,
            section="render",
            path=path,
        ),
        label_mode=_parse_choice(
            render_values,
            "label_mode",
            DEFAULT_CONFIG.render.label_mode,
            ATOM_LABEL_VALUES,
            section="render",
            path=path,
        ),
        label_placement=_parse_choice(
            render_values,
            "label_placement",
            DEFAULT_CONFIG.render.label_placement,
            ATOM_LABEL_PLACEMENT_VALUES,
            section="render",
            path=path,
        ),
        label_size=_parse_int(
            render_values,
            "label_size",
            DEFAULT_CONFIG.render.label_size,
            minimum=8,
            maximum=28,
            section="render",
            path=path,
        ),
        positive_color=_canonical_color_name(
            render_values.get("positive_color", DEFAULT_CONFIG.render.positive_color),
            colors,
            path,
        ),
        negative_color=_canonical_color_name(
            render_values.get("negative_color", DEFAULT_CONFIG.render.negative_color),
            colors,
            path,
        ),
    )
    view = ViewSettings(
        zoom=_parse_float(
            view_values,
            "zoom",
            DEFAULT_CONFIG.view.zoom,
            minimum=0.35,
            maximum=3.0,
            section="view",
            path=path,
        ),
        sync_views=_parse_bool(
            view_values,
            "sync_views",
            DEFAULT_CONFIG.view.sync_views,
            section="view",
            path=path,
        ),
        show_axes=_parse_bool(
            view_values,
            "show_axes",
            DEFAULT_CONFIG.view.show_axes,
            section="view",
            path=path,
        ),
        auto_render=_parse_bool(
            view_values,
            "auto_render",
            DEFAULT_CONFIG.view.auto_render,
            section="view",
            path=path,
        ),
    )
    return AppConfig(path=path, resources=resources, render=render, view=view, colors=colors)
