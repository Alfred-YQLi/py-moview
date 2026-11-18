"""Orbital rendering pipeline and caches, independent of Qt.

``RenderService`` turns ``(spin, orbital, grid, margin, isovalue)`` requests into
isosurface meshes.  It is safe to call from worker threads:

* fields are computed on demand with the separable evaluator
  (``core.grid``): ~10-120 ms per orbital, no basis-function cache needed;
* scalar fields are cached per ``(spin, orbital, grid, margin)``, so changing
  only the isovalue re-runs marching cubes and nothing else;
* surfaces are cached per isovalue as well; both share one memory budget;
* only the orbitals next to the one on screen are pre-rendered, so A / D
  stepping is instant without spending memory on a large frontier window;
* everything is tagged with a wavefunction *generation* so work started for a
  previous file can never populate the caches of the current one.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from ..config import ResourceSettings
from ..core.grid import GridSpec, OrbitalGrid, SeparableTerms, compute_orbital_grids, make_grid_spec
from ..core.surface import SurfaceMesh, extract_isosurfaces
from ..core.wavefunction import Wavefunction

# Orbitals on each side of the displayed one that are pre-rendered (surfaces only).
PREFETCH_NEIGHBORS = 1
# Full scalar fields kept: enough for a 9-view comparison plus a little history.
FIELD_CACHE_ENTRIES = 12


class Cancelled(Exception):
    """Raised inside a job whose result is no longer wanted."""


@dataclass
class RenderResult:
    spin: str
    index: int
    spec: GridSpec
    grid: OrbitalGrid | None  # scalar field, if it is still cached
    positive: SurfaceMesh
    negative: SurfaceMesh
    level: float

    @property
    def n_faces(self) -> int:
        return self.positive.n_faces + self.negative.n_faces


class ByteLRU:
    """Thread-safe LRU cache bounded by total bytes and entry count."""

    def __init__(self, max_bytes: int, max_entries: int | None = None):
        self.max_bytes = int(max_bytes)
        self.max_entries = max_entries
        self._items: OrderedDict[Hashable, tuple[object, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def get(self, key: Hashable):
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            self._items.move_to_end(key)
            return item[0]

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key: Hashable, value: object, nbytes: int) -> bool:
        with self._lock:
            self.pop(key)
            if nbytes > self.max_bytes:
                return False
            self._items[key] = (value, int(nbytes))
            self._bytes += int(nbytes)
            while self._items and (
                self._bytes > self.max_bytes or (self.max_entries is not None and len(self._items) > self.max_entries)
            ):
                _key, (_value, size) = self._items.popitem(last=False)
                self._bytes -= size
            return key in self._items

    def pop(self, key: Hashable) -> None:
        with self._lock:
            item = self._items.pop(key, None)
            if item is not None:
                self._bytes -= item[1]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def remove_where(self, predicate: Callable[[Hashable], bool]) -> None:
        with self._lock:
            for key in [key for key in self._items if predicate(key)]:
                self.pop(key)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)


def _margin_key(margin: float) -> float:
    return round(float(margin), 6)


def _iso_key(iso: float) -> float:
    return float(f"{float(iso):.9g}")


class RenderService:
    def __init__(self, resources: ResourceSettings):
        self.resources = resources
        self.wavefunction: Wavefunction | None = None
        self.generation = 0
        self._specs: dict[tuple, GridSpec] = {}
        self._terms: tuple[int, SeparableTerms] | None = None
        self._lock = threading.RLock()
        # Surfaces are small and cached generously; full scalar fields only for
        # the few orbitals on screen (they make isovalue changes instant and
        # are cheap to recompute otherwise).
        self._surface_cache = ByteLRU(resources.render_cache_bytes, resources.render_cache_entries)
        self._field_cache = ByteLRU(resources.render_cache_bytes, FIELD_CACHE_ENTRIES)

    # ----------------------------------------------------------- lifecycle
    def load(self, wavefunction: Wavefunction) -> int:
        with self._lock:
            self.generation += 1
            self.wavefunction = wavefunction
            self._specs.clear()
            self._terms = None
            self._surface_cache.clear()
            self._field_cache.clear()
            return self.generation

    def is_current(self, generation: int) -> bool:
        return generation == self.generation and self.wavefunction is not None

    def grid_spec(self, grid_size: int, margin: float) -> GridSpec:
        assert self.wavefunction is not None
        key = (self.generation, int(grid_size), _margin_key(margin))
        with self._lock:
            spec = self._specs.get(key)
            if spec is None:
                spec = make_grid_spec(self.wavefunction, grid_size, margin)
                self._specs[key] = spec
            return spec

    # -------------------------------------------------------------- caches
    def _field_key(self, generation, spin, index, grid_size, margin):
        return ("field", generation, spin, int(index), int(grid_size), _margin_key(margin))

    def _surface_key(self, generation, spin, index, grid_size, margin, iso):
        return ("surface", generation, spin, int(index), int(grid_size), _margin_key(margin), _iso_key(iso))

    def cached_result(self, spin: str, index: int, grid_size: int, margin: float, iso: float) -> RenderResult | None:
        generation = self.generation
        surfaces = self._surface_cache.get(self._surface_key(generation, spin, index, grid_size, margin, iso))
        if surfaces is None:
            return None
        grid = self._field_cache.get(self._field_key(generation, spin, index, grid_size, margin))
        positive, negative = surfaces
        return RenderResult(spin, int(index), self.grid_spec(grid_size, margin), grid, positive, negative, float(iso))

    def has_field(self, spin: str, index: int, grid_size: int, margin: float) -> bool:
        return self._field_key(self.generation, spin, index, grid_size, margin) in self._field_cache

    @property
    def cache_bytes(self) -> int:
        return self._surface_cache.nbytes + self._field_cache.nbytes

    # ------------------------------------------------------------- compute
    def _separable_terms(self, generation: int, wavefunction: Wavefunction) -> SeparableTerms:
        """Primitive/component table of the loaded basis, built once per file."""
        with self._lock:
            if self._terms is not None and self._terms[0] == generation:
                return self._terms[1]
            terms = SeparableTerms.from_wavefunction(wavefunction)
            if self.is_current(generation):
                self._terms = (generation, terms)
            return terms

    def render(
        self,
        generation: int,
        spin: str,
        indices: Sequence[int],
        grid_size: int,
        margin: float,
        iso: float,
        cancelled: Callable[[], bool] = lambda: False,
        *,
        keep_fields: bool = True,
    ) -> list[RenderResult]:
        """Compute (or fetch) isosurfaces for ``indices``; results keep the input order.

        ``keep_fields=False`` (background pre-render) caches only the surfaces.
        """
        wavefunction = self.wavefunction
        if wavefunction is None or not self.is_current(generation):
            raise Cancelled()
        spec = self.grid_spec(grid_size, margin)
        order = [int(i) for i in indices]
        surfaces: dict[int, tuple[SurfaceMesh, SurfaceMesh]] = {}
        grids: dict[int, OrbitalGrid] = {}
        for index in dict.fromkeys(order):
            cached = self._surface_cache.get(self._surface_key(generation, spin, index, grid_size, margin, iso))
            if cached is not None:
                surfaces[index] = cached
            grid = self._field_cache.get(self._field_key(generation, spin, index, grid_size, margin))
            if grid is not None:
                grids[index] = grid
        missing = [i for i in dict.fromkeys(order) if i not in surfaces and i not in grids]
        if missing:
            if cancelled():
                raise Cancelled()
            computed = compute_orbital_grids(
                wavefunction,
                spin,
                missing,
                spec,
                dtype=np.float32,
                chunk_points=self.resources.grid_chunk_points,
                terms=self._separable_terms(generation, wavefunction),
            )
            for grid in computed:
                grids[grid.orbital_index0] = grid
        current = self.is_current(generation)
        for index in dict.fromkeys(order):
            if index not in surfaces:
                if cancelled():
                    raise Cancelled()
                positive, negative, _level = extract_isosurfaces(grids[index], iso)
                surfaces[index] = (positive, negative)
                if current:
                    key = self._surface_key(generation, spin, index, grid_size, margin, iso)
                    self._surface_cache.put(key, surfaces[index], positive.nbytes + negative.nbytes)
            if keep_fields and current and index in grids:
                self._field_cache.put(self._field_key(generation, spin, index, grid_size, margin), grids[index], grids[index].nbytes)
        return [
            RenderResult(spin, index, spec, grids.get(index) if keep_fields else None, *surfaces[index], float(iso))
            for index in order
        ]

    # ------------------------------------------------------------ prefetch
    def neighbors_to_prefetch(self, spin: str, grid_size: int, margin: float, iso: float, displayed: Sequence[int]) -> list[int]:
        """Uncached orbitals next to a single displayed orbital (higher one first)."""
        wavefunction = self.wavefunction
        if wavefunction is None or len(displayed) != 1:
            return []
        center = int(displayed[0])
        candidates = []
        for distance in range(1, PREFETCH_NEIGHBORS + 1):
            candidates += [center + distance, center - distance]
        return [
            index
            for index in candidates
            if 0 <= index < wavefunction.n_orbitals(spin) and self.cached_result(spin, index, grid_size, margin, iso) is None
        ]


def estimate_points_text(spec: GridSpec) -> str:
    nx, ny, nz = spec.shape
    return f"{nx}×{ny}×{nz} = {spec.n_points:,} points"


def format_bytes(nbytes: float) -> str:
    value = float(max(0.0, nbytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit in ("B", "KiB") else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"  # pragma: no cover


def high_grid_warning_text(spec: GridSpec) -> str:
    return (
        f"Grid {spec.grid_size} samples {estimate_points_text(spec)}.\n\n"
        f"Each orbital field needs about {format_bytes(spec.scalar_nbytes(np.float32))} before surface "
        "extraction; the calculation may be slow and use substantial extra memory.\n\n"
        "Continue with this resolution?"
    )
