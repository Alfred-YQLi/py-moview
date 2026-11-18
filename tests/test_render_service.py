"""Qt-free rendering pipeline: caching, isovalue reuse, staleness, prefetch planning."""

from __future__ import annotations

import unittest
from dataclasses import replace

from moview.config import DEFAULT_CONFIG
from moview.gui.render_service import ByteLRU, Cancelled, RenderService

from helpers import make_wavefunction


class ByteLRUTests(unittest.TestCase):
    def test_evicts_least_recently_used_by_bytes_and_count(self) -> None:
        cache = ByteLRU(max_bytes=100, max_entries=3)
        for key in "abc":
            cache.put(key, key, 30)
        cache.get("a")  # refresh "a"
        cache.put("d", "d", 30)  # over the byte budget: evicts "b"
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "a")
        self.assertFalse(cache.put("huge", 1, 1000))
        self.assertLessEqual(cache.nbytes, 100)


class RenderServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wf = make_wavefunction()
        self.service = RenderService(DEFAULT_CONFIG.resources)
        self.generation = self.service.load(self.wf)

    def test_render_caches_fields_and_surfaces(self) -> None:
        first = self.service.render(self.generation, "alpha", [4, 5], 24, 3.0, 0.05)
        self.assertEqual([r.index for r in first], [4, 5])
        self.assertGreater(first[0].n_faces, 0)
        cached = self.service.cached_result("alpha", 4, 24, 3.0, 0.05)
        self.assertIs(cached.grid, first[0].grid)
        self.assertIs(cached.positive, first[0].positive)

    def test_new_isovalue_reuses_the_scalar_field(self) -> None:
        first = self.service.render(self.generation, "alpha", [4], 24, 3.0, 0.05)[0]
        self.assertIsNone(self.service.cached_result("alpha", 4, 24, 3.0, 0.02))
        second = self.service.render(self.generation, "alpha", [4], 24, 3.0, 0.02)[0]
        self.assertIs(second.grid, first.grid)  # only marching cubes reran
        self.assertGreater(second.positive.n_faces + second.negative.n_faces, 0)

    def test_terms_are_built_once_per_file(self) -> None:
        self.service.render(self.generation, "alpha", [4], 20, 3.0, 0.05)
        terms = self.service._terms
        self.service.render(self.generation, "alpha", [5], 20, 3.0, 0.05)
        self.assertIs(self.service._terms, terms)
        self.service.load(make_wavefunction(seed=2))
        self.assertIsNone(self.service._terms)

    def test_stale_generation_is_cancelled_and_never_cached(self) -> None:
        old = self.generation
        self.service.load(make_wavefunction(seed=9))
        with self.assertRaises(Cancelled):
            self.service.render(old, "alpha", [4], 24, 3.0, 0.05)
        with self.assertRaises(Cancelled):
            self.service.render(self.service.generation, "alpha", [4], 24, 3.0, 0.05, cancelled=lambda: True)
        self.assertIsNone(self.service.cached_result("alpha", 4, 24, 3.0, 0.05))

    def test_background_renders_cache_surfaces_but_not_fields(self) -> None:
        result = self.service.render(self.generation, "alpha", [6], 24, 3.0, 0.05, keep_fields=False)[0]
        self.assertIsNone(result.grid)
        self.assertFalse(self.service.has_field("alpha", 6, 24, 3.0))
        cached = self.service.cached_result("alpha", 6, 24, 3.0, 0.05)
        self.assertIs(cached.positive, result.positive)
        self.assertEqual(cached.spec.shape, result.spec.shape)
        # A foreground request for a new isovalue recomputes and keeps the field.
        again = self.service.render(self.generation, "alpha", [6], 24, 3.0, 0.02)[0]
        self.assertIsNotNone(again.grid)
        self.assertTrue(self.service.has_field("alpha", 6, 24, 3.0))

    def test_field_cache_is_bounded(self) -> None:
        from moview.gui.render_service import FIELD_CACHE_ENTRIES

        self.service.render(self.generation, "alpha", list(range(FIELD_CACHE_ENTRIES + 6)), 16, 2.0, 0.05)
        self.assertEqual(len(self.service._field_cache), FIELD_CACHE_ENTRIES)
        self.assertEqual(len(self.service._surface_cache), FIELD_CACHE_ENTRIES + 6)

    def test_only_uncached_neighbors_are_prefetched(self) -> None:
        self.assertEqual(self.service.neighbors_to_prefetch("alpha", 24, 3.0, 0.05, [4]), [5, 3])
        self.service.render(self.generation, "alpha", [5], 24, 3.0, 0.05)
        self.assertEqual(self.service.neighbors_to_prefetch("alpha", 24, 3.0, 0.05, [4]), [3])
        self.assertEqual(self.service.neighbors_to_prefetch("alpha", 24, 3.0, 0.05, [0]), [1])
        self.assertEqual(self.service.neighbors_to_prefetch("alpha", 24, 3.0, 0.05, [3, 4]), [])  # compare mode


if __name__ == "__main__":
    unittest.main()
