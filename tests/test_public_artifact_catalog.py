"""Pinned source catalogs must remain consistent with the OCD verifier shape."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainingday.public_artifact_catalog import BASE, CACHE, CATALOG, build_catalog


class PublicArtifactCatalogTests(unittest.TestCase):
    def test_catalog_is_exactly_derived_from_source_pins(self):
        generated = build_catalog()
        self.assertEqual(generated, json.loads(CATALOG.read_text(encoding="utf-8")))
        entries = generated["artifacts"]
        self.assertEqual(len(entries), 7)
        self.assertEqual(sum(e["size"] for e in entries), 11_732_880_841)
        self.assertEqual(len({e["destination_relpath"].casefold() for e in entries}), 7)
        self.assertTrue(all(e["url"].startswith("https://huggingface.co/") for e in entries))

    def test_changed_hash_is_reflected_without_silent_stale_catalog(self):
        with tempfile.TemporaryDirectory() as folder:
            altered = Path(folder) / "base.json"
            data = json.loads(BASE.read_text(encoding="utf-8"))
            data["files"][0]["sha256"] = "0" * 64
            altered.write_text(json.dumps(data), encoding="utf-8")
            self.assertNotEqual(build_catalog(altered), build_catalog())

    def test_unsafe_path_and_invalid_revision_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = json.loads(BASE.read_text(encoding="utf-8"))
            bad = Path(folder) / "bad.json"
            for changed in (
                lambda x: x["files"][0].update(path="../escape"),
                lambda x: x["source"].update(revision="latest"),
                lambda x: x["files"][0].update(size_bytes=-1),
            ):
                data = json.loads(BASE.read_text(encoding="utf-8"))
                changed(data)
                bad.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_catalog(bad)

    def test_cache_total_disagreement_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            altered = Path(folder) / "cache.json"
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            data["total_weight_bytes"] -= 1
            altered.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_catalog(cache_path=altered)

    def test_duplicate_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            altered = Path(folder) / "base.json"
            altered.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate manifest key"):
                build_catalog(base_path=altered)


if __name__ == "__main__":
    unittest.main()
