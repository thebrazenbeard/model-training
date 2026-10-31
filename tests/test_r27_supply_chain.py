"""Public, bounded supply-chain falsifiers; never downloads or executes inputs."""

import hashlib
import io
import stat
import tarfile
import unittest
import zipfile

from trainingday.r27_supply_chain import assess_supply_chain


REQUIREMENTS = "numpy==2.5.1\npytest==8.3.5\n"
_MISSING = object()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_for(name: str, payload: bytes, role: str = "runtime-config") -> dict:
    return {
        "schema": "TRAININGDAY_R27_STATIC_SUPPLY_CHAIN_V1",
        "requirements_sha256": sha(REQUIREMENTS.encode("utf-8")),
        "components": [{
            "name": name,
            "role": role,
            "locator": "public/example-artifact",
            "source_revision": "a" * 40,
            "artifact_sha256": sha(payload),
            "declared_bytes": len(payload),
            "license_spdx": "MIT",
        }],
    }


def zipper(entries: list[tuple[str, bytes, int | None]]) -> bytes:
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content, mode in entries:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            z.writestr(info, content)
    return sink.getvalue()


def tarball(entries: list[tuple[str, bytes, int | None]]) -> bytes:
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w:gz") as archive:
        for name, content, entry_type in entries:
            item = tarfile.TarInfo(name)
            if entry_type is None:
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
            else:
                item.type = entry_type
                item.linkname = "../escape"
                archive.addfile(item)
    return sink.getvalue()


def tarball_pax(headers: dict[str, str], *, global_headers: bool) -> bytes:
    sink = io.BytesIO()
    with tarfile.open(
        fileobj=sink, mode="w:gz", format=tarfile.PAX_FORMAT,
        pax_headers=headers if global_headers else None,
    ) as archive:
        info = tarfile.TarInfo("public.bin")
        payload = b"public"
        info.size = len(payload)
        if not global_headers:
            info.pax_headers = headers
        archive.addfile(info, io.BytesIO(payload))
    return sink.getvalue()


class StaticSupplyChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = b"public toy artifact bytes"
        self.manifest = manifest_for("toy-config", self.payload)

    def assess(self, manifest=_MISSING, *, observed=None, requirements=REQUIREMENTS):
        return assess_supply_chain(
            self.manifest if manifest is _MISSING else manifest,
            requirements_text=requirements,
            observed_artifacts={"toy-config": self.payload} if observed is None else observed,
        )

    def test_exact_bytes_and_dependency_pins_are_static_only(self):
        result = self.assess()
        self.assertEqual(result.reason, "STATIC_BYTES_MATCH_ONLY")
        self.assertTrue(result.static_receipt_consistent)
        self.assertTrue(result.artifact_bytes_matched)
        self.assertTrue(result.direct_requirement_versions_pinned)
        self.assertFalse(result.dependency_wheel_hashes_verified)
        self.assertFalse(result.reproducible_execution_verified)
        self.assertFalse(result.licenses_authenticated)
        self.assertFalse(result.binary_execution_authorized)
        self.assertFalse(result.model_release_authorized)

    def test_missing_or_different_observed_bytes_never_satisfy_hash(self):
        self.assertEqual(self.assess(observed={}).reason, "MISSING_OBSERVED_ARTIFACT")
        self.assertEqual(self.assess(observed={"toy-config": b"forged"}).reason,
                         "ARTIFACT_HASH_MISMATCH")
        self.assertEqual(self.assess(observed={"toy-config": self.payload, "extra": b"x"}).reason,
                         "UNDECLARED_OBSERVED_ARTIFACT")

    def test_requirements_are_pin_and_digest_bound(self):
        self.assertEqual(self.assess(requirements="numpy==1.0\n").reason,
                         "REQUIREMENTS_DIGEST_MISMATCH")
        for content in ("torch>=2.0\n", "-r other.txt\n", "git+https://host/repo\n",
                        "torch==2.0; python_version>'3.11'\n", "torch==2.0\ntorch==2.0\n",
                        "torch==latest\n"):
            bad = dict(self.manifest, requirements_sha256=sha(content.encode("utf-8")))
            with self.subTest(content=content):
                self.assertEqual(self.assess(bad, requirements=content).reason,
                                 "UNPINNED_REQUIREMENT")

    def test_manifest_inputs_are_exact_bounded_types(self):
        for value in (None, [], {"schema": "anything"}, object()):
            with self.subTest(value=str(type(value))):
                self.assertEqual(self.assess(value).reason, "MALFORMED_MANIFEST")
        forged = dict(self.manifest, external_independent_review=True)
        self.assertEqual(self.assess(forged).reason, "MALFORMED_MANIFEST")
        nested = dict(self.manifest, components=[object()])
        self.assertEqual(self.assess(nested).reason, "MALFORMED_COMPONENT")

        class EqualAll:
            def __eq__(self, other):
                return True

        forged_schema = dict(self.manifest, schema=EqualAll())
        self.assertEqual(self.assess(forged_schema).reason, "MALFORMED_MANIFEST")

    def test_unsafe_or_unpinned_component_inputs_rejected(self):
        edits = (
            {"source_revision": "main"}, {"locator": "../escape"},
            {"locator": "https://host/model?token=value"},
            {"artifact_sha256": "f" * 63}, {"declared_bytes": True},
            {"declared_bytes": 10 ** 1000}, {"declared_bytes": float("nan")},
            {"license_spdx": "UNKNOWN\nsecrets"},
            {"role": "run-executable-now"},
        )
        for fields in edits:
            with self.subTest(fields=fields):
                bad_component = dict(self.manifest["components"][0], **fields)
                bad = dict(self.manifest, components=[bad_component])
                self.assertEqual(self.assess(bad).reason, "MALFORMED_COMPONENT")

    def test_archives_statically_checked_without_extracting(self):
        safe = zipper([("a/model.bin", b"public toy bytes", None)])
        manifest = manifest_for("toy-archive", safe, "archive")
        result = assess_supply_chain(manifest, requirements_text=REQUIREMENTS,
                                     observed_artifacts={"toy-archive": safe})
        self.assertEqual(result.reason, "STATIC_BYTES_MATCH_ONLY")
        self.assertTrue(result.archive_metadata_safe)
        self.assertFalse(result.binary_execution_authorized)

    def test_archive_path_escape_symlink_duplicate_and_bomb_rejected(self):
        malicious = [
            zipper([("../escape", b"a", None)]),
            zipper([("/absolute/a", b"a", None)]),
            zipper([("C:\\escape", b"a", None)]),
            zipper([("link", b"target", stat.S_IFLNK | 0o777)]),
            zipper([("a.txt", b"a", None), ("A.txt", b"b", None)]),
            zipper([("payload.bin", b"z" * 100_000, None)]),
            zipper([("nested.zip", b"PK", None)]),
        ]
        for content in malicious:
            with self.subTest(sha=sha(content)):
                man = manifest_for("toy-archive", content, "archive")
                result = assess_supply_chain(man, requirements_text=REQUIREMENTS,
                                             observed_artifacts={"toy-archive": content})
                self.assertEqual(result.reason, "ARCHIVE_METADATA_REJECTED")
                self.assertFalse(result.binary_execution_authorized)

    def test_too_many_archive_entries_rejected(self):
        content = zipper([(f"s{i}.txt", b"x", None) for i in range(65)])
        man = manifest_for("toy-archive", content, "archive")
        result = assess_supply_chain(man, requirements_text=REQUIREMENTS,
                                     observed_artifacts={"toy-archive": content})
        self.assertEqual(result.reason, "ARCHIVE_METADATA_REJECTED")

    def test_tar_gz_is_metadata_only_and_rejects_escape_symlinks(self):
        good = tarball([("bin/llama-server", b"public toy executable placeholder", None)])
        for payload, expected in (
            (good, "STATIC_BYTES_MATCH_ONLY"),
            (tarball([("../../escape", b"a", None)]), "ARCHIVE_METADATA_REJECTED"),
            (tarball([("bin/link", b"", tarfile.SYMTYPE)]), "ARCHIVE_METADATA_REJECTED"),
            (tarball([("bin/hardlink", b"", tarfile.LNKTYPE)]), "ARCHIVE_METADATA_REJECTED"),
        ):
            with self.subTest(digest=sha(payload)):
                man = manifest_for("toy-archive", payload, "archive")
                result = assess_supply_chain(man, requirements_text=REQUIREMENTS,
                                             observed_artifacts={"toy-archive": payload})
                self.assertEqual(result.reason, expected)
                self.assertFalse(result.binary_execution_authorized)

    def test_windows_device_aliases_and_trailing_periods_rejected_in_both_archives(self):
        unsafe_names = (
            "CON", "NUL.txt", "COM1", "LPT9.prn", "aux.", "a.",
            "foo/CON", "sub/prn.md", "foo/COM9.tar", "LPT1", "a.txt.",
            "COM\u00b9", "LPT\u00b2.txt", "COM\u00b3.bin",
        )
        for unsafe_name in unsafe_names:
            for format_name, archive_factory in (("zip", zipper), ("tar.gz", tarball)):
                with self.subTest(name=unsafe_name, format=format_name):
                    content = archive_factory([(unsafe_name, b"public", None)])
                    man = manifest_for("toy-archive", content, "archive")
                    result = assess_supply_chain(
                        man, requirements_text=REQUIREMENTS,
                        observed_artifacts={"toy-archive": content},
                    )
                    self.assertEqual(result.reason, "ARCHIVE_METADATA_REJECTED")

    def test_zip_directory_mode_with_payload_is_rejected(self):
        content = zipper([("folder", b"should-not-be-a-directory", stat.S_IFDIR | 0o755)])
        man = manifest_for("toy-archive", content, "archive")
        result = assess_supply_chain(man, requirements_text=REQUIREMENTS,
                                     observed_artifacts={"toy-archive": content})
        self.assertEqual(result.reason, "ARCHIVE_METADATA_REJECTED")

    def test_tar_pax_extended_global_metadata_refused(self):
        for headers in (
            {"SCHILY.xattr.security.selinux": "system_u:object_r:bin_t:s0"},
            {"linkpath": "../../escape"},
        ):
            for global_headers in (False, True):
                with self.subTest(headers=headers, global_headers=global_headers):
                    content = tarball_pax(headers, global_headers=global_headers)
                    man = manifest_for("toy-archive", content, "archive")
                    result = assess_supply_chain(
                        man, requirements_text=REQUIREMENTS,
                        observed_artifacts={"toy-archive": content},
                    )
                    self.assertEqual(result.reason, "ARCHIVE_METADATA_REJECTED")
                    self.assertFalse(result.binary_execution_authorized)

    def test_observed_fixture_has_strict_size_and_type(self):
        bad = {"toy-config": b"x" * (4 * 1024 * 1024 + 1)}
        self.assertEqual(self.assess(observed=bad).reason, "OVERSIZED_OBSERVATION")
        self.assertEqual(self.assess(observed={"toy-config": bytearray(self.payload)}).reason,
                         "MALFORMED_OBSERVATION")

    def test_self_declared_license_does_not_make_rights_verified(self):
        for license_id in ("MIT", "Apache-2.0", "UNKNOWN"):
            component = dict(self.manifest["components"][0], license_spdx=license_id)
            result = self.assess(dict(self.manifest, components=[component]))
            self.assertFalse(result.licenses_authenticated)
            self.assertFalse(result.model_release_authorized)


if __name__ == "__main__":
    unittest.main()
