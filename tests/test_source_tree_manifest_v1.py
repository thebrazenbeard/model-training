"""Metadata-only pinned-source inventory: no donor imports, file copying or execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest

from source_tree_manifest_v1 import (
    source_tree_manifest, SourceTreeError, _decode_tree
)

def git(root,*args):
    p=subprocess.run(["git","-C",str(root),*args],capture_output=True,text=True,timeout=10)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout.strip()


class SourceTreeManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/"source"
        self.root.mkdir()
        git(self.root,"init","--quiet")
        git(self.root,"config","user.email","training-test@example.invalid")
        git(self.root,"config","user.name","Training Test")
        (self.root/"one").mkdir()
        (self.root/"two").mkdir()
        self.marker=self.root/"one"/"runner.py"
        self.marker.write_text('raise RuntimeError("DO_NOT_EXECUTE_DONOR")\n',encoding="utf8")
        (self.root/"two"/"README.md").write_text("metadata-only",encoding="utf8")
        git(self.root,"add",".")
        git(self.root,"commit","--quiet","-m","Fixture only")
        self.sha=git(self.root,"rev-parse","HEAD")
        self.assertEqual(len(self.sha),40)

    def test_exact_commit_inventory_lists_missing_without_copying_content(self):
        original={str(f.relative_to(self.root)) for f in self.root.rglob("*") if f.is_file()}
        result=source_tree_manifest(self.root,commit=self.sha,
            expected_folders=("one","two","absent"))
        self.assertEqual(result["present_folders"],["one","two"])
        self.assertEqual(result["missing_folders"],["absent"])
        self.assertEqual(result["copies_performed"],0)
        self.assertEqual(result["content_bytes_published"],0)
        self.assertFalse(result["source_files_executed"])
        self.assertFalse(result["compatibility_qualified"])
        self.assertFalse(result["source_identity_authenticated"])
        self.assertNotIn("DO_NOT_EXECUTE_DONOR",str(result))
        self.assertEqual({str(f.relative_to(self.root)) for f in self.root.rglob("*") if f.is_file()},original)
        self.assertEqual(len(result["manifest_sha256"]),64)

    def test_dirty_worktree_payload_is_not_used_in_pinned_inventory(self):
        baseline=source_tree_manifest(self.root,commit=self.sha,expected_folders=("one",))
        self.marker.write_text('raise RuntimeError("CHANGED_UNTRACKED")\n',encoding="utf8")
        reread=source_tree_manifest(self.root,commit=self.sha,expected_folders=("one",))
        self.assertEqual(baseline,reread)
        self.assertNotIn("CHANGED_UNTRACKED",str(reread))

    def test_executable_source_remains_metadata_only_even_in_git_tree(self):
        git(self.root,"update-index","--chmod=+x","one/runner.py")
        git(self.root,"commit","--quiet","-m","mark executable")
        sha=git(self.root,"rev-parse","HEAD")
        result=source_tree_manifest(self.root,commit=sha,expected_folders=("one",))
        item=result["entries"][0]
        self.assertEqual(item["handling"],"EXECUTABLE_NO_COPY")
        self.assertEqual(item["mode"],"100755")
        self.assertFalse(result["source_files_executed"])
        self.assertNotEqual(result["manifest_sha256"],
                            source_tree_manifest(self.root,commit=self.sha,expected_folders=("one",))["manifest_sha256"])

    def test_rejects_unpinned_floating_ref_and_bad_folder_requests(self):
        for bad in ("main","HEAD","../branch","0"*39):
            with self.subTest(commit=bad),self.assertRaises(SourceTreeError):
                source_tree_manifest(self.root,commit=bad,expected_folders=("one",))
        for dirs in (("one","one"),("../escape",),("one/subdir",),tuple()):
            with self.subTest(dirs=dirs),self.assertRaises(SourceTreeError):
                source_tree_manifest(self.root,commit=self.sha,expected_folders=dirs)

    def test_symlink_and_submodule_treated_as_metadata_not_content(self):
        oid="a"*40
        raw=(f"120000 blob {oid}\tone/link\0"
             f"160000 commit {oid}\ttwo/submodule\0").encode("ascii")
        rows=_decode_tree(raw)
        self.assertEqual(rows[0]["handling"],"SYMLINK_NO_COPY")
        self.assertEqual(rows[1]["handling"],"SUBMODULE_NO_COPY")
        with self.assertRaises(SourceTreeError):
            _decode_tree((f"120000 blob {oid}\tone/../escape\0").encode("ascii"))
        with self.assertRaises(SourceTreeError):
            _decode_tree(b"100644 blob "+oid.encode()+b"\tone/file")

if __name__=="__main__":
    unittest.main()
