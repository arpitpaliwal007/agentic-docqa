"""The standalone Colab repair must contain the tested code and preserve results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repair_file", ROOT / "scripts/repair_eval_judge.py")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class RepairFileTests(unittest.TestCase):
    def test_bundle_matches_sources_and_patch_preserves_saved_results(self):
        files = repair.patch_files()
        for name, content in files.items():
            self.assertEqual(content, (ROOT / name).read_bytes(), name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            repo = root / "project"
            (repo / "eval").mkdir(parents=True)
            (repo / "app").mkdir()
            (repo / "eval/run_eval.py").write_text("old evaluation code")
            results = root / "Drive/evaluation-results"
            results.mkdir(parents=True)
            original = results / "original.partial.json"
            saved = json.dumps({"metadata": {"fingerprint": "old"}, "results": {
                "q01": {"e2e": {"answer": "Saved answer", "evidence": "Actual evidence"}}}})
            original.write_text(saved)
            (results / "rejudged").mkdir()
            (results / "rejudged/new.partial.json").write_text("should not be selected")
            source, _ = repair.select_source(results)
            self.assertEqual(source, original)
            self.assertEqual(repair.find_repo(repo), repo)
            backup = repair.install_patch(repo, results / "rejudged/code-backups")
            self.assertEqual((backup / "eval/run_eval.py").read_text(), "old evaluation code")
            self.assertEqual(original.read_text(), saved)
            for name, content in files.items():
                self.assertEqual((repo / name).read_bytes(), content)

    def test_symlinked_module_cannot_overwrite_another_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            repo = root / "project"
            (repo / "eval").mkdir(parents=True)
            unrelated = root / "unrelated.txt"
            unrelated.write_text("keep")
            (repo / "eval/judge.py").symlink_to(unrelated)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                repair.install_patch(repo)
            self.assertEqual(unrelated.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
