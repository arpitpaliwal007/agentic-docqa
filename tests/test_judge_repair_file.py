"""The standalone Colab repair must contain the tested code and preserve results."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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

    def test_main_resolves_shared_drive_shortcut_before_backing_up_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            repo=root/'project'
            (repo/'eval').mkdir(parents=True)
            (repo/'app').mkdir()
            (repo/'eval/run_eval.py').write_text('old code')
            target=root/'shared-folder/evaluation-results'
            target.mkdir(parents=True)
            shortcut=root/'MyDrive/Agentic DocQA'
            shortcut.parent.mkdir()
            shortcut.symlink_to(target.parent, target_is_directory=True)
            source=target/'original.partial.json'
            source.write_text(json.dumps({'metadata':{},'results':{'q01':{'e2e':{'answer':'saved','evidence':'actual'}}}}))
            before=source.read_bytes()
            process=MagicMock()
            process.__enter__.return_value=process
            process.stdout=[]
            process.wait.return_value=0
            with patch.object(repair.subprocess,'Popen',return_value=process), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(repair.main(['--repo',str(repo),'--results',str(shortcut/'evaluation-results')]),0)
            self.assertEqual(source.read_bytes(),before)
            backups=list((target/'v2/rejudged-all/code-backups').glob('*/eval/run_eval.py'))
            self.assertEqual(len(backups),1)
            self.assertEqual(backups[0].read_text(),'old code')


if __name__ == "__main__":
    unittest.main()
