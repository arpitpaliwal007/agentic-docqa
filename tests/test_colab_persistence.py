"""Offline checks for the Colab runner's Drive backups (no GPU or packages needed)."""
import importlib.util
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_agentic_docqa_colab.py"
SPEC = importlib.util.spec_from_file_location("colab_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class DrivePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.runtime = self.base / "runtime"
        self.runtime.mkdir()
        self.store = runner.DriveStore(self.base / "Drive")
        self.store.restore(self.runtime)

    def test_committed_wal_data_restores_with_documents_and_account(self):
        documents = self.runtime / "data" / "users" / "demo" / "uploads"
        documents.mkdir(parents=True)
        (documents / "example.md").write_text("# Evidence\nAn answer to remember.")
        (self.runtime / "notebook-session.json").write_text('{"cookies": {"docqa_session": "test"}}')
        database = self.runtime / "qdrant_db" / "collection" / "storage.sqlite"
        database.parent.mkdir(parents=True)
        connection = sqlite3.connect(database)
        self.addCleanup(connection.close)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE chunks (text TEXT)")
        connection.execute("INSERT INTO chunks VALUES ('committed vector payload')")
        connection.commit()
        (database.parent / ".lock").write_text("runtime lock")
        checkpoint = self.store.save(self.runtime)
        with zipfile.ZipFile(checkpoint) as bundle:
            self.assertFalse(any(name.endswith(("-wal", "-shm", ".lock")) for name in bundle.namelist()))
        restored = self.base / "new-runtime"
        restored.mkdir()
        self.store.restore(restored)
        self.assertEqual((restored / documents.relative_to(self.runtime) / "example.md").read_text(), "# Evidence\nAn answer to remember.")
        with closing(sqlite3.connect(restored / database.relative_to(self.runtime))) as copied:
            self.assertEqual(copied.execute("SELECT * FROM chunks").fetchall(), [("committed vector payload",)])
        self.assertEqual(json.loads((restored / "notebook-session.json").read_text())["cookies"]["docqa_session"], "test")

    def test_existing_work_is_not_overwritten_and_older_checkpoints_remain(self):
        data = self.runtime / "data"
        data.mkdir()
        state = data / "chats.json"
        state.write_text("old")
        first = self.store.save(self.runtime)
        state.write_text("new")
        second = self.store.save(self.runtime)
        self.store.restore(self.runtime)
        self.assertEqual(state.read_text(), "new")
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists())

    def test_failed_upload_leaves_last_successful_checkpoint_selected(self):
        (self.runtime / "app.log").write_text("healthy")
        self.store.save(self.runtime)
        previous = (self.store.folder / "latest.json").read_bytes()
        with patch.object(runner.shutil, "copyfile", side_effect=OSError("Drive disconnected")):
            with self.assertRaisesRegex(OSError, "Drive disconnected"):
                self.store.save(self.runtime)
        self.assertEqual((self.store.folder / "latest.json").read_bytes(), previous)

    def test_corrupt_checkpoint_does_not_restore(self):
        (self.runtime / "app.log").write_text("healthy")
        checkpoint = self.store.save(self.runtime)
        checkpoint.write_bytes(b"incomplete upload")
        fresh = self.base / "fresh"
        fresh.mkdir()
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.store.restore(fresh)
        self.assertFalse((fresh / "app.log").exists())

    def test_checkpoint_cannot_write_outside_runtime(self):
        archive = self.store.snapshots / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../outside.txt", "must not escape")
        (self.store.folder / "latest.json").write_text(json.dumps({"filename": archive.name, "sha256": runner.file_digest(archive)}))
        with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
            self.store.restore(self.runtime)
        self.assertFalse((self.base / "outside.txt").exists())

    def test_runtime_cannot_be_reused_for_another_drive_folder(self):
        other = runner.DriveStore(self.base / "OtherDrive")
        with self.assertRaisesRegex(RuntimeError, "another Drive folder"):
            other.restore(self.runtime)

    def test_runner_bundles_current_app_and_evaluation(self):
        destination = self.base / "project"
        runner.unpack_project(destination)
        project = Path(__file__).resolve().parents[1]
        for source in [*(project / "app").rglob("*.py"), *(project / "eval").rglob("*.py"),
                       project / "scripts/repair_eval_judge.py", project / "eval/colab_eval.ipynb"]:
            self.assertEqual(source.read_bytes(), (destination / source.relative_to(project)).read_bytes())
        notebook = json.loads((destination / "eval/colab_eval.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)

    def test_expired_cookie_logs_into_the_existing_account(self):
        account = {"username": "colab_persistent", "password": "internal-generated-test-password"}
        (self.runtime / "notebook-account.json").write_text(json.dumps(account))
        (self.runtime / "notebook-session.json").write_text('{"cookies": {"docqa_session": "expired"}}')
        class Cookies(dict):
            def get_dict(self):
                return dict(self)
        session = Mock()
        session.cookies = Cookies()
        session.get.return_value.status_code = 401
        def login(url, **kwargs):
            self.assertTrue(url.endswith("/login"))
            self.assertEqual(kwargs["data"], account)
            session.cookies["docqa_session"] = "renewed"
            return Mock()
        session.post.side_effect = login
        session.request.return_value = Mock(ok=True)
        session.request.return_value.json.return_value = {"selected": "c1"}
        with patch.dict("sys.modules", {"requests": types.SimpleNamespace(Session=lambda: session)}):
            client = runner.DocQA("http://127.0.0.1:8000", self.runtime, [], self.store)
        self.assertEqual(client.chat_id, "c1")
        self.assertEqual(json.loads((self.runtime / "notebook-account.json").read_text()), account)
        self.assertTrue((self.store.folder / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
