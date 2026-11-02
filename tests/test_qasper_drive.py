"""Persistence tests exercise server readback and recovery without real accounts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eval.qasper.drive_backup import DriveBackup, encoded
from eval.storage import Results


class DriveBackupTests(unittest.TestCase):
    def test_wrong_account_stops_before_writing(self):
        service = Mock()
        service.about().get().execute.return_value = {'user': {'emailAddress': 'wrong@example.com'}}
        with self.assertRaisesRegex(ValueError, 'account mismatch'):
            DriveBackup(service, 'parent', 'right@example.com')
        service.files.assert_not_called()

    def test_upload_requires_exact_server_readback(self):
        backup = object.__new__(DriveBackup)
        backup.service = Mock()
        backup.service.files().create().execute.return_value = {'id': 'cloud-file'}
        backup.children = Mock(return_value=[])
        backup.download = Mock(return_value=b'correct')
        fake_http = SimpleNamespace(MediaIoBaseUpload=lambda *args, **kwargs: object())
        with patch.dict(sys.modules, {'googleapiclient.http': fake_http}):
            receipt = backup.upload('parent', 'q.json', b'correct')
            self.assertEqual(receipt['sha256'], hashlib.sha256(b'correct').hexdigest())
            backup.download.return_value = b'truncated'
            with self.assertRaisesRegex(IOError, 'checksum mismatch'):
                backup.upload('parent', 'q.json', b'correct')

    def test_cloud_question_recovers_after_runtime_loss_and_detects_corruption(self):
        metadata = {'fingerprint': 'same', 'run_id': 'qasper-test', 'sampled_n': 50, 'sample': {}}
        record = {'id': 'q1', 'status': 'complete', 'answer': 'saved answer'}
        content = encoded({'fingerprint': 'same', 'record': record})
        checksum = hashlib.sha256(content).hexdigest()
        backup = object.__new__(DriveBackup)
        backup.folder = Mock(side_effect=lambda parent, name: name)
        backup.parent_id = 'parent'
        def children(parent, name=None):
            return [{'id': 'metadata'}] if name == 'metadata.json' else [
                {'id': 'question', 'name': 'q1-' + checksum + '.json', 'createdTime': '2026-11-02T12:00:00Z'}]
        backup.children = children
        backup.download = Mock(side_effect=lambda file_id: encoded(metadata) if file_id == 'metadata' else content)
        with tempfile.TemporaryDirectory() as directory:
            state = Results(directory, 'qasper-test', metadata)
            backup.bind(state, resume=True)
            self.assertEqual(state.value['results']['q1'], record)
            self.assertTrue(Results(directory, 'qasper-test', metadata).completed('q1'))
            with self.assertRaisesRegex(ValueError, 'use --resume'):
                backup.bind(state, resume=False)
            backup.download.side_effect = lambda file_id: encoded(metadata) if file_id == 'metadata' else b'broken'
            with self.assertRaisesRegex(IOError, 'checksum mismatch'):
                backup.bind(state, resume=True)
