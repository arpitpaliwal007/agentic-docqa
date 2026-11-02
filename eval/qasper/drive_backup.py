"""Acknowledged Drive API backups, independent of the mounted Drive write cache.

Colab must authenticate with google.colab.auth.authenticate_user() first. No
credentials are saved in project files. Each question is an immutable object;
the runner stops on any upload/readback failure before starting another one.
"""
import hashlib
import io
import json

from eval.storage import atomic_json


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


class DriveBackup:
    def __init__(self, service, parent_id, account):
        self.service = service
        self.parent_id = parent_id
        actual = service.about().get(fields='user(emailAddress)').execute(num_retries=3)['user']['emailAddress']
        if actual.casefold() != account.casefold():
            raise ValueError(f'Drive account mismatch: expected {account}, got {actual}')
        parent = service.files().get(fileId=parent_id, fields='id,mimeType,capabilities(canAddChildren)', supportsAllDrives=True).execute(num_retries=3)
        if parent['mimeType'] != 'application/vnd.google-apps.folder' or not parent.get('capabilities', {}).get('canAddChildren'):
            raise ValueError('Drive backup parent must be a writable folder')

    @classmethod
    def connect(cls, parent_id, account):
        import google.auth
        from googleapiclient.discovery import build
        credentials, _ = google.auth.default()
        return cls(build('drive', 'v3', credentials=credentials, cache_discovery=False), parent_id, account)

    def children(self, parent, name=None):
        # IDs are supplied by Google; names are controlled by this evaluation.
        query = f"'{parent}' in parents and trashed = false"
        if name is not None:
            escaped = name.replace('\\', '\\\\').replace("'", "\\'")
            query += f" and name = '{escaped}'"
        result, token = [], None
        while True:
            page = self.service.files().list(q=query, pageSize=100, pageToken=token,
                fields='nextPageToken,files(id,name,mimeType,createdTime)', supportsAllDrives=True,
                includeItemsFromAllDrives=True).execute(num_retries=3)
            result.extend(page.get('files', []))
            token = page.get('nextPageToken')
            if not token:
                return result

    def folder(self, parent, name):
        matches = self.children(parent, name)
        if len(matches) > 1:
            raise ValueError(f'Ambiguous Drive backup folder: {name}')
        if matches:
            if matches[0]['mimeType'] != 'application/vnd.google-apps.folder':
                raise ValueError(f'Drive backup name is not a folder: {name}')
            return matches[0]['id']
        return self.service.files().create(body={'name': name, 'parents': [parent],
            'mimeType': 'application/vnd.google-apps.folder'}, fields='id', supportsAllDrives=True).execute(num_retries=3)['id']

    def download(self, file_id):
        return self.service.files().get_media(fileId=file_id, supportsAllDrives=True).execute(num_retries=3)

    def upload(self, parent, name, content, *, immutable=False, mime='application/json'):
        from googleapiclient.http import MediaIoBaseUpload
        matches = self.children(parent, name)
        if len(matches) > 1:
            raise ValueError(f'Ambiguous Drive backup file: {name}')
        if matches and immutable:
            file_id = matches[0]['id']
        else:
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime, resumable=True)
            if matches:
                request = self.service.files().update(fileId=matches[0]['id'], media_body=media,
                    fields='id', supportsAllDrives=True)
            else:
                request = self.service.files().create(body={'name': name, 'parents': [parent]},
                    media_body=media, fields='id', supportsAllDrives=True)
            file_id = request.execute(num_retries=3)['id']
        # Verify bytes read from Google's server, not the runtime's mounted cache.
        remote = self.download(file_id)
        checksum = hashlib.sha256(content).hexdigest()
        if hashlib.sha256(remote).hexdigest() != checksum:
            raise IOError(f'Drive readback checksum mismatch: {name}')
        return {'id': file_id, 'name': name, 'sha256': checksum, 'bytes': len(content)}

    def bind(self, state, *, resume):
        self.run_folder = self.folder(self.parent_id, state.value['metadata']['run_id'])
        self.record_folder = self.folder(self.run_folder, 'questions')
        existing = self.children(self.run_folder, 'metadata.json')
        if existing:
            metadata = json.loads(self.download(existing[0]['id']))
            if metadata['fingerprint'] != state.value['metadata']['fingerprint']:
                raise ValueError('Cloud run fingerprint differs')
            state.value['metadata'] = metadata
            records = self.children(self.record_folder)
            if records and not resume:
                raise ValueError('Cloud results exist; use --resume')
            # Immutable retry versions are ordered by server creation time.
            for item in sorted(records, key=lambda item: (item['createdTime'], item['id'])):
                if not item['name'].endswith('.json'):
                    continue
                content = self.download(item['id'])
                expected = item['name'].removesuffix('.json').rsplit('-', 1)[-1]
                if hashlib.sha256(content).hexdigest() != expected:
                    raise IOError('Cloud question checksum mismatch')
                envelope = json.loads(content)
                if envelope['fingerprint'] != metadata['fingerprint']:
                    raise ValueError('Cloud question fingerprint differs')
                state.value['results'][envelope['record']['id']] = envelope['record']
            state.save()
        else:
            self.upload(self.run_folder, 'metadata.json', encoded(state.value['metadata']), immutable=True)
            self.upload(self.run_folder, 'sample.json', encoded(state.value['metadata']['sample']), immutable=True)
        state.value['metadata']['drive_backup_folder'] = self.run_folder
        state.save()
        print('Verified cloud folder: https://drive.google.com/drive/folders/' + self.run_folder, flush=True)

    def record(self, state, record):
        content = encoded({'fingerprint': state.value['metadata']['fingerprint'], 'record': record})
        checksum = hashlib.sha256(content).hexdigest()
        safe_id = hashlib.sha256(record['id'].encode()).hexdigest()[:20]
        receipt = self.upload(self.record_folder, safe_id + '-' + checksum + '.json', content, immutable=True)
        atomic_json(state.path.parent / 'cloud-receipts' / (safe_id + '.json'), receipt)
        print(f"CLOUD VERIFIED {len(state.value['results'])}/{state.value['metadata']['sampled_n']}: {record['id']} ({receipt['bytes']} bytes)", flush=True)

    def checkpoint(self, state):
        self.upload(self.run_folder, state.path.name, state.path.read_bytes())
        report = state.path.parent / (state.value['metadata']['report_basename'] + '.md')
        if report.is_file():
            self.upload(self.run_folder, 'report.md', report.read_bytes(), mime='text/markdown')
        print(f"CLOUD CHECKPOINT VERIFIED: {len(state.value['results'])} questions", flush=True)
