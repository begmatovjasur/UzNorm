"""Immutable Drive-API backups; a mounted Drive read is NOT a cloud commit.

No Google calls happen on import. All writes need an explicitly constructed
DriveStore. This module never changes source checkpoints or their manifests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import zipfile


class BackupError(RuntimeError):
    """Safe to show in notebook output; never embeds HTTP bodies or credentials."""


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    size = 0
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(data)
            md5.update(data)
            size += len(data)
    return {'sha256': sha.hexdigest(), 'md5': md5.hexdigest(), 'size': size}


def safe_member(root, name):
    root = Path(root).resolve()
    if (not isinstance(name, str) or not name or ':' in name or '\\' in name
            or Path(name).is_absolute() or '..' in Path(name).parts):
        raise BackupError('Unsafe archive member path')
    cursor = root
    for part in Path(name).parts:
        cursor /= part
        if cursor.is_symlink():
            raise BackupError('Symlink in artifact path')
    target = (root / name).resolve()
    if target == root or not target.is_relative_to(root):
        raise BackupError('Artifact path escapes its root')
    return target


def sealed_members(folder, marker='COMPLETE.json', required=()):
    folder = Path(folder)
    if folder.is_symlink():
        raise BackupError('Sealed directory is a symlink')
    manifest = read_json(folder / marker)
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or not set(required).issubset(files):
        raise BackupError('Artifact manifest is incomplete')
    result = {}
    for name, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise BackupError('Invalid artifact checksum')
        path = safe_member(folder, name)
        if not path.is_file():
            raise BackupError('Artifact member is missing: ' + name)
        result[name] = (path, expected)
    result[marker] = (folder / marker, hashes(folder / marker)['sha256'])
    return result


REQUIRED = ('config.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')


def checkpoint_members(run, checkpoint):
    """Include all transitive best-model dependencies without deserializing pickle."""
    if Path(checkpoint).is_symlink() or Path(run).is_symlink():
        raise BackupError('Symlink checkpoint/run is not allowed')
    run, checkpoint = Path(run).resolve(), Path(checkpoint).resolve()
    pending, seen, members = [checkpoint], set(), {}
    while pending:
        folder = pending.pop()
        if folder in seen:
            continue
        if (folder.parent not in (run, run / 'recovery')
                or not re.fullmatch(r'checkpoint-\d+', folder.name)):
            raise BackupError('Checkpoint is outside the selected run')
        seen.add(folder)
        required = REQUIRED + (('RECOVERY.json',) if folder.parent == run / 'recovery' else ())
        files = sealed_members(folder, required=required)
        if 'model.safetensors' not in files:
            if 'model.safetensors.index.json' not in files:
                raise BackupError('Model weights missing')
            mapping = read_json(folder / 'model.safetensors.index.json').get('weight_map', {})
            if not mapping or not set(mapping.values()).issubset(files):
                raise BackupError('Model shards missing')
        state = read_json(folder / 'trainer_state.json')
        if type(state.get('global_step')) is not int or state['global_step'] != int(folder.name.split('-')[-1]):
            raise BackupError('Checkpoint step mismatch')
        if folder.parent == run / 'recovery':
            info = read_json(folder / 'RECOVERY.json')
            if info.get('kind') != 'full_state_recovery' or info.get('global_step') != state['global_step']:
                raise BackupError('Recovery metadata mismatch')
        best = state.get('best_model_checkpoint')
        if best:
            best = Path(best)
            if not best.is_absolute() or best.parent.resolve() != run or best.is_symlink():
                raise BackupError('Best checkpoint outside the selected run')
            pending.append(best.resolve())
        for name, item in files.items():
            relative = folder.relative_to(run).as_posix() + '/' + name
            members[relative] = item
    return members


@dataclass(frozen=True)
class Binding:
    run_id: str
    signature: str
    storage_id: str

    def __post_init__(self):
        for value, pattern in ((self.run_id, r'[a-zA-Z0-9_-]{1,64}'),
                               (self.signature, r'[0-9a-f]{64}'), (self.storage_id, r'[0-9a-f]{32}')):
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise BackupError('Invalid backup binding')

    def properties(self):
        return {'uznorm_guard': 'v1', 'run_id': self.run_id,
                'signature': self.signature, 'storage_id': self.storage_id}


def build_archive(path, members, context):
    """Hash while copying: a changed/missing source aborts before cloud publication."""
    names = set(members)
    if 'CLOUD_MANIFEST.json' in names:
        raise BackupError('Reserved manifest name')
    for name in names:
        safe_member(Path(path).parent, name)
    manifest = {'schema': 1, **context, 'files': {n: item[1] for n, item in members.items()}}
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, (source, expected) in sorted(members.items()):
            digest = hashlib.sha256()
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o600 << 16
            with Path(source).open('rb') as incoming, archive.open(info, 'w', force_zip64=True) as outgoing:
                for data in iter(lambda: incoming.read(8 * 1024**2), b''):
                    digest.update(data)
                    outgoing.write(data)
            if digest.hexdigest() != expected:
                raise BackupError('Source changed or checksum failed: ' + name)
        archive.writestr('CLOUD_MANIFEST.json', json_bytes(manifest))
    return hashes(path)


def verify_archive(path, binding):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 20000 or len({i.filename for i in infos}) != len(infos):
            raise BackupError('Duplicate/excessive archive members')
        for info in infos:
            safe_member(Path(path).parent, info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise BackupError('Unsupported archive entry')
        manifest_info = archive.getinfo('CLOUD_MANIFEST.json')
        if manifest_info.file_size > 4 * 1024**2:
            raise BackupError('Archive manifest too large')
        manifest = json.loads(archive.read('CLOUD_MANIFEST.json'))
        if manifest.get('schema') != 1 or manifest.get('binding') != binding.properties():
            raise BackupError('Archive belongs to another run/storage policy')
        files = manifest.get('files')
        if not isinstance(files, dict) or set(files) | {'CLOUD_MANIFEST.json'} != {i.filename for i in infos}:
            raise BackupError('Archive inventory mismatch')
        for name, expected in files.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for data in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(data)
            if digest.hexdigest() != expected:
                raise BackupError('Downloaded archive checksum mismatch: ' + name)
    return manifest


def check_server_record(record, binding, digest=None, parent=None):
    """Server-computed checksum, not client-supplied appProperties alone."""
    props = record.get('appProperties', {})
    if record.get('trashed') or any(props.get(k) != v for k, v in binding.properties().items()):
        raise BackupError('Cloud file binding mismatch')
    if parent is not None and record.get('parents') != [parent]:
        raise BackupError('Cloud file is outside the dedicated backup folder')
    expected = digest or {'sha256': props.get('payload_sha256'), 'md5': props.get('payload_md5'),
                          'size': props.get('payload_size')}
    if digest and any(str(props.get('payload_' + key)) != str(value) for key, value in digest.items()):
        raise BackupError('Cloud manifest properties differ from the uploaded bytes')
    if (not re.fullmatch(r'[0-9a-f]{64}', str(expected.get('sha256')))
            or not re.fullmatch(r'[0-9a-f]{32}', str(expected.get('md5')))):
        raise BackupError('Cloud file has no usable checksum evidence')
    if int(record.get('size', -1)) != int(expected['size']):
        raise BackupError('Cloud file size does not match')
    if record.get('md5Checksum') != expected['md5']:
        raise BackupError('Google-server MD5 checksum does not match')
    if record.get('sha256Checksum') and record['sha256Checksum'] != expected['sha256']:
        raise BackupError('Google-server SHA-256 checksum does not match')
    return record


class DriveStore:
    """Google Drive API only. No access through /content/drive for remote proof."""
    FIELDS = 'id,name,size,md5Checksum,sha256Checksum,parents,trashed,appProperties,createdTime,mimeType'

    def __init__(self, service, root_id, binding, *, allow_writes=False, allow_prune=False,
                 max_bytes=120 * 1024**3, upload_timeout=1800):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', root_id):
            raise BackupError('Invalid Drive root ID')
        if not allow_writes:
            raise BackupError('Google Drive API backup consent is required')
        self.api, self.binding, self.allow_prune = service, binding, allow_prune
        self.root_id, self.max_bytes, self.upload_timeout = root_id, max_bytes, upload_timeout
        self.folder_id = None

    @staticmethod
    def request(request):
        try:
            return request.execute(num_retries=2)
        except Exception as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', 'unavailable')
            raise BackupError(f'Drive API request failed (HTTP {status}); training must not advance') from None

    def find(self, parent, *, name=None):
        query = f"'{parent}' in parents and trashed = false"
        if name is not None:
            query += " and name = '" + name.replace('\\', '\\\\').replace("'", "\\'") + "'"
        result, cursor, seen = [], None, set()
        while True:
            page = self.request(self.api.files().list(q=query, spaces='drive', pageSize=1000,
                                pageToken=cursor, fields='nextPageToken,files(' + self.FIELDS + ')'))
            result.extend(page.get('files', []))
            cursor = page.get('nextPageToken')
            if not cursor:
                return result
            if cursor in seen or len(result) > 20000:
                raise BackupError('Unexpected Drive pagination; stopped')
            seen.add(cursor)

    def prepare(self):
        root = self.request(self.api.files().get(fileId=self.root_id, fields='id,mimeType,trashed'))
        if root.get('trashed') or root.get('mimeType') != 'application/vnd.google-apps.folder':
            raise BackupError('Selected Drive project root is not a folder')
        markers = self.find(self.root_id, name='.uznorm-storage.json')
        if len(markers) != 1:
            raise BackupError('Cloud-side storage marker missing or ambiguous')
        if int(markers[0].get('size', 65537)) > 65536:
            raise BackupError('Cloud-side storage marker is too large')
        raw = self.request(self.api.files().get_media(fileId=markers[0]['id']))
        if len(raw) > 65536 or json.loads(raw).get('storage_id') != self.binding.storage_id:
            raise BackupError('Google API account/storage does not match training Drive')
        name = 'uznorm-cloud-' + self.binding.run_id
        folders = self.find(self.root_id, name=name)
        if not folders:
            folder = self.request(self.api.files().create(body={
                'name': name, 'parents': [self.root_id], 'mimeType': 'application/vnd.google-apps.folder',
                'appProperties': self.binding.properties()}, fields=self.FIELDS))
        elif len(folders) == 1:
            folder = folders[0]
        else:
            raise BackupError('Multiple backup folders; explicit selection required')
        if (folder.get('mimeType') != 'application/vnd.google-apps.folder'
                or folder.get('parents') != [self.root_id]
                or any(folder.get('appProperties', {}).get(k) != v for k, v in self.binding.properties().items())):
            raise BackupError('Backup folder belongs to another policy/run')
        self.folder_id = folder['id']
        return self.folder_id

    def inspect(self, file_id, digest=None):
        record = self.request(self.api.files().get(fileId=file_id, fields=self.FIELDS))
        return check_server_record(record, self.binding, digest, self.folder_id)

    def records(self):
        if not self.folder_id:
            raise BackupError('Cloud backup store was not prepared')
        return self.find(self.folder_id)

    def put(self, path, *, kind, step, digest, support_id=None):
        from googleapiclient.http import MediaFileUpload
        if kind not in ('support', 'checkpoint', 'final') or type(step) is not int or step < 0:
            raise BackupError('Invalid cloud snapshot kind/step')
        name = f'{kind}-{step}-{digest["sha256"][:24]}.zip'
        existing = self.find(self.folder_id, name=name)
        for item in existing:
            try:
                record = self.inspect(item['id'], digest)
                props = record['appProperties']
                if (props.get('kind') == kind and props.get('step') == str(step)
                        and props.get('support_id') == support_id):
                    return record
            except BackupError:
                continue
        records = self.records()
        if sum(int(r.get('size', 0)) for r in records) + digest['size'] > self.max_bytes:
            raise BackupError('Dedicated cloud-backup budget would be exceeded; no old backups were deleted')
        props = {**self.binding.properties(), 'kind': kind, 'step': str(step),
                 'payload_sha256': digest['sha256'], 'payload_md5': digest['md5'],
                 'payload_size': str(digest['size'])}
        if support_id:
            props['support_id'] = support_id
        media = MediaFileUpload(str(path), mimetype='application/zip', chunksize=32 * 1024**2, resumable=True)
        request = self.api.files().create(body={'name': name, 'parents': [self.folder_id],
                                               'appProperties': props}, media_body=media, fields=self.FIELDS)
        start, response, previous = time.monotonic(), None, -1
        try:
            while response is None:
                if time.monotonic() - start > self.upload_timeout:
                    raise BackupError('Cloud upload deadline exceeded; last confirmed backup retained')
                progress, response = request.next_chunk(num_retries=3)
                percent = int(progress.progress() * 100) if progress else 0
                if percent >= previous + 5:
                    print(f'CLOUD_UPLOAD {kind} step={step} {percent}%', flush=True)
                    previous = percent
        except BackupError:
            raise
        except Exception as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', 'unavailable')
            raise BackupError(f'Cloud upload failed (HTTP {status}); last confirmed backup retained') from None
        finally:
            media.stream().close()
        # A separate server request is mandatory, even after HTTP upload success.
        return self.inspect(response['id'], digest)

    def download(self, record, destination):
        from googleapiclient.http import MediaIoBaseDownload
        record = self.inspect(record['id'])
        destination = Path(destination)
        try:
            with destination.open('xb') as stream:
                job = MediaIoBaseDownload(stream, self.api.files().get_media(fileId=record['id']),
                                         chunksize=32 * 1024**2)
                done, started = False, time.monotonic()
                while not done:
                    if time.monotonic() - started > self.upload_timeout:
                        raise BackupError('Cloud download deadline exceeded')
                    _, done = job.next_chunk(num_retries=3)
        except BackupError:
            raise
        except Exception:
            raise BackupError('Cloud download failed; no restore was performed') from None
        digest = hashes(destination)
        if digest['sha256'] != record['appProperties']['payload_sha256']:
            raise BackupError('Downloaded cloud bytes failed SHA-256 verification')
        self.inspect(record['id'], digest)
        return destination

    def prune(self, newest, keep=2):
        """Only opt-in deletion of this policy's generated checkpoint archives.

        No old FUSE run files, support archives, final exports, or other folders
        are touched. At least two *server-verified* checkpoints remain.
        """
        if not self.allow_prune:
            return
        verified = []
        for record in self.records():
            if record.get('appProperties', {}).get('kind') != 'checkpoint':
                continue
            try:
                item = self.inspect(record['id'])
                step = int(item['appProperties']['step'])
                if step < 0:
                    continue
                verified.append(item)
            except (BackupError, ValueError, KeyError):
                continue  # Never delete an unknown/corrupt/foreign file.
        verified.sort(key=lambda r: (int(r['appProperties']['step']), r.get('createdTime', '')), reverse=True)
        if newest['id'] not in {r['id'] for r in verified}:
            raise BackupError('Newest archive could not be verified for retention')
        retain = {r['id'] for r in verified[:max(2, keep)]} | {newest['id']}
        if len(retain) < 2:
            return
        for old in verified:
            if old['id'] in retain:
                continue
            # Recheck retained copies immediately before irreversible retention.
            for item in verified:
                if item['id'] in retain:
                    self.inspect(item['id'])
            self.inspect(old['id'])
            self.request(self.api.files().delete(fileId=old['id']))
            print('CLOUD_RETENTION removed own old checkpoint archive:', old['id'], flush=True)


class BackupGate:
    def __init__(self, store, root, run, work, *, reserve_bytes=8 * 1024**3, max_unbacked_steps=50):
        self.store, self.binding = store, store.binding
        self.root, self.run, self.work = Path(root).resolve(), Path(run).resolve(), Path(work).resolve()
        if self.run == self.root or not self.run.is_relative_to(self.root):
            raise BackupError('Run must be inside the selected storage project')
        if self.work.is_relative_to(self.root):
            raise BackupError('Temporary backup archives must be on local disk, not Drive')
        self.reserve_bytes, self.max_unbacked_steps = reserve_bytes, max_unbacked_steps
        self.support, self.ready, self.last_cloud_step = None, False, -1
        self.policy_sha = hashes(Path(__file__))['sha256']
        self.policy_files = {Path(__file__).name: Path(__file__)}

    def boundary(self, step):
        if not self.ready or step - self.last_cloud_step >= self.max_unbacked_steps:
            raise BackupError('Cloud checkpoint is overdue; next optimizer step is blocked')
        self.disk_space(0)

    def disk_space(self, payload, copies=1):
        self.work.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.work).free < self.reserve_bytes + payload * copies:
            raise BackupError('Local disk reserve is low; training paused before more work is lost')

    def storage_members(self, files, base):
        base = Path(base).resolve()
        if not base.is_relative_to(self.root):
            raise BackupError('Archive source is outside storage project')
        relative = base.relative_to(self.root).as_posix()
        prefix = 'storage' if relative == '.' else 'storage/' + relative
        return {prefix.rstrip('/') + '/' + name: item for name, item in files.items()}

    def publish(self, members, kind, step, *, roundtrip=False):
        payload = sum(path.stat().st_size for path, _ in members.values())
        self.disk_space(payload, copies=2 if roundtrip else 1)
        stage = Path(tempfile.mkdtemp(prefix='cloud-guard-', dir=self.work))
        archive = stage / 'payload.zip'
        context = {'binding': self.binding.properties(), 'kind': kind, 'step': step,
                   'storage_extension_sha256': self.policy_sha,
                   'policy_files_sha256': {n: hashes(p)['sha256'] for n, p in self.policy_files.items()},
                   'support_id': self.support['id'] if self.support else None,
                   'source_run_relative': self.run.relative_to(self.root).as_posix()}
        digest = build_archive(archive, members, context)
        record = self.store.put(archive, kind=kind, step=step, digest=digest,
                                support_id=context['support_id'])
        # The transport contract is checked here too, so a buggy adapter cannot advance the gate.
        check_server_record(record, self.binding, digest)
        if roundtrip:
            downloaded = self.store.download(record, stage / 'roundtrip.zip')
            verify_archive(downloaded, self.binding)
            Path(downloaded).unlink()  # Only this invocation's verified temporary download.
        archive.unlink()  # Only this invocation's uploaded and server-verified temporary archive.
        stage.rmdir()    # Empty, uniquely created local directory; never a user run directory.
        return record

    def seed(self, checkpoint, smoke, frozen_zip, *, frozen_sha256):
        self.ready = False
        meta = read_json(self.run / 'run-meta.json')
        if meta.get('signature') != self.binding.signature or meta.get('wandb_id') != self.binding.run_id:
            raise BackupError('Local run does not match cloud binding')
        if hashes(frozen_zip)['sha256'] != frozen_sha256:
            raise BackupError('Frozen training release checksum mismatch')
        step = read_json(Path(checkpoint) / 'trainer_state.json')['global_step']
        for item in self.store.records():
            if item.get('appProperties', {}).get('kind') == 'checkpoint':
                try:
                    remote = self.store.inspect(item['id'])
                except BackupError:
                    continue
                if int(remote['appProperties']['step']) > step:
                    raise BackupError('A newer confirmed cloud checkpoint exists. Restore it; do not roll back silently.')
        smoke = Path(smoke).resolve()
        smoke_done = read_json(smoke / 'TRAINING_COMPLETE.json')
        if (smoke_done.get('smoke') is not True or smoke_done.get('global_step') != 2
                or smoke_done.get('gate_signature') != meta.get('gate_signature')):
            raise BackupError('Matching successful smoke is required')
        files = sealed_members(smoke / 'export', 'EXPORT_COMPLETE.json', ('config.json', 'training-provenance.json'))
        members = self.storage_members(files, smoke / 'export')
        for path in [smoke / 'TRAINING_COMPLETE.json', smoke / 'DRIVE_DURABILITY.json',
                     self.root / '.uznorm-storage.json', self.run.parent / (self.run.name + '-smoke-passed.json')]:
            members['storage/' + path.relative_to(self.root).as_posix()] = (path, hashes(path)['sha256'])
        members['frozen/uznorm-byt5-base-safe-v2.zip'] = (Path(frozen_zip), frozen_sha256)
        for name, path in self.policy_files.items():
            members['policy/' + name] = (path, hashes(path)['sha256'])
        self.support = self.publish(members, 'support', 0, roundtrip=True)
        record = self.checkpoint(checkpoint, roundtrip=True)
        self.ready = True
        print(f'CLOUD_ROUNDTRIP_OK step={step} backup_file_id={record["id"]}', flush=True)
        return record

    def checkpoint(self, checkpoint, *, roundtrip=False):
        try:
            return self._checkpoint(checkpoint, roundtrip=roundtrip)
        except BaseException:
            self.ready = False
            raise

    def _checkpoint(self, checkpoint, *, roundtrip=False):
        if not self.support:
            raise BackupError('Cloud support bundle has not been verified')
        self.store.inspect(self.support['id'])
        checkpoint = Path(checkpoint).resolve()
        files = checkpoint_members(self.run, checkpoint)
        meta_path = self.run / 'run-meta.json'
        files['run-meta.json'] = (meta_path, hashes(meta_path)['sha256'])
        step = read_json(checkpoint / 'trainer_state.json')['global_step']
        if step < self.last_cloud_step:
            raise BackupError('Refusing checkpoint rollback')
        members = self.storage_members(files, self.run)
        record = self.publish(members, 'checkpoint', step, roundtrip=roundtrip)
        self.last_cloud_step = step  # Only AFTER server verification / requested roundtrip.
        print(f'CLOUD_CONFIRMED step={step} file_id={record["id"]} size={record["size"]}', flush=True)
        self.store.prune(record)
        return record

    def final(self):
        if not self.support:
            raise BackupError('Missing cloud support archive')
        self.store.inspect(self.support['id'])
        files = sealed_members(self.run / 'export', 'EXPORT_COMPLETE.json', ('config.json', 'training-provenance.json'))
        members = self.storage_members(files, self.run / 'export')
        for name in ('run-meta.json', 'TRAINING_COMPLETE.json'):
            path = self.run / name
            members['storage/' + self.run.relative_to(self.root).as_posix() + '/' + name] = (path, hashes(path)['sha256'])
        step = read_json(self.run / 'TRAINING_COMPLETE.json')['global_step']
        record = self.publish(members, 'final', step, roundtrip=True)
        print('CLOUD_FINAL_VERIFIED file_id=' + record['id'], flush=True)
        return record
