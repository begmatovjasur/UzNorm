"""Frozen 5k full-FT on Colab L4, initialized ONLY from the real-392 model.

Import is inert. Cloud reads/writes require explicit CLI modes. The original
persistent-local pilot and all previous releases remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from stagequality import train_pilot as pilot
from durability.cloud_backup import Binding, DriveStore, BackupError, build_archive, safe_member, verify_archive

ROOT_ID = 'PRIVATE_DRIVE_ID_OMITTED'
STORAGE_ID = 'b7613af8c2594e13a711145af74cab12'
PARENT_RUN = 'ureal-14a56cf24f5b1116'
TOTAL = 157
STAGE = 'quality5k-full-real392-v1'
RUN_FILES = ('RUN.json', 'baseline-development.json', 'comparison.json', 'RESULT.json')


def package_check(package):
    package = Path(package).resolve()
    pilot.reject_links(package)
    manifest = pilot.read_json(package / 'PACKAGE.json')
    pilot.require(manifest.get('stage') == STAGE, 'Wrong 5k Colab package')
    inventory = manifest.get('files', {})
    pilot.require(inventory and set(inventory) | {'PACKAGE.json'} ==
                  {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()},
                  'Package inventory changed (use the original ZIP)')
    for name, expected in inventory.items():
        path = safe_member(package, name)
        pilot.reject_links(path)
        pilot.require(pilot.sha(path) == expected, 'Package hash mismatch: ' + name)
    pilot.require(not any('final-test' in name or name.endswith(('.pt', '.pth', '.safetensors'))
                          for name in inventory), 'Training kit must not contain weights or final test')
    train, dev = pilot.validate_dataset(package / 'dataset')
    provenance = pilot.read_json(package / 'provenance/real-PACKAGE.json')
    parent_signature = pilot.digest(provenance)
    pilot.require('ureal-' + parent_signature[:16] == PARENT_RUN, 'Parent release changed')
    signature = pilot.digest(manifest)
    binding = Binding('uq5k-' + signature[:16], signature, STORAGE_ID)
    parent_binding = Binding(PARENT_RUN, parent_signature, STORAGE_ID)
    return train, dev, binding, parent_binding


def google_service():
    import google.auth
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default()
    http = httplib2.Http(timeout=60)
    http.redirect_codes = http.redirect_codes - {308}  # Drive resumable status, not redirect.
    return build('drive', 'v3', http=google_auth_httplib2.AuthorizedHttp(credentials, http=http),
                 cache_discovery=False)


def open_store(service, binding, *, create=False):
    # DriveStore's constructor requires consent even for reads. For create=False,
    # only get/list calls are made; prepare/put/prune are never invoked.
    store = DriveStore(service, ROOT_ID, binding, allow_writes=True, allow_prune=False,
                       max_bytes=40 * 1024**3)
    if create:
        store.prepare()
        return store
    folders = store.find(ROOT_ID, name='uznorm-cloud-' + binding.run_id)
    pilot.require(len(folders) <= 1, 'Ambiguous cloud folder; no writes attempted')
    if not folders:
        return None
    folder = folders[0]
    pilot.require(folder.get('parents') == [ROOT_ID]
                  and folder.get('mimeType') == 'application/vnd.google-apps.folder'
                  and all(folder.get('appProperties', {}).get(k) == v for k, v in binding.properties().items()),
                  'Cloud folder/account binding mismatch')
    store.folder_id = folder['id']
    return store


def latest_record(store):
    if store is None:
        return None
    records = []
    for item in store.records():
        props = item.get('appProperties', {})
        pilot.require(all(props.get(k) == v for k, v in store.binding.properties().items()),
                      'Foreign/unbound file in dedicated cloud folder')
        if props.get('kind') not in ('checkpoint', 'final'):
            continue
        item = store.inspect(item['id'])  # Do not skip a broken newer upload.
        step = int(item['appProperties']['step'])
        pilot.require(step >= 0, 'Negative cloud step')
        records.append(item)
    if not records:
        return None
    step = max(int(r['appProperties']['step']) for r in records)
    newest = [r for r in records if int(r['appProperties']['step']) == step]
    for kind in ('checkpoint', 'final'):
        same = [r for r in newest if r['appProperties']['kind'] == kind]
        pilot.require(len({r['appProperties']['payload_sha256'] for r in same}) <= 1,
                      'Conflicting cloud writers; no rollback or overwrite')
    return max(newest, key=lambda r: r['appProperties']['kind'] == 'final')


def cloud_anchor(record):
    if record is None:
        return None
    p = record['appProperties']
    return int(p['step']), p['kind'], p['payload_sha256']


def parent_record(service, parent_binding):
    store = open_store(service, parent_binding)
    record = latest_record(store)
    pilot.require(record is not None and record['appProperties']['kind'] == 'final'
                  and record['appProperties']['step'] == '392',
                  'Completed real-392 cloud model required; no fallback to 3551/313')
    return store, record


def fetch_parent(service, parent_binding, attempt):
    store, record = parent_record(service, parent_binding)
    archive = attempt / 'parent-real-392.zip'
    print(f'PARENT_DOWNLOAD real-392 bytes={record["size"]}; old models untouched', flush=True)
    store.download(record, archive)
    print('PARENT_SHA256 all archive members...', flush=True)
    manifest = verify_archive(archive, parent_binding)
    pilot.require(manifest.get('kind') == 'final' and manifest.get('step') == 392, 'Wrong parent archive')
    with zipfile.ZipFile(archive) as bundle:
        def read(name):
            pilot.require(name in manifest['files'] and bundle.getinfo(name).file_size < 8 * 1024**2,
                          'Missing/oversized parent metadata: ' + name)
            return json.loads(bundle.read(name))
        state, result = read('checkpoint/trainer_state.json'), read('run/RESULT.json')
        meta, parent_kit = read('run/run-meta.json'), read('kit/PACKAGE.json')
        pilot.require(state.get('global_step') == 392 and state.get('epoch') == 1
                      and not state.get('best_model_checkpoint')
                      and result.get('training_finished') is True and result.get('optimizer_steps') == 392
                      and result.get('epochs') == 1
                      and result.get('wandb_id') == PARENT_RUN
                      and meta.get('binding') == parent_binding.properties()
                      and pilot.digest(parent_kit) == parent_binding.signature,
                      'Parent completion/lineage mismatch')
        seal = read('checkpoint/COMPLETE.json')['files']
        for name in pilot.FILES:
            pilot.require(seal.get(name) == manifest['files'].get('checkpoint/' + name)
                          and name in seal, 'Parent inner/outer seal mismatch')
        pilot.require(seal['model.safetensors'] == pilot.PARENTS['real-392'][2], 'Pinned real-392 weights changed')
        target = attempt / 'parent-model'
        target.mkdir(exist_ok=False)
        for name in pilot.FILES:  # No parent optimizer/RNG/pickle deserialization.
            with bundle.open('checkpoint/' + name) as incoming, (target / name).open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    pilot.write_new(target / 'LOCAL_MODEL.json', dict(run_id=PARENT_RUN, step=392,
                    files={name: seal[name] for name in pilot.FILES}))
    metadata = pilot.validate_parent(target, 'real-392')
    metadata.pop('path')
    metadata.update(source_file_id=record['id'], archive_sha256=record['appProperties']['payload_sha256'])
    print('PARENT_VERIFIED real-392; fresh optimizer/scheduler; full fine-tuning', flush=True)
    return target, metadata


def run_signature(binding):
    return pilot.digest(dict(binding=binding.properties(), config=pilot.CONFIG,
                             parent_weights=pilot.PARENTS['real-392'][2], train=pilot.TRAIN_SHA,
                             development=pilot.DEV_SHA))


def validate_run(meta, binding, environment=None):
    pilot.require(meta.get('binding') == binding.properties() and meta.get('signature') == run_signature(binding)
                  and meta.get('config') == pilot.CONFIG and meta.get('train_sha256') == pilot.TRAIN_SHA
                  and meta.get('development_sha256') == pilot.DEV_SHA
                  and meta.get('parent', {}).get('run_id') == PARENT_RUN
                  and meta.get('parent', {}).get('step') == 392
                  and meta.get('parent', {}).get('weights_sha256') == pilot.PARENTS['real-392'][2]
                  and meta.get('fresh_optimizer_scheduler') is True
                  and meta.get('final_test_used') is False,
                  'Resume code/data/parent/config binding changed')
    if environment is not None:
        pilot.require(meta.get('environment') == environment,
                      'Resume package versions/Python/GPU changed; do not silently restart')


def validate_result(output, binding):
    value = pilot.read_json(output / 'RESULT.json')
    pilot.require(value.get('signature') == run_signature(binding)
                  and value.get('step') == TOTAL and value.get('epoch') == 1
                  and value.get('final_test_used') is False,
                  'Final result mismatch')
    for field, name in [('baseline_sha256', 'baseline-development.json'),
                        ('comparison_sha256', 'comparison.json')]:
        pilot.require(value.get(field) == pilot.sha(output / name), 'Final report checksum mismatch')
    name = value.get('after_file', '')
    pilot.require(Path(name).name == name and name.startswith('after-development-')
                  and value.get('after_sha256') == pilot.sha(output / name), 'After report mismatch')
    pilot.validate_final_state(pilot.verify_checkpoint(output / 'checkpoint-157', run_signature(binding)))
    return value


def restore(store, record, package, attempt, binding):
    step, kind, _ = cloud_anchor(record)
    pilot.require(0 <= step <= TOTAL and (kind != 'final' or step == TOTAL), 'Invalid 5k cloud step')
    print(f'RESTORE_CLOUD step={step}/{TOTAL} kind={kind}; only cloud-confirmed progress resumes', flush=True)
    archive = attempt / 'restore.zip'
    store.download(record, archive)
    manifest = verify_archive(archive, binding)
    pilot.require(manifest.get('step') == step and manifest.get('kind') == kind,
                  'Cloud record/archive mismatch')
    package_manifest = pilot.read_json(package / 'PACKAGE.json')
    expected_kit = {'kit/' + n: v for n, v in package_manifest['files'].items()}
    expected_kit['kit/PACKAGE.json'] = pilot.sha(package / 'PACKAGE.json')
    pilot.require({n: v for n, v in manifest['files'].items() if n.startswith('kit/')} == expected_kit,
                  'Restored kit differs from this exact notebook release')
    output = attempt / 'run'
    output.mkdir(exist_ok=False)
    prefix = f'run/checkpoint-{step}/'
    required = {f'run/{n}' for n in ('RUN.json', 'baseline-development.json')}
    pilot.require(required.issubset(manifest['files']), 'Missing bound run/baseline')
    with zipfile.ZipFile(archive) as bundle:
        for name in manifest['files']:
            if name.startswith('kit/'):
                continue
            relative = name.removeprefix('run/')
            allowed = name.startswith(prefix) or relative in RUN_FILES or (
                relative.startswith('after-development-') and '/' not in relative and relative.endswith('.json'))
            pilot.require(name.startswith('run/') and allowed, 'Unexpected restored run member')
            target = safe_member(output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    meta = pilot.read_json(output / 'RUN.json')
    validate_run(meta, binding)
    checkpoint = output / f'checkpoint-{step}'
    state = pilot.verify_checkpoint(checkpoint, run_signature(binding))
    pilot.require(state['global_step'] == step, 'Restored step mismatch')
    if step == TOTAL:
        pilot.validate_final_state(state)
    if kind == 'final':
        validate_result(output, binding)
    return output, checkpoint, meta


class CloudGate:
    def __init__(self, store, output, package, binding, *, record=None, check_disk=True):
        self.store, self.output, self.package, self.binding = store, Path(output), Path(package), binding
        self.anchor = cloud_anchor(record)
        self.last = self.anchor[0] if self.anchor else -1
        self.ready = record is not None
        self.check_disk = check_disk

    def boundary(self, step):
        pilot.require(self.ready and self.last <= step < self.last + pilot.CONFIG['save_steps'],
                      'Cloud backup unconfirmed/overdue; next optimizer step blocked')
        if self.check_disk:
            pilot.check_disk(self.output)

    def publish(self, checkpoint, *, kind='checkpoint', roundtrip=False):
        self.ready = False
        signature = run_signature(self.binding)
        state = pilot.verify_checkpoint(checkpoint, signature)
        step = state['global_step']
        validate_run(pilot.read_json(self.output / 'RUN.json'), self.binding)
        if kind == 'final':
            validate_result(self.output, self.binding)
        pilot.require(kind in ('checkpoint', 'final') and 0 <= step <= TOTAL and step >= self.last,
                      'Cloud snapshot rollback/invalid kind')
        pilot.require(cloud_anchor(latest_record(self.store)) == self.anchor,
                      'Cloud changed since restore/save; possible second writer. No overwrite.')
        members = {}
        for path in Path(checkpoint).iterdir():
            members['run/' + checkpoint.name + '/' + path.name] = (path, pilot.sha(path))
        paths = [self.output / n for n in RUN_FILES]
        paths += sorted(self.output.glob('after-development-*.json'))
        for path in paths:
            if path.is_file():
                members['run/' + path.name] = (path, pilot.sha(path))
        inventory = pilot.read_json(self.package / 'PACKAGE.json')['files']
        for name, expected in inventory.items():
            members['kit/' + name] = (safe_member(self.package, name), expected)
        members['kit/PACKAGE.json'] = (self.package / 'PACKAGE.json', pilot.sha(self.package / 'PACKAGE.json'))
        payload = sum(p.stat().st_size for p, _ in members.values())
        if self.check_disk:
            pilot.require(shutil.disk_usage(self.output).free >= pilot.MIN_RESERVE + payload * (2 if roundtrip else 1),
                          'Insufficient local backup space; nothing deleted')
        stage = Path(tempfile.mkdtemp(prefix='upload-', dir=self.output.parent))
        archive = stage / 'checkpoint.zip'
        print(f'BACKUP_START step={step} kind={kind} payload_GiB={payload/1024**3:.2f}', flush=True)
        checksum = build_archive(archive, members, dict(binding=self.binding.properties(), kind=kind, step=step))
        record = self.store.put(archive, kind=kind, step=step, digest=checksum)
        if roundtrip:
            downloaded = self.store.download(record, stage / 'cloud-readback.zip')
            verify_archive(downloaded, self.binding)
            Path(downloaded).unlink()  # Only this newly verified temporary copy.
        self.anchor, self.last, self.ready = cloud_anchor(record), step, True
        print(f'QUALITY_CLOUD_CONFIRMED step={step} kind={kind} file_id={record["id"]}', flush=True)
        archive.unlink()  # Uploaded, server-verified temp only. Keep ALL checkpoints.
        stage.rmdir()
        return record


def attach_cloud(trainer, gate):
    from transformers import TrainerCallback
    from transformers.trainer_callback import ProgressCallback
    trainer.remove_callback(ProgressCallback)
    trainer.args.disable_tqdm = True  # Visible plain lines survive Colab subprocess pipes.
    class CloudProgress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # pilot.Safety has already saved/sealed step 0 before this callback.
            if not gate.ready:
                gate.publish(gate.output / f'checkpoint-{state.global_step}', roundtrip=True)
            gate.boundary(state.global_step)
            self.started, self.initial = time.monotonic(), state.global_step

        def on_step_begin(self, args, state, control, **kwargs):
            gate.boundary(state.global_step)

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gate.boundary(state.global_step)

        def on_save(self, args, state, control, **kwargs):
            gate.publish(gate.output / f'checkpoint-{state.global_step}')

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                print(f'LOSS step={state.global_step} loss={logs["loss"]} lr={logs.get("learning_rate")}', flush=True)

        def on_step_end(self, args, state, control, **kwargs):
            done = state.global_step - self.initial
            if done == 1 or state.global_step % 10 == 0 or state.global_step == state.max_steps:
                elapsed = time.monotonic() - self.started
                eta = elapsed / max(done, 1) * (state.max_steps - state.global_step)
                print(f'QUALITY_TRAIN {state.global_step}/{state.max_steps} epoch={state.epoch:.3f} '
                      f'elapsed_min={elapsed/60:.1f} ETA_min~{eta/60:.1f}; '
                      'past backup pauses included; final evaluation/upload extra', flush=True)
    trainer.add_callback(CloudProgress())


def environment(attempt):
    env = pilot.gpu_environment(attempt / 'run')
    pilot.require('L4' in env['gpu'], 'This frozen Colab pilot requires NVIDIA L4 + BF16')
    # Kernel hostname/platform path changes across Colab restarts are irrelevant.
    return {k: env[k] for k in ('python', 'packages', 'torch_cuda', 'gpu', 'gpu_bytes')}


def cloud_preflight(service, binding, parent_binding):
    # Read-only account check before any full model download or cloud creation.
    store = open_store(service, binding)
    newest = latest_record(store)
    if newest:
        print(f'EXISTING_5K_CLOUD step={cloud_anchor(newest)[0]} kind={cloud_anchor(newest)[1]}', flush=True)
    else:
        _, parent = parent_record(service, parent_binding)
        print(f'PARENT_AVAILABLE real-392 bytes={parent["size"]}; weights not yet downloaded/verified', flush=True)
    quota = DriveStore.request(service.about().get(fields='storageQuota')).get('storageQuota', {})
    if 'limit' in quota and 'usage' in quota:
        free = int(quota['limit']) - int(quota['usage'])
        print(f'DRIVE_FREE_GiB={free/1024**3:.1f}', flush=True)
        if not newest:
            pilot.require(free >= 30 * 1024**3, 'At least 30 GiB free Drive space requested; no cleanup performed')
    print('CLOUD_PREFLIGHT_OK; no training and no cloud writes in this check', flush=True)


def run(args):
    package = args.package.resolve()
    train, dev, binding, parent_binding = package_check(package)
    print(f'QUALITY_PACKAGE_OK parent=real-392 train=5000 development=60 epoch=1 steps={TOTAL} run={binding.run_id}', flush=True)
    if not args.train and not args.check_cloud:
        return
    if args.train:
        pilot.require(args.allow_cloud and args.old_stopped and args.allow_silver,
                      'Confirm Google backup use, all old training stopped, and Silver data')
    service = google_service()
    cloud_preflight(service, binding, parent_binding)
    if not args.train:
        return
    base = Path('/content')
    pilot.require(base.is_dir() and not base.is_symlink() and (any(k.startswith('COLAB_') for k in os.environ)
                  or Path('/var/colab/hostname').exists()), 'Use a separate Colab L4 runtime')
    # One writer per local runtime. Cross-runtime exclusion still requires user confirmation.
    lock_dir = base / ('.quality-lock-' + binding.run_id)
    lock_dir.mkdir(exist_ok=True)
    with pilot.persistent_lock(lock_dir):
        attempt = Path(tempfile.mkdtemp(prefix='uznorm-quality5k-', dir=base))
        env = environment(attempt)
        store = open_store(service, binding, create=True)
        newest = latest_record(store)
        checkpoint = None
        if newest:
            output, checkpoint, meta = restore(store, newest, package, attempt, binding)
            pilot.validate_baseline(output, run_signature(binding), dev)
            if newest['appProperties']['kind'] == 'final':
                print('QUALITY_ALREADY_COMPLETE; no model loaded and no optimizer step executed', flush=True)
                show_result(output, checkpoint, newest)
                return
            validate_run(meta, binding, env)
            model_path = checkpoint
        else:
            model_path, parent_meta = fetch_parent(service, parent_binding, attempt)
            output = attempt / 'run'; output.mkdir(exist_ok=False)
            meta = dict(binding=binding.properties(), signature=run_signature(binding), config=pilot.CONFIG,
                        environment=env, parent=parent_meta, fresh_optimizer_scheduler=True,
                        train_sha256=pilot.TRAIN_SHA, development_sha256=pilot.DEV_SHA,
                        human_gold=False, final_test_used=False)
            pilot.write_new(output / 'RUN.json', meta)
        pilot.offline()
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
        set_seed(pilot.CONFIG['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        print('MODEL_LOADING full ByT5-base; all parameters trainable', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                                    use_safetensors=True, dtype=torch.float32)
        pilot.require(sum(p.numel() for p in model.parameters()) == 581653248
                      and tokenizer.__class__.__name__ == 'ByT5Tokenizer', 'Expected ByT5-base')
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.config.use_cache = False; model.generation_config.use_cache = True
        trainer = pilot.make_trainer(model, tokenizer, train, dev, output, run_signature(binding))
        gate = CloudGate(store, output, package, binding, record=newest)
        attach_cloud(trainer, gate)
        # Smoke exercises two long rows, but never advances the optimizer.
        from stage10k.runner import smoke
        smoke(trainer, train)
        if not checkpoint:
            print('BASELINE_DEVELOPMENT_START 60; not final test', flush=True)
            metrics, predictions = pilot.evaluate_development(trainer, dev)
            pilot.write_new(output / 'baseline-development.json', dict(signature=run_signature(binding),
                            development_sha256=pilot.DEV_SHA, metrics=metrics, predictions=predictions))
            # Baseline decoding must not alter the seeded training RNG.
            set_seed(pilot.CONFIG['seed'])
        if not checkpoint or pilot.read_json(checkpoint / 'trainer_state.json')['global_step'] < TOTAL:
            print('TRAINING_START full fine-tuning; 157 steps; backup every 25 steps', flush=True)
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            checkpoint = output / 'checkpoint-157'
            if not checkpoint.exists():
                trainer._save_checkpoint(model, None)
                gate.publish(checkpoint)
        pilot.validate_final_state(pilot.verify_checkpoint(checkpoint, run_signature(binding)))
        print('AFTER_DEVELOPMENT_START 60; no weight updates', flush=True)
        metrics, predictions = pilot.evaluate_development(trainer, dev)
        after = pilot.write_evaluation_attempt(output, run_signature(binding), metrics, predictions)
        before = pilot.read_json(output / 'baseline-development.json')['metrics']
        comparison = dict(before=before, after=metrics, n=60, human_gold=False, final_test_used=False,
                          automatic_model_replacement=False)
        # Restored checkpoint archives have no final reports. A failed-final-upload
        # attempt is left on disk; a new attempt is restored from cloud, not overwritten.
        pilot.write_new(output / 'comparison.json', comparison)
        pilot.validate_dataset(package / 'dataset')
        if not newest:
            pilot.validate_parent(model_path, 'real-392')
        pilot.write_new(output / 'RESULT.json', dict(signature=run_signature(binding), step=TOTAL, epoch=1,
            after_file=after.name, after_sha256=pilot.sha(after), baseline_sha256=pilot.sha(output / 'baseline-development.json'),
            comparison_sha256=pilot.sha(output / 'comparison.json'), final_test_used=False, human_gold=False,
            parent_unchanged=True, automatic_model_replacement=False))
        record = gate.publish(checkpoint, kind='final', roundtrip=True)
        print('QUALITY_STAGE_COMPLETE_CLOUD_VERIFIED step=157 epoch=1', flush=True)
        show_result(output, checkpoint, record)


def show_result(output, checkpoint, record):
    print('MODEL:', checkpoint, '\nCOMPARISON:', output / 'comparison.json', flush=True)
    print('FINAL_BACKUP: PRIVATE_ARTIFACT_LINK_OMITTED' + record['id'] + '/view', flush=True)
    stats = pilot.read_json(output / 'comparison.json')
    for name in ('exact_match_pct', 'raw_cer_pct', 'spelling_cer_pct', 'content_wer_pct',
                 'punctuation_macro_f1_pct', 'identity_change_rate_pct'):
        before = stats['before']['model']['apostrophe_equivalent'].get(name)
        after = stats['after']['model']['apostrophe_equivalent'].get(name)
        print(f'{name}: {before} -> {after}', flush=True)
    print('60 Silver development examples, not general accuracy. Inspect raw predictions/meaning.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    for name in ('check-cloud', 'train', 'allow-cloud', 'old-stopped', 'allow-silver'):
        p.add_argument('--' + name, action='store_true')
    return p


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        print('QUALITY_INTERRUPTED: last cloud-confirmed checkpoint retained. Do not reset during upload.', flush=True)
        raise SystemExit(130)
    except Exception as exc:
        safe = str(exc) if isinstance(exc, (RuntimeError, BackupError)) else 'No raw network/auth body logged.'
        print('QUALITY_STOPPED:', type(exc).__name__, safe, flush=True)
        print('Do not delete backups or start a second writer. Full checkpoint resume only.', flush=True)
        raise SystemExit(1)
