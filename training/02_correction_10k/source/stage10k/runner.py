"""10k warm-start with local full-state checkpoints and synchronous Drive API backup.

No network, GPU, or filesystem mutations occur merely by importing this module.
Parent optimizer/scheduler are never loaded. Resume applies ONLY to this stage.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

from uznorm.tracking import privacy_defaults
privacy_defaults()

from durability.cloud_backup import (BackupError, Binding, DriveStore, build_archive,
                                    hashes, safe_member, sealed_members, verify_archive)
from uznorm.io import digest, read_json, read_jsonl, sha256, write_json, write_jsonl

PARENT = Binding('79e3c655',
    'b7c162b5626bf22f5edf441ead00c66c7e7b1594f9ae00b02d818ffaf92c0f5e',
    'b7613af8c2594e13a711145af74cab12')
PARENT_FILE = '14iXysxt0eLGQcNnWUhnHwLNPyQgkvIb5'
DRIVE_ROOT = 'PRIVATE_DRIVE_ID_OMITTED'
TRAIN_SHA = '51ea62e0b40e570ac269d7c08f0f3190e50dd54146523ae08d9472c31e727cea'
QUOTAS = dict(lexical=3500, mixed=2500, punctuation=1800, casing=1000,
              og_apostrophe=800, tutuq=400)
REQUIRED = ('config.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')


def google_service():
    import google.auth
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default()
    http = httplib2.Http(timeout=60)
    # Drive resumable uploads use 308 WITHOUT Location. It is not a redirect.
    http.redirect_codes = http.redirect_codes - {308}
    transport = google_auth_httplib2.AuthorizedHttp(credentials, http=http)
    return build('drive', 'v3', http=transport, cache_discovery=False)


def other_gpu_pids(text, own_pid):
    pids = {int(line.strip()) for line in text.splitlines() if line.strip()}
    return pids - {own_pid}


def package_check(package):
    from uznorm.data import CATEGORIES, validate_row, _canonical, _grams
    package = Path(package).resolve()
    manifest = read_json(package / 'PACKAGE.json')
    for name, expected in manifest['files'].items():
        if sha256(safe_member(package, name)) != expected:
            raise ValueError('Package checksum mismatch: ' + name)
    config = read_json(package / 'config.json')
    if sha256(package / 'data/train.jsonl') != TRAIN_SHA:
        raise ValueError('This stage requires the frozen 10k selection')
    train = list(read_jsonl(package / 'data/train.jsonl'))
    monitor = list(read_jsonl(package / 'data/monitor-128.jsonl'))
    model_config = SimpleNamespace(**config['model'])
    for split, rows in [('train', train), ('validation', monitor)]:
        for row in rows:
            validate_row(row, split, model_config)
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate IDs')
    if Counter(r['category'] for r in train) != QUOTAS:
        raise ValueError('10k category quotas changed')
    if Counter(r['category'] for r in monitor) != {c: 16 for c in CATEGORIES}:
        raise ValueError('The fixed development monitor must contain 16 per category')
    for field in ('id', 'document_id', 'group_id', 'source_id'):
        if {r[field] for r in train} & {r[field] for r in monitor}:
            raise ValueError('Train/development leakage: ' + field)
    train_text = {_canonical(r[f]) for r in train for f in ('input', 'target')}
    train_grams = {g for r in train for f in ('input', 'target') for g in _grams(r[f])}
    if any(_canonical(r[f]) in train_text or train_grams.intersection(_grams(r[f]))
           for r in monitor for f in ('input', 'target')):
        raise ValueError('Train/development text leakage')
    signature = digest(manifest)
    return config, train, monitor, Binding('u10k-' + signature[:16], signature, PARENT.storage_id)


def require_disk(path, payload=0, copies=1):
    if shutil.disk_usage(path).free < 8 * 1024**3 + payload * copies:
        raise BackupError('Local disk reserve is low. No old/cloud checkpoint was deleted.')


def latest_record(store):
    candidates = []
    for record in store.records():
        props = record.get('appProperties', {})
        if props.get('kind') not in ('checkpoint', 'final'):
            continue
        # Do not silently skip a broken newer checkpoint and roll back.
        record = store.inspect(record['id'])
        props = record['appProperties']
        step = int(props['step'])
        if step < 0:
            raise BackupError('Negative cloud step')
        candidates.append((step, props['kind'] == 'final', record))
    if not candidates:
        return None
    newest_step = max(c[0] for c in candidates)
    same = [c for c in candidates if c[0] == newest_step]
    # Multiple different payloads for the same kind/step mean possible writers.
    for kind in (False, True):
        if len({c[2]['appProperties']['payload_sha256'] for c in same if c[1] == kind}) > 1:
            raise BackupError('Conflicting cloud checkpoints. Do not run a second writer.')
    return max(candidates, key=lambda c: (c[0], c[1]))[2]


def unpack_verified(archive, binding, output):
    manifest = verify_archive(archive, binding)
    output = Path(output)
    output.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        require_disk(output, sum(i.file_size for i in bundle.infolist()))
        for name in manifest['files']:
            target = safe_member(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    return manifest


def fetch_parent(service, work):
    """Read-only access to the OLD cloud folder; no prepare/put/prune on it."""
    store = DriveStore(service, DRIVE_ROOT, PARENT, allow_writes=True, allow_prune=False)
    folders = store.find(DRIVE_ROOT, name='uznorm-cloud-' + PARENT.run_id)
    if len(folders) != 1:
        raise BackupError('Old cloud folder missing/ambiguous; do not start from random weights')
    folder = folders[0]
    if (folder.get('parents') != [DRIVE_ROOT]
            or folder.get('mimeType') != 'application/vnd.google-apps.folder'
            or any(folder.get('appProperties', {}).get(k) != v for k, v in PARENT.properties().items())):
        raise BackupError('Parent folder binding mismatch')
    store.folder_id = folder['id']
    record = store.inspect(PARENT_FILE)
    if record['appProperties'].get('step') != '3551' or record['appProperties'].get('kind') != 'checkpoint':
        raise BackupError('Expected the confirmed parent checkpoint-3551 archive')
    print('PARENT_DOWNLOAD: cloud checkpoint-3551 (~4.34 GiB); original files unchanged.', flush=True)
    archive = Path(work) / 'parent-3551.zip'
    require_disk(work, int(record['size']), 3)
    store.download(record, archive)
    manifest = verify_archive(archive, PARENT)
    # Extract only model/tokenizer/config and the non-executable JSON state for lineage.
    # Parent optimizer.pt/scheduler.pt/RNG are NEVER deserialized for warm-start.
    suffix = '/recovery/checkpoint-3551/trainer_state.json'
    state_names = [n for n in manifest['files'] if n.endswith(suffix)]
    if len(state_names) != 1:
        raise BackupError('3551 state missing/ambiguous inside parent archive')
    prefix = state_names[0].removesuffix('trainer_state.json')
    target = Path(work) / 'parent-model'
    target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        state = json.loads(bundle.read(state_names[0]))
        if state.get('global_step') != 3551:
            raise BackupError('Parent trainer state is not 3551')
        for name in manifest['files']:
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            if (relative.endswith('.safetensors') or relative.endswith('.json')) and relative not in ('COMPLETE.json', 'RECOVERY.json'):
                dest = safe_member(target, relative)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as incoming, dest.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    if not (target / 'model.safetensors').exists() and not (target / 'model.safetensors.index.json').exists():
        raise BackupError('Parent safetensors missing')
    print('PARENT_VERIFIED: step=3551. Only model/tokenizer will initialize the new stage.', flush=True)
    return target, {'file_id': record['id'], 'sha256': record['appProperties']['payload_sha256'],
                    'parent_step': 3551, 'parent_epoch': state.get('epoch')}


class StageGate:
    def __init__(self, store, run, package, *, interval=50, check_disk=True):
        self.store, self.run, self.package = store, Path(run), Path(package)
        self.interval, self.check_disk = interval, check_disk
        self.last_cloud_step, self.ready = -1, False

    def boundary(self, step):
        if not self.ready or step < self.last_cloud_step or step - self.last_cloud_step >= self.interval:
            raise BackupError('Cloud checkpoint overdue/unverified; next optimizer step blocked')
        if self.check_disk:
            require_disk(self.run)

    def publish(self, checkpoint, *, kind='checkpoint', roundtrip=False):
        self.ready = False
        files = sealed_members(checkpoint, required=REQUIRED)
        state = read_json(Path(checkpoint) / 'trainer_state.json')
        step = state['global_step']
        if type(step) is not int or state.get('best_model_checkpoint'):
            raise BackupError('New-stage snapshot must be independent of any old best checkpoint')
        newest = latest_record(self.store)
        if newest and int(newest['appProperties']['step']) > step:
            raise BackupError('A newer cloud snapshot exists. Concurrent writer/rollback refused.')
        members = {'checkpoint/' + name: item for name, item in files.items()}
        for name in ('run-meta.json', 'baseline.json', 'baseline-predictions.jsonl',
                     'after.json', 'after-predictions.jsonl', 'comparison.json', 'RESULT.json'):
            path = self.run / name
            if path.exists():
                members['run/' + name] = (path, sha256(path))
        # Small self-contained code/config/data in every archive; no old 3000 weights.
        package_manifest = read_json(self.package / 'PACKAGE.json')
        for name, expected in package_manifest['files'].items():
            members['kit/' + name] = (safe_member(self.package, name), expected)
        members['kit/PACKAGE.json'] = (self.package / 'PACKAGE.json', sha256(self.package / 'PACKAGE.json'))
        payload = sum(path.stat().st_size for path, _ in members.values())
        if self.check_disk:
            require_disk(self.run, payload, 2 if roundtrip else 1)
        stage = Path(tempfile.mkdtemp(prefix='upload-', dir=self.run.parent))
        archive = stage / 'checkpoint.zip'
        checksum = build_archive(archive, members, {'binding': self.store.binding.properties(),
                                                   'kind': kind, 'step': step})
        record = self.store.put(archive, kind=kind, step=step, digest=checksum)
        if roundtrip:
            downloaded = stage / 'cloud-readback.zip'
            self.store.download(record, downloaded)
            verify_archive(downloaded, self.store.binding)
            downloaded.unlink()  # This unique, verified temporary download only.
        self.last_cloud_step, self.ready = step, True
        print(f'STAGE_CLOUD_CONFIRMED step={step} kind={kind} file_id={record["id"]}', flush=True)
        archive.unlink()  # Unique local temporary archive; checkpoint/cloud file retained.
        stage.rmdir()
        return record


def training_args(config, output, *, cpu=False):
    from transformers import Seq2SeqTrainingArguments
    t = config['training']
    return Seq2SeqTrainingArguments(
        output_dir=str(output), num_train_epochs=t['epochs'],
        per_device_train_batch_size=t['micro_batch'], gradient_accumulation_steps=t['accumulation'],
        per_device_eval_batch_size=1, learning_rate=t['learning_rate'], warmup_ratio=t['warmup_ratio'],
        optim='adafactor', lr_scheduler_type='linear', weight_decay=0, max_grad_norm=1.0,
        bf16=not cpu, fp16=False, tf32=False, use_cpu=cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='no', save_strategy='steps', save_steps=t['save_steps'], save_total_limit=None,
        save_only_model=False, save_safetensors=True, load_best_model_at_end=False,
        logging_steps=10, report_to=[] if cpu else ['wandb'], run_name=config['name'],
        predict_with_generate=True, generation_num_beams=1,
        generation_max_length=config['model']['max_target_tokens'] + 1,
        dataloader_num_workers=0, eval_accumulation_steps=1,
        seed=t['seed'], data_seed=t['seed'], disable_tqdm=False)


def make_trainer(model, tokenizer, train, monitor, config, output, gate, *, cpu=False):
    import torch
    from transformers import Seq2SeqTrainer, TrainerCallback, DataCollatorForSeq2Seq
    from uznorm.training import SourceOnlyTrainer
    from uznorm.checkpoints import complete_checkpoint
    from uznorm.data import TextDataset
    from uznorm.metrics import make_compute_metrics

    class LocalTrainer(SourceOnlyTrainer):
        def _save_checkpoint(self, model, trial):
            checkpoint = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
            if checkpoint.exists():
                raise BackupError('Refusing to overwrite an existing local checkpoint')
            if gate.check_disk:
                require_disk(output, sum(p.numel() * p.element_size() for p in model.parameters()), 2)
            # Deliberately replace only the old FUSE save wrapper, not generation/loss safety.
            Seq2SeqTrainer._save_checkpoint(self, model, trial)
            complete_checkpoint(checkpoint)
            gate.publish(checkpoint, roundtrip=self.state.global_step == 0)

    class Boundary(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step == 0 and not gate.ready:
                trainer._save_checkpoint(trainer.model, None)
            gate.boundary(state.global_step)

        def on_step_begin(self, args, state, control, **kwargs):
            gate.boundary(state.global_step)

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            # Finite loss is checked in SourceOnlyTrainer; gradients must also be finite.
            finite = [torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None]
            if not finite or not torch.stack(finite).all().item():
                raise FloatingPointError('Non-finite gradient; optimizer step blocked')

    mc = SimpleNamespace(**config['model'])
    trainer = LocalTrainer(model=model, args=training_args(config, output, cpu=cpu),
        train_dataset=TextDataset(train, tokenizer, mc), eval_dataset=TextDataset(monitor, tokenizer, mc),
        processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        compute_metrics=make_compute_metrics(tokenizer, monitor), callbacks=[Boundary()])
    return trainer


def evaluate(trainer, rows, path, prefix):
    import numpy as np
    result = trainer.predict(trainer.eval_dataset, metric_key_prefix=prefix,
                             max_length=trainer.args.generation_max_length, num_beams=1, do_sample=False)
    ids = np.where(result.predictions == -100, trainer.processing_class.pad_token_id, result.predictions)
    predictions = trainer.processing_class.batch_decode(ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
    write_json(Path(path) / (prefix + '.json'), result.metrics)
    write_jsonl(Path(path) / (prefix + '-predictions.jsonl'),
                ({'id': r['id'], 'input': r['input'], 'target': r['target'], 'prediction': p,
                  'category': r['category']} for r, p in zip(rows, predictions)))
    return result.metrics


def gpu_preflight():
    import importlib.metadata
    import torch
    from packaging.version import Version
    from uznorm.hardware import environment
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Bitta NVIDIA GPU kerak. L4 runtime tanlang.')
    if Version(torch.__version__.split('+')[0]) < Version('2.6'):
        raise RuntimeError('torch >= 2.6 kerak; eski pickle checkpoint xavfsiz yuklanmaydi.')
    if 'L4' not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Ushbu muzlatilgan config NVIDIA L4 + BF16 uchun. GPU/configni yashirincha almashtirmang.')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    # Device capability queries may have created THIS fresh process's CUDA context.
    # The current process has not loaded a model yet; every OTHER GPU PID blocks startup.
    if other_gpu_pids(result.stdout, os.getpid()):
        raise RuntimeError('GPUda boshqa jarayon bor. Uni avtomatik to‘xtatmaymiz.')
    result = environment()
    result['packages'].update({name: importlib.metadata.version(name) for name in
        ('google-api-python-client', 'google-auth-httplib2', 'google-auth', 'httplib2', 'safetensors', 'numpy')})
    return {**result, 'gpu': torch.cuda.get_device_name(0),
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory}


def smoke(trainer, rows):
    """Two long training rows, forward/backward; never applies an optimizer step."""
    import torch
    from transformers import set_seed
    indexes = sorted(range(len(rows)), key=lambda i: max(len(rows[i]['input'].encode()),
                      len(rows[i]['target'].encode())), reverse=True)[:trainer.args.per_device_train_batch_size]
    batch = trainer.data_collator([trainer.train_dataset[i] for i in indexes])
    batch = trainer._prepare_inputs(batch)
    model = trainer.model
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.zero_grad(set_to_none=True)
    with trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(model, batch)
    loss.backward()
    finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not finite or not torch.stack(finite).all().item():
        raise FloatingPointError('GPU smoke: non-finite gradients')
    value = float(loss.detach().cpu())
    model.zero_grad(set_to_none=True)
    del loss, batch
    torch.cuda.empty_cache()
    set_seed(trainer.args.seed)
    print(f'GPU_SMOKE_OK finite_loss={value:.6f}; optimizer steps=0; weights unchanged.', flush=True)


def run(package, *, allow_cloud=False, old_stopped=False, allow_non_gold=False, work='/content'):
    if not allow_cloud or not old_stopped or not allow_non_gold:
        raise RuntimeError('Cloud backup, barcha eski treninglar to‘xtagani va non-Gold data roziligi kerak.')
    config, train, monitor, binding = package_check(package)
    if not os.environ.get('WANDB_API_KEY', '').strip():
        raise RuntimeError('Colab Secrets: WANDB_API_KEY va shu notebook uchun Notebook access kerak.')
    env = gpu_preflight()
    base = Path(work).resolve()
    if base != Path('/content') or not base.is_dir():
        raise RuntimeError('Production work must be the Colab local /content disk, not mounted Drive.')
    require_disk(base, 24 * 1024**3)
    service = google_service()
    store = DriveStore(service, DRIVE_ROOT, binding, allow_writes=True,
                       allow_prune=False, max_bytes=40 * 1024**3)
    store.prepare()  # Checks the cloud-side storage marker/account BEFORE any model load.
    newest = latest_record(store)
    attempt = Path(tempfile.mkdtemp(prefix='uznorm-10k-', dir=base))
    output = attempt / 'run'
    output.mkdir()
    resume = None
    if newest:
        print('RESTORE_NEW_STAGE step=' + newest['appProperties']['step'], flush=True)
        archive = attempt / 'restore.zip'
        require_disk(attempt, int(newest['size']), 3)
        store.download(newest, archive)
        unpack_verified(archive, binding, attempt / 'restore')
        saved = attempt / 'restore/run'
        meta = read_json(saved / 'run-meta.json')
        if meta.get('binding') != binding.properties() or meta.get('environment') != env or meta.get('config') != config:
            raise BackupError('New-stage config/code/data/environment differs; resume refused.')
        for path in saved.iterdir():
            if path.is_file():
                shutil.copyfile(path, output / path.name)
        model_path = attempt / 'restore/checkpoint'
        from uznorm.io import verify_seal
        verify_seal(model_path, 'COMPLETE.json', REQUIRED)
        state = read_json(model_path / 'trainer_state.json')
        if state['global_step'] != int(newest['appProperties']['step']) or state.get('best_model_checkpoint'):
            raise BackupError('Restored new-stage checkpoint state mismatch')
        if newest['appProperties']['kind'] == 'final':
            if not (output / 'RESULT.json').is_file():
                raise BackupError('Final cloud archive has no result')
            print('STAGE_ALREADY_COMPLETE. No optimizer step executed.', flush=True)
            print('Comparison:', output / 'comparison.json', flush=True)
            return output
        resume = model_path
    else:
        model_path, parent = fetch_parent(service, attempt)
        meta = {'schema': 1, 'binding': binding.properties(), 'config': config, 'environment': env,
                'parent': parent, 'initialization': 'parent_model_and_tokenizer_only',
                'fresh_optimizer_scheduler': True, 'data_quality': 'screened_synthetic_not_gold'}
        write_json(output / 'run-meta.json', meta)

    import torch
    import wandb
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    set_seed(config['training']['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True)
    if model.config.model_type != 't5' or sum(p.numel() for p in model.parameters()) != 581653248 or tokenizer.__class__.__name__ != 'ByT5Tokenizer':
        raise BackupError('Expected the verified ByT5-base model/tokenizer, not a different base model')
    model.config.use_cache = False
    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=False)
    tracking = wandb.init(project=config['wandb_project'], id=binding.run_id, name=config['name'],
        resume='allow', mode='online', dir=str(attempt), config={'stage': meta, 'global_step_is_new_stage': True},
        save_code=False, settings=wandb.Settings(disable_code=True, console='off', disable_job_creation=True))
    # The stable stage ID allows recovery if interrupted between W&B init and checkpoint-0.
    # It never targets the old W&B ID. Cloud checkpoints, not W&B, determine model recovery.
    gate = StageGate(store, output, package, interval=config['training']['save_steps'])
    if resume:
        gate.last_cloud_step, gate.ready = state['global_step'], True
    trainer = make_trainer(model, tokenizer, train, monitor, config, output, gate)
    try:
        smoke(trainer, train)
        if not resume:
            print('BASELINE: parent-3551 on fixed 128 old validation records (not Gold).', flush=True)
            baseline = evaluate(trainer, monitor, output, 'baseline')
            tracking.log(baseline)
        elif not (output / 'baseline.json').exists():
            raise BackupError('Original before-training baseline missing from cloud snapshot')
        total = math.ceil(config['training']['epochs'] * math.ceil(
            math.ceil(len(train) / config['training']['micro_batch']) / config['training']['accumulation']))
        current = state['global_step'] if resume else 0
        print(f'STAGE_TRAIN parent=3551 new_step={current}/{total} epochs={config["training"]["epochs"]} wandb={binding.run_id}', flush=True)
        if current < total:
            trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            checkpoint = output / f'checkpoint-{trainer.state.global_step}'
            if not checkpoint.exists():
                trainer._save_checkpoint(trainer.model, None)
        else:
            checkpoint = resume
            trainer.state.global_step = current
        if trainer.state.global_step != total:
            raise RuntimeError('Epoch incomplete; resume later from last STAGE_CLOUD_CONFIRMED.')
        after_metrics = evaluate(trainer, monitor, output, 'after')
        tracking.log(after_metrics)
        before, after = read_json(output / 'baseline.json'), read_json(output / 'after.json')
        comparison = {key.removeprefix('after_'): {'before': before.get('baseline_' + key.removeprefix('after_')),
                      'after': value} for key, value in after.items()
                      if key.startswith('after_') and 'baseline_' + key.removeprefix('after_') in before}
        write_json(output / 'comparison.json', {'monitor_n': len(monitor), 'not_independent_gold_test': True,
                                               'metrics': comparison})
        write_json(output / 'RESULT.json', {'optimizer_steps': total, 'epochs': 1,
                    'parent_step': 3551, 'wandb_id': binding.run_id, 'comparison': 'comparison.json',
                    'training_finished': True, 'remote_confirmation_requires_final_archive': True})
        record = gate.publish(checkpoint, kind='final', roundtrip=True)
        print(f'STAGE_COMPLETE_CLOUD_VERIFIED step={total} file_id={record["id"]}', flush=True)
        print('COMPARISON:', output / 'comparison.json', flush=True)
        return output
    finally:
        tracking.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--allow-cloud', action='store_true')
    parser.add_argument('--old-stopped', action='store_true')
    parser.add_argument('--allow-non-gold', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        config, train, monitor, binding = package_check(args.package)
        print(f'PACKAGE_OK train={len(train)} monitor={len(monitor)} run={binding.run_id}')
        return
    try:
        run(args.package, allow_cloud=args.allow_cloud, old_stopped=args.old_stopped,
            allow_non_gold=args.allow_non_gold)
    except KeyboardInterrupt:
        print('PAUSED: resume from the last STAGE_CLOUD_CONFIRMED, not an incomplete local save.', flush=True)
        raise SystemExit(130)
    except BackupError as exc:
        print('BACKUP_BLOCKED:', exc, flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
