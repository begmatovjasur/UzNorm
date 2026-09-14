"""Child of the completed correction10k step-313 model; never resumes its optimizer."""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import tempfile
import time
import zipfile

from stage10k.runner import (StageGate, google_service, require_disk, latest_record,
    unpack_verified, gpu_preflight, smoke, make_trainer, evaluate, DRIVE_ROOT, REQUIRED)
from durability.cloud_backup import Binding, DriveStore, BackupError, safe_member, verify_archive
from uznorm.io import digest, read_json, read_jsonl, sha256, write_json, verify_seal
from stagereal.prepare import check_splits, keys

PARENT = Binding('u10k-7c8dd4ae7f962a9d',
    '7c8dd4ae7f962a9de88103a3c47015b819f06e7503d5be47072207c366a6a202',
    'b7613af8c2594e13a711145af74cab12')
PARENT_STEP = 313


def package_check(package):
    package = Path(package).resolve()
    manifest = read_json(package / 'PACKAGE.json')
    if manifest.get('stage') != 'real-reviews-v1':
        raise ValueError('Wrong stage package')
    for name, expected in manifest['files'].items():
        if sha256(safe_member(package, name)) != expected:
            raise ValueError('Package checksum mismatch: ' + name)
    config = read_json(package / 'config.json')
    report = read_json(package / 'provenance/split.json')
    splits = {s: list(read_jsonl(package / f'data/{s}.jsonl')) for s in ('train', 'validation', 'test')}
    check_splits(splits)
    if {s: len(v) for s, v in splits.items()} != report['counts']:
        raise ValueError('Split counts changed')
    real = list(read_jsonl(package / 'data/real-monitor-144.jsonl'))
    development = {r['id']: r for r in splits['validation']}
    if len(real) != 144 or len({r['id'] for r in real}) != 144 or any(development.get(r['id']) != r for r in real):
        raise ValueError('Real monitor is not an exact validation subset')
    synthetic = list(read_jsonl(package / 'data/synthetic-monitor-128.jsonl'))
    if len(synthetic) != 128:
        raise ValueError('Old-format regression monitor changed')
    from types import SimpleNamespace
    from uznorm.data import validate_row
    for row in synthetic:
        validate_row(row, 'validation', SimpleNamespace(**config['model']))
    train_keys = {key for r in splits['train'] for key in keys(r)}
    if any(train_keys & keys(r) for r in synthetic):
        raise ValueError('Real train / synthetic monitor text overlap')
    monitor = real + [{**r, 'evaluation_domain': 'synthetic'} for r in synthetic]
    if len({r['id'] for r in monitor}) != len(monitor):
        raise ValueError('Monitor IDs overlap')
    signature = digest(manifest)
    return config, splits['train'], monitor, Binding('ureal-' + signature[:16], signature, PARENT.storage_id)


def validate_parent_metadata(manifest, result, meta, state, kit_manifest):
    if (manifest.get('kind') != 'final' or manifest.get('step') != PARENT_STEP
        or result.get('training_finished') is not True or result.get('optimizer_steps') != PARENT_STEP
        or result.get('epochs') != 1 or result.get('wandb_id') != PARENT.run_id
        or meta.get('binding') != PARENT.properties() or digest(kit_manifest) != PARENT.signature
        or state.get('global_step') != PARENT_STEP or state.get('epoch') != 1
        or state.get('best_model_checkpoint')):
        raise BackupError('Expected completed 10k model step=313 epoch=1. No fallback to 3551.')


def fetch_parent(service, work):
    # The DriveStore constructor requires consent, but NO prepare/put/prune is called on parent.
    store = DriveStore(service, DRIVE_ROOT, PARENT, allow_writes=True, allow_prune=False)
    folders = store.find(DRIVE_ROOT, name='uznorm-cloud-' + PARENT.run_id)
    if len(folders) != 1:
        raise BackupError('Completed 10k cloud folder missing/ambiguous; original weights not used as fallback.')
    folder = folders[0]
    if (folder.get('parents') != [DRIVE_ROOT] or folder.get('mimeType') != 'application/vnd.google-apps.folder'
            or any(folder.get('appProperties', {}).get(k) != v for k, v in PARENT.properties().items())):
        raise BackupError('Parent folder/account binding mismatch')
    store.folder_id = folder['id']
    record = latest_record(store)
    if not record or record['appProperties'].get('kind') != 'final' or record['appProperties'].get('step') != str(PARENT_STEP):
        raise BackupError('Final cloud-confirmed 10k step-313 is required; training was not started.')
    archive = Path(work) / 'parent-10k-313.zip'
    require_disk(work, int(record['size']), 3)
    print('PARENT_DOWNLOAD 10k step=313 bytes=' + record['size'] + '; checkpoint-3551 is NOT selected.', flush=True)
    store.download(record, archive)
    print('PARENT_VERIFY all archive SHA-256 checks...', flush=True)
    manifest = verify_archive(archive, PARENT)
    target = Path(work) / 'parent-model'
    with zipfile.ZipFile(archive) as bundle:
        def js(name):
            if name not in manifest['files']:
                raise BackupError('Parent archive metadata missing: ' + name)
            return json.loads(bundle.read(name))
        state = js('checkpoint/trainer_state.json')
        validate_parent_metadata(manifest, js('run/RESULT.json'), js('run/run-meta.json'),
                                 state, js('kit/PACKAGE.json'))
        complete = js('checkpoint/COMPLETE.json')['files']
        if not set(REQUIRED).issubset(complete):
            raise BackupError('Incomplete parent state seal')
        for name, expected in complete.items():
            if manifest['files'].get('checkpoint/' + name) != expected:
                raise BackupError('Parent inner seal differs from cloud archive')
        target.mkdir(exist_ok=False)
        for name in manifest['files']:
            if not name.startswith('checkpoint/'):
                continue
            relative = name.removeprefix('checkpoint/')
            if relative == 'COMPLETE.json' or relative not in complete:
                continue
            if relative.endswith(('.json', '.safetensors')):
                path = safe_member(target, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as incoming, path.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
    if not (target / 'model.safetensors').is_file() and not (target / 'model.safetensors.index.json').is_file():
        raise BackupError('Parent model weights missing')
    print('PARENT_VERIFIED run=' + PARENT.run_id + ' step=313 epoch=1; fresh optimizer for real reviews.', flush=True)
    return target, {'run_id': PARENT.run_id, 'parent_step': 313, 'parent_epoch': 1,
        'file_id': record['id'], 'sha256': record['appProperties']['payload_sha256']}


def scored_evaluation(trainer, monitor, output, prefix):
    """Predict ONCE; report real reviews and synthetic regression separately."""
    from uznorm.metrics import score, numeric_metrics
    print(prefix.upper() + f'_EVAL samples={len(monitor)} (144 real + 128 synthetic; not Gold)', flush=True)
    combined = evaluate(trainer, monitor, output, prefix)
    predictions = list(read_jsonl(Path(output) / (prefix + '-predictions.jsonl')))
    if [p['id'] for p in predictions] != [r['id'] for r in monitor]:
        raise ValueError('Prediction alignment mismatch')
    domains = {}
    for domain in ('real', 'synthetic'):
        indices = [i for i, r in enumerate(monitor) if r['evaluation_domain'] == domain]
        domains[domain] = numeric_metrics(score([monitor[i] for i in indices],
                                [predictions[i]['prediction'] for i in indices], breakdown=False))
    value = {'combined': combined, 'domains': domains, 'not_gold': True}
    write_json(Path(output) / (prefix + '.json'), value)
    return {prefix + '_' + domain + '_' + key: val for domain, stats in domains.items() for key, val in stats.items()}


def comparison(before, after):
    return {'not_independent_gold_test': True, 'test_set_used': False,
        'metrics': {domain: {key: {'before': value, 'after': after['domains'][domain].get(key)}
                   for key, value in stats.items()} for domain, stats in before['domains'].items()}}


def progress_callback():
    from transformers import TrainerCallback
    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started, self.start_step = time.monotonic(), state.global_step
        def on_step_end(self, args, state, control, **kwargs):
            done = state.global_step - self.start_step
            if done == 1 or state.global_step % 10 == 0 or state.global_step == state.max_steps:
                elapsed = time.monotonic() - self.started
                eta = elapsed / max(done, 1) * (state.max_steps - state.global_step)
                print(f'REAL_TRAIN step={state.global_step}/{state.max_steps} '
                      f'epoch={state.epoch:.3f} elapsed_min={elapsed/60:.1f} '
                      f'ETA_training_min~{eta/60:.1f} (recent backup pauses included; final eval extra)', flush=True)
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                print(f'LOSS step={state.global_step} loss={logs["loss"]} lr={logs.get("learning_rate")}', flush=True)
        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
            self.eval_done = getattr(self, 'eval_done', 0) + 1
            total = len(eval_dataloader) if eval_dataloader is not None else 0
            if self.eval_done == 1 or self.eval_done % 32 == 0 or self.eval_done == total:
                print(f'EVAL_PROGRESS {self.eval_done}/{total}', flush=True)
            if self.eval_done == total:
                self.eval_done = 0
    return Progress()


def validate_restore(meta, state, record, binding, config, env, total):
    if (meta.get('binding') != binding.properties() or meta.get('environment') != env
        or meta.get('config') != config or meta.get('parent', {}).get('run_id') != PARENT.run_id
        or meta.get('parent', {}).get('parent_step') != 313 or meta.get('fresh_optimizer_scheduler') is not True):
        raise BackupError('New real-stage config/code/data/environment/lineage differs. Resume refused.')
    step = state.get('global_step')
    if type(step) is not int or not 0 <= step <= total or str(step) != record['appProperties']['step'] or state.get('best_model_checkpoint'):
        raise BackupError('Real-stage restored state mismatch')


def run(package, *, allow_cloud=False, old_stopped=False, allow_non_gold=False):
    if not (allow_cloud and old_stopped and allow_non_gold):
        raise RuntimeError('Confirm all previous training stopped, Silver data, and Google backup use.')
    package = Path(package).resolve()
    config, train, monitor, binding = package_check(package)
    if not os.environ.get('WANDB_API_KEY', '').strip():
        raise RuntimeError('Colab Secrets WANDB_API_KEY / Notebook access required.')
    total = math.ceil(len(train) / (config['training']['micro_batch'] * config['training']['accumulation']))
    if config['training']['epochs'] != 1:
        raise ValueError('This release is frozen to 1 epoch')
    env = gpu_preflight()
    base = Path('/content')
    if not base.is_dir() or base.is_symlink():
        raise RuntimeError('Use Colab local /content with L4, not Drive FUSE.')
    require_disk(base, 32 * 1024**3)
    service = google_service()
    store = DriveStore(service, DRIVE_ROOT, binding, allow_writes=True, allow_prune=False, max_bytes=40 * 1024**3)
    store.prepare()
    newest = latest_record(store)
    attempt = Path(tempfile.mkdtemp(prefix='uznorm-real-', dir=base))
    output = attempt / 'run'
    output.mkdir()
    resume = None
    print(f'REAL_STAGE run={binding.run_id} train={len(train)} epoch=1 steps={total} local={output}', flush=True)
    if newest:
        print('RESTORE_REAL_STAGE step=' + newest['appProperties']['step'], flush=True)
        archive = attempt / 'restore.zip'
        require_disk(attempt, int(newest['size']), 3)
        store.download(newest, archive)
        unpack_verified(archive, binding, attempt / 'restore')
        saved = attempt / 'restore/run'
        meta = read_json(saved / 'run-meta.json')
        model_path = attempt / 'restore/checkpoint'
        verify_seal(model_path, 'COMPLETE.json', REQUIRED)
        state = read_json(model_path / 'trainer_state.json')
        validate_restore(meta, state, newest, binding, config, env, total)
        for path in saved.iterdir():
            if path.is_file():
                shutil.copyfile(path, output / path.name)
        if not (output / 'baseline.json').is_file():
            raise BackupError('Original before-training baseline missing')
        if newest['appProperties']['kind'] == 'final':
            result = read_json(output / 'RESULT.json')
            if result.get('optimizer_steps') != total or result.get('training_finished') is not True or result.get('wandb_id') != binding.run_id or state.get('epoch') != 1:
                raise BackupError('Final archive result inconsistent')
            print('REAL_STAGE_ALREADY_COMPLETE; no optimizer step executed.', flush=True)
            print('MODEL:', model_path, 'COMPARISON:', output / 'comparison.json', flush=True)
            return output
        resume = model_path
    else:
        model_path, parent = fetch_parent(service, attempt)
        meta = {'schema': 1, 'binding': binding.properties(), 'config': config, 'environment': env,
            'parent': parent, 'initialization': 'parent_model_and_tokenizer_only',
            'fresh_optimizer_scheduler': True, 'data_quality': 'real_reviews_silver_not_gold',
            'train_n': len(train), 'real_monitor_n': 144, 'synthetic_monitor_n': 128, 'test_used': False}
        write_json(output / 'run-meta.json', meta)
    import torch
    import wandb
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    set_seed(config['training']['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True, trust_remote_code=False)
    if model.config.model_type != 't5' or sum(p.numel() for p in model.parameters()) != 581653248 or tokenizer.__class__.__name__ != 'ByT5Tokenizer':
        raise BackupError('Expected verified ByT5-base')
    model.config.use_cache = False
    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=False)
    tracking = wandb.init(project=config['wandb_project'], id=binding.run_id, name=config['name'], resume='allow',
        mode='online', dir=str(attempt), config={'stage': meta}, save_code=False,
        settings=wandb.Settings(disable_code=True, console='off', disable_job_creation=True))
    gate = StageGate(store, output, package, interval=config['training']['save_steps'])
    if resume:
        gate.last_cloud_step, gate.ready = state['global_step'], True
    trainer = make_trainer(model, tokenizer, train, monitor, config, output, gate)
    trainer.add_callback(progress_callback())
    # Plain interval progress is reliably visible through the notebook's stdout relay.
    trainer.args.disable_tqdm = True
    from transformers.trainer_callback import ProgressCallback
    trainer.remove_callback(ProgressCallback)
    try:
        smoke(trainer, train)
        if not resume:
            tracking.log(scored_evaluation(trainer, monitor, output, 'baseline'))
        current = state['global_step'] if resume else 0
        print(f'REAL_TRAIN_BEGIN step={current}/{total}; parent10k=313; new optimizer.', flush=True)
        if current < total:
            trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            checkpoint = output / f'checkpoint-{trainer.state.global_step}'
            if not checkpoint.exists():
                trainer._save_checkpoint(trainer.model, None)
        else:
            checkpoint = resume
            trainer.state.global_step, trainer.state.epoch = current, state.get('epoch')
        if trainer.state.global_step != total:
            raise BackupError('Epoch not finished; restore last confirmed real-stage checkpoint.')
        tracking.log(scored_evaluation(trainer, monitor, output, 'after'))
        write_json(output / 'comparison.json', comparison(read_json(output / 'baseline.json'), read_json(output / 'after.json')))
        write_json(output / 'RESULT.json', {'optimizer_steps': total, 'epochs': 1, 'parent_step': 313,
            'parent_run_id': PARENT.run_id, 'wandb_id': binding.run_id, 'training_finished': True,
            'not_gold': True, 'test_set_used': False, 'remote_confirmation_requires_final_archive': True})
        record = gate.publish(checkpoint, kind='final', roundtrip=True)
        print(f'REAL_STAGE_COMPLETE_CLOUD_VERIFIED step={total} file_id={record["id"]}', flush=True)
        print('MODEL:', checkpoint, 'COMPARISON:', output / 'comparison.json', flush=True)
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
        _, train, monitor, binding = package_check(args.package)
        print(f'PACKAGE_OK real_train={len(train)} development={len(monitor)} run={binding.run_id}', flush=True)
        return
    try:
        run(args.package, allow_cloud=args.allow_cloud, old_stopped=args.old_stopped, allow_non_gold=args.allow_non_gold)
    except KeyboardInterrupt:
        print('PAUSED. Use the same kit to restore the last STAGE_CLOUD_CONFIRMED. Do not reset during upload.', flush=True)
        raise SystemExit(130)
    except BackupError as exc:
        print('BACKUP_BLOCKED:', exc, flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
