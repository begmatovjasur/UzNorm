"""Offline, persistent-local-NVIDIA training handoff; checks only by default.

Only the frozen train/development files are opened. No final-test code path.
The GPU/VRAM checks are eligibility checks, not proof that training will fit.
No parent optimizer is loaded. Resume is confined to this exact new run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA = '0f4f8e356d3b1040bb8f3ae45c8af5f8ddf59731793ab70ca5fa353cef8eba6a'
DEV_SHA = '50a4e04099b678eeac5adca5c95df578839c2504a339ace4aa4d121471fa4136'
MANIFEST_SHA = '27e76f6d661f961fcc06d10b77ff5b202f358b957032c1081504b98972c1095f'
PARENTS = {
    'old-3551': ('79e3c655', 3551, '450eef6d037b7f1f789c76935f128981d93bdbe3be5e02385bd5d951859317bc'),
    'real-392': ('ureal-14a56cf24f5b1116', 392, 'e2c1771a5436578c1d9e37c35ac4ad0910359945f38e8051492b6731eecf403e'),
}
FILES = ('model.safetensors', 'config.json', 'generation_config.json', 'added_tokens.json',
         'special_tokens_map.json', 'tokenizer_config.json')
QUOTAS = dict(lexical=2000, chat_mixed=1000, format=750, identity=1000, protected_entities=250)
CONFIG = dict(epochs=1, micro_batch=2, accumulation=16, learning_rate=1e-5,
              warmup_ratio=.05, seed=3407, save_steps=25, max_source_tokens=512,
              max_target_tokens=512, optimizer='adafactor', precision='bf16', keep_all=True)
MIN_FREE = 30 * 1024**3
MIN_RESERVE = 8 * 1024**3
SPECIAL = re.compile(r'<(?:extra_id_\d+|pad|/s|unk)>')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def reject_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink() and not (hasattr(item, 'is_junction') and item.is_junction()),
                'Symlink/junction path refused: ' + str(item))


def validate_paths(dataset, parent, output, resume=False):
    require(not any(key.startswith('COLAB_') for key in os.environ)
            and not Path('/var/colab/hostname').exists(), 'Colab is unsupported; persistent local GPU only')
    for path in (dataset, parent, output):
        reject_links(path)
        require('content' not in Path(path).absolute().parts[:2], '/content is unsupported')
    dataset, parent, output = (Path(p).resolve() for p in (dataset, parent, output))
    require(dataset.is_dir() and parent.is_dir(), 'Dataset and parent directories must exist')
    for protected in (dataset, parent):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                'Output must be separate from parent and frozen dataset')
    # Also forbid writing under any sibling production model, not only selected parent.
    models = ROOT / 'local_model' / 'models'
    require(not output.is_relative_to(models.resolve()), 'Output must not be inside production models')
    for ancestor in output.parents:
        require(not any((ancestor / marker).is_file() for marker in
                        ('MANIFEST.json', 'manifest.json', 'PACKAGE.json', 'DELIVERY.json',
                         'LOCAL_MODEL.json', 'RUN.json', 'TRAINING_COMPLETE.json')),
                'Output must not be nested inside another frozen artifact or run')
    require(output.is_dir() if resume else not output.exists(),
            'Resume needs an existing run; fresh training requires a new output directory')
    return dataset, parent, output


def validate_dataset(folder):
    reject_links(folder / 'MANIFEST.json')
    require(sha(folder / 'MANIFEST.json') == MANIFEST_SHA, 'Frozen manifest mismatch')
    manifest = read_json(folder / 'MANIFEST.json')
    declared = {name.replace('\\', '/'): value for name, value in manifest['files'].items()}
    require(manifest.get('train_rows') == 5000 and manifest.get('quotas_met') is True,
            'Incomplete quality pilot')
    rows = {}
    for split, expected, count in [('train', TRAIN_SHA, 5000), ('development', DEV_SHA, 60)]:
        name = 'data/' + split + '.jsonl'; path = folder / name
        reject_links(path)
        require(declared.get(name) == expected and sha(path) == expected, 'Pinned data mismatch: ' + split)
        rows[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require(len(rows[split]) == count, 'Unexpected row count')
        require(len({row['id'] for row in rows[split]}) == count, 'Duplicate IDs')
        for row in rows[split]:
            require(row.get('split') == split and row.get('category') in QUOTAS, 'Wrong split/category')
            require(bool(row.get('group_id')), 'Missing source family')
            require(type(row.get('is_identity')) is bool
                    and row['is_identity'] == (row['input'] == row['target']), 'Identity flag mismatch')
            for field in ('input', 'target'):
                text = row[field]
                require(isinstance(text, str) and text.strip() and '\x00' not in text
                        and not SPECIAL.search(text) and len(text.encode()) + 1 <= 512,
                        'Invalid text or forbidden truncation')
    require(Counter(r['category'] for r in rows['train']) == QUOTAS, '5k quotas changed')
    require(sum(r['is_identity'] for r in rows['train']) == 1000, 'Identity quota changed')
    for field in ('id', 'group_id'):
        require(not {r[field] for r in rows['train']} & {r[field] for r in rows['development']},
                'Train/development family leakage')
    return rows['train'], rows['development']


def validate_parent(folder, choice):
    run, step, weights = PARENTS[choice]
    reject_links(folder)
    require({p.name for p in folder.iterdir()} == set(FILES) | {'LOCAL_MODEL.json'},
            'Parent must contain exactly six inference files and LOCAL_MODEL.json')
    reject_links(folder / 'LOCAL_MODEL.json')
    meta = read_json(folder / 'LOCAL_MODEL.json')
    require(meta.get('run_id') == run and meta.get('step') == step
            and set(meta.get('files', {})) == set(FILES)
            and meta['files']['model.safetensors'] == weights, 'Pinned parent lineage mismatch')
    for name in (*FILES, 'LOCAL_MODEL.json'):
        reject_links(folder / name)
        require((folder / name).is_file(), 'Parent file missing')
        if name in FILES:
            require(sha(folder / name) == meta['files'][name], 'Parent checksum mismatch: ' + name)
    config, tokenizer = (read_json(folder / n) for n in ('config.json', 'tokenizer_config.json'))
    require(config.get('model_type') == 't5' and config.get('architectures') == ['T5ForConditionalGeneration']
            and tokenizer.get('tokenizer_class') == 'ByT5Tokenizer'
            and not config.get('auto_map') and not tokenizer.get('auto_map'), 'Unexpected model/custom code')
    return dict(choice=choice, path=str(folder), run_id=run, step=step, weights_sha256=weights,
                manifest_sha256=sha(folder / 'LOCAL_MODEL.json'), files=meta['files'])


def offline():
    for key, value in dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                           WANDB_DISABLED='true', WANDB_MODE='disabled', USE_TF='0', USE_FLAX='0',
                           TOKENIZERS_PARALLELISM='false').items():
        os.environ[key] = value
    for path in (ROOT, ROOT / 'src', ROOT / 'local_model'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def gpu_environment(output, *, resume=False):
    offline()
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one local CUDA GPU required')
    require(torch.cuda.is_bf16_supported(), 'BF16 GPU required; no precision fallback')
    gpu = torch.cuda.get_device_properties(0)
    require(gpu.total_memory >= 16 * 1024**3, 'At least 16 GiB GPU memory required; actual fit remains untested')
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    check_disk(parent, initial=not resume)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'accelerate', 'safetensors', 'numpy', 'rapidfuzz')}
    require(versions['transformers'] == '4.57.1' and versions['accelerate'] == '1.10.1',
            'Use the tested transformers 4.57.1 / accelerate 1.10.1 API')
    from packaging.version import Version
    require(Version(torch.__version__.split('+')[0]) >= Version('2.6'), 'PyTorch >= 2.6 required')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Distributed training is unsupported')
    executable = shutil.which('nvidia-smi')
    if executable:
        result = subprocess.run([executable, '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=30, check=True)
        pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        require(not pids - {os.getpid()}, 'Another GPU process is active; no process was stopped')
        print('GPU_PROCESS_CHECK_OK; no other reported compute process', flush=True)
    else:
        print('GPU_PROCESS_CHECK_UNAVAILABLE; --confirm-old-stopped remains required', flush=True)
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                torch_cuda=torch.version.cuda, gpu=gpu.name, gpu_bytes=gpu.total_memory)


def check_disk(path, *, initial=False):
    minimum = MIN_FREE if initial else MIN_RESERVE
    require(shutil.disk_usage(path).free >= minimum,
            ('At least 30 GiB initial space' if initial else '8 GiB ongoing disk reserve') +
            ' required; no checkpoint deletion attempted')


def code_hashes():
    paths = [Path(__file__), *sorted((ROOT / 'src' / 'uznorm').glob('*.py')),
             *(ROOT / 'local_model' / n for n in ('evaluate.py', 'engine.py', 'prepare_benchmark.py'))]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


@contextmanager
def persistent_lock(output):
    """OS lock is released on close/crash; its lock file is never deleted."""
    path = output / '.run.lock'
    reject_links(path)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Run is locked; no lock removal attempted') from exc
            try:
                yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Run is locked; no lock removal attempted') from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def verify_checkpoint(folder, signature):
    reject_links(folder)
    seal = read_json(folder / 'COMPLETE.json')['files']
    required = set(FILES) | {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'PILOT_CHECKPOINT.json'}
    require(required.issubset(seal), 'Checkpoint is incomplete')
    require({p.name for p in folder.iterdir()} == set(seal) | {'COMPLETE.json'}, 'Unsealed checkpoint inventory')
    for name, expected in seal.items():
        require(Path(name).name == name, 'Nested/unsafe checkpoint member')
        reject_links(folder / name)
        require(sha(folder / name) == expected, 'Checkpoint checksum mismatch: ' + name)
    state, marker = read_json(folder / 'trainer_state.json'), read_json(folder / 'PILOT_CHECKPOINT.json')
    step = state.get('global_step')
    baseline_hash = validate_baseline(folder.parent, signature)
    require(type(step) is int and folder.name == 'checkpoint-' + str(step)
            and marker == dict(signature=signature, step=step, baseline_sha256=baseline_hash,
                               development_sha256=DEV_SHA)
            and not state.get('best_model_checkpoint'), 'Checkpoint run/step binding mismatch')
    return state


def validate_baseline(output, signature, rows=None):
    path = output / 'baseline-development.json'
    reject_links(path)
    baseline = read_json(path)
    require(baseline.get('signature') == signature and baseline.get('development_sha256') == DEV_SHA,
            'Baseline run/development binding mismatch')
    predictions = baseline.get('predictions')
    metrics = baseline.get('metrics', {})
    require(isinstance(predictions, list) and predictions and metrics.get('n') == len(predictions)
            and metrics.get('split') == 'development' and metrics.get('final_test_used') is False,
            'Baseline inventory/scope mismatch')
    require(all(isinstance(row.get('prediction'), str) for row in predictions), 'Invalid baseline predictions')
    if rows is not None:
        require(len(predictions) == len(rows)
                and all(all(saved.get(k) == row[k] for k in ('id', 'input', 'target'))
                        for saved, row in zip(predictions, rows)), 'Baseline prediction/data alignment mismatch')
    return sha(path)


def validate_final_state(state):
    require(type(state.get('global_step')) is int and state['global_step'] == 157
            and type(state.get('epoch')) in (int, float)
            and math.isclose(state['epoch'], 1.0, abs_tol=1e-9, rel_tol=0),
            'Final checkpoint must be exactly step 157 / epoch 1')


def resume_checkpoint(output, binding):
    require(read_json(output / 'RUN.json') == binding, 'Run config/data/parent/code/environment changed')
    require(not (output / 'TRAINING_COMPLETE.json').exists(), 'Run already complete; do not overwrite it')
    candidates = [p for p in output.iterdir() if p.name.startswith('checkpoint-')]
    require(candidates and all(re.fullmatch(r'checkpoint-\d+', p.name) for p in candidates),
            'No unambiguous checkpoint; use a new run, not a rollback')
    newest = max(candidates, key=lambda p: int(p.name.split('-')[1]))
    state = verify_checkpoint(newest, binding['signature'])
    require(0 <= state['global_step'] <= 157, 'Checkpoint step outside this pilot')
    if state['global_step'] == 157:
        validate_final_state(state)
    return newest


def make_trainer(model, tokenizer, train, dev, output, signature, *, cpu=False, config=None):
    """CPU override is for isolated tiny random-model tests, never exposed by CLI."""
    offline()
    import torch
    from types import SimpleNamespace
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, DataCollatorForSeq2Seq
    from uznorm.training import SourceOnlyTrainer
    from uznorm.data import TextDataset
    from uznorm.checkpoints import complete_checkpoint
    config = config or CONFIG
    args = Seq2SeqTrainingArguments(output_dir=str(output), num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['micro_batch'], gradient_accumulation_steps=config['accumulation'],
        per_device_eval_batch_size=1, learning_rate=config['learning_rate'], warmup_ratio=config['warmup_ratio'],
        optim='adafactor', lr_scheduler_type='linear', weight_decay=0, max_grad_norm=1.,
        bf16=not cpu, fp16=False, tf32=False, use_cpu=cpu, report_to=[],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='no', save_strategy='steps', save_steps=config['save_steps'], save_total_limit=None,
        save_only_model=False, save_safetensors=True, load_best_model_at_end=False,
        logging_steps=10, predict_with_generate=True, generation_num_beams=1,
        generation_max_length=config['max_target_tokens'] + 1, dataloader_num_workers=0,
        seed=config['seed'], data_seed=config['seed'], disable_tqdm=cpu)

    class LocalTrainer(SourceOnlyTrainer):
        def _save_checkpoint(self, model, trial):
            folder = output / ('checkpoint-' + str(self.state.global_step))
            require(not folder.exists(), 'Refusing existing checkpoint overwrite')
            if not cpu:
                check_disk(output)
            baseline_hash = validate_baseline(output, signature, dev)
            require(baseline_hash == self.pilot_baseline_sha256, 'Baseline changed during this training process')
            Seq2SeqTrainer._save_checkpoint(self, model, trial)
            write_new(folder / 'PILOT_CHECKPOINT.json', dict(signature=signature, step=self.state.global_step,
                       baseline_sha256=baseline_hash, development_sha256=DEV_SHA))
            complete_checkpoint(folder)
            verify_checkpoint(folder, signature)
            print('LOCAL_CHECKPOINT_VERIFIED step=' + str(self.state.global_step), flush=True)

    class Safety(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.train_started = time.monotonic(); self.start_step = state.global_step
            trainer.pilot_baseline_sha256 = validate_baseline(output, signature, dev)
            if state.global_step == 0 and not (output / 'checkpoint-0').exists():
                trainer._save_checkpoint(trainer.model, None)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0 or state.global_step == state.max_steps:
                elapsed = time.monotonic() - self.train_started
                done = max(1, state.global_step - self.start_step)
                eta = elapsed / done * max(0, state.max_steps - state.global_step)
                print(f'TRAIN_PROGRESS {state.global_step}/{state.max_steps} epoch={state.epoch:.4f} '
                      f'elapsed={elapsed:.1f}s train_eta={eta:.1f}s; final evaluation extra', flush=True)

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
            self.eval_done = getattr(self, 'eval_done', 0) + 1
            total = len(eval_dataloader) if eval_dataloader is not None else 0
            if self.eval_done == 1 or self.eval_done % 10 == 0 or self.eval_done == total:
                print(f'DEVELOPMENT_PROGRESS {self.eval_done}/{total}', flush=True)
            if self.eval_done == total:
                self.eval_done = 0

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            finite = [torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None]
            require(finite and torch.stack(finite).all().item(), 'Non-finite gradients; optimizer step blocked')
            if not cpu:
                check_disk(output)

    trainer = LocalTrainer(model=model, args=args,
        train_dataset=TextDataset(train, tokenizer, SimpleNamespace(**config)),
        eval_dataset=TextDataset(dev, tokenizer, SimpleNamespace(**config)), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True), callbacks=[Safety()])
    return trainer


def evaluate_development(trainer, rows):
    import numpy as np
    from evaluate import measured
    result = trainer.predict(trainer.eval_dataset, max_length=513, num_beams=1, do_sample=False)
    ids = np.where(result.predictions < 0, trainer.processing_class.pad_token_id, result.predictions)
    predictions = trainer.processing_class.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    categories = {c: measured([r for r in rows if r['category'] == c],
                              [p for r, p in zip(rows, predictions) if r['category'] == c])
                  for c in sorted({r['category'] for r in rows})}
    return dict(split='development', n=len(rows), model=measured(rows, predictions), categories=categories,
                unchanged_input_baseline=measured(rows, [r['input'] for r in rows]),
                missing_eos_n=sum(trainer.processing_class.eos_token_id not in row for row in ids),
                human_gold=False, final_test_used=False), [dict(id=r['id'], input=r['input'], target=r['target'],
                                                             prediction=p) for r, p in zip(rows, predictions)]


def write_evaluation_attempt(output, signature, metrics, predictions):
    after = output / ('after-development-' + uuid.uuid4().hex + '.json')
    write_new(after, dict(signature=signature, development_sha256=DEV_SHA,
                         metrics=metrics, predictions=predictions))
    return after


def commit_completion(output, value):
    """Publish a complete file atomically without clobber; retain its unique draft.

    Persistent storage must support same-directory hard links (e.g. NTFS/ext4).
    Failure leaves checkpoints/reports available, never a partial final marker.
    """
    final = output / 'TRAINING_COMPLETE.json'
    require(not final.exists(), 'Completion already exists; no overwrite')
    draft = output / ('.completion-' + uuid.uuid4().hex + '.json')
    write_new(draft, value)
    require(not final.exists(), 'Completion appeared; preserve draft without overwriting')
    os.link(draft, final, follow_symlinks=False)


def run(args):
    dataset, parent, output = validate_paths(args.dataset, args.parent_dir, args.output, args.resume)
    train, dev = validate_dataset(dataset)
    parent_meta = validate_parent(parent, args.parent)
    print('DATA_PARENT_VERIFIED; frozen 5000 train / 60 development; final test unopened', flush=True)
    env = gpu_environment(output, resume=args.resume)
    payload = dict(schema=1, config=CONFIG, parent=parent_meta, dataset=str(dataset), output=str(output),
                   manifest_sha256=MANIFEST_SHA, train_sha256=TRAIN_SHA, development_sha256=DEV_SHA,
                   environment=env, code=code_hashes(), planned_optimizer_steps=157,
                   persistent_local_only=True, remote_backup=False)
    binding = dict(payload, signature=digest(payload))
    checkpoint = resume_checkpoint(output, binding) if args.resume else None
    if checkpoint:
        validate_baseline(output, binding['signature'], dev)
    print(json.dumps(dict(status='PREFLIGHT_VERIFIED_GPU_TRAINING_UNTESTED', signature=binding['signature'],
                          parent=args.parent, train_rows=5000, development_rows=60, output=str(output),
                          estimated_capacity_proven=False, training_started=False), indent=2), flush=True)
    if not args.train:
        return
    require(args.allow_silver and args.confirm_old_stopped and args.confirm_persistent_storage,
            'Training requires --allow-silver --confirm-old-stopped --confirm-persistent-storage')
    if not args.resume:
        output.mkdir(mode=0o777, parents=True, exist_ok=False)
    with persistent_lock(output):
        if args.resume:
            checkpoint = resume_checkpoint(output, binding)
            validate_baseline(output, binding['signature'], dev)
        else:
            write_new(output / 'RUN.json', binding)
        offline()
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
        set_seed(CONFIG['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model_path = checkpoint or parent
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, dtype=torch.float32)
        require(model.config.model_type == 't5' and tokenizer.__class__.__name__ == 'ByT5Tokenizer'
                and sum(p.numel() for p in model.parameters()) == 581653248, 'Expected full ByT5-base')
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.config.use_cache = False; model.generation_config.use_cache = True
        trainer = make_trainer(model, tokenizer, train, dev, output, binding['signature'])
        try:
            if not checkpoint:
                print('BASELINE_DEVELOPMENT_START', flush=True)
                metrics, predictions = evaluate_development(trainer, dev)
                write_new(output / 'baseline-development.json', dict(signature=binding['signature'],
                          development_sha256=DEV_SHA, metrics=metrics, predictions=predictions))
            # Longest examples probe forward/backward only; no hidden batch/precision fallback.
            indexes = sorted(range(len(train)), key=lambda i: max(len(train[i]['input'].encode()),
                             len(train[i]['target'].encode())), reverse=True)[:2]
            model.train(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            batch = trainer._prepare_inputs(trainer.data_collator([trainer.train_dataset[i] for i in indexes]))
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(model, batch)
            loss.backward()
            require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                    'Non-finite smoke gradients')
            model.zero_grad(set_to_none=True); del loss, batch; torch.cuda.empty_cache(); set_seed(CONFIG['seed'])
            print('GPU_SMOKE_OK; no optimizer step applied by smoke', flush=True)
            if not checkpoint or read_json(checkpoint / 'trainer_state.json')['global_step'] < 157:
                trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
                require(trainer.state.global_step == 157, 'One epoch incomplete')
                checkpoint = output / 'checkpoint-157'
                if not checkpoint.exists():
                    trainer._save_checkpoint(model, None)
            validate_final_state(verify_checkpoint(checkpoint, binding['signature']))
            print('AFTER_DEVELOPMENT_START', flush=True)
            metrics, predictions = evaluate_development(trainer, dev)
            after = write_evaluation_attempt(output, binding['signature'], metrics, predictions)
            require(validate_parent(parent, args.parent) == parent_meta, 'Parent changed during training')
            require(sha(dataset / 'data/train.jsonl') == TRAIN_SHA and sha(dataset / 'data/development.jsonl') == DEV_SHA,
                    'Dataset changed during training')
            commit_completion(output, dict(signature=binding['signature'], optimizer_steps=157,
                epoch=1, checkpoint=str(checkpoint), checkpoint_manifest_sha256=sha(checkpoint / 'COMPLETE.json'),
                baseline_sha256=validate_baseline(output, binding['signature'], dev), after_file=after.name, after_sha256=sha(after),
                final_test_used=False, parent_unchanged=True, dataset_unchanged=True,
                remote_backup=False))
            print('LOCAL_TRAINING_COMPLETE step=157; sealed final test unused', flush=True)
        except BaseException:
            print('STOPPED: preserve this run; no fallback, overwrite, cleanup, or CPU training attempted.', flush=True)
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', choices=sorted(PARENTS), required=True)
    p.add_argument('--parent-dir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for flag in ('train', 'resume', 'allow-silver', 'confirm-old-stopped', 'confirm-persistent-storage'):
        p.add_argument('--' + flag, action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
