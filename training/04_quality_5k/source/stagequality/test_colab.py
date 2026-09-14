"""Offline fake-cloud and tiny random CPU integration; no production model/API."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from stagequality import colab_runner as colab
from stagequality import train_pilot as pilot
from durability.cloud_backup import Binding, BackupError, build_archive, hashes, verify_archive


class FakeStore:
    def __init__(self, root, binding):
        self.root, self.binding = root, binding
        root.mkdir()
        self.items = []
        self.fail = False
        self.downloads = 0

    def records(self):
        return copy.deepcopy(self.items)

    def inspect(self, file_id):
        item = next(r for r in self.items if r['id'] == file_id)
        if hashes(self.root / file_id)['sha256'] != item['appProperties']['payload_sha256']:
            raise BackupError('Server checksum mismatch')
        return copy.deepcopy(item)

    def put(self, path, kind, step, digest):
        if self.fail:
            raise BackupError('Injected upload failure')
        file_id = 'file-' + str(len(self.items))
        with Path(path).open('rb') as src, (self.root / file_id).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        item = dict(id=file_id, size=str(digest['size']), appProperties=dict(
            self.binding.properties(), kind=kind, step=str(step), payload_sha256=digest['sha256']))
        self.items.append(item)
        return self.inspect(file_id)

    def download(self, item, target):
        self.downloads += 1
        self.inspect(item['id'])
        with (self.root / item['id']).open('rb') as src, Path(target).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert hashes(target)['sha256'] == item['appProperties']['payload_sha256']
        return target


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='quality-colab-test-'))
        self.binding = Binding('fixture', 'a' * 64, 'b' * 32)
        self.store = FakeStore(self.root / 'cloud', self.binding)
        self.package = self.root / 'kit'; self.package.mkdir()
        pilot.write_new(self.package / 'PACKAGE.json', dict(files={}))
        self.output = self.root / 'run'; self.output.mkdir()
        self.signature = colab.run_signature(self.binding)

    def tearDown(self):
        assert self.root.resolve().parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith('quality-colab-test-')
        shutil.rmtree(self.root)

    def meta(self):
        return dict(binding=self.binding.properties(), signature=colab.run_signature(self.binding),
            config=copy.deepcopy(pilot.CONFIG), train_sha256=pilot.TRAIN_SHA,
            development_sha256=pilot.DEV_SHA, parent=dict(run_id=colab.PARENT_RUN, step=392,
            weights_sha256=pilot.PARENTS['real-392'][2]), fresh_optimizer_scheduler=True,
            final_test_used=False, environment={'fixture': True})

    def baseline(self, rows=None):
        rows = rows or [dict(id='1', input='salom', target='Salom', category='lexical')]
        pilot.write_new(self.output / 'RUN.json', self.meta())
        pilot.write_new(self.output / 'baseline-development.json', dict(signature=colab.run_signature(self.binding),
            development_sha256=pilot.DEV_SHA, metrics=dict(n=len(rows), split='development', final_test_used=False),
            predictions=[dict(r, prediction=r['input']) for r in rows]))

    def checkpoint(self, step, epoch=.5):
        folder = self.output / ('checkpoint-' + str(step)); folder.mkdir()
        for name in set(pilot.FILES) | {'optimizer.pt', 'scheduler.pt', 'rng_state.pth'}:
            pilot.write_new(folder / name, dict(fixture=True))
        pilot.write_new(folder / 'trainer_state.json', dict(global_step=step, epoch=epoch))
        pilot.write_new(folder / 'PILOT_CHECKPOINT.json', dict(signature=colab.run_signature(self.binding),
            step=step, baseline_sha256=pilot.sha(self.output / 'baseline-development.json'),
            development_sha256=pilot.DEV_SHA))
        pilot.write_new(folder / 'COMPLETE.json', dict(files={p.name: pilot.sha(p) for p in folder.iterdir()}))
        return folder

    def gate(self, record=None):
        return colab.CloudGate(self.store, self.output, self.package, self.binding,
                               record=record, check_disk=False)

    def test_import_inert_and_cli_opt_in(self):
        before = dict(os.environ)
        spec = importlib.util.spec_from_file_location('inert_quality_colab', colab.__file__)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(before, dict(os.environ))
        args = colab.parser().parse_args(['--package', str(self.package)])
        self.assertFalse(args.train or args.check_cloud or args.allow_cloud)
        with patch.object(colab, 'package_check', return_value=([], [], self.binding, self.binding)), \
             patch.object(colab, 'google_service') as google:
            colab.run(args)
            google.assert_not_called()
            args.train = True
            with self.assertRaisesRegex(RuntimeError, 'Confirm'):
                colab.run(args)
            google.assert_not_called()

    def test_parent_has_no_old_model_fallback(self):
        with patch.object(colab, 'open_store', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                colab.parent_record(None, self.binding)

    def test_read_only_cloud_preflight_never_creates_or_uploads(self):
        quota_request = SimpleNamespace(execute=lambda **kw: {'storageQuota': {
            'limit': str(100 * 1024**3), 'usage': '0'}})
        service = SimpleNamespace(about=lambda: SimpleNamespace(get=lambda **kw: quota_request))
        with patch.object(colab, 'open_store', return_value=None) as opened, \
             patch.object(colab, 'parent_record', return_value=(self.store, {'size': '123'})):
            colab.cloud_preflight(service, self.binding, self.binding)
            opened.assert_called_once_with(service, self.binding)
            self.assertEqual(self.store.items, [])
            self.assertEqual(self.store.downloads, 0)

    def test_foreign_and_conflicting_latest_records_rejected(self):
        self.baseline(); record = self.gate().publish(self.checkpoint(0))
        self.store.items[0]['appProperties']['run_id'] = 'foreign'
        with self.assertRaisesRegex(RuntimeError, 'Foreign'):
            colab.latest_record(self.store)
        self.store.items[0] = copy.deepcopy(record)
        other = copy.deepcopy(record)
        other['id'] = 'other'; other['appProperties']['payload_sha256'] = '9' * 64
        with patch.object(self.store, 'records', return_value=[record, other]), \
             patch.object(self.store, 'inspect', side_effect=lambda i: record if i == record['id'] else other):
            with self.assertRaisesRegex(RuntimeError, 'Conflicting'):
                colab.latest_record(self.store)

    def test_gate_blocks_until_confirmed_and_after_interval(self):
        self.baseline(); zero = self.checkpoint(0)
        gate = self.gate()
        with self.assertRaises(RuntimeError): gate.boundary(0)
        gate.publish(zero, roundtrip=True)
        self.assertEqual(self.store.downloads, 1)
        for step in (0, 24): gate.boundary(step)
        for step in (-1, 25):
            with self.assertRaises(RuntimeError): gate.boundary(step)
        self.assertTrue(zero.exists())

    def test_upload_failure_cannot_advance_optimizer(self):
        self.baseline(); gate = self.gate()
        gate.publish(self.checkpoint(0))
        self.store.fail = True
        with self.assertRaises(BackupError): gate.publish(self.checkpoint(25))
        self.assertFalse(gate.ready)
        with self.assertRaises(RuntimeError): gate.boundary(25)
        self.assertEqual(colab.cloud_anchor(colab.latest_record(self.store))[0], 0)

    def test_restore_new_directory_full_state_binding(self):
        self.baseline(); gate = self.gate()
        saved = self.checkpoint(25)
        record = gate.publish(saved)
        attempt = self.root / 'new-runtime'; attempt.mkdir()
        out, restored, meta = colab.restore(self.store, record, self.package, attempt, self.binding)
        self.assertNotEqual(out, self.output)
        self.assertEqual(pilot.verify_checkpoint(restored, self.signature)['global_step'], 25)
        self.assertEqual({p.name: pilot.sha(p) for p in saved.iterdir()},
                         {p.name: pilot.sha(p) for p in restored.iterdir()})
        colab.validate_run(meta, self.binding, {'fixture': True})
        with self.assertRaisesRegex(RuntimeError, 'versions'):
            colab.validate_run(meta, self.binding, {'fixture': False})

    def test_broken_newest_is_not_skipped(self):
        self.baseline(); gate = self.gate()
        gate.publish(self.checkpoint(0)); gate.publish(self.checkpoint(25))
        with (self.store.root / self.store.items[-1]['id']).open('ab') as f: f.write(b'broken')
        with self.assertRaises(BackupError): colab.latest_record(self.store)

    def test_conflicting_writers_or_rollback_fail_closed(self):
        self.baseline(); gate = self.gate()
        zero = self.checkpoint(0); saved = gate.publish(zero)
        stale = self.gate(saved)
        gate.publish(self.checkpoint(25))
        with self.assertRaisesRegex(RuntimeError, 'second writer'):
            stale.publish(self.output / 'checkpoint-25')
        with self.assertRaisesRegex(RuntimeError, 'rollback'):
            gate.publish(zero)

    def test_restored_kit_mismatch_rejected(self):
        self.baseline(); record = self.gate().publish(self.checkpoint(0))
        with (self.package / 'PACKAGE.json').open('a') as f: f.write(' ')
        attempt = self.root / 'new'; attempt.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'exact notebook'):
            colab.restore(self.store, record, self.package, attempt, self.binding)

    def test_parent_download_extracts_only_safe_model_files(self):
        source = self.root / 'parent-source'; (source / 'checkpoint').mkdir(parents=True)
        (source / 'run').mkdir(); (source / 'kit').mkdir()
        provenance = dict(fixture=True)
        binding = Binding(colab.PARENT_RUN, pilot.digest(provenance), colab.STORAGE_ID)
        store = FakeStore(self.root / 'parent-cloud', binding)
        for name in pilot.FILES:
            pilot.write_new(source / 'checkpoint' / name, dict(model_type='t5',
                architectures=['T5ForConditionalGeneration'], tokenizer_class='ByT5Tokenizer'))
        seal = {p.name: pilot.sha(p) for p in (source / 'checkpoint').iterdir()}
        pilot.write_new(source / 'checkpoint/COMPLETE.json', dict(files=seal))
        pilot.write_new(source / 'checkpoint/trainer_state.json', dict(global_step=392, epoch=1))
        pilot.write_new(source / 'checkpoint/optimizer.pt', {'never': 'deserialize'})
        pilot.write_new(source / 'run/run-meta.json', dict(binding=binding.properties()))
        pilot.write_new(source / 'run/RESULT.json', dict(training_finished=True, optimizer_steps=392,
                                                       epochs=1, wandb_id=colab.PARENT_RUN))
        pilot.write_new(source / 'kit/PACKAGE.json', provenance)
        archive = self.root / 'parent.zip'
        digest = build_archive(archive, {p.relative_to(source).as_posix(): (p, pilot.sha(p))
            for p in source.rglob('*') if p.is_file()}, dict(binding=binding.properties(), kind='final', step=392))
        record = store.put(archive, kind='final', step=392, digest=digest)
        attempt = self.root / 'parent-attempt'; attempt.mkdir()
        with patch.object(colab, 'parent_record', return_value=(store, record)), \
             patch.dict(pilot.PARENTS, {'real-392': (colab.PARENT_RUN, 392, seal['model.safetensors'])}):
            folder, meta = colab.fetch_parent(None, binding, attempt)
            self.assertEqual(meta['step'], 392)
            self.assertEqual({p.name for p in folder.iterdir()}, set(pilot.FILES) | {'LOCAL_MODEL.json'})
            self.assertNotIn('path', meta)

    def test_final_cloud_roundtrip_and_idempotent_restore(self):
        self.baseline(); checkpoint = self.checkpoint(157, epoch=1)
        after = pilot.write_evaluation_attempt(self.output, self.signature, {}, [])
        pilot.write_new(self.output / 'comparison.json', {'before': {}, 'after': {}})
        pilot.write_new(self.output / 'RESULT.json', dict(signature=self.signature, step=157, epoch=1,
            final_test_used=False, after_file=after.name, after_sha256=pilot.sha(after),
            baseline_sha256=pilot.sha(self.output / 'baseline-development.json'),
            comparison_sha256=pilot.sha(self.output / 'comparison.json')))
        record = self.gate().publish(checkpoint, kind='final', roundtrip=True)
        attempt = self.root / 'finished'; attempt.mkdir()
        output, _, _ = colab.restore(self.store, record, self.package, attempt, self.binding)
        self.assertEqual(colab.validate_result(output, self.binding)['step'], 157)
        self.assertEqual(len(self.store.items), 1)

    def test_tiny_cpu_cloud_roundtrip_resume_exact_weights(self):
        pilot.offline()
        import torch
        from transformers import T5Config, T5ForConditionalGeneration, ByT5Tokenizer
        torch.set_num_threads(2); torch.manual_seed(3407)
        model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=16, d_kv=8, d_ff=32,
            num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
            eos_token_id=1, pad_token_id=0, use_cache=False, dropout_rate=0.1))
        tokenizer = ByT5Tokenizer()
        rows = [dict(id=str(i), input='salom ' + str(i), target='Salom ' + str(i), category='lexical') for i in range(9)]
        config = dict(pilot.CONFIG, micro_batch=1, accumulation=2, save_steps=2,
                      max_source_tokens=32, max_target_tokens=32)
        with patch.object(pilot, 'CONFIG', config):
            signature = colab.run_signature(self.binding)
            self.baseline(rows)
            trainer = pilot.make_trainer(model, tokenizer, rows, rows, self.output, signature, cpu=True)
            gate = self.gate(); colab.attach_cloud(trainer, gate)
            trainer.train()
            self.assertEqual([int(r['appProperties']['step']) for r in self.store.items], [0, 2, 4, 5])
            self.assertEqual(self.store.downloads, 1)
            expected = {n: t.detach().clone() for n, t in model.state_dict().items()}
            # Branch the fake cloud at step 2 to emulate a session lost after that commit.
            branch = FakeStore(self.root / 'resume-cloud', self.binding)
            branch.items = copy.deepcopy(self.store.items[:2])
            for item in branch.items:
                shutil.copyfile(self.store.root / item['id'], branch.root / item['id'])
            record = colab.latest_record(branch)
            attempt = self.root / 'resume-attempt'; attempt.mkdir()
            out, checkpoint, _ = colab.restore(branch, record, self.package, attempt, self.binding)
            restored = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True)
            resumed = pilot.make_trainer(restored, tokenizer, rows, rows, out, signature, cpu=True)
            colab.attach_cloud(resumed, colab.CloudGate(branch, out, self.package, self.binding,
                                                       record=record, check_disk=False))
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 5)
            for name, value in expected.items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=1e-6, atol=1e-7)
            self.assertEqual([int(r['appProperties']['step']) for r in branch.items], [0, 2, 4, 5])


if __name__ == '__main__':
    unittest.main()
