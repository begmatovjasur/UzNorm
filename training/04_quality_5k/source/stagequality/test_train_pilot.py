"""Offline fixtures plus tiny random CPU T5; never trains a production parent."""
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
import uuid

from stagequality import train_pilot as pilot


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / ('quality-pilot-test-' + uuid.uuid4().hex)
        self.root.mkdir(mode=0o777)

    def tearDown(self):
        assert self.root.resolve().parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith('quality-pilot-test-')
        shutil.rmtree(self.root)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        pilot.write_new(path, value)

    def test_import_no_execution_or_environment_mutation(self):
        before = dict(os.environ)
        spec = importlib.util.spec_from_file_location('pure_import_check', pilot.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(set(module.PARENTS), {'old-3551', 'real-392'})

    def test_parser_is_checks_only_and_parent_required(self):
        args = pilot.parser().parse_args(['--parent', 'old-3551', '--parent-dir', 'model',
                                         '--dataset', 'data', '--output', 'new'])
        self.assertFalse(args.train)
        self.assertFalse(args.resume)
        self.assertFalse(args.allow_silver)

    def test_paths_reject_overlap_existing_colab_and_links(self):
        dataset, model = self.root / 'dataset', self.root / 'model'
        dataset.mkdir(); model.mkdir()
        pilot.validate_paths(dataset, model, self.root / 'new')
        for output in (dataset / 'new', model / 'new', self.root):
            with self.assertRaises(RuntimeError):
                pilot.validate_paths(dataset, model, output)
        with patch.dict(os.environ, {'COLAB_RELEASE_TAG': 'test'}):
            with self.assertRaises(RuntimeError):
                pilot.validate_paths(dataset, model, self.root / 'new')
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(RuntimeError):
                pilot.validate_paths(dataset, model, self.root / 'new')
        frozen = self.root / 'another-frozen'; frozen.mkdir()
        self.save(frozen / 'PACKAGE.json', {})
        with self.assertRaises(RuntimeError):
            pilot.validate_paths(dataset, model, frozen / 'new')

    def dataset(self):
        folder = self.root / 'dataset'; (folder / 'data').mkdir(parents=True)
        rows = []
        for category, count in pilot.QUOTAS.items():
            for i in range(count):
                name = category + str(i)
                rows.append(dict(id=name, input='matn', target='matn' if category == 'identity' else 'Matn',
                                 category=category, group_id=name, split='train', is_identity=category == 'identity'))
        dev = [dict(id='dev' + str(i), input='soz', target='So‘z', category='lexical',
                    group_id='dev' + str(i), split='development', is_identity=False) for i in range(60)]
        for name, values in [('train', rows), ('development', dev)]:
            with (folder / 'data' / (name + '.jsonl')).open('x', encoding='utf-8') as stream:
                for row in values:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        train_sha = pilot.sha(folder / 'data/train.jsonl'); dev_sha = pilot.sha(folder / 'data/development.jsonl')
        # Missing final-test payload is intentional and must never be opened.
        self.save(folder / 'MANIFEST.json', dict(train_rows=5000, quotas_met=True, files={
            'data\\train.jsonl': train_sha, 'data\\development.jsonl': dev_sha,
            'data\\final-test-sealed.jsonl': '0' * 64, 'reports\\quality.json': '0' * 64}))
        return folder, dict(TRAIN_SHA=train_sha, DEV_SHA=dev_sha, MANIFEST_SHA=pilot.sha(folder / 'MANIFEST.json'))

    def test_dataset_pins_without_final_payload(self):
        folder, pins = self.dataset()
        with patch.multiple(pilot, **pins):
            train, dev = pilot.validate_dataset(folder)
            self.assertEqual((len(train), len(dev)), (5000, 60))
            with (folder / 'data/train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(RuntimeError, 'Pinned data'):
                pilot.validate_dataset(folder)

    def test_parent_only_whitelisted_hashes(self):
        folder = self.root / 'parent'; folder.mkdir()
        for name in pilot.FILES:
            self.save(folder / name, {'model_type': 't5', 'architectures': ['T5ForConditionalGeneration'],
                                      'tokenizer_class': 'ByT5Tokenizer'})
        hashes = {name: pilot.sha(folder / name) for name in pilot.FILES}
        self.save(folder / 'LOCAL_MODEL.json', dict(run_id='fixture', step=5, files=hashes))
        with patch.dict(pilot.PARENTS, fixture=('fixture', 5, hashes['model.safetensors'])):
            self.assertEqual(pilot.validate_parent(folder, 'fixture')['step'], 5)
            self.save(folder / 'optimizer.pt', {})
            with self.assertRaisesRegex(RuntimeError, 'exactly six'):
                pilot.validate_parent(folder, 'fixture')

    def test_non_cuda_fails_before_model_load(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict('sys.modules', {'torch': fake}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                pilot.gpu_environment(self.root / 'new')

    def test_initial_and_ongoing_disk_reserves(self):
        for free, initial, passes in [(30, True, True), (29, True, False), (9, False, True), (7, False, False)]:
            with patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=free * 1024**3)):
                if passes:
                    pilot.check_disk(self.root, initial=initial)
                else:
                    with self.assertRaises(RuntimeError):
                        pilot.check_disk(self.root, initial=initial)

    def test_resume_gpu_preflight_uses_ongoing_not_initial_reserve(self):
        fake = SimpleNamespace(__version__='2.6.0', version=SimpleNamespace(cuda='fixture'),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                 is_bf16_supported=lambda: True,
                                 get_device_properties=lambda _: SimpleNamespace(total_memory=24 * 1024**3, name='fixture')))
        versions = {'transformers': '4.57.1', 'accelerate': '1.10.1'}
        with patch.dict('sys.modules', {'torch': fake}), \
             patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=15 * 1024**3)), \
             patch.object(shutil, 'which', return_value=None), \
             patch.object(pilot.importlib.metadata, 'version', side_effect=lambda n: versions.get(n, '1.0.0')):
            with self.assertRaisesRegex(RuntimeError, '30 GiB'):
                pilot.gpu_environment(self.root / 'run', resume=False)
            self.assertEqual(pilot.gpu_environment(self.root / 'run', resume=True)['gpu'], 'fixture')

    def test_checks_only_creates_no_output(self):
        data, parent = self.root / 'dataset', self.root / 'parent'; data.mkdir(); parent.mkdir()
        args = pilot.parser().parse_args(['--parent', 'old-3551', '--parent-dir', str(parent),
                                         '--dataset', str(data), '--output', str(self.root / 'new')])
        with patch.object(pilot, 'validate_dataset', return_value=([], [])), \
             patch.object(pilot, 'validate_parent', return_value={}), \
             patch.object(pilot, 'gpu_environment', return_value={}), \
             patch.object(pilot, 'code_hashes', return_value={}):
            pilot.run(args)
            self.assertFalse(args.output.exists())
            args.train = True
            with self.assertRaisesRegex(RuntimeError, 'requires'):
                pilot.run(args)
            self.assertFalse(args.output.exists())

    def test_persistent_lock_keeps_file_and_releases(self):
        with pilot.persistent_lock(self.root):
            self.assertTrue((self.root / '.run.lock').is_file())
            with self.assertRaises(RuntimeError):
                with pilot.persistent_lock(self.root):
                    pass
        with pilot.persistent_lock(self.root):
            pass
        self.assertTrue((self.root / '.run.lock').is_file())

    def test_resume_no_silent_rollback_or_binding_change(self):
        binding = dict(signature='fixture')
        self.save(self.root / 'RUN.json', binding)
        (self.root / 'checkpoint-1').mkdir(); (self.root / 'checkpoint-2').mkdir()
        with patch.object(pilot, 'verify_checkpoint', side_effect=RuntimeError('newest incomplete')) as verify:
            with self.assertRaisesRegex(RuntimeError, 'newest incomplete'):
                pilot.resume_checkpoint(self.root, binding)
            self.assertEqual(verify.call_args.args[0].name, 'checkpoint-2')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            pilot.resume_checkpoint(self.root, dict(signature='other'))

    def test_write_never_overwrites(self):
        path = self.root / 'proof.json'; pilot.write_new(path, {'old': True})
        with self.assertRaises(FileExistsError):
            pilot.write_new(path, {'new': True})
        self.assertEqual(pilot.read_json(path), {'old': True})

    def test_final_state_requires_completed_epoch(self):
        pilot.validate_final_state(dict(global_step=157, epoch=1.0))
        for state in (dict(global_step=156, epoch=1.), dict(global_step=157, epoch=.99),
                      dict(global_step=157, epoch=True)):
            with self.assertRaises(RuntimeError):
                pilot.validate_final_state(state)

    def test_crash_after_evaluation_preserves_and_retries_without_overwrite(self):
        first = pilot.write_evaluation_attempt(self.root, 'fixture', {}, [])
        before = pilot.sha(first)
        # Emulate the interruption before completion publication.
        second = pilot.write_evaluation_attempt(self.root, 'fixture', {}, [])
        self.assertNotEqual(first, second)
        self.assertEqual(pilot.sha(first), before)
        pilot.commit_completion(self.root, dict(after_file=second.name, after_sha256=pilot.sha(second)))
        self.assertEqual(pilot.read_json(self.root / 'TRAINING_COMPLETE.json')['after_file'], second.name)
        with self.assertRaises(RuntimeError):
            pilot.commit_completion(self.root, {})

    def test_completion_publication_cannot_clobber_racing_destination(self):
        original_link = os.link
        def racing_link(source, target, **kwargs):
            pilot.write_new(target, {'other': True})
            original_link(source, target, **kwargs)
        with patch.object(os, 'link', side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                pilot.commit_completion(self.root, {'ours': True})
        self.assertEqual(pilot.read_json(self.root / 'TRAINING_COMPLETE.json'), {'other': True})

    def test_dynamic_development_categories(self):
        pilot.offline()
        import numpy as np
        from transformers import ByT5Tokenizer
        tokenizer = ByT5Tokenizer()
        rows = [dict(id=str(i), input='matn', target='Matn', category=category)
                for i, category in enumerate(('chat_mixed', 'format', 'protected_entities', 'lexical'))]
        predictions = np.array(tokenizer([row['target'] for row in rows], padding=True)['input_ids'])
        trainer = SimpleNamespace(processing_class=tokenizer, eval_dataset=rows,
                                  predict=lambda *a, **kw: SimpleNamespace(predictions=predictions))
        metrics, pairs = pilot.evaluate_development(trainer, rows)
        self.assertEqual(set(metrics['categories']), {row['category'] for row in rows})
        self.assertEqual(metrics['missing_eos_n'], 0)
        self.assertFalse(metrics['final_test_used'])
        self.assertEqual([p['id'] for p in pairs], [r['id'] for r in rows])

    def baseline(self, output, rows):
        self.save(output / 'baseline-development.json', dict(signature='fixture', development_sha256=pilot.DEV_SHA,
            metrics=dict(n=len(rows), split='development', final_test_used=False),
            predictions=[dict(id=r['id'], input=r['input'], target=r['target'], prediction=r['input']) for r in rows]))

    def test_tiny_random_cpu_save_and_same_stage_resume(self):
        pilot.offline()
        import torch
        from transformers import T5Config, T5ForConditionalGeneration, ByT5Tokenizer
        torch.set_num_threads(2)
        torch.manual_seed(3407)
        model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=16, d_kv=8, d_ff=32,
            num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
            eos_token_id=1, pad_token_id=0, use_cache=False, dropout_rate=0.0))
        tokenizer = ByT5Tokenizer()
        rows = [dict(id=str(i), input='salom ' + str(i), target='Salom ' + str(i), category='lexical') for i in range(9)]
        config = dict(pilot.CONFIG, micro_batch=1, accumulation=2, save_steps=2,
                      max_source_tokens=32, max_target_tokens=32)
        output = self.root / 'tiny'; output.mkdir()
        self.baseline(output, rows)
        trainer = pilot.make_trainer(model, tokenizer, rows, rows, output, 'fixture', cpu=True, config=config)
        self.assertEqual(trainer.args.report_to, [])
        trainer.train()
        self.assertEqual(trainer.state.global_step, 5)
        for step in (0, 2, 4, 5):
            state = pilot.verify_checkpoint(output / ('checkpoint-' + str(step)), 'fixture')
            self.assertEqual(state['global_step'], step)
        zero = torch.load(output / 'checkpoint-0/optimizer.pt', weights_only=True)
        self.assertEqual(zero['state'], {})
        expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        checkpoint = output / 'checkpoint-2'
        hashes = {path.name: pilot.sha(path) for path in checkpoint.iterdir()}
        restored = T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True,
                                                             use_safetensors=True)
        newout = self.root / 'tiny-test-branch'; newout.mkdir()
        shutil.copyfile(output / 'baseline-development.json', newout / 'baseline-development.json')
        resumed = pilot.make_trainer(restored, tokenizer, rows, rows, newout, 'fixture', cpu=True, config=config)
        resumed.train(resume_from_checkpoint=str(checkpoint))
        for name, value in expected.items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=1e-6, atol=1e-7)
        self.assertEqual(hashes, {path.name: pilot.sha(path) for path in checkpoint.iterdir()})
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            trainer._save_checkpoint(model, None)
        with (output / 'baseline-development.json').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(RuntimeError, 'binding mismatch'):
            pilot.verify_checkpoint(output / 'checkpoint-5', 'fixture')


if __name__ == '__main__':
    unittest.main()
