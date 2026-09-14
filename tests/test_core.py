"""Small fixtures only: no checkpoint download, no credentials, no network."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock
import zipfile

from uznorm_studio.artifacts import FILES, ModelError, check_name, import_archive, inspect_model, write_new
from uznorm_studio.config import Settings, policy
from uznorm_studio.service import Corrector, Prediction, SessionLock, validate_text


def blob(value):
    return json.dumps(value, sort_keys=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    """A fake sealed training archive with the same schema, never a usable model."""
    expected = policy()
    prefix = f"run/checkpoint-{expected['step']}/"
    payload = {prefix + name: b"fake-weights" if name == "model.safetensors" else blob({}) for name in FILES}
    payload[prefix + "config.json"] = blob({"model_type": "t5", "architectures": ["T5ForConditionalGeneration"]})
    payload[prefix + "tokenizer_config.json"] = blob({"tokenizer_class": "ByT5Tokenizer"})
    payload[prefix + "trainer_state.json"] = blob({"global_step": 157, "epoch": 1.0, "best_model_checkpoint": None})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"):
        payload[prefix + name] = b"NEVER deserialize or import me"
    payload["run/baseline-development.json"] = blob({"test_fixture": True})
    payload["run/comparison.json"] = blob({"test_fixture": True})
    after = "after-development-test.json"
    payload["run/" + after] = blob({"test_fixture": True})
    payload[prefix + "PILOT_CHECKPOINT.json"] = blob(dict(signature=expected["run_signature"], step=157,
        baseline_sha256=digest(payload["run/baseline-development.json"]), development_sha256=expected["development_sha256"]))
    payload[prefix + "COMPLETE.json"] = blob({"files": {n.removeprefix(prefix): digest(v) for n, v in payload.items() if n.startswith(prefix)}})
    payload["run/RUN.json"] = blob(dict(binding=expected["binding"], signature=expected["run_signature"],
        parent=dict(run_id=expected["parent_run_id"], step=392)))
    payload["run/RESULT.json"] = blob(dict(signature=expected["run_signature"], step=157, epoch=1, final_test_used=False,
        baseline_sha256=digest(payload["run/baseline-development.json"]), comparison_sha256=digest(payload["run/comparison.json"]),
        after_file=after, after_sha256=digest(payload["run/" + after])))
    payload["kit/PACKAGE.json"] = blob({"test_fixture": True})
    expected["package_sha256"] = digest(payload["kit/PACKAGE.json"])
    return expected, payload


def archive(path, expected, payload, manifest_changes=None, corrupt=None):
    manifest = dict(schema=1, binding=expected["binding"], kind="final", step=157,
                    files={name: digest(value) for name, value in payload.items()})
    manifest.update(manifest_changes or {})
    with zipfile.ZipFile(path, "x") as target:
        for name, value in payload.items():
            target.writestr(name, value + b"damage" if name == corrupt else value)
        target.writestr("CLOUD_MANIFEST.json", blob(manifest))


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uznorm-test-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.expected, self.payload = fixture()
        self.zip = self.root / "fixture.zip"
        self.model = self.root / "models/quality5k-157"

    def test_valid_import_only_inference_files(self):
        archive(self.zip, self.expected, self.payload)
        import_archive(self.zip, self.model, expected=self.expected)
        self.assertEqual({p.name for p in self.model.iterdir()}, set(FILES) | {"LOCAL_MODEL.json"})
        self.assertEqual(inspect_model(self.model, expected=self.expected)["step"], 157)
        self.assertFalse((self.model / "optimizer.pt").exists())
        with self.assertRaises(ModelError):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_corrupted_member_rejected_before_publish(self):
        archive(self.zip, self.expected, self.payload, corrupt="run/checkpoint-157/model.safetensors")
        with self.assertRaisesRegex(ModelError, "buzilgan"):
            import_archive(self.zip, self.model, expected=self.expected)
        self.assertFalse(self.model.exists())

    def test_old_model_and_checkpoint_kind_rejected(self):
        for changes in ({"step": 392}, {"kind": "checkpoint"}, {"binding": {"run_id": "old"}}):
            with self.subTest(changes=changes):
                path = self.root / (str(len(list(self.root.iterdir()))) + ".zip")
                archive(path, self.expected, self.payload, changes)
                with self.assertRaises(ModelError):
                    import_archive(path, self.model, expected=self.expected)
                self.assertFalse(self.model.exists())

    def test_wrong_parent_rejected_even_if_outer_hash_matches(self):
        data = json.loads(self.payload["run/RUN.json"])
        data["parent"]["run_id"] = "other"
        self.payload["run/RUN.json"] = blob(data)
        archive(self.zip, self.expected, self.payload)
        with self.assertRaisesRegex(ModelError, "lineage"):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_inner_hash_mismatch_rejected(self):
        self.payload["run/checkpoint-157/model.safetensors"] = b"new different bytes"
        archive(self.zip, self.expected, self.payload)
        with self.assertRaisesRegex(ModelError, "Ichki/tashqi"):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_baseline_binding_rejected(self):
        self.payload["run/baseline-development.json"] = blob({"changed": True})
        archive(self.zip, self.expected, self.payload)
        with self.assertRaisesRegex(ModelError, "baseline"):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_traversal_rejected(self):
        self.payload["../escape.txt"] = b"no"
        archive(self.zip, self.expected, self.payload)
        with self.assertRaisesRegex(ModelError, "yo‘l"):
            import_archive(self.zip, self.model, expected=self.expected)
        self.assertFalse((self.root / "escape.txt").exists())

    def test_unsafe_names(self):
        for name in ("../a", "a/../b", "/abs", "C:/a", "a\\b", "a//b", "./a", "a/./b", ""):
            with self.subTest(name=name), self.assertRaises(ModelError):
                check_name(name)

    def test_duplicate_zip_members_rejected(self):
        archive(self.zip, self.expected, self.payload)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(self.zip, "a") as target:
                target.writestr("kit/PACKAGE.json", b"duplicate")
        with self.assertRaises(ModelError):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_local_tamper_detected(self):
        archive(self.zip, self.expected, self.payload)
        import_archive(self.zip, self.model, expected=self.expected)
        (self.model / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ModelError, "buzilgan"):
            inspect_model(self.model, expected=self.expected)

    def test_extra_local_code_rejected(self):
        archive(self.zip, self.expected, self.payload)
        import_archive(self.zip, self.model, expected=self.expected)
        (self.model / "custom.py").write_text("never execute")
        with self.assertRaises(ModelError):
            inspect_model(self.model, expected=self.expected)

    def test_no_disk_space_no_publication(self):
        archive(self.zip, self.expected, self.payload)
        with patch("uznorm_studio.artifacts.shutil.disk_usage", return_value=types.SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ModelError, "joy"):
                import_archive(self.zip, self.model, expected=self.expected)
        self.assertFalse(self.model.exists())

    def test_failed_staging_never_becomes_installed_model(self):
        archive(self.zip, self.expected, self.payload)
        with patch("uznorm_studio.artifacts.inspect_model", side_effect=ModelError("late verification failure")):
            with self.assertRaises(ModelError):
                import_archive(self.zip, self.model, expected=self.expected)
        self.assertFalse(self.model.exists())
        self.assertTrue(self.zip.exists())

    def test_missing_inventory_file_rejected(self):
        archive(self.zip, self.expected, self.payload, {"files": {"missing.json": "0" * 64}})
        with self.assertRaises(ModelError):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_unexpected_symlink_zip_entry_rejected(self):
        archive(self.zip, self.expected, self.payload)
        with zipfile.ZipFile(self.zip, "a") as target:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            target.writestr(info, "outside")
        with self.assertRaises(ModelError):
            import_archive(self.zip, self.model, expected=self.expected)

    def test_json_export_never_overwrites(self):
        target = self.root / "result.json"
        write_new(target, {"input": "hello"})
        with self.assertRaises(FileExistsError):
            write_new(target, {"input": "changed"})
        self.assertEqual(json.loads(target.read_text())["input"], "hello")


class ServiceTests(unittest.TestCase):
    def test_byte_limit_not_character_limit(self):
        self.assertEqual(validate_text("a" * 511), 511)
        self.assertEqual(validate_text("‘" * 170), 510)
        with self.assertRaises(ValueError):
            validate_text("‘" * 171)

    def test_invalid_inputs_rejected(self):
        for text in ("", "  \n", "x\x00", "<extra_id_0>", "<pad>", "</s>", "\ud800", None):
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                validate_text(text)

    def test_normal_input_not_normalized(self):
        self.assertEqual(validate_text("  o'zbek  "), len("  o'zbek  ".encode()))

    def test_settings_thread_bounds(self):
        with self.assertRaises(ValueError):
            Settings(Path.cwd(), 0)
        with self.assertRaises(ValueError):
            Settings(Path.cwd(), 9)

    def test_session_lock_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = SessionLock(Path(directory) / "lock"), SessionLock(Path(directory) / "lock")
            try:
                first.acquire()
                with self.assertRaises(ModelError):
                    second.acquire()
                first.release()
                second.acquire()
            finally:
                first.release()
                second.release()

    def test_missing_model_does_not_fallback_or_hold_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Corrector(Settings(Path(directory)))
            with self.assertRaises(ModelError):
                engine.load()
            self.assertFalse(engine.ready)
            self.assertIsNone(engine._lock.stream)

    def test_correct_requires_loaded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelError):
                Corrector(Settings(Path(directory))).correct("salom")

    def test_prediction_keeps_raw_output_and_warns_on_missing_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Corrector(Settings(Path(directory)))
            engine.model, engine.tokenizer = MagicMock(), MagicMock()
            engine.metadata = {"run_id": "fixture", "step": 157}
            engine.torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
            engine.tokenizer.return_value = {"input_ids": types.SimpleNamespace(shape=(1, 8))}
            engine.tokenizer.eos_token_id = 1
            engine.model.generate.return_value = [types.SimpleNamespace(tolist=lambda: [0, 4, 5])]
            engine.tokenizer.decode.return_value = "  O'zBeK?!  "
            with self.assertLogs("uznorm_studio", level="INFO") as logs:
                result = engine.correct("privateText123")
            self.assertEqual(result.output, "  O'zBeK?!  ")
            self.assertFalse(result.ended_with_eos)
            self.assertFalse(result.postprocessing)
            kwargs = engine.model.generate.call_args.kwargs
            self.assertEqual(kwargs["max_length"], 513)
            self.assertFalse(kwargs["do_sample"])
            self.assertEqual(kwargs["num_beams"], 1)
            self.assertNotIn("privateText123", str(logs.output))
            self.assertNotIn("O'zBeK", str(logs.output))

    def test_package_import_does_not_load_model_libraries(self):
        code = "import sys; import uznorm_studio; from uznorm_studio import cli; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; print('INERT_OK')"
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_loader_is_local_safetensors_only_and_loads_once(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            self.skipTest("Optional inference libraries not installed")
        with tempfile.TemporaryDirectory() as directory:
            engine = Corrector(Settings(Path(directory)))
            tokenizer = type("ByT5Tokenizer", (), {})()
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = model
            model.parameters.return_value = [types.SimpleNamespace(numel=lambda: 581653248)]
            try:
                with patch("uznorm_studio.service.inspect_model", return_value={"run_id": "fixture", "step": 157}), \
                     patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer) as tok_load, \
                     patch.object(AutoModelForSeq2SeqLM, "from_pretrained", return_value=model) as load:
                    engine.load()
                    engine.load()
                    self.assertEqual(load.call_count, 1)
                    self.assertTrue(load.call_args.kwargs["local_files_only"])
                    self.assertTrue(load.call_args.kwargs["use_safetensors"])
                    self.assertFalse(load.call_args.kwargs["trust_remote_code"])
                    self.assertEqual(load.call_args.kwargs["dtype"], torch.float32)
                    self.assertTrue(tok_load.call_args.kwargs["local_files_only"])
                    self.assertFalse(tok_load.call_args.kwargs["trust_remote_code"])
                    self.assertTrue(engine.ready)
            finally:
                engine.close()

    def test_tiny_random_t5_backend_contract_not_language_quality(self):
        try:
            import torch
            from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration
        except ImportError:
            self.skipTest("Optional inference libraries not installed")
        with tempfile.TemporaryDirectory() as directory:
            torch.set_num_threads(2)
            torch.manual_seed(7)
            engine = Corrector(Settings(Path(directory)))
            engine.model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=16, d_ff=32,
                num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
                pad_token_id=0, eos_token_id=1)).eval()
            engine.tokenizer, engine.torch = ByT5Tokenizer(), torch
            engine.metadata = {"run_id": "tiny-random-test-not-user-model", "step": 157}
            result = engine.correct("Salom.")
            self.assertIsInstance(result.output, str)
            self.assertEqual(result.input, "Salom.")
            self.assertEqual(result.device, "cpu")
            self.assertGreater(result.seconds, 0)
            engine.close()


class GuiTests(unittest.TestCase):
    def test_window_and_raw_prediction_without_model(self):
        from uznorm_studio.tk_runtime import prepare_tk
        prepare_tk()
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception as exc:
            self.skipTest("GUI display unavailable: " + type(exc).__name__)
        from uznorm_studio.gui import Studio
        with tempfile.TemporaryDirectory() as directory:
            try:
                root.withdraw()
                view = Studio(root, Settings(Path(directory)))
                self.assertEqual(str(view.correct_button["state"]), "disabled")
                view.input_box.insert("1.0", "xato 😊")
                root.update()
                result = Prediction("xato 😊", "  Xato 😊!  ", 0.1, True, "fixture", 157)
                view.show_prediction(result)
                self.assertEqual(view.output_box.get("1.0", "end-1c"), result.output)
                self.assertEqual(str(view.output_box["state"]), "disabled")
                view.clear()
                self.assertEqual(view.input_box.get("1.0", "end-1c"), "")
                self.assertIsNone(view.prediction)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
