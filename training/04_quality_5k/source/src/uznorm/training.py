"""Single-GPU full fine-tuning with source-only generation and explicit resume."""

import logging
import math
from pathlib import Path
import time
import uuid

from .tracking import privacy_defaults, require_tracking, start_tracking

privacy_defaults()

import torch
from transformers import (DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback, set_seed)

from .checkpoints import complete_checkpoint, latest_complete, retain_regular
from .data import CATEGORIES, MONITOR, TextDataset, load_split, validate_data
from .hardware import environment, inspect_hardware
from .io import digest, read_json, read_jsonl, run_lock, seal, sha256, verify_seal, write_json
from .metrics import make_compute_metrics, numeric_metrics, score
from .recovery import RecoveryCallback
from .storage import guard_output, require_durability_check

LOG = logging.getLogger(__name__)


class SourceOnlyTrainer(Seq2SeqTrainer):
    """Targets are used for loss only; generate() receives only encoder text/mask."""

    def _save_checkpoint(self, model, trial):
        guard_output(self.args.output_dir)
        return super()._save_checkpoint(model, trial)

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        # Rotation is deferred until SafetyCallback commits COMPLETE.json.
        # Older best-model dependencies may still be needed by recovery snapshots.
        return

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite training loss; inspect data, learning rate and precision")
        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        if prediction_loss_only or not self.args.predict_with_generate:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        inputs = self._prepare_inputs(inputs)
        kwargs = dict(getattr(self, "_gen_kwargs", {}))
        kwargs.update(gen_kwargs)
        kwargs.update(do_sample=False, use_cache=True)
        with torch.no_grad(), self.compute_loss_context_manager():
            generated = self.model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **kwargs)
            loss = model(**inputs).loss.mean().detach() if "labels" in inputs else None
        if loss is not None and not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        length = self.args.generation_max_length
        if generated.shape[-1] < length:
            generated = self._pad_tensors_to_max_len(generated, length)
        labels = inputs.get("labels")
        if labels is not None and labels.shape[-1] < length:
            labels = self._pad_tensors_to_max_len(labels, length)
        return loss, generated, labels


class SafetyCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        guard_output(args.output_dir)
        complete_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        retain_regular(args.output_dir, args.save_total_limit, state.best_model_checkpoint)

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm", "eval_loss"):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise FloatingPointError(f"Non-finite {key}; training stopped")


class ThroughputCallback(TrainerCallback):
    """Local performance evidence, excluding evaluation/checkpoint time; never a fixed cost promise."""

    def __init__(self):
        self.durations = []
        self.start = None

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.start is not None:
            self.durations.append(time.monotonic() - self.start)
            self.start = None

    def report(self, hw, smoke):
        # First optimizer step may include setup/warmup overhead. Two smoke steps are too few for ETA.
        stable = self.durations[1:]
        return {"optimizer_steps_observed": len(self.durations),
                "mean_optimizer_step_seconds_excluding_first": sum(stable) / len(stable) if stable else None,
                "effective_batch_size_this_run": hw["micro_batch"] * (1 if smoke else hw["accumulation"]),
                "evaluation_checkpoint_time_included": False, "smoke_only": smoke,
                "full_training_eta_reliable": False,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if hw["device"] == "cuda" else None,
                "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if hw["device"] == "cuda" else None}


def training_args(config, output, hw, *, smoke=False):
    train = config.training
    return Seq2SeqTrainingArguments(
        output_dir=str(output), num_train_epochs=train.epochs, max_steps=2 if smoke else -1,
        per_device_train_batch_size=hw["micro_batch"],
        gradient_accumulation_steps=1 if smoke else hw["accumulation"],
        per_device_eval_batch_size=train.eval_batch_size,
        learning_rate=train.learning_rate, warmup_ratio=train.warmup_ratio, weight_decay=train.weight_decay,
        optim="adafactor", lr_scheduler_type="linear", max_grad_norm=1.0,
        bf16=hw["bf16"], fp16=False, tf32=False,
        gradient_checkpointing=train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", save_strategy="steps", eval_steps=2 if smoke else train.eval_save_steps,
        save_steps=2 if smoke else train.eval_save_steps, logging_steps=1 if smoke else train.logging_steps,
        logging_first_step=True, logging_nan_inf_filter=False,
        predict_with_generate=True, generation_max_length=config.model.max_target_tokens + 1,
        generation_num_beams=config.model.num_beams,
        load_best_model_at_end=True, metric_for_best_model="category_macro_cer_pct", greater_is_better=False,
        save_total_limit=train.save_total_limit, save_safetensors=True,
        restore_callback_states_from_checkpoint=True,
        seed=train.seed, data_seed=train.seed,
        report_to=[] if config.tracking.mode == "disabled" else ["wandb"],
        dataloader_num_workers=0, eval_accumulation_steps=1, remove_unused_columns=True,
        push_to_hub=False, label_smoothing_factor=0.0,
    )


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def run_signatures(config, hw, env, smoke):
    gate = digest({"config": config.signature_dict(), "hardware": hw, "environment": env, "code": code_hashes()})
    return gate, digest({"gate": gate, "smoke": smoke})


def check_smoke(folder, gate):
    if folder is None:
        raise ValueError("A successful matching smoke run is required: --smoke-run PATH")
    done = read_json(Path(folder) / "TRAINING_COMPLETE.json")
    if done.get("smoke") is not True or done.get("gate_signature") != gate or done.get("global_step") != 2:
        raise ValueError("Smoke test must match the current code, dataset, config and hardware")
    verify_seal(Path(folder) / "export", "EXPORT_COMPLETE.json", ("training-provenance.json", "config.json"))


def select_smoke_rows(rows, per_category=4):
    selected = []
    for category in CATEGORIES:
        pool = [row for row in rows if row["category"] == category]
        if not pool:
            raise ValueError(f"Smoke fixture missing {category}")
        selected.extend(pool[:per_category])
    return selected


def run(config, output, *, smoke=False, resume=False, smoke_run=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output = Path(output).resolve()
    storage = guard_output(output)
    data_root = Path(config.data.root).resolve()
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("Run directory must be separate from the frozen dataset")
    hw = inspect_hardware(config)
    LOG.info("Selected model: %s @ %s; hardware: %s", config.model.name, config.model.revision, hw)
    env = environment()
    require_tracking(config)
    if config.tracking.mode != "disabled":
        LOG.info("W&B sends metrics/config only; raw examples stay local")
    gate, signature = run_signatures(config, hw, env, smoke)
    if not smoke:
        check_smoke(smoke_run, gate)
        require_durability_check(smoke_run, storage)
    # Scan all three splits for integrity; no test examples enter trainer or model selection.
    report = validate_data(config, full=True)
    with run_lock(output):
        meta_path = output / "run-meta.json"
        checkpoint = None
        if meta_path.exists():
            meta = read_json(meta_path)
            if meta.get('storage', {}).get('storage_id') != storage.get('storage_id'):
                raise ValueError('Run boshqa Drive storage ID bilan yaratilgan; hisob/papkani tekshiring.')
            if meta.get("signature") != signature:
                raise ValueError("Code/data/config/environment changed. Use a new run directory.")
            if (output / "TRAINING_COMPLETE.json").exists():
                verify_seal(output / "export", "EXPORT_COMPLETE.json")
                completion = read_json(output / "TRAINING_COMPLETE.json")
                provenance = read_json(output / "export" / "training-provenance.json")
                if completion.get("signature") != signature or provenance.get("signature") != signature:
                    raise ValueError("Completed run/export signature mismatch")
                LOG.info("Run already complete: %s", output / "export")
                return
            if not resume:
                raise ValueError("Existing run: pass --resume or use a new directory")
            # Recover the small final-marker write if export was already committed.
            if (output / "export").exists():
                verify_seal(output / "export", "EXPORT_COMPLETE.json")
                provenance = read_json(output / "export" / "training-provenance.json")
                if provenance["signature"] != signature:
                    raise ValueError("Export/run signature mismatch")
                write_json(output / "TRAINING_COMPLETE.json", provenance["completion"])
                LOG.info("Recovered committed export; no training was repeated")
                return
            checkpoint = latest_complete(output)
            if checkpoint is None:
                raise ValueError("No complete checkpoint to resume. Keep this run and choose a new directory.")
        else:
            if resume or any(p.name != ".run.lock" for p in output.iterdir()):
                raise ValueError("New training requires an empty output directory; resume requires run metadata")
            meta = {"signature": signature, "gate_signature": gate, "wandb_id": uuid.uuid4().hex[:8],
                    "config": config.as_dict(), "hardware": hw, "environment": env,
                    "code_sha256": code_hashes(), "smoke": smoke, "storage": storage}
            write_json(meta_path, meta)
        write_json(output / "data-validation.json", report)
        write_json(output / "environment.json", env)
        write_json(output / "config.resolved.json", config.as_dict())
        wb = start_tracking(config, meta["wandb_id"], output.name, bool(checkpoint),
                            {"config": config.signature_dict(), "hardware": hw, "smoke": smoke}, output)
        try:
            if wb:
                LOG.info("W&B: %s", wb.url or "offline")
            set_seed(config.training.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if hw["device"] == "cuda":
                torch.cuda.reset_peak_memory_stats()
            tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
            # This pinned official revision has pytorch_model.bin, not safetensors.
            # Use PyTorch's restricted weights-only loader; all new saves use safetensors.
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model.name, revision=config.model.revision,
                                                        use_safetensors=False, weights_only=True)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.config.use_cache = False
            model.generation_config.use_cache = True
            parameters = sum(p.numel() for p in model.parameters())
            LOG.info("Full fine-tune: %s; %d trainable parameters", config.model.name, parameters)
            train_rows = load_split(config, "train")
            monitor = list(read_jsonl(data_root / MONITOR))
            if smoke:
                train_rows = select_smoke_rows(train_rows)
                monitor = select_smoke_rows(monitor, 1)
            throughput = ThroughputCallback()
            recovery = RecoveryCallback(resume_checkpoint=checkpoint)
            trainer = SourceOnlyTrainer(
                model=model, args=training_args(config, output, hw, smoke=smoke),
                train_dataset=TextDataset(train_rows, tokenizer, config.model),
                eval_dataset=TextDataset(monitor, tokenizer, config.model), processing_class=tokenizer,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100,
                                                   pad_to_multiple_of=8),
                compute_metrics=make_compute_metrics(tokenizer, monitor),
                callbacks=[SafetyCallback(), throughput, *([] if smoke else [recovery]), EarlyStoppingCallback(
                    early_stopping_patience=config.training.early_stopping_patience)],
            )
            recovery.trainer = trainer
            baseline = score(monitor, [row["input"] for row in monitor])
            write_json(output / "copy-baseline-monitor.json", baseline)
            if wb:
                wb.config.update({"trainable_parameters": parameters,
                                  "train_rows": len(train_rows), "monitor_rows": len(monitor)})
                wb.log({"copy_baseline/" + k: v for k, v in numeric_metrics(baseline).items()})
            start = time.monotonic()
            result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)
            performance = throughput.report(hw, smoke)
            write_json(output / "performance.json", performance)
            LOG.info("Performance (smoke is not a full-run ETA): %s", performance)
            if wb:
                wb.log({"performance/" + key: value for key, value in numeric_metrics(performance).items()})
            if latest_complete(output) is None:
                raise RuntimeError("No complete checkpoint after training; export is blocked")
            temporary_export = output / f".export-{uuid.uuid4().hex}"
            guard_output(output)
            trainer.save_model(str(temporary_export))
            tokenizer.save_pretrained(temporary_export)
            completion = {"signature": signature, "gate_signature": gate, "smoke": smoke,
                          "global_step": trainer.state.global_step, "best_metric": trainer.state.best_metric,
                          "best_checkpoint": trainer.state.best_model_checkpoint,
                          "elapsed_this_session_seconds": time.monotonic() - start,
                          "wandb_url": wb.url if wb else None}
            write_json(temporary_export / "training-provenance.json", {
                "signature": signature, "config": config.as_dict(), "completion": completion,
                "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
                "targets_gold": False, "meaning_preservation_measured": False,
                "dataset_sha256": config.data.manifest_sha256, "smoke_only": smoke})
            with (temporary_export / "README.md").open("w", encoding="utf-8") as stream:
                stream.write("# Experimental Uzbek ByT5 normalization\n\n"
                             "Full fine-tune; output is a correction suggestion. Targets are not verified Gold.\n"
                             "No demonstrated real-chat grammar or meaning-preservation accuracy.\n"
                             "Review names, negation, numbers and code-switched text. No redistribution rights implied.\n"
                             "See training-provenance.json and the run's evaluation reports.\n")
            seal(temporary_export, "EXPORT_COMPLETE.json", ("config.json", "training-provenance.json"))
            temporary_export.rename(output / "export")
            write_json(output / "TRAINING_COMPLETE.json", completion)
            if wb:
                wb.summary["training_complete"] = True
            LOG.info("Saved %s", output / "export")
        except BaseException:
            if wb:
                wb.finish(exit_code=1)
            raise
        else:
            if wb:
                wb.finish()
