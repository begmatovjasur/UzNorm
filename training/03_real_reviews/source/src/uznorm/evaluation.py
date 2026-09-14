"""Frozen validation/test evaluation, copy baseline and local human-review artifacts."""

from collections import Counter
import logging
from pathlib import Path
import time
import uuid

from .data import CATEGORIES, load_split, release_manifest
from .inference import Normalizer
from .hardware import environment, inspect_hardware
from .io import digest, read_json, run_lock, sha256, verify_seal, write_json, write_jsonl, seal
from .metrics import score

LOG = logging.getLogger(__name__)


def evaluate(config, output, split="validation", *, accept_test=False, device="auto"):
    if split not in ("validation", "test"):
        raise ValueError("Only validation or test can be evaluated")
    if split == "test" and not accept_test:
        raise ValueError("Freeze model/config on validation first, then explicitly pass --accept-test")
    output = Path(output).resolve()
    completion = read_json(output / "TRAINING_COMPLETE.json")
    meta = read_json(output / "run-meta.json")
    if completion["smoke"]:
        raise ValueError("Smoke results are not final model quality")
    if meta["config"]["data"]["manifest_sha256"] != config.data.manifest_sha256:
        raise ValueError("Cannot evaluate a different dataset under this run")
    if {k: meta["config"][k] for k in ("model", "training")} != {
            k: config.as_dict()[k] for k in ("model", "training")}:
        raise ValueError("Use the run's frozen model/training configuration for evaluation")
    release_manifest(config)
    export = output / "export"
    verify_seal(export, "EXPORT_COMPLETE.json", ("config.json", "training-provenance.json"))
    provenance = read_json(export / "training-provenance.json")
    if provenance["signature"] != meta["signature"] or completion["signature"] != meta["signature"]:
        raise ValueError("Export/run signature mismatch")
    # Never return a stale report for changed code, model or dataset.
    signature = digest({"export": sha256(export / "EXPORT_COMPLETE.json"),
                        "data": config.data.manifest_sha256, "split": split, "device": device,
                        "runtime": environment(), "hardware": inspect_hardware(config, require_cuda=False),
                        "code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    report_dir = output / "evaluation" / f"{split}-{signature[:12]}"
    with run_lock(report_dir):
        if (report_dir / "EVALUATION_COMPLETE.json").exists():
            verify_seal(report_dir, "EVALUATION_COMPLETE.json", ("metrics.json", "predictions.jsonl"))
            report = read_json(report_dir / "metrics.json")
            if report["signature"] != signature:
                raise ValueError("Evaluation signature mismatch")
            return report_dir
        normalizer = Normalizer(export, device=device)
        rows = load_split(config, split)
        predictions, finished = [], []
        start = time.monotonic()
        batch = config.training.eval_batch_size
        for offset in range(0, len(rows), batch):
            predicted, done = normalizer.predict_batch([r["input"] for r in rows[offset:offset + batch]])
            predictions.extend(predicted)
            finished.extend(done)
            if offset % 200 == 0:
                LOG.info("Generated %d/%d %s examples", len(predictions), len(rows), split)
        baseline = score(rows, [r["input"] for r in rows])
        measured = score(rows, predictions)
        reduction = {}
        for name in ("raw_cer_pct", "spelling_cer_pct", "content_wer_pct"):
            reduction[name] = 100 * (baseline[name] - measured[name]) / baseline[name] if baseline[name] else None
        by_category = {}
        for category in CATEGORIES:
            indices = [i for i, row in enumerate(rows) if row["category"] == category]
            by_category[category] = score([rows[i] for i in indices], [predictions[i] for i in indices], breakdown=False)
        report = {"signature": signature, "split": split, "n": len(rows), "copy_baseline": baseline,
                  "model": measured, "relative_error_reduction_pct": reduction, "by_category": by_category,
                  "source_counts": dict(Counter(r["source_kind"] for r in rows)),
                  "generation_missing_eos_pct": 100 * (len(rows) - sum(finished)) / len(rows),
                  "generation_wall_seconds": time.monotonic() - start,
                  "dataset_sha256": config.data.manifest_sha256,
                  "export_manifest_sha256": sha256(export / "EXPORT_COMPLETE.json"),
                  "environment": environment(), "hardware": normalizer.hardware,
                  "inference_device": normalizer.device,
                  "meaning_preservation_measured": False, "gold_grammar_accuracy_measured": False,
                  "warning": "Synthetic reconstruction of unverified targets; not real-chat Gold performance."}
        # Stage then commit to allow safe retry after interruption; no previous completed result is replaced.
        stage = report_dir / f".pending-{uuid.uuid4().hex}"
        write_json(stage / "metrics.json", report)
        write_jsonl(stage / "predictions.jsonl", ({"id": row["id"], "category": row["category"],
                     "input": row["input"], "target": row["target"], "prediction": pred, "has_eos": done}
                     for row, pred, done in zip(rows, predictions, finished)))
        review = []
        for category in CATEGORIES:
            candidates = [i for i, row in enumerate(rows) if row["category"] == category]
            for i in sorted(candidates, key=lambda i: digest(["human-review", rows[i]["id"]]))[:25]:
                review.append({"id": rows[i]["id"], "category": category, "input": rows[i]["input"],
                               "reference_unverified": rows[i]["target"], "prediction": predictions[i],
                               "meaning_preserved": None, "grammar_correct": None, "spelling_correct": None,
                               "names_numbers_preserved": None, "reviewer": None, "notes": None})
        write_jsonl(stage / "manual-review-200.jsonl", review)
        for path in stage.iterdir():
            path.replace(report_dir / path.name)
        stage.rmdir()
        # The lock itself is excluded from the artifact seal.
        seal(report_dir, "EVALUATION_COMPLETE.json", ("metrics.json", "predictions.jsonl"))
        return report_dir
