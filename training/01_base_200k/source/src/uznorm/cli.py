"""Thin CLI: validate -> preflight -> smoke -> train -> validation -> frozen test."""

import argparse
import json
import logging
import os
from pathlib import Path

from .config import load_config
from .io import write_json


def parser():
    root = argparse.ArgumentParser(prog="uznorm", description=__doc__)
    root.add_argument("--debug", action="store_true", help="Show full diagnostic traceback (keep logs private)")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("validate-data", "preflight", "smoke", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in ("validate-data", "preflight"):
            command.add_argument("--report", type=Path)
        if name == "validate-data":
            command.add_argument("--quick", action="store_true", help="Skip all-file and shared 8-word audits")
        if name in ("smoke", "train", "evaluate"):
            command.add_argument("--output", type=Path, required=True)
        if name in ("smoke", "train"):
            command.add_argument("--resume", action="store_true")
        if name == "train":
            command.add_argument("--smoke-run", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=("validation", "test"), default="validation")
            command.add_argument("--accept-test", action="store_true")
            command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict = commands.add_parser("predict")
    predict.add_argument("--export", type=Path, required=True)
    predict.add_argument("--text", required=True, help="For sensitive text use Python API, not shell history")
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "predict":
            from .inference import Normalizer
            print(Normalizer(args.export, device=args.device).correct(args.text))
            return 0
        config = load_config(args.config)
        if args.command == "validate-data":
            from .data import validate_data
            result = validate_data(config, full=not args.quick)
        elif args.command == "preflight":
            from .data import release_manifest
            from .hardware import environment, inspect_hardware
            from .tracking import require_tracking
            release_manifest(config)
            hardware = inspect_hardware(config)
            require_tracking(config)
            result = {"hardware": hardware, "environment": environment(),
                      "model": config.model.name, "model_revision": config.model.revision,
                      "training": config.as_dict()["training"],
                      "dataset_sha256": config.data.manifest_sha256, "tracking": config.tracking.mode,
                      "network_auth_checked": False, "note": "Smoke verifies actual GPU/model/W&B/checkpoint use."}
        elif args.command in ("smoke", "train"):
            from .training import run
            run(config, args.output, smoke=args.command == "smoke", resume=args.resume,
                smoke_run=getattr(args, "smoke_run", None))
            return 0
        else:
            from .evaluation import evaluate
            result = {"report_dir": str(evaluate(config, args.output, args.split,
                                                 accept_test=args.accept_test, device=args.device))}
        if getattr(args, "report", None):
            if args.report.exists():
                raise ValueError("Report already exists; choose a new filename")
            write_json(args.report, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        if args.debug:
            raise
        # Third-party exception messages can contain credentials. Do not print their body here.
        if type(exc) in (ValueError, RuntimeError, FloatingPointError, FileNotFoundError):
            message = str(exc)
            for name in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                secret = os.environ.get(name)
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            logging.error("%s: %s", type(exc).__name__, message)
        else:
            logging.error("%s. Use --debug privately for a full traceback; redact credentials before sharing.", type(exc).__name__)
        return 1
