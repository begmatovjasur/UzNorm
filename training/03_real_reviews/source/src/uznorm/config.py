"""Strict, version-controlled configuration. Secrets do not belong here."""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import re

from .io import read_json

MODEL_REVISIONS = {
    "google/byt5-small": "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
    "google/byt5-base": "92d8c008d55cf7c254915bac165171dfe6c20c44",
}


@dataclass(frozen=True)
class DataConfig:
    root: str
    manifest_sha256: str
    allow_unverified_targets: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str = "google/byt5-small"
    revision: str = "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
    max_source_tokens: int = 512
    max_target_tokens: int = 512
    num_beams: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 3407
    epochs: int = 3
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    effective_batch_size: int = 32
    micro_batch_size: int | str = "auto"
    eval_batch_size: int = 2
    eval_save_steps: int = 250
    logging_steps: int = 25
    early_stopping_patience: int = 4
    save_total_limit: int = 3
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    mode: str = "online"
    project: str = "uznorm-byt5-v2"
    entity: str | None = None


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig

    def as_dict(self):
        return asdict(self)

    def signature_dict(self):
        value = self.as_dict()
        # Moving an unchanged release between Colab and Drive must not change its identity.
        value["data"].pop("root")
        return value


def _construct(cls, value):
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise ValueError(f"Unknown or invalid keys in {cls.__name__}")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"Missing configuration fields in {cls.__name__}") from exc


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def from_dict(raw, base: Path) -> Config:
    if not isinstance(raw, dict) or set(raw) != {"data", "model", "training", "tracking"}:
        raise ValueError("Config requires exactly data/model/training/tracking sections")
    data = _construct(DataConfig, raw["data"])
    model = _construct(ModelConfig, raw["model"])
    train = _construct(TrainingConfig, raw["training"])
    tracking = _construct(TrackingConfig, raw["tracking"])
    if not isinstance(data.root, str) or not data.root:
        raise ValueError("data.root is required")
    if not isinstance(data.manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", data.manifest_sha256):
        raise ValueError("Pin the exact dataset manifest SHA-256")
    if type(data.allow_unverified_targets) is not bool or type(train.gradient_checkpointing) is not bool:
        raise ValueError("Boolean configuration fields require true/false")
    if not isinstance(model.name, str) or model.name not in MODEL_REVISIONS:
        raise ValueError("Supported models: google/byt5-small and google/byt5-base")
    if not isinstance(model.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", model.revision):
        raise ValueError("Model revision must be an immutable commit SHA")
    if model.revision != MODEL_REVISIONS[model.name]:
        raise ValueError("Model/revision mismatch. Use the pinned revision for the selected model.")
    for name in ("max_source_tokens", "max_target_tokens", "num_beams"):
        positive_int(getattr(model, name), name)
    for name in ("epochs", "effective_batch_size", "eval_batch_size", "eval_save_steps", "logging_steps",
                 "early_stopping_patience", "save_total_limit"):
        positive_int(getattr(train, name), name)
    if type(train.seed) is not int or not 0 <= train.seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if train.save_total_limit < 2:
        raise ValueError("Keep at least two checkpoints for interruption recovery")
    for name in ("learning_rate", "warmup_ratio", "weight_decay"):
        value = getattr(train, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name}")
    if not 0 < train.learning_rate < 1 or not 0 <= train.warmup_ratio < 1:
        raise ValueError("Invalid learning rate or warmup ratio")
    if train.micro_batch_size != "auto":
        positive_int(train.micro_batch_size, "micro_batch_size")
        if train.effective_batch_size % train.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    if tracking.mode not in ("online", "offline", "disabled"):
        raise ValueError("tracking.mode must be online, offline or disabled")
    if not isinstance(tracking.project, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", tracking.project):
        raise ValueError("Use a simple W&B project name")
    if tracking.entity is not None and (not isinstance(tracking.entity, str) or not tracking.entity):
        raise ValueError("tracking.entity must be null or a nonempty string")
    resolved = str((Path(base) / data.root).resolve())
    return Config(DataConfig(resolved, data.manifest_sha256, data.allow_unverified_targets), model, train, tracking)


def load_config(path: Path) -> Config:
    return from_dict(read_json(path), Path(path).resolve().parent)
