"""Explicit, portable paths and conservative laptop inference settings."""
from dataclasses import dataclass
import json
import os
from pathlib import Path


def policy():
    return json.loads(Path(__file__).with_name("model-policy.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Settings:
    home: Path
    threads: int = 4

    def __post_init__(self):
        if not 1 <= self.threads <= 8:
            raise ValueError("CPU oqimlari 1–8 oralig‘ida bo‘lishi kerak.")

    @property
    def model_dir(self):
        return self.home / "models" / policy()["model_directory"]

    @property
    def log_dir(self):
        return self.home / "logs"

    @classmethod
    def default(cls, home=None, threads=4):
        default = Path(__file__).resolve().parents[2]
        if not (default / "pyproject.toml").is_file():
            default = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "UzNormStudio"
        return cls(Path(home or os.environ.get("UZNORM_STUDIO_HOME", default)).resolve(), threads)

