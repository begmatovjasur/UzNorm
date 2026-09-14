"""One CPU model per application home, resident for the session, offline only."""
from dataclasses import asdict, dataclass
import logging
import os
from pathlib import Path
import re
import threading
import time

from .artifacts import ModelError, inspect_model

LOG = logging.getLogger("uznorm_studio")
SPECIAL = re.compile(r"<(?:extra_id_\d+|pad|/s|unk)>")


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Bo‘sh bo‘lmagan matn kiriting.")
    if "\x00" in text or SPECIAL.search(text):
        raise ValueError("Maxsus model tokenlari qabul qilinmaydi.")
    try:
        count = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise ValueError("Matnda noto‘g‘ri Unicode belgisi bor.") from None
    if count > 511:
        raise ValueError(f"Matn {count} bayt. Chegara 511 UTF-8 bayt; matn avtomatik kesilmaydi.")
    return count


class SessionLock:
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def acquire(self):
        if self.stream:
            raise ModelError("Model sessiyasi allaqachon ochiq.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise ModelError("Boshqa UzNorm sessiyasi modelni ishlatyapti. Avval uni yoping.") from None
        self.stream = stream

    def release(self):
        if self.stream:
            self.stream.close()  # OS releases the lock; the file is deliberately retained.
            self.stream = None


@dataclass(frozen=True)
class Prediction:
    input: str
    output: str
    seconds: float
    ended_with_eos: bool
    run_id: str
    step: int
    device: str = "cpu"
    dtype: str = "float32"
    postprocessing: bool = False

    def to_dict(self):
        return asdict(self)


class Corrector:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.tokenizer = self.torch = self.metadata = None
        self._guard = threading.Lock()
        self._lock = SessionLock(settings.home / "models" / ".session.lock")

    @property
    def ready(self):
        return self.model is not None

    def load(self, progress=lambda _: None):
        with self._guard:
            if self.ready:
                return
            self._lock.acquire()
            try:
                self.metadata = inspect_model(self.settings.model_dir, progress)
                for key, value in dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                        USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false").items():
                    os.environ[key] = value
                progress("ByT5 CPU xotirasiga yuklanmoqda…")
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                torch.set_num_threads(self.settings.threads)
                tokenizer = AutoTokenizer.from_pretrained(self.settings.model_dir, local_files_only=True, trust_remote_code=False)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.model_dir, local_files_only=True,
                    use_safetensors=True, trust_remote_code=False, dtype=torch.float32).to("cpu").eval()
                if tokenizer.__class__.__name__ != "ByT5Tokenizer" or sum(p.numel() for p in model.parameters()) != 581653248:
                    raise ModelError("Kutilgan ByT5-base arxitekturasi emas.")
                self.torch, self.tokenizer, self.model = torch, tokenizer, model
                LOG.info("model_ready run=%s step=%s device=cpu threads=%s", self.metadata["run_id"], self.metadata["step"], self.settings.threads)
            except BaseException:
                self.model = self.tokenizer = self.torch = None
                self._lock.release()
                raise
            progress("MODEL TAYYOR · CPU FP32 · internet talab qilinmaydi")

    def correct(self, text):
        byte_count = validate_text(text)
        with self._guard:
            if not self.ready:
                raise ModelError("Avval modelni yuklang.")
            started = time.perf_counter()
            encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[-1] > 512:
                raise ValueError("Token uzunligi limitdan oshgan; avtomatik kesilmadi.")
            with self.torch.inference_mode():
                ids = self.model.generate(**encoded, max_length=513, num_beams=1, do_sample=False, use_cache=True)
            output = self.tokenizer.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ended = self.tokenizer.eos_token_id in ids[0].tolist()
            elapsed = time.perf_counter() - started
            LOG.info("inference_done input_bytes=%s seconds=%.3f ended_with_eos=%s", byte_count, elapsed, ended)
            return Prediction(text, output, elapsed, ended, self.metadata["run_id"], self.metadata["step"])

    def close(self):
        with self._guard:
            self.model = self.tokenizer = self.torch = None
            self._lock.release()

