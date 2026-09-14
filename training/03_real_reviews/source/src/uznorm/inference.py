"""Local exported-model inference. No implicit rewrite rules or semantic guarantees."""

from contextlib import nullcontext
from pathlib import Path

from .config import from_dict
from .data import SPECIAL
from .hardware import inspect_hardware
from .io import read_json, verify_seal
from .tracking import privacy_defaults


class Normalizer:
    def __init__(self, export, *, device="auto"):
        privacy_defaults()
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        export = Path(export)
        verify_seal(export, "EXPORT_COMPLETE.json", ("config.json", "training-provenance.json"))
        provenance = read_json(export / "training-provenance.json")
        if provenance["smoke_only"]:
            raise ValueError("Smoke export is for infrastructure testing, not inference or quality claims")
        self.config = from_dict(provenance["config"], export)
        self.hardware = inspect_hardware(self.config, require_cuda=False)
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = self.hardware["device"] if device == "auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(export, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(export, local_files_only=True,
                                                        use_safetensors=True).to(self.device).eval()

    def predict_batch(self, texts):
        import torch

        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("Provide a nonempty list of strings")
        if any(SPECIAL.search(text) or "\x00" in text for text in texts):
            raise ValueError("Literal model special tokens/NUL are unsupported")
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.config.model.max_source_tokens:
            raise ValueError("Input is too long. Split into complete sentences; automatic truncation is forbidden.")
        cast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" and self.hardware["bf16"] else nullcontext()
        with torch.inference_mode(), cast:
            result = self.model.generate(
                **{key: value.to(self.device) for key, value in encoded.items()},
                max_new_tokens=self.config.model.max_target_tokens, num_beams=self.config.model.num_beams,
                do_sample=False, use_cache=True)
        predictions = self.tokenizer.batch_decode(result, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        finished = [self.tokenizer.eos_token_id in sequence for sequence in result.tolist()]
        return predictions, finished

    def correct(self, text: str) -> str:
        """Return a raw suggestion; reject empty/truncated generation rather than silently accepting it."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return text
        predictions, finished = self.predict_batch([text])
        if not finished[0] or not predictions[0].strip():
            raise RuntimeError("Incomplete or empty generation; keep the original text for review")
        return predictions[0]
