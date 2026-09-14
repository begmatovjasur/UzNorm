"""Offline FP32 CPU inference. No API, network, training, or output rewriting."""
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MODEL = Path(__file__).resolve().parent / 'models' / 'real-392'
SPECIAL = re.compile(r'<(?:extra_id_\d+|pad|/s|unk)>')

def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Bo‘sh bo‘lmagan matn kiriting.')
    if '\x00' in text or SPECIAL.search(text):
        raise ValueError('Maxsus model tokenlari qabul qilinmaydi.')
    if len(text.encode('utf-8')) + 1 > 512:
        raise ValueError('Matn juda uzun: 511 UTF-8 baytdan oshirmang. Avtomatik kesilmaydi.')

class Corrector:
    def __init__(self, model_dir=MODEL, threads=4, *, expected_run_id='ureal-14a56cf24f5b1116', expected_step=392):
        started = time.perf_counter()
        folder = Path(model_dir)
        self.manifest = json.loads((folder / 'LOCAL_MODEL.json').read_text(encoding='utf-8'))
        if self.manifest['run_id'] != expected_run_id or self.manifest['step'] != expected_step:
            raise RuntimeError('Kutilgan run/qadamdagi model emas.')
        declared = self.manifest['files']
        if {p.name for p in folder.iterdir()} != set(declared) | {'LOCAL_MODEL.json'}:
            raise RuntimeError('Model papkasida manifestga kirmagan fayl bor.')
        for name, expected in declared.items():
            if Path(name).name != name or (folder / name).is_symlink():
                raise RuntimeError('Noto‘g‘ri model yo‘li.')
            if file_sha(folder / name) != expected:
                raise RuntimeError('Model fayli buzilgan: ' + name)
        print('MODEL_SHA256_OK; CPU ga yuklanmoqda...', flush=True)
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32).to('cpu').eval()
        self.lock = threading.Lock()
        self.backend = {'device': 'cpu', 'dtype': 'float32', 'threads': threads,
                        'torch': torch.__version__, 'transformers': transformers.__version__,
                        'num_beams': 1, 'do_sample': False, 'max_length': 513,
                        'postprocessing': False, 'network': False}
        self.load_seconds = time.perf_counter() - started
        print(f'MODEL_READY step={expected_step} CPU; yuklash {self.load_seconds:.1f}s', flush=True)

    def correct(self, text):
        validate_text(text)
        with self.lock:
            start = time.perf_counter()
            batch = self.tokenizer(text, return_tensors='pt', truncation=False)
            if batch['input_ids'].shape[-1] > 512:
                raise ValueError('512 ByT5 token chegarasi oshdi.')
            with self.torch.inference_mode():
                ids = self.model.generate(**batch, max_length=513, num_beams=1,
                                          do_sample=False, use_cache=True)
            prediction = self.tokenizer.decode(ids[0], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            return {'input': text, 'output': prediction,
                    'seconds': round(time.perf_counter() - start, 3),
                    'ended_with_eos': bool(ids[0, -1].item() == self.tokenizer.eos_token_id)}
