"""One-GPU baseline. Native BF16 when supported, otherwise FP32; never FP16/TF32."""

import importlib.metadata
import platform


def inspect_hardware(config, *, require_cuda=True):
    import torch
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        raise RuntimeError("torch >= 2.6 is required for checkpoint loading")
    cuda = torch.cuda.is_available()
    if require_cuda and not cuda:
        raise RuntimeError("GPU yo‘q. Colab: Runtime > Change runtime type > GPU.")
    if cuda and torch.cuda.device_count() != 1:
        raise RuntimeError("This baseline requires exactly one visible GPU")
    native_bf16 = False
    if cuda:
        try:
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
    micro = config.training.micro_batch_size
    if micro == "auto":
        proposed = 4 if gb >= 35 and native_bf16 else (2 if gb >= 22 and native_bf16 else 1)
        micro = next(n for n in (proposed, 2, 1) if n <= proposed and config.training.effective_batch_size % n == 0)
    return {"device": "cuda" if cuda else "cpu", "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "vram_gib": round(gb, 2), "bf16": native_bf16, "fp16": False, "tf32": False,
            "micro_batch": micro, "accumulation": config.training.effective_batch_size // micro}


def environment():
    import torch
    return {"python": platform.python_version(), "platform": platform.system(), "cuda": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "accelerate", "wandb", "rapidfuzz")}}
