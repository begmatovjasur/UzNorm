"""Fail-closed import: verify the cloud archive; extract six inference files only.

Never executes pickle or remote code. Hashes establish integrity and lineage
consistency, not a publisher digital signature. Use the supplied private Drive URL.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

from .config import policy

FILES = ("model.safetensors", "config.json", "generation_config.json", "added_tokens.json",
         "special_tokens_map.json", "tokenizer_config.json")
JSON_LIMIT = 4 * 1024**2
ARCHIVE_LIMIT = 8 * 1024**3


class ModelError(RuntimeError):
    """A user-readable error without raw credentials or model inputs."""


def require(condition, message):
    if not condition:
        raise ModelError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    require(Path(path).stat().st_size <= JSON_LIMIT, "JSON fayli juda katta.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_name(name):
    require(isinstance(name, str), "Arxiv fayl nomi matn bo‘lishi kerak.")
    parts = PurePosixPath(name).parts
    require(name and not name.startswith("/") and "\\" not in name and ":" not in name
            and ".." not in parts and "." not in name.split("/") and "//" not in name,
            "Arxivda xavfli yoki noaniq fayl yo‘li bor.")


def inspect_model(folder, progress=lambda _: None, expected=None):
    expected = expected or policy()
    folder = Path(folder)
    require(folder.is_dir(), "Yangi 5k model hali import qilinmagan. Avval yakuniy ZIPni tanlang.")
    require(not folder.is_symlink(), "Model papkasi symlink bo‘lishi mumkin emas.")
    require({p.name for p in folder.iterdir()} == set(FILES) | {"LOCAL_MODEL.json"},
            "Model papkasida yetishmagan yoki ortiqcha fayl bor.")
    require(not any(p.is_symlink() or not p.is_file() for p in folder.iterdir()), "Model yo‘li noto‘g‘ri.")
    meta = read_json(folder / "LOCAL_MODEL.json")
    require(meta.get("schema") == 1 and meta.get("run_id") == expected["binding"]["run_id"]
            and meta.get("step") == expected["step"] and meta.get("epoch") == expected["epoch"]
            and meta.get("run_signature") == expected["run_signature"]
            and meta.get("parent_run_id") == expected["parent_run_id"]
            and meta.get("source_kind") == "final" and set(meta.get("files", {})) == set(FILES),
            "Model versiyasi mos emas. Eski real-392/3551 avtomatik tanlanmaydi.")
    for name, digest in meta["files"].items():
        progress("Tekshirilmoqda: " + name)
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest)
                and sha(folder / name) == digest, "Model fayli buzilgan: " + name)
    config, tokenizer = read_json(folder / "config.json"), read_json(folder / "tokenizer_config.json")
    require(config.get("model_type") == "t5" and config.get("architectures") == ["T5ForConditionalGeneration"]
            and tokenizer.get("tokenizer_class") == "ByT5Tokenizer"
            and not config.get("auto_map") and not tokenizer.get("auto_map"),
            "Faqat standart ByT5 modeli qabul qilinadi.")
    return meta


def import_archive(source, target, progress=lambda _: None, expected=None):
    expected = expected or policy()
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink(), "Yuklab olingan ZIP fayli topilmadi.")
    require(not target.exists(), "Model papkasi allaqachon mavjud. Ustiga yozilmaydi.")
    require(not any(p.is_symlink() for p in (target, *target.parents)), "Model yo‘lida symlink bor.")
    progress("Arxiv tarkibi tekshirilmoqda…")
    with zipfile.ZipFile(source) as bundle:
        infos = bundle.infolist()
        names = {i.filename for i in infos}
        require(len(infos) <= 2000 and len(names) == len(infos)
                and sum(i.file_size for i in infos) <= ARCHIVE_LIMIT, "ZIP hajmi/tarkibi kutilganidan farq qiladi.")
        for info in infos:
            check_name(info.filename)
            require(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16), "ZIP entry turi noto‘g‘ri.")
        def js(name):
            require(name in names and bundle.getinfo(name).file_size <= JSON_LIMIT,
                    "Arxiv metama’lumoti yetishmaydi: " + name)
            return json.loads(bundle.read(name))
        manifest = js("CLOUD_MANIFEST.json")
        require(manifest.get("schema") == 1 and manifest.get("binding") == expected["binding"]
                and manifest.get("step") == expected["step"] and manifest.get("kind") == "final",
                "Aynan yangi 5k modelning 157-qadam FINAL arxivi kerak.")
        inventory = manifest.get("files", {})
        require(set(inventory) | {"CLOUD_MANIFEST.json"} == names, "ZIP manifesti va fayllari mos emas.")
        for index, (name, digest) in enumerate(inventory.items(), 1):
            require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "Noto‘g‘ri SHA-256.")
            progress(f"SHA-256 {index}/{len(inventory)}: {name}")
            with bundle.open(name) as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == digest, "ZIP fayli buzilgan: " + name)
        require(inventory.get("kit/PACKAGE.json") == expected["package_sha256"], "Trening paketi boshqa.")
        run, result = js("run/RUN.json"), js("run/RESULT.json")
        prefix = f"run/checkpoint-{expected['step']}/"
        state, seal, marker = js(prefix + "trainer_state.json"), js(prefix + "COMPLETE.json"), js(prefix + "PILOT_CHECKPOINT.json")
        require(run.get("binding") == expected["binding"] and run.get("signature") == expected["run_signature"]
                and run.get("parent", {}).get("run_id") == expected["parent_run_id"]
                and run.get("parent", {}).get("step") == expected["parent_step"], "Model lineage mos emas.")
        require(type(state.get("global_step")) is int and state["global_step"] == expected["step"]
                and state.get("epoch") == 1 and not state.get("best_model_checkpoint")
                and result.get("step") == expected["step"] and result.get("epoch") == 1
                and result.get("signature") == expected["run_signature"] and result.get("final_test_used") is False,
                "Trening yakuni tasdiqlanmagan.")
        require(marker == dict(signature=expected["run_signature"], step=expected["step"],
                baseline_sha256=inventory.get("run/baseline-development.json"),
                development_sha256=expected["development_sha256"]), "Checkpoint/baseline bog‘liqligi noto‘g‘ri.")
        for field, name in (("baseline_sha256", "baseline-development.json"), ("comparison_sha256", "comparison.json")):
            require(result.get(field) == inventory.get("run/" + name) and "run/" + name in inventory,
                    "Yakuniy hisobot checksum mos emas.")
        after = result.get("after_file", "")
        check_name(after)
        require("/" not in after and after.startswith("after-development-") and after.endswith(".json")
                and result.get("after_sha256") == inventory.get("run/" + after)
                and "run/" + after in inventory, "Yakuniy baholash fayli mos emas.")
        required = set(FILES) | {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "PILOT_CHECKPOINT.json"}
        require(required.issubset(seal.get("files", {})), "To‘liq checkpoint yetishmaydi.")
        require({n.removeprefix(prefix) for n in inventory if n.startswith(prefix)} == set(seal["files"]) | {"COMPLETE.json"},
                "Checkpoint inventory mos emas.")
        for name, digest in seal["files"].items():
            check_name(name)
            require("/" not in name and inventory.get(prefix + name) == digest, "Ichki/tashqi SHA-256 mos emas.")
        target.parent.mkdir(parents=True, exist_ok=True)
        size = sum(bundle.getinfo(prefix + n).file_size for n in FILES)
        require(shutil.disk_usage(target.parent).free > size + 1024**3, "Model uchun diskda joy yetarli emas.")
        # Normal mkdir inherits the project ACL on Windows. mkdtemp mode 0700
        # may create a protected ACL that other legitimate local sessions cannot read.
        staging = target.parent / (".import-" + uuid.uuid4().hex)
        staging.mkdir(mode=0o755, exist_ok=False)
        for name in FILES:
            progress("Model ko‘chirilmoqda: " + name)
            with bundle.open(prefix + name) as incoming, (staging / name).open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
        progress("Manba arxivining yakuniy SHA-256 tekshiruvi…")
        write_new(staging / "LOCAL_MODEL.json", dict(schema=1, run_id=expected["binding"]["run_id"],
            step=expected["step"], epoch=1, run_signature=expected["run_signature"],
            parent_run_id=expected["parent_run_id"], source_kind="final", source_archive_sha256=sha(source),
            files={n: seal["files"][n] for n in FILES}, inference_only=True))
    inspect_model(staging, progress, expected)
    require(not target.exists(), "Boshqa import tugagan; uning ustiga yozilmaydi.")
    staging.rename(target)
    progress("MODEL IMPORT QILINDI: " + str(target))
    return target
