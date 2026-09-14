"""Fail closed on Colab: an ordinary /content/drive directory is not durable storage.

A FUSE mount and read-back are necessary checks, not a guarantee of remote sync.
The user confirms the persistent storage ID in Drive's web UI before training.
"""

import os
from pathlib import Path
import re
import uuid

from .io import read_json, write_json

DRIVE = Path("/content/drive")
MARKER = ".uznorm-storage.json"


def mount_details(mountinfo, mountpoint):
    expected = str(Path(mountpoint).resolve())
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 6 or len(filesystem) < 2:
            continue
        decoded = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if decoded == expected and filesystem[0] in ("fuse.drive", "fuse.drivefs"):
            return {"mount_id": fields[0], "filesystem": filesystem[0], "mountpoint": decoded}
    raise RuntimeError("Google Drive FUSE mount topilmadi. Oddiy /content/drive papkasiga yozish taqiqlangan. Drive mount katagini qayta bajaring.")


def require_mount():
    try:
        info = mount_details(Path("/proc/self/mountinfo").read_text(), DRIVE)
    except OSError as exc:
        raise RuntimeError("Drive mount holatini tekshirib bo‘lmadi; trening boshlanmaydi.") from exc
    if not (DRIVE / "MyDrive").is_dir():
        raise RuntimeError("Mounted Drive ichida MyDrive topilmadi.")
    return info


def drive_project(root):
    require_mount()
    root = Path(root).resolve()
    mydrive = (DRIVE / "MyDrive").resolve()
    if root == mydrive or not root.is_relative_to(mydrive):
        raise RuntimeError("Alohida MyDrive loyiha papkasini tanlang; vaqtinchalik /content mumkin emas.")
    return root


def prepare_storage(root, expected_id=""):
    """Explicit notebook setup only. Never initialize a missing identity on resume."""
    root = drive_project(root)
    marker = root / MARKER
    if not marker.is_file():
        if expected_id:
            raise RuntimeError("Oldingi Drive storage ID topilmadi. Boshqa hisob yoki papka; yangi ID avtomatik yaratilmaydi.")
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"storage_id": uuid.uuid4().hex, "purpose": "uznorm durable run storage"})
    value = read_json(marker)
    identity = value.get("storage_id")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise RuntimeError("Drive storage marker noto‘g‘ri; ustidan yozilmaydi.")
    if expected_id and identity != expected_id:
        raise RuntimeError("Drive storage ID mos emas. Treningdagi hisob va loyiha papkasini tekshiring.")
    # Tiny probe checks immediate read-back; it does not prove cloud-side commit.
    probe = root / (".write-probe-" + uuid.uuid4().hex + ".json")
    try:
        write_json(probe, {"nonce": identity})
        require_mount()
        if read_json(probe) != {"nonce": identity}:
            raise RuntimeError("Drive write/read probe mos emas.")
    finally:
        if probe.is_file():
            probe.unlink()  # Only this invocation's uniquely named tiny probe.
    return {"storage_id": identity, "marker": str(marker), **require_mount(),
            "remote_sync_guaranteed": False}


def guard_output(output):
    """Called before run/log/checkpoint writes and every optimizer step in Colab."""
    output = Path(output).resolve()
    required = (os.environ.get("UZNORM_REQUIRE_DRIVE") == "1"
                or "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ
                or Path('/var/colab/hostname').is_file()
                or output.is_relative_to(Path("/content")))
    if not required:
        return {"mode": "local_non_colab", "remote_sync_guaranteed": False}
    require_mount()
    expected = os.environ.get("UZNORM_STORAGE_ID", "")
    root_value = os.environ.get("UZNORM_STORAGE_ROOT", "")
    if not expected or not root_value:
        raise RuntimeError("Drive storage ID tasdiqlanmagan. Notebook storage tekshiruvi katagini bajaring.")
    root = drive_project(root_value)
    if output == root or not output.is_relative_to(root):
        raise RuntimeError("Output tasdiqlangan Drive loyiha papkasining ichida bo‘lishi kerak.")
    try:
        actual = read_json(root / MARKER)["storage_id"]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError("Drive storage marker o‘qilmadi. Vaqtinchalik papkaga yozishni davom ettirmaymiz.") from exc
    if actual != expected:
        raise RuntimeError("Drive hisobi/papkasi almashgan: storage ID mos emas.")
    return {"mode": "google_drive_fuse", "storage_id": actual, "root": str(root),
            "remote_sync_guaranteed": False}


def require_durability_check(smoke_run, storage):
    if storage['mode'] != 'google_drive_fuse':
        return
    try:
        evidence = read_json(Path(smoke_run) / 'DRIVE_DURABILITY.json')
        valid = (evidence.get('storage_id') == storage['storage_id']
                 and evidence.get('verified_after_flush_and_remount') is True)
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError('Smoke Drive flush/remount tekshiruvidan o‘tmagan. Notebookdagi saqlash testi katagini bajaring.')
