"""UTF-8 I/O, atomic metadata, and bounded, checksum-verified artifacts."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import uuid


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path.name}:{number}")
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL: {path.name}:{number}") from exc


@contextmanager
def atomic_text(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value):
    with atomic_text(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: Path, values):
    with atomic_text(path) as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def contained(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlink escapes on Windows and Linux."""
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Invalid artifact member path")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Artifact member escapes its root")
    target = (root / part).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Artifact member escapes its root")
    return target


def seal(folder: Path, marker: str, required=()):
    folder = Path(folder)
    for name in required:
        if not contained(folder, name).is_file():
            raise ValueError(f"Incomplete artifact: missing {name}")
    files = {}
    for path in sorted(folder.rglob("*")):
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue  # Cooperative locks and abandoned staging directories are not artifacts.
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in sealed artifacts")
        if path.is_file() and path.name != marker and not path.name.endswith(".tmp"):
            files[path.relative_to(folder).as_posix()] = sha256(path)
    if not files:
        raise ValueError("Cannot seal an empty artifact")
    write_json(folder / marker, {"files": files})


def verify_seal(folder: Path, marker: str, required=()):
    files = read_json(Path(folder) / marker)["files"]
    if not isinstance(files, dict) or not files or not set(required).issubset(files):
        raise ValueError("Incomplete artifact manifest")
    for name, expected in files.items():
        path = contained(folder, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return files


@contextmanager
def run_lock(folder: Path):
    """Cooperative one-writer lock. Never automatically remove a potentially live lock."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".run.lock"
    token = uuid.uuid4().hex
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "token": token}, stream)
    except FileExistsError as exc:
        raise RuntimeError("Run is locked. Check the original process before manually removing .run.lock.") from exc
    try:
        yield
    finally:
        if path.exists() and read_json(path).get("token") == token:
            path.unlink()
