"""Verified local checkpoints only; no guessing which interrupted write is usable."""

from pathlib import Path
import re
import logging
import shutil

from .io import contained, read_json, seal, verify_seal
from .storage import guard_output

REQUIRED = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json")


def _model_members(folder, files):
    if "model.safetensors" in files:
        return
    index = "model.safetensors.index.json"
    if index not in files:
        raise ValueError("Safe model weights missing")
    mapping = read_json(contained(folder, index))["weight_map"]
    if not mapping or not set(mapping.values()).issubset(files):
        raise ValueError("Model shards missing")


def complete_checkpoint(folder):
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    _model_members(folder, files)
    seal(folder, "COMPLETE.json", REQUIRED)


def latest_complete(output):
    output = Path(output).resolve()
    folders = [p for base in (output, output / "recovery") for p in base.glob("checkpoint-*")
               if p.is_dir() and not p.is_symlink() and re.fullmatch(r"checkpoint-\d+", p.name)]
    # At equal steps prefer the post-evaluation checkpoint (metrics/callback state).
    for folder in sorted(folders, key=lambda p: (int(p.name.split("-")[-1]), p.parent == output), reverse=True):
        try:
            files = verify_seal(folder, "COMPLETE.json", REQUIRED)
            _model_members(folder, files)
            state = read_json(folder / "trainer_state.json")
            if state["global_step"] != int(folder.name.split("-")[-1]):
                continue
            if folder.parent == output / "recovery":
                info = read_json(folder / "RECOVERY.json")
                if info.get("global_step") != state["global_step"] or info.get("kind") != "full_state_recovery":
                    continue
            best = state.get("best_model_checkpoint")
            if best:
                # Resume uses the same output directory. Never deserialize an arbitrary external checkpoint.
                best_path = Path(best).resolve()
                if best_path.parent != Path(output).resolve():
                    raise ValueError("Best checkpoint points outside the current run")
                best_files = verify_seal(best_path, "COMPLETE.json", REQUIRED)
                _model_members(best_path, best_files)
            return str(folder)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def retain_regular(output, keep, best=None):
    """Prune only after commit, retaining best-model dependencies of fallback states.

HF rotation occurs before on_save seals the new checkpoint. Deferring rotation
prevents a kill at that point from deleting the last recoverable dependency.
"""
    output = Path(output).resolve()
    valid = {}
    recovery_states = []
    for base in (output, output / 'recovery'):
        for folder in base.glob('checkpoint-*'):
            if not re.fullmatch(r'checkpoint-\d+', folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            try:
                files = verify_seal(folder, 'COMPLETE.json', REQUIRED)
                _model_members(folder, files)
                state = read_json(folder / 'trainer_state.json')
            except (OSError, ValueError, KeyError, TypeError):
                continue  # Never delete incomplete or corrupt artifacts here.
            if base == output:
                valid[folder] = state
            else:
                recovery_states.append(state)
    selected = set(sorted(valid, key=lambda p: int(p.name.split('-')[-1]), reverse=True)[:keep])
    pending = [s.get('best_model_checkpoint') for s in recovery_states]
    pending += [valid[p].get('best_model_checkpoint') for p in selected]
    pending.append(best)
    while pending:
        value = pending.pop()
        if not value:
            continue
        path = Path(value).resolve()
        if path.parent != output or path not in valid:
            return  # Unresolved dependency: stop cleanup, do not guess.
        if path not in selected:
            selected.add(path)
            pending.append(valid[path].get('best_model_checkpoint'))
    for folder in set(valid) - selected:
        guard_output(output)
        if folder.resolve().parent != output or folder.is_symlink():
            raise RuntimeError('Refusing unsafe regular checkpoint retention target')
        logging.getLogger(__name__).info('Checkpoint retention removes %s; fallback dependencies retained', folder)
        shutil.rmtree(folder)
