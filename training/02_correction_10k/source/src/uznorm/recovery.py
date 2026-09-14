"""Independent, committed recovery snapshots before slow evaluation (single GPU).

Snapshots preserve optimizer, scheduler, RNG and callbacks. They do not replace
evaluated checkpoints or change the validation set/best-model criterion.
"""

import copy
import logging
from pathlib import Path
import re
import shutil
import time
import uuid

from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState

from .checkpoints import complete_checkpoint, REQUIRED
from .io import read_json, verify_seal, write_json
from .storage import guard_output

LOG = logging.getLogger(__name__)


class RecoveryCallback(TrainerCallback):
    def __init__(self, steps=50, seconds=600, keep=2, resume_checkpoint=None):
        self.steps, self.seconds, self.keep = steps, seconds, keep
        self.trainer = None
        self.last_step, self.last_time = 0, time.monotonic()
        self.recheck_evaluation = False
        if resume_checkpoint and (Path(resume_checkpoint) / "RECOVERY.json").is_file():
            info = read_json(Path(resume_checkpoint) / "RECOVERY.json")
            self.last_step = info["global_step"]
            self.recheck_evaluation = info["evaluation_pending"]

    def on_step_begin(self, args, state, control, **kwargs):
        guard_output(args.output_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if self.recheck_evaluation:
            # Trainer resumes at the next optimizer step. Recheck there instead of
            # silently waiting another 250 steps for the interrupted evaluation.
            control.should_evaluate = True
            control.should_save = True
            self.recheck_evaluation = False
        due = (state.global_step == 1 or state.global_step - self.last_step >= self.steps
               or time.monotonic() - self.last_time >= self.seconds or control.should_evaluate)
        if due and state.global_step > self.last_step:
            self.snapshot(args, state, control)
        return control

    def snapshot(self, args, state, control):
        if self.trainer is None or args.world_size != 1 or args.save_only_model:
            raise RuntimeError("Recovery requires the bound single-GPU full-state trainer")
        output = Path(args.output_dir).resolve()
        guard_output(output)
        store = output / "recovery"
        store.mkdir(exist_ok=True)
        destination = store / f"checkpoint-{state.global_step}"
        if destination.exists():
            raise RuntimeError("Recovery checkpoint already exists; never overwrite an existing snapshot")
        staging = store / (".incomplete-" + uuid.uuid4().hex)
        staging.mkdir()
        trainer = self.trainer
        trainer.save_model(str(staging), _internal_call=True)
        trainer._save_optimizer_and_scheduler(str(staging))
        trainer._save_scaler(str(staging))
        trainer._save_rng_state(str(staging))
        snapshot_state = copy.deepcopy(state)
        for cb in trainer.callback_handler.callbacks + [control]:
            if isinstance(cb, ExportableState):
                snapshot_state.stateful_callbacks[cb.__class__.__name__] = cb.state()
        snapshot_state.save_to_json(str(staging / "trainer_state.json"))
        write_json(staging / "RECOVERY.json", {"global_step": state.global_step,
                   "evaluation_pending": bool(control.should_evaluate), "kind": "full_state_recovery",
                   "wall_time": time.time(), "remote_sync_guaranteed": False})
        guard_output(output)
        complete_checkpoint(staging)
        verify_seal(staging, "COMPLETE.json", REQUIRED)
        guard_output(output)
        staging.rename(destination)  # Incomplete staging is never a resume candidate.
        self.last_step, self.last_time = state.global_step, time.monotonic()
        write_json(output / "RECOVERY_LATEST.json", {"checkpoint": str(destination),
                   "global_step": state.global_step, "remote_sync_guaranteed": False})
        LOG.info("Recovery committed: step=%s path=%s (read-back verified; remote sync not guaranteed)",
                 state.global_step, destination)
        self.prune(store, destination)

    def prune(self, store, newest):
        # Only own sealed recovery snapshots, after a replacement was committed.
        # Never touch ordinary checkpoints, user folders, or interrupted staging.
        valid = []
        for folder in store.iterdir():
            if not re.fullmatch(r"checkpoint-\d+", folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            try:
                verify_seal(folder, "COMPLETE.json", REQUIRED)
                info = read_json(folder / "RECOVERY.json")
                if info.get("kind") == "full_state_recovery":
                    valid.append(folder)
            except (OSError, ValueError, KeyError):
                continue
        valid.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)
        for old in valid[self.keep:]:
            guard_output(store.parent)
            if old.resolve().parent != store.resolve() or old == newest:
                raise RuntimeError("Refusing unsafe recovery retention target")
            LOG.info("Recovery retention removes %s; latest %s full snapshots retained", old, self.keep)
            shutil.rmtree(old)
