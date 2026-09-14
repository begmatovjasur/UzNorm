"""W&B metrics/config only. No raw texts, secrets, code or model artifact uploads."""

import os


def privacy_defaults():
    for key, value in {"WANDB_LOG_MODEL": "false", "WANDB_WATCH": "false", "WANDB_DISABLE_CODE": "true",
                       "WANDB_CONSOLE": "off", "TOKENIZERS_PARALLELISM": "false",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "USE_TF": "0", "USE_FLAX": "0"}.items():
        os.environ[key] = value


def require_tracking(config):
    if config.tracking.mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
        raise RuntimeError("Enable WANDB_API_KEY in Colab Secrets. Never paste the key into code or chat.")


def start_tracking(config, run_id, name, resume, metadata, output):
    privacy_defaults()
    require_tracking(config)
    if config.tracking.mode == "disabled":
        return None
    import wandb

    if config.tracking.mode == "online":
        # New-style W&B keys are not necessarily 40 characters long.
        wandb.login(key=os.environ["WANDB_API_KEY"].strip(), relogin=False)
    return wandb.init(project=config.tracking.project, entity=config.tracking.entity,
                      mode=config.tracking.mode, id=run_id, name=name,
                      resume="must" if resume and config.tracking.mode == "online" else None,
                      dir=str(output), config=metadata, save_code=False,
                      settings=wandb.Settings(disable_code=True, console="off", disable_job_creation=True))
