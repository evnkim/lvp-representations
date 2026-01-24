"""
Small wandb helpers for the linear probe script.
"""

from dataclasses import dataclass
import os


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    mode: str = "online"  # "online" or "offline"


def init_wandb(cfg: WandbConfig, config_dict: dict):
    if not cfg.enabled:
        return None
    import wandb

    os.environ.setdefault("WANDB_MODE", cfg.mode)
    return wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.name,
        config=config_dict,
    )


def log_wandb(run, metrics: dict, step: int | None = None):
    if run is None:
        return
    run.log(metrics, step=step)
