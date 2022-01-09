import re
from pathlib import Path
from typing import Optional

import torch
import wandb
from wandb.apis.public import Run

RUN_PATH_PATTERN = re.compile(r"^([^/]+)/([^/]+)/([^/]+)$")


def resolve_ckpt(ckpt_or_run_path: str) -> str:
    """Resolve the run path or ckpt to a checkpoint.

    Args:
        ckpt_or_run_path: run identifier or checkpoint path

    Returns:
        an existing path towards the best checkpoint
    """
    if Path(ckpt_or_run_path).exists():
        return ckpt_or_run_path

    try:
        api = wandb.Api()
        run: Run = api.run(path=ckpt_or_run_path)
        ckpt_or_run_path = run.config["paths/checkpoints/best"]
        return ckpt_or_run_path
    except wandb.errors.CommError:
        raise ValueError(f"Checkpoint or run not found: {ckpt_or_run_path}")


def resolve_run_path(ckpt_or_run_path: str) -> str:
    """Resolve the run path or ckpt to a run path.

    Args:
        ckpt_or_run_path: run identifier or checkpoint path

    Returns:
        an wandb run path identifier
    """
    if RUN_PATH_PATTERN.match(ckpt_or_run_path):
        return ckpt_or_run_path

    try:
        return torch.load(ckpt_or_run_path)["run_path"]
    except FileNotFoundError:
        raise ValueError(f"Checkpoint or run not found: {ckpt_or_run_path}")


def resolve_run_version(ckpt_or_run_path: Optional[str] = None, run_path: Optional[str] = None) -> str:
    """Resolve the run path or ckpt to the wandb run version.

    Args:
        ckpt_or_run_path: run identifier or checkpoint path
        run_path: the run path if already available

    Returns:
        a wandb run version
    """
    if run_path is None:
        run_path = resolve_run_path(ckpt_or_run_path)
    return RUN_PATH_PATTERN.match(run_path).group(3)