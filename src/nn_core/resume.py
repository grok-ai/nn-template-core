from pathlib import Path

import wandb
from wandb.apis.public import Run


def resolve_ckpt(ckpt_or_run_path: str) -> str:
    """Resolve the run or ckpt to a checkpoint

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
