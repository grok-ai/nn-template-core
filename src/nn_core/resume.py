import logging
import re
import tempfile
from operator import xor
from pathlib import Path
from typing import Optional, Tuple

import wandb
from omegaconf import DictConfig
from wandb.apis.public import Run

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import environ
from nn_core.serialization import NNCheckpointIO

pylogger = logging.getLogger(__name__)

RUN_PATH_PATTERN = re.compile(r"^([^/]+)/([^/]+)/([^/]+)$")

RESUME_MODES = {
    None: {
        "logging": False,
        "trainer": False,
        "weights": False,
    },
    "finetune": {
        "logging": False,
        "trainer": False,
        "weights": True,
    },
    "hotstart": {
        "logging": False,
        "trainer": True,
        "weights": True,
    },
    "continue": {
        "logging": True,
        "trainer": True,
        "weights": True,
    },
}


def resolve_ckpt(ckpt_or_run_path: str) -> str:
    """Resolve the run path or ckpt to a checkpoint.

    Args:
        ckpt_or_run_path: run identifier or checkpoint path

    Returns:
        an existing path towards the best checkpoint
    """
    if RUN_PATH_PATTERN.match(ckpt_or_run_path):
        # If WANDB_DIR is set (as it is the case with our hydra configuration), the run dir is created by wandb in the
        # project's root folder instead of in a temp directory.
        with tempfile.TemporaryDirectory() as tmp_dir, environ(WANDB_DIR=tmp_dir):
            # We are resolving the path from a wandb run id
            try:
                api = wandb.Api()
                run: Run = api.run(path=ckpt_or_run_path)
                ckpt_or_run_path = run.config["paths/checkpoints/best"]
                return ckpt_or_run_path
            except wandb.errors.CommError:
                raise ValueError(f"Checkpoint or run not found: {ckpt_or_run_path}")

    _ckpt_or_run_path: Path = Path(ckpt_or_run_path)
    # If the path is relative, it is wrt the PROJECT_ROOT, so it is prepended.
    if not _ckpt_or_run_path.is_absolute():
        _ckpt_or_run_path = PROJECT_ROOT / _ckpt_or_run_path

    if _ckpt_or_run_path.exists():
        return ckpt_or_run_path


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
        return NNCheckpointIO.load(path=Path(ckpt_or_run_path))["run_path"]
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


# TODO: Refactor returning type to include restore mode too.
def parse_restore(restore_cfg: DictConfig) -> Tuple[Optional[str], Optional[str]]:
    if restore_cfg is None:
        return None, None

    ckpt_or_run_path = restore_cfg.ckpt_or_run_path
    resume_mode = restore_cfg.mode

    resume_ckpt_path = None
    resume_run_version = None

    if xor(bool(ckpt_or_run_path), bool(resume_mode)):
        pylogger.warning(f"Inconsistent resume modality {resume_mode} and checkpoint path '{ckpt_or_run_path}'")
    else:
        pylogger.info(f"Restoring with mode: <{resume_mode}>")

    if resume_mode not in RESUME_MODES:
        message = f"Unsupported resume mode {resume_mode}. Available resume modes are: {RESUME_MODES}"
        pylogger.error(message)
        raise ValueError(message)

    flags = RESUME_MODES[resume_mode]

    if ckpt_or_run_path is not None:
        resume_ckpt_path = resolve_ckpt(ckpt_or_run_path)
        pylogger.info(f"Resolved checkpoint path: '{resume_ckpt_path}'")

        if flags["logging"]:
            run_path = resolve_run_path(ckpt_or_run_path)
            resume_run_version = resolve_run_version(run_path=run_path)
            pylogger.info(f"Resume logging to: '{run_path}'")

    return resume_ckpt_path, resume_run_version
