import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
from lightning.pytorch import Callback, Trainer
from omegaconf import DictConfig

from nn_core.model_logging import NNLogger
from nn_core.resume import parse_restore
from nn_core.serialization import METADATA_KEY, NNCheckpointIO

pylogger = logging.getLogger(__name__)


class NNTemplateCore(Callback):
    def __init__(self, restore_cfg: Optional[DictConfig]):
        self.resume_ckpt_path, self.resume_run_version = parse_restore(restore_cfg)
        self.restore_mode: Optional[str] = restore_cfg.get("mode", None) if restore_cfg is not None else None
        self.restore_strict: bool = restore_cfg.get("strict", True) if restore_cfg is not None else True

    @property
    def resume_id(self) -> Optional[str]:
        return self.resume_run_version

    @property
    def trainer_ckpt_path(self) -> Optional[str]:
        return self.resume_ckpt_path if self.restore_mode != "finetune" else None

    @staticmethod
    def _is_nnlogger(trainer: Trainer) -> bool:
        return isinstance(trainer.logger, NNLogger)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_nnlogger(trainer):
            trainer.logger: NNLogger
            trainer.logger.upload_source()
            trainer.logger.log_configuration(model=pl_module)
            trainer.logger.watch_model(pl_module=pl_module)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self.restore_mode == "finetune":
            checkpoint = NNCheckpointIO.load(path=Path(self.resume_ckpt_path))

            pl_module.load_state_dict(checkpoint["state_dict"], strict=self.restore_strict)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_nnlogger(trainer):
            trainer.logger: NNLogger
            trainer.logger.upload_run_files()

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        if self._is_nnlogger(trainer):
            trainer.logger.on_save_checkpoint(trainer=trainer, pl_module=pl_module, checkpoint=checkpoint)
        metadata = getattr(trainer.datamodule, "metadata", None)
        if metadata is not None:
            checkpoint[METADATA_KEY] = metadata
