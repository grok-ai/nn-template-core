import logging
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer

from nn_core.model_logging import NNLogger

pylogger = logging.getLogger(__name__)


class NNTemplateCore(Callback):
    @staticmethod
    def _is_nnlogger(trainer: Trainer) -> bool:
        return isinstance(trainer.logger, NNLogger)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_nnlogger(trainer):
            trainer.logger: NNLogger
            trainer.logger.upload_source()
            trainer.logger.log_configuration(model=pl_module)
            trainer.logger.watch_model(pl_module=pl_module)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._is_nnlogger(trainer):
            trainer.logger: NNLogger
            trainer.logger.upload_run_files()

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        if self._is_nnlogger(trainer):
            trainer.logger.on_save_checkpoint(trainer=trainer, pl_module=pl_module, checkpoint=checkpoint)
