import dataclasses
import logging
from typing import Any, Dict, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from nn_core.common import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


@dataclasses.dataclass
class Upload:
    checkpoint: bool = True
    source: bool = True


class NNLoggerConfiguration(Callback):
    def __init__(self, upload: Optional[Dict[str, bool]], logger: Optional[DictConfig], **kwargs):
        self.upload: Upload = Upload(**upload)
        self.logger_cfg = logger
        self.kwargs = kwargs

        self.wandb: bool = self.logger_cfg["_target_"].endswith("WandbLogger")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.logger.log_configuration(model=pl_module)

        if "wandb_watch" in self.kwargs:
            trainer.logger.wrapped.watch(pl_module, **self.kwargs["wandb_watch"])

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> dict:
        data = [
            ("best_model_path", trainer.checkpoint_callback.best_model_path),
            ("best_model_score", str(trainer.checkpoint_callback.best_model_score.detach().cpu().item())),
        ]
        trainer.logger.log_text(key="storage_info", columns=["key", "value"], data=data)

        return checkpoint

    # on_init_end can be employed since the Trainer doesn't use the logger until then.
    def on_init_end(self, trainer: "pl.Trainer") -> None:
        if self.logger_cfg is None:
            return

        pylogger.info(f"Instantiating <{self.logger_cfg['_target_'].split('.')[-1]}>")

        if trainer.fast_dev_run and self.wandb:
            # Switch wandb mode to offline to prevent online logging
            self.logger_cfg.mode = "offline"

        logger: LightningLoggerBase = hydra.utils.instantiate(self.logger_cfg)

        if self.upload.source:
            if self.wandb:
                logger.experiment.log_code(
                    root=PROJECT_ROOT,
                    name=None,
                    include_fn=(
                        lambda path: path.startswith(
                            (
                                str(PROJECT_ROOT / "conf"),
                                str(PROJECT_ROOT / "src"),
                                str(PROJECT_ROOT / "setup.cfg"),
                                str(PROJECT_ROOT / "env.yaml"),
                            )
                        )
                        and path.endswith((".py", ".yaml", ".yml", ".toml", ".cfg"))
                    ),
                )

        trainer.logger.wrapped = logger
