import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import hydra
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger
from omegaconf import DictConfig, OmegaConf

from nn_core.common import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


_STATS_KEY: str = "stats"


class NNLogger(Logger):

    __doc__ = Logger.__doc__

    def __init__(self, logging_cfg: DictConfig, cfg: DictConfig, resume_id: Optional[str]):
        super().__init__()
        self.logging_cfg = logging_cfg
        self.cfg = cfg
        self.resume_id = resume_id

        self.storage_dir: str = cfg.core.storage_dir
        self.wandb: bool = self.logging_cfg.logger["_target_"].endswith("WandbLogger")

        if self.cfg.train.trainer.fast_dev_run and self.wandb:
            # Switch wandb mode to offline to prevent online logging
            pylogger.info("Setting the logger in 'offline' mode")
            self.logging_cfg.logger.mode = "offline"

        pylogger.info(f"Instantiating <{self.logging_cfg.logger['_target_'].split('.')[-1]}>")
        self.wrapped: Logger = hydra.utils.instantiate(
            self.logging_cfg.logger,
            version=self.resume_id,
            dir=os.getenv("WANDB_DIR", "."),
        )

        # force experiment lazy initialization
        _ = self.wrapped.experiment

    def __getattr__(self, item: str) -> Any:
        if self.wrapped is not None:
            pylogger.debug(f"Delegation with '__getattr__': {self.wrapped.__class__.__qualname__}.{item}")
            return getattr(self.wrapped, item)

    def watch_model(self, pl_module: LightningModule):
        if self.wandb and "wandb_watch" in self.logging_cfg:
            pylogger.info("Starting to 'watch' the module")
            self.wrapped.watch(pl_module, **self.logging_cfg["wandb_watch"])

    def upload_source(self) -> None:
        if self.logging_cfg.upload.source and self.wandb:
            pylogger.info("Uploading source code to W&B")
            self.wrapped.experiment.log_code(
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

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        # Attach to each checkpoint saved the configuration and the wandb run path (to resume logging from
        # only the checkpoint)
        pylogger.debug("Attaching 'cfg' to the checkpoint")
        checkpoint["cfg"] = OmegaConf.to_container(trainer.logger.cfg, resolve=True)

        pylogger.debug("Attaching 'run_path' to the checkpoint")
        checkpoint[
            "run_path"
        ] = f"{trainer.logger.experiment.entity}/{trainer.logger.experiment.project_name()}/{trainer.logger.version}"

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # Log the checkpoint meta information
        self.add_path(obj_id="checkpoints/best", obj_path=checkpoint_callback.best_model_path)
        self.add_path(
            obj_id="checkpoints/best_score",
            obj_path=str(checkpoint_callback.best_model_score.detach().cpu().item()),
        )

    def add_path(self, obj_id: str, obj_path: str) -> None:
        key = f"paths/{obj_id}"
        pylogger.debug(f"Logging '{key}'")
        self.experiment.config.update({key: str(obj_path)}, allow_val_change=True)

    @property
    def save_dir(self) -> Optional[str]:
        return self.storage_dir

    @property
    def experiment(self) -> Any:
        """Return the experiment object associated with this logger."""
        return self.wrapped.experiment

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Records metrics.

        This method logs metrics as as soon as it received them. If you want to aggregate
        metrics for one specific `step`, use the
        :meth:`~lightning.pytorch.loggers.base.Logger.agg_and_log_metrics` method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        return self.wrapped.log_metrics(metrics=metrics, step=step)

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keywoard arguments, depends on the specific logger being used
        """
        raise RuntimeError(
            "This method is called automatically by PyTorch Lightning if save_hyperparameters(logger=True) is called. "
            "The whole configuration is already logged by logger.log_configuration, set logger=False"
        )

    def log_text(self, *args, **kwargs) -> None:
        """Log text.

        Arguments are directly passed to the logger.
        """
        return self.wrapped.log_text(*args, **kwargs)

    def log_image(self, *args, **kwargs) -> None:
        """Log image.

        Arguments are directly passed to the logger.
        """
        return self.wrapped.log_image(*args, **kwargs)

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self.wrapped.name

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        return self.wrapped.version

    @property
    def run_dir(self) -> str:
        # TODO: verify remote URLs handling
        return os.path.join(*map(str, (self.storage_dir, self.name, self.version)))

    def log_configuration(
        self,
        model: LightningModule,
        cfg: Union[Dict[str, Any], argparse.Namespace, DictConfig] = None,
        *args,
        **kwargs,
    ):
        """This method controls which parameters from Hydra config are saved by Lightning loggers.

        Additionally saves:
            - sizes of train, val, test dataset
            - number of trainable model parameters

        Args:
            cfg (DictConfig): [description]
            model (pl.LightningModule): [description]
            trainer (pl.Trainer): [description]
        """
        if cfg is None:
            cfg = OmegaConf.create(self.cfg)

        if isinstance(cfg, DictConfig):
            cfg: Union[Dict[str, Any], argparse.Namespace, DictConfig] = OmegaConf.to_container(cfg, resolve=True)

        # Store the YaML config separately into the wandb dir
        yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
        run_dir: Path = Path(self.run_dir)
        run_dir.mkdir(exist_ok=True, parents=True)
        config_save_path = run_dir / "config.yaml"
        pylogger.debug(f"Saving the configuration in: {config_save_path}")
        config_save_path.write_text(yaml_conf)

        # save number of model parameters
        pylogger.debug("Injecting model statistics in the 'cfg'")
        cfg[f"{_STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
        cfg[f"{_STATS_KEY}/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg[f"{_STATS_KEY}/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        # send hparams to all loggers
        pylogger.debug("Logging 'cfg'")
        self.wrapped.log_hyperparams(cfg)

    def upload_run_files(self):
        if self.logging_cfg.upload.run_files:
            if self.wandb:
                pylogger.info("Uploading run files to W&B")
                shutil.copytree(self.run_dir, f"{self.wrapped.experiment.dir}/run_files")

                # FIXME: symlink not working for some reason
                # os.symlink(self.run_dir, f"{self.wrapped.experiment.dir}/run_files", target_is_directory=True)
