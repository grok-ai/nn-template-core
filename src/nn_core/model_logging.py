import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase

_STATS_KEY: str = "stats"


class NNLogger(LightningLoggerBase):

    __doc__ = LightningLoggerBase.__doc__

    def __init__(self, logger: Optional[LightningLoggerBase], storage_dir: str, cfg):
        super().__init__()
        self.wrapped: LightningLoggerBase = logger
        self.storage_dir: str = storage_dir
        self.cfg = cfg

    def __getattr__(self, item):
        if self.wrapped is not None:
            return getattr(self.wrapped, item)

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
        :meth:`~pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics` method.

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
            "This method is called automatically by PyTorch Lightning if save_hyperparameters(logger=True) is called. The whole configuration is already logged by logger.log_configuration, set logger=False"
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
        model: pytorch_lightning.LightningModule,
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
        (run_dir / "config.yaml").write_text(yaml_conf)

        # save number of model parameters
        cfg[f"{_STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
        cfg[f"{_STATS_KEY}/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg[f"{_STATS_KEY}/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        # send hparams to all loggers
        self.wrapped.log_hyperparams(cfg)
