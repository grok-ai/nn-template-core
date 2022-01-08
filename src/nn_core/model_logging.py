import argparse
from typing import Any, Dict, Optional, Union

import pytorch_lightning
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase

STATS_KEY: str = "stats"


class NNLogger(LightningLoggerBase):
    __doc__ = LightningLoggerBase.__doc__

    def __init__(self, logger: LightningLoggerBase):
        super().__init__()
        self.logger: LightningLoggerBase = logger

    def __getattr__(self, item):
        return getattr(self.logger, item)

    @property
    def experiment(self) -> Any:
        """Return the experiment object associated with this logger."""
        return self.logger.experiment

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Records metrics.

        This method logs metrics as as soon as it received them. If you want to aggregate
        metrics for one specific `step`, use the
        :meth:`~pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics` method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        return self.logger.log_metrics(metrics=metrics, step=step)

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
        return self.logger.log_text(*args, **kwargs)

    def log_image(self, *args, **kwargs) -> None:
        """Log image.

        Arguments are directly passed to the logger.
        """
        return self.logger.log_image(*args, **kwargs)

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self.logger.name

    @property
    def version(self) -> Union[int, str]:
        """Return the experiment version."""
        return self.logger.version

    def log_configuration(
        self,
        cfg: Union[Dict[str, Any], argparse.Namespace, DictConfig],
        model: pytorch_lightning.LightningModule,
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
        if isinstance(cfg, DictConfig):
            cfg: Union[Dict[str, Any], argparse.Namespace, DictConfig] = OmegaConf.to_container(cfg, resolve=True)

        # save number of model parameters
        cfg[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
        cfg[f"{STATS_KEY}/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg[f"{STATS_KEY}/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        # send hparams to all loggers
        self.logger.log_hyperparams(cfg)
