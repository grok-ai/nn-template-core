from typing import Callable, Dict, NoReturn

from omegaconf import DictConfig, OmegaConf


class OnSaveCheckpointInjection:
    def __init__(
        self,
        cfg: DictConfig,
        on_save_checkpoint: Callable[[Dict], NoReturn],
    ):
        """Inject the configuration into the checkpoint monkey patching the on_save_checkpoint hook.

        Args:
            cfg: the configuration to inject
            on_save_checkpoint: the on_save_checkpoint to monkey patch
        """
        self.cfg = cfg
        self.on_save_checkpoint = on_save_checkpoint

    def __call__(self, checkpoint: Dict) -> None:
        self.on_save_checkpoint(checkpoint)
        checkpoint["cfg"] = OmegaConf.to_container(self.cfg, resolve=True)
