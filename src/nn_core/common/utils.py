import logging
import os
from contextlib import contextmanager
from typing import List, Optional

import dotenv
import numpy as np
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from rich.prompt import Prompt

pylogger = logging.getLogger(__name__)


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """Safely read an environment variable.

    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            message = f"{env_name} not defined and no default value is present!"
            pylogger.error(message)
            raise KeyError(message)
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            message = f"{env_name} has yet to be configured and no default value is present!"
            pylogger.error(message)
            raise ValueError(message)
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """Load all the environment variables defined in the `env_file`.

    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    if env_file is None:
        env_file = dotenv.find_dotenv(usecwd=True)
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


@contextmanager
def environ(**kwargs):
    """Temporarily set the process environment variables.

    https://stackoverflow.com/a/34333710

    >>> with environ(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type kwargs: dict[str, unicode]
    :param kwargs: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(kwargs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def enforce_tags(tags: Optional[List[str]]) -> List[str]:
    if tags is None:
        if "id" in HydraConfig().cfg.hydra.job:
            # We are in multi-run setting (either via a sweep or a scheduler)
            message: str = "You need to specify 'core.tags' in a multi-run setting!"
            pylogger.error(message)
            raise ValueError(message)

        pylogger.warning("No tags provided, asking for tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="develop")
        tags = [x.strip() for x in tags.split(",")]

    pylogger.info(f"Tags: {tags if tags is not None else []}")
    return tags


def seed_index_everything(train_cfg: DictConfig, sampling_seed: int = 42) -> Optional[int]:
    if "seed_index" in train_cfg and train_cfg.seed_index is not None:
        seed_index = train_cfg.seed_index
        np.random.seed(sampling_seed)
        seeds = np.random.randint(np.iinfo(np.int32).max, size=max(42, seed_index + 1))
        seed = seeds[seed_index]
        seed_everything(seed)
        pylogger.info(f"Setting seed {seed} from seeds[{seed_index}]")
        return seed
    else:
        pylogger.warning("The seed has not been set! The reproducibility is not guaranteed.")
        return None
