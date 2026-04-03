"""
Entrypoint base class.
"""

from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf

from common.logger import get_logger


class Entrypoint(ABC):
    """
    Entrypoint method is invoked to start the execution of the program.
    """

    def __init__(self, config: DictConfig):
        self.config = config.copy()
        OmegaConf.set_readonly(self.config, True)
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def entrypoint(self):
        pass
