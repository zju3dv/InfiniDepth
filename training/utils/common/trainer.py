"""
Trainer base class.
"""

from abc import abstractmethod

from common.entrypoint import Entrypoint


class Trainer(Entrypoint):
    """
    Trainer defines the complete training procedure.
    The abstract methods are defined to enforce codebase formality.
    """

    @abstractmethod
    def configure_dataloaders(self):
        pass

    @abstractmethod
    def configure_models(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def training_loop(self):
        pass
