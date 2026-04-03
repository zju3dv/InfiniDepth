"""
Evaluator base class.
"""

from abc import abstractmethod

from common.entrypoint import Entrypoint


class Evaluator(Entrypoint):
    """
    Evaluator defines the complete evaluation procedure.
    The abstract methods are defined to enforce codebase formality.
    """

    @abstractmethod
    def configure_dataloaders(self):
        pass

    @abstractmethod
    def configure_models(self):
        pass

    @abstractmethod
    def configure_metrics(self):
        pass

    @abstractmethod
    def evaluation_loop(self):
        pass
