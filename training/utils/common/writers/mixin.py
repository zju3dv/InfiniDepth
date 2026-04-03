from common.distributed import get_global_rank
from common.entrypoint import Entrypoint

from .base import Writer
from .collection import CollectionWriter
from .dummy import DummyWriter


class WriterMixin:
    """
    Provide writer capabilities.
    Config must contain a "writer" key.
    Currently only rank 0 is logged. Other ranks are dummy writers.
    """

    # ----------------- Example Config -----------------
    # writer:
    #   wandb:
    #     project: my-project
    #     name: my-name
    # --------------------------------------------------

    writer: Writer

    def configure_writer(self):
        assert isinstance(self, Entrypoint)
        self.writer = (
            CollectionWriter.from_config(self.config.writer)
            if get_global_rank() == 0
            else DummyWriter()
        )
