from typing import Optional

from common.entrypoint import Entrypoint

from .dataclass import PersistedTrainingState
from .manager import PersistenceManager


class PersistenceMixin:
    """
    Provide persistence capability.
    Config must contain a "persistence" key.
    """

    # ----------------- Example Config -----------------
    # persistence:
    #   path: hdfs://path/to/location  (required)
    # --------------------------------------------------

    persistence: PersistenceManager
    resume: Optional[PersistedTrainingState]

    def configure_persistence(self):
        assert isinstance(self, Entrypoint)
        self.persistence = PersistenceManager(path=self.config.persistence.path)
        self.persistence.save_config(self.config)
        self.resume = self.persistence.load_last_step()
