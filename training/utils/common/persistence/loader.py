"""
Model loader.
"""

from typing import Optional

from .dataclass import PersistedModel
from .manager import PersistenceManager


def load_model_from_path(
    path: str,
    name: str = "model",
    step: Optional[int] = None,
) -> PersistedModel:
    """
    Load a persisted model from a given path and name.
    """
    manager = PersistenceManager(path)
    archive = manager.load_step(step)
    if archive is None:
        raise ValueError(f"Model not found under path: {path} and step: {step}")
    if name not in archive.models.keys():
        raise ValueError(f"Model name: {name} not found in archive.")
    return archive.models[name]
