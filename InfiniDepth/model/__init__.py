from .registry import MODEL_REGISTRY, register_model
from .model import InfiniDepth, InfiniDepth_DC

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "InfiniDepth",
    "InfiniDepth_DC",
]
