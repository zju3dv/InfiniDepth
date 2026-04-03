from .registry import MODEL_REGISTRY, register_model
from .model import InfiniDepth, InfiniDepth_DepthSensor

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "InfiniDepth",
    "InfiniDepth_DepthSensor",
]


def __getattr__(name):
    if name in {"InfiniDepth", "InfiniDepth_DepthSensor"}:
        from .model import InfiniDepth, InfiniDepth_DepthSensor

        return {
            "InfiniDepth": InfiniDepth,
            "InfiniDepth_DepthSensor": InfiniDepth_DepthSensor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
