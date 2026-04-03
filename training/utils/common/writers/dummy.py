from typing import Any, Dict, List, Optional, Tuple
import torch
from omegaconf import DictConfig

from .base import Writer


class DummyWriter(Writer):
    def log_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        step: int,
    ) -> None:
        pass

    @staticmethod
    def from_config(config: DictConfig):
        return DummyWriter()

    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        captions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass

    def log_videos(
        self,
        videos: Dict[str, torch.Tensor],
        step: int,
        captions: Optional[Dict[str, List[str]]] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
        fps: int = 4,
    ) -> None:
        pass

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        pass
