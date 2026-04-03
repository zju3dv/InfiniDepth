from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from omegaconf import DictConfig

from .base import Writer
from .dummy import DummyWriter
from .tensorboard import TensorBoardWriter
from .wandb import WandbWriter


class CollectionWriter(Writer):
    def __init__(self, writers: List[Writer]):
        self.writers = writers

    @staticmethod
    def from_config(config: DictConfig):
        writers = []
        if "wandb" in config:
            writers.append(WandbWriter.from_config(config.wandb))
        if "tensorboard" in config:
            writers.append(TensorBoardWriter.from_config(config.tensorboard))
        if "dummy" in config:
            writers.append(DummyWriter.from_config(config.dummy))
        return CollectionWriter(writers)

    def log_metrics(
        self,
        metrics: Dict[str, Union[torch.Tensor, float]],
        step: int,
    ) -> None:
        for writer in self.writers:
            writer.log_metrics(metrics, step)

    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        captions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        for writer in self.writers:
            writer.log_images(images, step, value_range, captions)

    def log_videos(
        self,
        videos: Dict[str, torch.Tensor],
        step: int,
        captions: Optional[Dict[str, List[str]]] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
        fps: int = 4,
    ) -> None:
        for writer in self.writers:
            writer.log_videos(videos, step, captions, value_range, fps)

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        for writer in self.writers:
            writer.log_hyperparams(params)
