"""Abstract base class used to build new loggers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


class Writer(ABC):
    """Base class for experiment loggers"""

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Union[torch.Tensor, float]],
        step: int,
    ) -> None:
        """
        Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """

    @abstractmethod
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        captions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Log images.
        Args:
            images: [N C H W]
            step: global step
            value_range: the value range of image tensor
            captions: captions of images
        """

    @abstractmethod
    def log_videos(
        self,
        videos: Dict[str, torch.Tensor],
        step: int,
        captions: Optional[Dict[str, List[str]]] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
        fps: int = 4,
    ) -> None:
        """
        Log videos.

        Args:
            videos: math:`(N, T, C, H, W)`.
            step: global step
            value_range: the value range of video tensor
            captions: captions of videos
            fps: save fps
        """

    @abstractmethod
    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """
        Record hyperparameters.

        Args:
            params: :class: `Dict` containing the hyperparameters
        """
