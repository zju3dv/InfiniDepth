from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from common.fs import mkdir

from .base import Writer
from .utils import normalize


class TensorBoardWriter(Writer):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        mkdir(self.root_dir)
        self.writer = SummaryWriter(self.root_dir)

    @staticmethod
    def from_config(config: DictConfig):
        return TensorBoardWriter(config.root_dir)

    def log_metrics(
        self,
        metrics: Dict[str, Union[torch.Tensor, float]],
        step: int,
    ) -> None:
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.writer.add_scalar(k, v, step)

    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        captions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        # Captions not supported here.
        for k, v in images.items():
            v = normalize(v.detach(), value_range)
            self.writer.add_images(k, v, step, dataformats="NCHW")

    def log_videos(
        self,
        videos: Dict[str, torch.Tensor],
        step: int,
        captions: Optional[Dict[str, List[str]]] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
        fps: int = 4,
    ) -> None:
        # Captions not supported here.
        for k, v in videos.items():
            v = normalize(v.detach(), value_range)
            self.writer.add_video(k, v, step, fps=fps)

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        """
        Record hyperparameters. TensorBoard logs with and without saved hyperparameters are
        incompatible, the hyperparameters are then not displayed in the TensorBoard.

        Args:
            params: a dictionary-like container with the hyperparameters

        """
        exp, ssi, sei = hparams(params, metric_dict={"hp_metric": -1})
        writer = self.writer._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
