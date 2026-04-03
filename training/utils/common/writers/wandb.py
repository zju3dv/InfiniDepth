import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import wandb
from omegaconf import DictConfig

from common.fs import is_hdfs_path, mkdir

from .base import Writer
from .utils import normalize


class WandbWriter(Writer):
    """
    Wandb logger for experiment loggers.
    Hdfs supported partially.
    """

    def __init__(self, project: str, name: str, root_dir: str = ".", **kwargs):
        assert not is_hdfs_path(root_dir)
        path = os.path.join(root_dir, project, name)
        mkdir(path)
        self.writer = wandb.init(project=project, name=name, dir=path, **kwargs)

    @staticmethod
    def from_config(config: DictConfig):
        return WandbWriter(**config)

    def log_metrics(
        self,
        metrics: Dict[str, Union[torch.Tensor, float]],
        step: int,
    ) -> None:
        self.writer.log(metrics, step=step)

    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: int,
        value_range: Tuple[float, float] = (0.0, 1.0),
        captions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        metrics = {}
        for k, v in images.items():
            v = normalize(v.detach(), value_range).unbind(dim=0)  # C H W
            v_cap = captions[k] if captions is not None and k in captions else [None] * len(v)
            if len(v_cap) != len(v):
                raise ValueError(f"Expected {len(v)} items but only found {len(v_cap)} for {k}")
            metrics[k] = [wandb.Image(img, caption=cap) for img, cap in zip(v, v_cap)]
        self.writer.log(metrics, step)

    def log_videos(
        self,
        videos: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        step: int,
        captions: Optional[Dict[str, List[str]]] = None,
        value_range: Tuple[float, float] = (0.0, 1.0),
        fps: int = 4,
    ) -> None:
        metrics = {}
        for k, v in videos.items():
            if isinstance(v, list):
                v = [
                    normalize(vv.detach(), value_range).mul_(255).round_().to(torch.uint8)
                    for vv in v
                ]
            elif isinstance(v, torch.Tensor):
                v = (
                    normalize(v.detach(), value_range)
                    .mul_(255)
                    .round_()
                    .to(torch.uint8)
                    .unbind(dim=0)
                )
            v_cap = captions[k] if captions is not None and k in captions else [None] * len(v)
            if len(v_cap) != len(v):
                raise ValueError(f"Expected {len(v)} items but only found {len(v_cap)} for {k}")
            metrics[k] = [
                wandb.Video(vid.cpu().numpy(), caption=cap, fps=fps) for vid, cap in zip(v, v_cap)
            ]
        self.writer.log(metrics, step)

    def log_hyperparams(
        self,
        params: Dict[str, Any],
    ) -> None:
        self.writer.config.update(params, allow_val_change=True)
