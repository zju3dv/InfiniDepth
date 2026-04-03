"""
Metrics (FID, FVD, etc,.)
"""

import os.path
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from einops import rearrange
from torch import nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchmetrics.multimodal.clip_score import CLIPScore as CLIPScore_

from common.distributed import get_global_rank
from common.fs import download, download_and_extract, exists, is_hdfs_path, mkdir, move


def load_target_info(file_path: str):
    if file_path is None or not exists(file_path):
        return None
    local_path = download(file_path) if is_hdfs_path(file_path) else file_path
    return torch.load(local_path)


def save_target_info(info: Any, file_path: str):
    if file_path is None:
        return
    dir_path = os.path.dirname(file_path)
    mkdir(dir_path)
    if is_hdfs_path(file_path):
        file_name = os.path.basename(file_path)
        torch.save(info, file_name)
        move(file_name, dir_path)
    else:
        torch.save(info, file_path)


def filter(inputs: torch.Tensor, mask: torch.BoolTensor) -> Optional[torch.Tensor]:
    if inputs is None:
        return None
    if mask is None:
        return inputs
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return inputs[mask]


class FID(FrechetInceptionDistance):
    def __init__(
        self,
        *,
        extractor_path: str,
        resolution: int = 299,
        target_info_path: Optional[str] = None,
        interpolation: str = "bicubic",
        **kwargs,
    ):
        assert exists(extractor_path), "The extractor for FID is not found."
        download(
            extractor_path,
            dirname="/home/tiger/.cache/torch/hub/checkpoints/",
            filename="weights-inception-2015-12-05-6726825d.pth",
        )
        super().__init__(**kwargs)
        self.target_info_path = target_info_path
        self.target_info = load_target_info(target_info_path)
        self.resolution = resolution
        self.interpolation = interpolation

    def process(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            # hack: video-like image, [B, C, T, H, W] where T=1
            image = image[:, :, 0]
        return F.interpolate(
            image, (self.resolution, self.resolution), mode=self.interpolation, antialias=True
        )

    def update(self, preds: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None):
        if preds is not None:
            super().update(self.process(preds), real=False)
        if target is not None:
            super().update(self.process(target), real=True)

    def compute(self) -> torch.Tensor:
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        if self.target_info is not None:
            mean_real = self.target_info["mu"].to(mean_fake)
            cov_real = self.target_info["sigma"].to(cov_fake)
        else:
            mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
            cov_real_num = (
                self.real_features_cov_sum
                - self.real_features_num_samples * mean_real.t().mm(mean_real)
            )
            cov_real = cov_real_num / (self.real_features_num_samples - 1)
            if self.target_info_path is not None and get_global_rank() == 0:
                save_target_info(
                    {"mu": mean_real.cpu(), "sigma": cov_real.cpu()}, self.target_info_path
                )
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(
            self.orig_dtype
        )


class FVD(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    def __init__(
        self,
        *,
        extractor_path: str,
        resolution: int = 224,
        target_info_path: Optional[str] = None,
        feature_dim: int = 400,
        interpolation: str = "bicubic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert exists(extractor_path), "The extractor for FVD is not found."
        self.extractor = self.configurate_extractor(extractor_path)
        self.target_info_path = target_info_path
        self.target_info = load_target_info(target_info_path)
        self.resolution = resolution
        self.interpolation = getattr(TVF.InterpolationMode, interpolation.upper())

        # Initialize state variables used to compute FID
        num_features = feature_dim
        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum"
        )
        self.add_state(
            "real_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum"
        )
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state(
            "fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum"
        )
        self.add_state(
            "fake_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum"
        )
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def process(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B, C, T, H, W] in range(0, 1)
        """
        video = video[:, :, : video.size(2) - video.size(2) % 16]
        num_frames = video.size(2)
        # NOTE: I know it's not elegant, but we need to add this logic to
        # avoid someone forgetting to slice the frame sequence.
        video = rearrange(video, "b c t h w -> (b t) c h w")
        video = TVF.resize(
            video,
            size=[self.resolution],
            interpolation=self.interpolation,
            antialias=True,
        )
        video = TVF.center_crop(video, output_size=[self.resolution, self.resolution])
        video = rearrange(video, "(b t) c h w -> b c t h w", t=num_frames)
        video = (video - 0.5) * 2.0  # -> [-1, 1]
        features = self.extractor(
            video.to(self.device), rescale=False, resize=False, return_features=True
        )
        return features

    def update(self, preds: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None):
        if preds is not None:
            pred_feat = self.process(preds)
            self.orig_dtype = pred_feat.dtype
            pred_feat = pred_feat.double()
            if pred_feat.dim() == 1:
                pred_feat = pred_feat.unsqueeze(0)
            self.fake_features_sum += pred_feat.sum(dim=0)
            self.fake_features_cov_sum += pred_feat.t().mm(pred_feat)
            self.fake_features_num_samples += preds.shape[0]

        if target is not None:
            real_feat = self.process(target)
            self.orig_dtype = real_feat.dtype
            real_feat = real_feat.double()
            if real_feat.dim() == 1:
                real_feat = real_feat.unsqueeze(0)
            self.real_features_sum += real_feat.sum(dim=0)
            self.real_features_cov_sum += real_feat.t().mm(real_feat)
            self.real_features_num_samples += target.shape[0]

    def compute(self) -> torch.Tensor:
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        if self.target_info is not None:
            mean_real = self.target_info["mu"].to(mean_fake)
            cov_real = self.target_info["sigma"].to(cov_fake)
        else:
            mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
            cov_real_num = (
                self.real_features_cov_sum
                - self.real_features_num_samples * mean_real.t().mm(mean_real)
            )
            cov_real = cov_real_num / (self.real_features_num_samples - 1)
            if self.target_info_path is not None and get_global_rank() == 0:
                save_target_info(
                    {"mu": mean_real.cpu(), "sigma": cov_real.cpu()}, self.target_info_path
                )
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(
            self.orig_dtype
        )

    def configurate_extractor(self, path: str) -> nn.Module:
        return torch.jit.load(download(path)).eval().to(self.device)


class CLIPScore(CLIPScore_):
    def __init__(
        self,
        *,
        extractor_path: str,
        resolution: int = 224,
        interpolation: str = "bicubic",
        **kwargs,
    ):
        assert exists(extractor_path), "The extractor for CLIP is not found."
        model_path = download_and_extract(extractor_path)
        super().__init__(model_name_or_path=model_path, **kwargs)
        self.resolution = resolution
        self.interpolation = interpolation

    def process(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            # hack: video-like image, [B, C, T, H, W] where T=1
            image = image[:, :, 0]
        return F.interpolate(
            image, (self.resolution, self.resolution), mode=self.interpolation, antialias=True
        )

    def update(self, preds: torch.Tensor, text: List[str]) -> None:
        """Update CLIP score on a batch of images and text.

        Args:
            images: a single [N, C, H, W] tensor
            text: a list of captions

        Raises:
            ValueError:
                If not all images have format [C, H, W]
            ValueError:
                If the number of images and captions do not match

        """
        images = (self.process(preds) * 255).type(torch.uint8)
        return super().update(images, text)


class ModifiedMetricCollection(MetricCollection):
    def update(self, *args: Any, mask: Optional[torch.BoolTensor] = None, **kwargs: Any) -> None:
        return super().update(
            *(filter(arg, mask) for arg in args),
            **{key: filter(kwarg, mask) for key, kwarg in kwargs.items()},
        )

    def compute_and_reset(self):
        result = self.compute()
        self.reset()
        return result
