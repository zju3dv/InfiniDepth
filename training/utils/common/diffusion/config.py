"""
Utility functions for creating schedules and samplers from config.
"""

import torch
from omegaconf import DictConfig

from .samplers.base import Sampler
from .samplers.euler import EulerSampler
from .schedules.base import Schedule
from .schedules.cos import CosineSchedule
from .schedules.lerp import LinearInterpolationSchedule
from .schedules.vp import DiscreteVariancePreservingSchedule
from .timesteps.base import SamplingTimesteps, TrainingTimesteps
from .timesteps.sampling.matching import SNRMatchingSamplingTimesteps
from .timesteps.sampling.trailing import UniformTrailingSamplingTimesteps
from .timesteps.training.logitnormal import LogitNormalTrainingTimesteps
from .timesteps.training.matching import SNRMatchingTrainingTimesteps
from .timesteps.training.uniform import UniformTrainingTimesteps


def create_schedule_from_config(
    config: DictConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Schedule:
    """
    Create a schedule from configuration.
    """
    if config.type == "vp":
        schedule = DiscreteVariancePreservingSchedule.from_preset(
            name=config.name,
            steps=config.get("steps", 1000),
            device=device,
            dtype=dtype,
        )
        if config.get("zsnr", False):
            schedule = schedule.to_zsnr()
        if config.get("shift_snr", None):
            schedule = schedule.shift_snr(config.shift_snr)
        return schedule

    if config.type == "cos":
        return CosineSchedule(T=config.get("T", 1.0))

    if config.type == "lerp":
        return LinearInterpolationSchedule(T=config.get("T", 1.0))

    raise NotImplementedError


def create_sampler_from_config(
    config: DictConfig,
    schedule: Schedule,
    timesteps: SamplingTimesteps,
) -> Sampler:
    """
    Create a sampler from configuration.
    """
    if config.type == "euler":
        return EulerSampler(
            schedule=schedule,
            timesteps=timesteps,
            prediction_type=config.prediction_type,
        )

    raise NotImplementedError


def create_training_timesteps_from_config(
    config: DictConfig,
    schedule: Schedule,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> TrainingTimesteps:
    if config.type == "uniform":
        return UniformTrainingTimesteps(T=schedule.T)
    if config.type == "logitnormal":
        return LogitNormalTrainingTimesteps(
            T=schedule.T,
            loc=config.loc,
            scale=config.scale,
        )
    if config.type == "snr_matching":
        source_schedule = create_schedule_from_config(
            config=config.source_schedule,
            device=device,
            dtype=dtype,
        )
        source_timesteps = create_training_timesteps_from_config(
            config=config.source_timesteps,
            schedule=source_schedule,
            device=device,
            dtype=dtype,
        )
        return SNRMatchingTrainingTimesteps(
            source_timesteps=source_timesteps,
            source_schedule=source_schedule,
            target_schedule=schedule,
        )

    raise NotImplementedError


def create_sampling_timesteps_from_config(
    config: DictConfig,
    schedule: Schedule,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SamplingTimesteps:
    if config.type == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=config.steps,
            device=device,
        )
    if config.type == "snr_matching":
        source_schedule = create_schedule_from_config(
            config=config.source_schedule,
            device=device,
            dtype=dtype,
        )
        source_timesteps = create_sampling_timesteps_from_config(
            config=config.source_timesteps,
            schedule=source_schedule,
            device=device,
            dtype=dtype,
        )
        return SNRMatchingSamplingTimesteps(
            source_timesteps=source_timesteps,
            source_schedule=source_schedule,
            target_schedule=schedule,
        )
