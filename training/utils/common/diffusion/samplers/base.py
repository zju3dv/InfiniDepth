"""
Sampler base class.
"""

from abc import ABC, abstractmethod
import torch

from ..schedules.base import Schedule
from ..timesteps.base import SamplingTimesteps
from ..types import PredictionType, SamplingDirection
from ..utils import assert_schedule_timesteps_compatible


class Sampler(ABC):
    """
    Samplers are ODE/SDE solvers.
    """

    def __init__(
        self,
        schedule: Schedule,
        timesteps: SamplingTimesteps,
        prediction_type: PredictionType,
    ):
        assert_schedule_timesteps_compatible(
            schedule=schedule,
            timesteps=timesteps,
        )
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type

    @abstractmethod
    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step to the next timestep.
        """

    def get_next_timestep(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the next sample timestep.
        Support multiple different timesteps t in a batch.
        If no more steps, return out of bound value -1 or T+1.
        """
        T = self.timesteps.T
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1
        bound = -1 if self.timesteps.direction == SamplingDirection.backward else T + 1

        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        s = s.where(next_idx < steps, bound)
        return s
