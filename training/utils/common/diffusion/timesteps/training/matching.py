from typing import Sequence
import torch

from ...schedules.base import Schedule
from ...utils import assert_schedule_timesteps_compatible
from ..base import TrainingTimesteps


class SNRMatchingTrainingTimesteps(TrainingTimesteps):
    """
    Match the SNR distribution of a source schedule and its training timesteps
    to the target schedule.
    """

    def __init__(
        self,
        source_timesteps: TrainingTimesteps,
        source_schedule: Schedule,
        target_schedule: Schedule,
    ):
        assert_schedule_timesteps_compatible(
            schedule=source_schedule,
            timesteps=source_timesteps,
        )
        super().__init__(T=target_schedule.T)
        self.source_timesteps = source_timesteps
        self.source_schedule = source_schedule
        self.target_schedule = target_schedule

    def sample(
        self,
        size: Sequence[int],
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        src_t = self.source_timesteps.sample(size, device)
        tgt_t = self.target_schedule.isnr(self.source_schedule.snr(src_t))
        return tgt_t
