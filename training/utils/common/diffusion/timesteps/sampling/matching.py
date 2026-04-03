from ...schedules.base import Schedule
from ...utils import assert_schedule_timesteps_compatible
from ..base import SamplingTimesteps


class SNRMatchingSamplingTimesteps(SamplingTimesteps):
    """
    Match the SNR sampling step of of the source schedule to the target schedule.
    """

    def __init__(
        self,
        source_timesteps: SamplingTimesteps,
        source_schedule: Schedule,
        target_schedule: Schedule,
    ):
        assert_schedule_timesteps_compatible(
            schedule=source_schedule,
            timesteps=source_timesteps,
        )
        super().__init__(
            T=target_schedule.T,
            timesteps=target_schedule.isnr(source_schedule.snr(source_timesteps.timesteps)),
            direction=source_timesteps.direction,
        )
