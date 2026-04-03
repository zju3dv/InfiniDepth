from ...types import SamplingDirection
from ..base import SamplingTimesteps


class InverseSamplingTimesteps(SamplingTimesteps):
    """
    Inverse the sampling timesteps.
    This can be used for noise inversion.
    """

    def __init__(self, timesteps: SamplingTimesteps):
        super().__init__(
            T=timesteps.T,
            timesteps=timesteps.timesteps.flip(0),
            direction=SamplingDirection.reverse(timesteps.direction),
        )
