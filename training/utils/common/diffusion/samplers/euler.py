"""
Euler ODE solver.
"""

import torch

from ..utils import expand_dims
from .base import Sampler


class EulerSampler(Sampler):
    """
    The Euler method is the simplest ODE solver.
    <https://en.wikipedia.org/wiki/Euler_method>
    """

    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Step to the next timestep.
        """
        return self.step_to(pred, x_t, t, self.get_next_timestep(t), **kwargs)

    def step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        t = expand_dims(t, x_t.ndim)
        s = expand_dims(s, x_t.ndim)
        T = self.schedule.T
        # Step from x_t to x_s.
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        # Clamp x_s to x_0 and x_T if s is out of bound.
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        if 'ret_x0' in kwargs:
            return pred_x_s, pred_x_0
        return pred_x_s
