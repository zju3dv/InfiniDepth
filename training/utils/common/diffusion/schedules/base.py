"""
Schedule base class.
"""

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Union
import torch

from ..types import PredictionType
from ..utils import expand_dims


class Schedule(ABC):
    """
    Diffusion schedules are uniquely defined by T, A, B:

        x_t = A(t) * x_0 + B(t) * x_T, where t in [0, T]

    Schedules can be continuous or discrete.
    """

    @abstractproperty
    def T(self) -> Union[int, float]:
        """
        Maximum timestep inclusive.
        Schedule is continuous if float, discrete if int.
        """

    @abstractmethod
    def A(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolation coefficient A.
        Returns tensor with the same shape as t.
        """

    @abstractmethod
    def B(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolation coefficient B.
        Returns tensor with the same shape as t.
        """

    # ----------------------------------------------------

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal to noise ratio.
        Returns tensor with the same shape as t.
        """
        return (self.A(t) ** 2) / (self.B(t) ** 2)

    def isnr(self, snr: torch.Tensor) -> torch.Tensor:
        """
        Inverse signal to noise ratio.
        Returns tensor with the same shape as snr.
        Subclass may implement.
        """
        raise NotImplementedError

    # ----------------------------------------------------

    def is_continuous(self) -> bool:
        """
        Whether the schedule is continuous.
        """
        return isinstance(self.T, float)

    def forward(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion forward function.
        """
        t = expand_dims(t, x_0.ndim)
        return self.A(t) * x_0 + self.B(t) * x_T

    def convert_from_pred(
        self, pred: torch.Tensor, pred_type: PredictionType, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert from prediction. Return predicted x_0 and x_T.   
        """
        t = expand_dims(t, x_t.ndim)
        A_t = self.A(t)
        B_t = self.B(t)
        # x_0 --> latents (Depth),   x_T --> noise (Gaussian/Gray/RGB)
        if pred_type == PredictionType.x_T:
            pred_x_T = pred
            pred_x_0 = (x_t - B_t * pred_x_T) / A_t
        elif pred_type == PredictionType.x_0:
            pred_x_0 = pred
            pred_x_T = (x_t - A_t * pred_x_0) / B_t
        elif pred_type == PredictionType.v_cos:
            pred_x_0 = A_t * x_t - B_t * pred
            pred_x_T = A_t * pred + B_t * x_t
        elif pred_type == PredictionType.v_lerp:
            pred_x_0 = (x_t - B_t * pred) / (A_t + B_t)
            pred_x_T = (x_t + A_t * pred) / (A_t + B_t)
        else:
            raise NotImplementedError

        return pred_x_0, pred_x_T

    def convert_to_pred(
        self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor, pred_type: PredictionType
    ) -> torch.FloatTensor:
        """
        Convert to prediction target given x_0 and x_T.
        """
        if pred_type == PredictionType.x_T:
            return x_T
        if pred_type == PredictionType.x_0:
            return x_0
        if pred_type == PredictionType.v_cos:
            t = expand_dims(t, x_0.ndim)
            return self.A(t) * x_T - self.B(t) * x_0
        if pred_type == PredictionType.v_lerp:
            return x_T - x_0
        raise NotImplementedError
