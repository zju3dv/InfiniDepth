import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
# from lib.network.mas.schedulers import DDPM3DScheduler
from .utils import randlike_shape


class PipelineHelper:
    @staticmethod
    def sch_get_scheduler(scheduler_type=None, scheduler_opt=None):
        if scheduler_type == "ddim":
            return DDIMScheduler(**scheduler_opt)
        elif scheduler_type == "ddpm":
            return DDPMScheduler(**scheduler_opt)
        elif scheduler_type == "ddpm3d":
            return DDPM3DScheduler(**scheduler_opt)

    def sch_get_var(self, t):
        scheduler = self.te_scheduler
        if self.args.scheduler_type == "ddim":
            # "t_prev < 0" is handled in the DDIM scheduler
            t_prev = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            return scheduler._get_variance(t, t_prev)
        elif self.args.scheduler_type == "ddpm":
            return scheduler._get_variance(t)

    # ========== Sample ========== #
    @staticmethod
    def prepare_extra_step_kwargs(scheduler, generator, eta=0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_x(self, shape, generator):
        x = randlike_shape(shape, generator)

        # scale the initial noise by the standard deviation required by the scheduler
        assert self.te_scheduler.init_noise_sigma == 1.0, "Do not match the original implementaiton."
        x = x * self.te_scheduler.init_noise_sigma
        return x

    def get_prog_bar(self, total):
        return ProgBarWrapper(total=total, leave=False, bar_format="{l_bar}{bar:10}{r_bar}")


class ProgBarWrapper:
    def __init__(self, **kwargs):
        self.prog_bar = self.get_prog_bar(**kwargs)

    @rank_zero_only
    def get_prog_bar(self, **kwargs):
        return tqdm(**kwargs)

    @rank_zero_only
    def update(self, **kwargs):
        self.prog_bar.update(**kwargs)
