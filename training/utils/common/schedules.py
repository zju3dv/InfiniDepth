"""
Schedule utility for learning rate or loss weighting, etc.
"""

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch
from omegaconf import DictConfig, ListConfig


class Schedule(ABC):
    """
    Schedule base class
    """

    def __init__(self, steps: int):
        self.steps = steps

    def __len__(self):
        return self.steps

    @abstractmethod
    def __getitem__(self, step: int) -> float:
        pass


class LazyInitializable(ABC):
    @abstractmethod
    def _lazy_initialize(self, start: float):
        pass


class SequentialSchedule(Schedule):
    """
    A Sequential collection of schedules.
    Example:
        1. First 1000 steps linearly increase from 0.0 to 1.0
        2. Next 500 steps stays constant at 1.0
        3. Next 2000 steps consine anneal from 1.0 to 0.5
        4. Stays at 0.5 afterward

        SequentialSchedule([
            LinearSchedule(steps=1000, start=0.0, end=1.0),
            ConstantSchedule(steps=500, value=1.0),
            CosineSchedule(steps=2000, start=1.0, end=0.5)
        ])

        Equivalently, it supports lazy initialization for non-first schedules.
        This infers the start value from the last schedule's end value.

        SequentialSchedule([
            LinearSchedule(steps=1000, start=0.0, end=1.0),
            ConstantSchedule(steps=500),
            CosineSchedule(steps=2000, end=0.5)
        ])
    """

    def __init__(
        self,
        schedules: List[Schedule],
    ):
        assert len(schedules) > 0
        super().__init__(sum(len(s) for s in schedules))
        self.schedules = schedules

        # Lazy initialize all schedules.
        for a, b in zip(self.schedules[:-1], self.schedules[1:]):
            if isinstance(b, LazyInitializable):
                b._lazy_initialize(a[len(a) - 1])

    def __getitem__(self, step: int) -> float:
        assert step >= 0, "Step cannot be negative"
        for i, schedule in enumerate(self.schedules):
            if (step < len(schedule)) or (i == len(self.schedules) - 1):
                return schedule[step]
            step -= schedule.steps


class ConstantSchedule(Schedule, LazyInitializable):
    """
    Constant schedule always returns the same value on any steps.
    """

    def __init__(self, steps: int, value: Optional[float] = None):
        super().__init__(steps)
        self.value = value

    def _lazy_initialize(self, start: float):
        if self.value is None:
            self.value = start

    def __getitem__(self, step: int) -> float:
        assert step >= 0, "Step cannot be negative"
        assert self.value is not None, "Constant schedule is not initialized."
        return self.value


class LinearSchedule(Schedule, LazyInitializable):
    """
    Linear schedule linearly changes from start to end over defined steps.
    When step < 0, return the start value. When step >= steps, return the end value.
    """

    def __init__(self, steps: int, end: float, *, start: Optional[float] = None):
        super().__init__(steps)
        self.start = start
        self.end = end

    def _lazy_initialize(self, start: float):
        if self.start is None:
            self.start = start

    def __getitem__(self, step: int) -> float:
        if step >= len(self) - 1:
            return self.end
        assert step >= 0, "Step cannot be negative"
        assert self.start is not None, "Linear schedule is not initialized."
        t = step / (len(self) - 1)
        return self.start * (1 - t) + self.end * t


class CosineSchedule(Schedule, LazyInitializable):
    """
    Cosine schedules changes from start to end over a cosine curve over defined steps.
    When step < 0, return the start value. When step >= steps, return the end value.
    """

    def __init__(self, steps: int, end: float, *, start: Optional[float] = None):
        super().__init__(steps)
        self.start = start
        self.end = end

    def _lazy_initialize(self, start: float):
        if self.start is None:
            self.start = start

    def __getitem__(self, step: int) -> float:
        if step >= len(self) - 1:
            return self.end
        assert step >= 0, "Step cannot be negative"
        assert self.start is not None, "Cosine schedule is not initialized."
        t = step / (len(self) - 1)
        return self.end + (self.start - self.end) * (1 + math.cos(math.pi * t)) / 2


# -------------------------------------------


def apply_lr(
    optimizer: torch.optim.Optimizer,
    schedule: Schedule,
    step: int,
    param_group: Optional[int] = None,
) -> float:
    """
    Apply schedule as the learning rate to an optimizer.
    """
    lr = schedule[step]
    groups = optimizer.param_groups
    if param_group is not None:
        groups = [groups[param_group]]
    for group in groups:
        group["lr"] = lr
    return lr


def create_schedule_from_config(config: Union[DictConfig, ListConfig]):
    """
    Create a schedule from a config.
    """
    if isinstance(config, ListConfig):
        return SequentialSchedule([create_schedule_from_config(c) for c in config])

    if isinstance(config, DictConfig):
        if config.type == "constant":
            return ConstantSchedule(
                steps=config.steps,
                value=config.get("value"),
            )
        if config.type == "linear":
            return LinearSchedule(
                steps=config.steps,
                start=config.get("start"),
                end=config.end,
            )
        if config.type == "cosine":
            return CosineSchedule(
                steps=config.steps,
                start=config.get("start"),
                end=config.end,
            )

    raise NotImplementedError
