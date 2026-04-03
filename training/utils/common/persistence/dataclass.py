"""
Persistence dataclasses
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from common.config import create_object, load_config
from common.fs import copy, download, exists, mkdir

from ..decorators import assert_only_global_rank_zero
from .utils import convert_dtype, get_local_path


@dataclass
class PersistedConfig:
    path: str

    def load(self, distributed: bool = True) -> DictConfig:
        return load_config(download(self.path, distributed=distributed))

    @assert_only_global_rank_zero
    def save(self, config: DictConfig):
        local_path = get_local_path(self.path)
        OmegaConf.save(config, local_path, resolve=True)
        mkdir(os.path.dirname(self.path))
        copy(local_path, self.path)


@dataclass
class PersistedStates:
    path: str

    def load(self, device: torch.device = None, distributed: bool = True) -> Any:
        return torch.load(download(self.path, distributed=distributed), map_location=device)

    @assert_only_global_rank_zero
    def save(self, states: Any, *, dtype: Optional[torch.dtype] = None, blocking: bool = False):
        local_path = get_local_path(self.path)
        convert_dtype(states, dtype)
        torch.save(states, local_path)
        mkdir(os.path.dirname(self.path))
        copy(local_path, self.path, blocking=blocking)


@dataclass
class PersistedModel:
    config: PersistedConfig
    states: PersistedStates

    def create(self, device: torch.device = "cpu") -> torch.nn.Module:
        config = self.config.load()
        states = self.states.load(device)
        model = create_object(config).to(device)
        model.load_state_dict(states)
        return model


@dataclass
class PersistedOptimizer:
    states: PersistedStates


@dataclass
class PersistedTrainingState:
    step: int
    models: Dict[str, PersistedModel]
    optimizers: Dict[str, PersistedOptimizer]


@dataclass
class PersistedMetric:
    path: str

    def load(self, distributed: bool = True) -> Optional[Dict[str, Any]]:
        if exists(self.path):
            return pd.read_csv(
                download(self.path, overwrite=True, distributed=distributed)
            ).to_dict()
        else:
            return None

    @assert_only_global_rank_zero
    def save(self, step: int, metric: Dict[str, Any]):
        local_path = get_local_path(self.path)
        record = self.load(distributed=False)
        if record is not None:
            record = pd.DataFrame(record)
            for key in metric:
                if key not in record.columns:
                    record[key] = None
            if step in record["step"].values:
                index = record["step"] == step
                for key, value in metric.items():
                    record.loc[index, key] = value
            else:
                record.loc[len(record)] = {"step": step, **metric}
        else:
            record = pd.DataFrame([{"step": step, **metric}])
        record.to_csv(local_path, index=False)
        mkdir(os.path.dirname(self.path))
        copy(local_path, self.path)
