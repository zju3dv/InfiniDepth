"""
Configuration utility functions
"""

import importlib
from typing import Any, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)


def load_config(path: str, argv: List[str] = None) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration. Will resolve inheritance.
    """
    config = OmegaConf.load(path)
    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_inheritance(config)
    return config


def resolve_inheritance(config: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)
        if inherit:
            assert isinstance(inherit, str)
            config = OmegaConf.merge(load_config(inherit), config)
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_inheritance(v)
        return config

    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_inheritance(v)
        return config

    raise NotImplementedError


def import_item(path: str, name: str) -> Any:
    """
    Import a python item. Example: import_item("path.to.file", "MyClass") -> MyClass
    """
    return getattr(importlib.import_module(path), name)


def create_object(config: DictConfig) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)
    """
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")


def create_dataset(path: str, *args, **kwargs) -> Any:
    """
    Create a dataset. Requires the file to contain a "create_dataset" function.
    """
    return import_item(path, "create_dataset")(*args, **kwargs)
