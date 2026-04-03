# Copyright (c) 2024, Haotong Lin. All rights reserved.
import argparse
import os
import time
from datetime import datetime
from typing import List
from omegaconf import DictConfig, ListConfig, OmegaConf
from training.utils.logger import Log

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


def process_cfg_file(cfg):
    """
    Process the configuration file to handle 'CFG:' references.

    This function recursively processes the configuration object to resolve
    any 'CFG:' references by merging the referenced configuration file into
    the current configuration.

    Args:
        cfg: The configuration object which can be of type int, float, str,
             list, ListConfig, DictConfig, or dict.

    Returns:
        The processed configuration object with all 'CFG:' references resolved.

    Raises:
        ValueError: If the type of cfg is unsupported.
    """

    def process_value(value):
        if isinstance(value, str) and value.startswith("CFG:"):
            return merge_cfg(value.split(":")[-1], OmegaConf.create())
        else:
            return value

    if isinstance(cfg, int) or isinstance(cfg, float):
        return cfg

    if isinstance(cfg, str):
        return process_value(cfg)

    if isinstance(cfg, list) or isinstance(cfg, List) or isinstance(cfg, ListConfig):
        return ListConfig([process_cfg_file(item) for item in cfg])

    if isinstance(cfg, DictConfig) or isinstance(cfg, dict):
        cfg = OmegaConf.to_container(DictConfig(cfg))
        for key in cfg:
            try:
                value = cfg[key]
            except KeyError:
                pass
            cfg[key] = process_cfg_file(value)
        return DictConfig(cfg)

    raise ValueError(f"Unsupported type: {type(cfg)}")


def parse_cfg(cfg, args):
    """
    Parse the configuration file and merge options from the command line.

    This function updates the configuration object with additional information
    such as local rank, entry point, and experiment name. It also handles
    dynamic replacements in the experiment name based on the current file name,
    git branch, git commit, and current date.

    Args:
        cfg: The configuration object to be updated.
        args: Command line arguments containing additional options.

    Returns:
        The updated configuration object.
    """
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.exp_name = cfg.exp_name.replace("FILENAME", os.path.splitext(os.path.basename(args.cfg_file))[0])
    cfg.exp_name = cfg.exp_name.replace("GITBRANCH", os.popen("git describe --all").read().strip()[6:])
    cfg.exp_name = cfg.exp_name.replace("GITCOMMIT", os.popen("git describe --tags --always").read().strip())
    cfg.exp_name = cfg.exp_name.replace("TODAY", datetime.now().strftime("%Y-%m-%d"))

    if "DEBUG" in cfg.exp_name:
        timestamp_file = "/tmp/training_TIMESTAMP.txt"
        if cfg.local_rank == 0:
            timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
            with open(timestamp_file, "w") as f:
                f.write(timestamp)
        while not os.path.exists(timestamp_file):
            time.sleep(1)
        with open(timestamp_file) as f:
            timestamp = f.read().strip()
        cfg.exp_name = cfg.exp_name.replace("DEBUG", timestamp)

    if cfg.pl_trainer.get("precision", 32) == "bf16-mixed":
        import torch

        if not torch.cuda.is_bf16_supported():
            Log.warn("BF16 is not supported on this device, falling back to 16-mixed precision")
            cfg.pl_trainer.precision = "16-mixed"
    return cfg


def merge_cfg(cfg_file, cfg, loaded_cfg_files=None):
    """
    Merge the configuration file with options from the command line.

    This function loads a configuration file, processes it to resolve any
    'CFG:' references, and merges it with the existing configuration object.
    It also handles any included configuration files specified within the
    current configuration.

    Args:
        cfg_file: The path to the configuration file to be loaded.
        cfg: The existing configuration object to be merged with.
        loaded_cfg_files: A list to keep track of loaded configuration files.

    Returns:
        The merged configuration object.

    Raises:
        ValueError: If there is an error loading or merging the configuration file.
    """
    try:
        current_cfg = OmegaConf.load(cfg_file)
        current_cfg = process_cfg_file(current_cfg)
    except Exception as e:
        raise ValueError(f"Error loading {cfg_file}: {e}")

    if "__include__" in current_cfg:
        for included_file in current_cfg.__include__:
            cfg = merge_cfg(included_file, cfg, loaded_cfg_files)
        del current_cfg.__include__

    try:
        cfg = OmegaConf.merge(cfg, current_cfg)
    except Exception as e:
        try:
            Log.warn(f"Warning merging {cfg_file} with {loaded_cfg_files}: {e}")
            cfg = OmegaConf.unsafe_merge(cfg, current_cfg)
        except Exception as e:
            raise ValueError(f"Error merging {cfg_file} with {loaded_cfg_files}: {e}")

    if loaded_cfg_files is not None:
        loaded_cfg_files.append(cfg_file)

    return cfg


def make_cfg(args):
    """
    Create and configure the OmegaConf configuration object.

    This function initializes a new configuration object, merges it with
    the specified configuration file, and applies any additional options
    from the command line arguments.

    Args:
        args: Command line arguments containing configuration options.

    Returns:
        The fully configured OmegaConf object.
    """
    cfg = OmegaConf.create()
    loaded_cfg_files = []
    cfg = merge_cfg(args.cfg_file, cfg, loaded_cfg_files)
    for included_file in args.include:
        cfg = merge_cfg(included_file, cfg, loaded_cfg_files)
    Log.info("Loaded cfg files:\n" + "\n".join(loaded_cfg_files))
    cfg.entry = args.entry
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    cfg = parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument("--entry", type=str, default="train_net")
parser.add_argument("--include", type=str, action="append", default=[])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)
