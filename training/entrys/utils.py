import hydra
import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree
import os

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from training.utils.pylogger import monitor_process_wrapper
from training.utils.logger import Log

# ================================ #
#      utils & get components      #
# ================================ #


@monitor_process_wrapper
def get_data(cfg: DictConfig, wo_train: bool = False) -> pl.LightningDataModule:
    datamodule = hydra.utils.instantiate(cfg.data, wo_train=wo_train, _recursive_=False)
    return datamodule


@monitor_process_wrapper
def get_model(cfg: DictConfig) -> pl.LightningModule:
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if hasattr(cfg, "exp_name"):
        # TODO: Maybe not elegant!
        model.exp_name = cfg.exp_name
    return model


@monitor_process_wrapper
def get_callbacks(cfg: DictConfig) -> list:
    if not hasattr(cfg, "callbacks"):
        return None
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks


@rank_zero_only
def print_cfg(cfg: DictConfig, use_rich: bool = False):
    if use_rich:
        print_order = ("data", "model", "callbacks", "logger", "pl_trainer", "exp")
        style = "dim"
        tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

        # add fields from `print_order` to queue
        # add all the other fields to queue (not specified in `print_order`)
        queue = []
        for field in print_order:
            queue.append(field) if field in cfg else Log.warn(f"Field '{field}' not found in config. Skipping.")
        for field in cfg:
            if field not in queue:
                queue.append(field)

        # generate config tree from queue
        for field in queue:
            branch = tree.add(field, style=style, guide_style=style)
            config_group = cfg[field]
            if isinstance(config_group, DictConfig):
                branch_content = OmegaConf.to_yaml(config_group, resolve=False)
            else:
                branch_content = str(config_group)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        rich.print(tree)
    else:
        Log.info(OmegaConf.to_yaml(cfg, resolve=False))
        
@monitor_process_wrapper
def delete_output_dir(resume_training, output_dir, confirm_delete_previous_dir):
    '''
    resume_training: bool
    output_dir: str
    confirm_delete_previous_dir: bool
    return None
    delete the output_dir if not resume_training
    '''
    if not resume_training and os.path.exists(output_dir):
        Log.warn("Not resume_training, training from scratch.")
        Log.warn("Deleting the output path: {}".format(output_dir))
        if confirm_delete_previous_dir: 
            os.system('rm -rf {}'.format(output_dir))
            Log.info("Deleted the output path: {}".format(output_dir))
        else:
            while True:
                user_input = input("Delete the output path: {}? (y/n)".format(output_dir))
                if user_input == "y": os.system('rm -rf {}'.format(output_dir)); break
                elif user_input == "n": Log.warn("Not deleting the output path."); break
                else: Log.warn("Invalid input, please input again.")