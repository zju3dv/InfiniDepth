import os
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from training.entrys.utils import delete_output_dir, get_callbacks, get_data, get_model, print_cfg
from training.utils.logger import Log
from training.utils.net_utils import find_last_ckpt_path, load_pretrained_model

torch.set_float32_matmul_precision("medium")


def train_net(cfg: DictConfig) -> None:
    """
    Instantiate the trainer, and then train the model.
    """
    if cfg.print_cfg:
        print_cfg(cfg, use_rich=True)
    callbacks = get_callbacks(cfg)
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )
    # seed everything before loading data
    pl.seed_everything(cfg.seed)
    # pl.seed_everything(cfg.seed + trainer.node_rank*8 + trainer.local_rank)
    datamodule: pl.LightningDataModule = get_data(cfg)
    model: pl.LightningModule = get_model(cfg)

    # load pretrained model
    delete_output_dir(cfg.resume_training, cfg.output_dir, cfg.confirm_delete_previous_dir)
    ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    load_pretrained_model(model, ckpt_path)
    if ckpt_path is None and "ckpt_path" in cfg:
        assert os.path.exists(cfg.ckpt_path), f"checkpoint {cfg.ckpt_path} does not exist"
        Log.info(f"Using checkpoint {cfg.ckpt_path} for validation")
        ckpt_path = cfg.ckpt_path
        if "ckpt_type" in cfg:
            ckpt_type = cfg.ckpt_type
        else:
            ckpt_type = None
        load_pretrained_model(model, ckpt_path, ckpt_type)
        pretrained_model = torch.load(ckpt_path, map_location="cpu")
        if "optimizer" not in pretrained_model or cfg.finetune_only:
            ckpt_path = None

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
