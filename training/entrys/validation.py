import os
from typing import Tuple
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from training.entrys.utils import get_callbacks, get_data, get_model, print_cfg
from training.utils.net_utils import find_last_ckpt_path, load_pretrained_model
from training.utils.pylogger import Log
from tqdm.auto import tqdm

torch.set_float32_matmul_precision("medium")


def setup_trainer(cfg: DictConfig) -> Tuple[pl.Trainer, pl.LightningModule, pl.LightningDataModule]:
    """
    Set up the PyTorch Lightning trainer, model, and data module.
    """
    if cfg.print_cfg:
        print_cfg(cfg, use_rich=True)
    pl.seed_everything(cfg.seed)
    # preparation
    datamodule = get_data(cfg, wo_train=True)
    model = get_model(cfg)
    ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    if "ckpt_path" in cfg:
        assert os.path.exists(cfg.ckpt_path), f"checkpoint {cfg.ckpt_path} does not exist"
        Log.info(f"Using checkpoint {cfg.ckpt_path} for validation")
        ckpt_path = cfg.ckpt_path
    load_pretrained_model(model, ckpt_path)

    # PL callbacks and logger
    callbacks = get_callbacks(cfg)
    cfg_logger = DictConfig.copy(cfg.logger)
    cfg_logger.update({"version": "val_metrics"})
    logger = hydra.utils.instantiate(cfg_logger, _recursive_=False)

    # PL-Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    return trainer, model, datamodule


def val(cfg: DictConfig) -> None:
    """
    Validate the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.validate(model, datamodule.val_dataloader())


def predict(cfg: DictConfig) -> None:
    """
    Predict using the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.predict(model, datamodule.val_dataloader())


def test(cfg: DictConfig) -> None:
    """
    Test the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.test(model, datamodule.test_dataloader())


def debug_train_dataloader(cfg: DictConfig) -> None:
    """
    Debug the training dataloader.
    """
    # trainer, model, datamodule = setup_trainer(cfg)
    cfg.data.train_loader_opts.num_workers = 0
    datamodule = get_data(cfg)
    dataloader = datamodule.train_dataloader()
    for data in tqdm(iter(dataloader)):
        Log.info(data["disparity_mask"].sum())
        if data["disparity_mask"].sum() <= 10:
            pass
        # for k in data.keys():
        #     if isinstance(data[k], np.ndarray):
        #         Log.info(f"{k}: {data[k].shape} {data[k].dtype}")
        #     elif isinstance(data[k], torch.Tensor):
        #         Log.info(f"{k}: {data[k].shape} {data[k].dtype}")
        #     elif isinstance(data[k], dict):
        #         for kk in data[k].keys():
        #             if isinstance(data[k][kk], np.ndarray):
        #                 Log.info(f"{k}.{kk}: {data[k][kk].shape}")
        #             elif isinstance(data[k][kk], torch.Tensor):
        #                 Log.info(f"{k}.{kk}: {data[k][kk].shape} {data[k][kk].dtype}")
        #     else:
        #         Log.info(f"{k}: {type(data[k])}")
        #         Log.info(data[k])


def debug_val_dataloader(cfg: DictConfig) -> None:
    """
    Debug the training dataloader.
    """
    # trainer, model, datamodule = setup_trainer(cfg)
    cfg.data.val_loader_opts = cfg.data.get("val_loader_opts", DictConfig({}))
    cfg.data.val_loader_opts.num_workers = 0
    datamodule = get_data(cfg, wo_train=True)
    dataloader = iter(datamodule.val_dataloader())
    for data in tqdm(dataloader):
        pass
        # Log.info(data.keys())


def debug_cfg(cfg: DictConfig) -> None:
    """
    Debug the config.
    """
    print_cfg(cfg, use_rich=True)
