import json
import os
from os.path import join
from typing import Any, Dict, List
import numpy as np
import pytorch_lightning as pl
import torch
import pandas as pd
from hydra.utils import instantiate
from training.utils.logger import Log
from training.utils.vis_utils import visualize_depth


class Model(pl.LightningModule):
    def __init__(
        self,
        pipeline,  # The pipeline is the model itself
        optimizer,  # The optimizer is the optimizer used to train the model
        lr_table=None,  # The lr_table is the learning rate table
        output_dir: str = None,
        output_tag: str = "default",
        clear_output_dir=False,
        scheduler_cfg=None,  # The scheduler_cfg is the scheduler configuration
        ignored_weights_prefix=["pipeline.text_encoder", "pipeline.vae"],
        image_step=1000,
        **kwargs,
    ):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.optimizer = instantiate(optimizer)
        if lr_table is not None:
            self.lr_table = instantiate(lr_table)
        self.scheduler_cfg = scheduler_cfg
        self.image_step = image_step

        if clear_output_dir:
            Log.warn(f"Clear output dir: {join(output_dir, output_tag)}")
            os.system(f"rm -rf {join(output_dir, output_tag)}")
        self.output_dir = join(output_dir, output_tag)
        self.metrics_dict = {}
        self.ignored_weights_prefix = ignored_weights_prefix

        self.test_step = self.validation_step

    def training_step(self, batch, batch_idx):
        output = self.pipeline.forward_train(batch)
        if not isinstance(self.trainer.train_dataloader, List):
            B = self.trainer.train_dataloader.batch_size
        else:
            B = np.sum([dataloader.batch_size for dataloader in self.trainer.train_dataloader])
        loss = output["loss"]
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"Loss is NaN or Inf: {loss}")
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=B, sync_dist=True
        )
        # Log other metrics
        for k, v in output.items():
            if (k != "loss" and k.endswith("_vis")) or "loss" in k:
                self.log(
                    f"train/{k[:-4]}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=B,
                    sync_dist=True,
                )

        if self.global_step % self.image_step == 0:
            if "depth" in output and "image" in batch:
                depth_np = output["depth"][0][0].float().detach().cpu().numpy()
                rgb_np = batch["image"][0].detach().cpu().numpy().transpose((1, 2, 0))
                depth_vis = visualize_depth(depth_np)
                rgb_vis = (rgb_np * 255.0).astype(np.uint8)
                vis_img = np.concatenate([rgb_vis, depth_vis], axis=1)
                self.logger.experiment.add_image("train/depth_vis", vis_img.transpose((2, 0, 1)), self.global_step)
        if "depth" in output:
            del output["depth"]
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        raise NotImplementedError

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """Task-specific validation step. CAP or GEN."""
        raise NotImplementedError

    # ============== Utils ================= #
    def configure_optimizers(self):
        group_table = {}
        params = []
        if self.lr_table is not None:
            for k, v in self.pipeline.named_parameters():
                if v.requires_grad:
                    group, lr = self.lr_table.get_lr(k)
                    if lr == 0:
                        v.requires_grad = False
                    if group not in group_table:
                        group_table[group] = len(group_table)
                        params.append({"params": [v], "lr": lr, "name": group})
                    else:
                        params[group_table[group]]["params"].append(v)
        else:
            for k, v in self.pipeline.named_parameters():
                if v.requires_grad:
                    params.append({"params": [v]})
        optimizer = self.optimizer(params=params)
        if self.scheduler_cfg is None:
            return optimizer
        scheduler_cfg = dict(self.scheduler_cfg)
        scheduler_cfg["scheduler"] = instantiate(scheduler_cfg["scheduler"], optimizer=optimizer)
        return [optimizer], [scheduler_cfg]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for ig_keys in self.ignored_weights_prefix:
            Log.debug(f"Remove key `{ig_keys}' from checkpoint.")
            for k in list(checkpoint["state_dict"].keys()):
                if k.startswith(ig_keys):
                    checkpoint["state_dict"].pop(k)
        super().on_save_checkpoint(checkpoint)

    def load_pretrained_model(self, ckpt_path, ckpt_type):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"Loading ckpt type `{ckpt_type}': {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = []
        for k in missing:
            miss = True
            for ig_keys in self.ignored_weights_prefix:
                if k.startswith(ig_keys):
                    miss = False
            if miss:
                real_missing.append(k)
        if len(real_missing) > 0:
            Log.warn(f"Missing keys: {real_missing}")
        if len(unexpected) > 0:
            Log.error(f"Unexpected keys: {unexpected}")

    def on_validation_epoch_end(self):
        metrics_to_save = {}
        if self.trainer and hasattr(self.trainer, "logged_metrics"):
            # Convert tensors to Python numbers for JSON serialization
            for key, value in self.trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics_to_save[key] = value.item()
                else:
                    metrics_to_save[key] = value
        else:
            Log.warn("Warning: Trainer or logged_metrics not available in on_test_epoch_end.")
            return  # Or handle as an error

        try:
            output_json_path = join(self.output_dir, "metrics/metrics.json")
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(metrics_to_save, f, indent=4)
            Log.info(f"Successfully saved metrics to {output_json_path}")

        except Exception as e:
            Log.error(f"Error saving average metrics: {e}")

        try:
            if not self.training and len(self.scene_metrics) > 0 and self.save_metrics:
                csv_save_path = join(self.output_dir, "metrics/all_scenes.csv")
                df = pd.DataFrame(self.scene_metrics)
                df.to_csv(csv_save_path, index=False)
                Log.info(f"Successfully saved {len(df)} scene metrics to {csv_save_path}")

        except Exception as e:
            Log.error(f"Error saving metrics of every scene: {e}")
