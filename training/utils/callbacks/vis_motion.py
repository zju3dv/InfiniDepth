import os
import decord
import pytorch_lightning as pl
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_only

from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

from training.utils.logger import Log
from training.utils.skeleton_motion_visualization import SkeletonAnimationGenerator

import matplotlib

matplotlib.use("Agg")


class VisMotion(pl.Callback):
    def __init__(self, video_dir, fps=30):
        super().__init__()

        self.fps = fps
        self.video_dir = video_dir
        # prepare the directory
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        Log.info("Visualize motion init")

    @rank_zero_only
    def vis_motion(self, trainer, outputs):
        # Log.info("Visualize motion")
        b = 0  # batch visualization is not support now
        eg_motion = outputs["pred_ayfz_motion"][b].cpu().numpy()
        video_path = os.path.join(self.video_dir, f"{trainer.global_step:07d}.mp4")

        # save the video
        sag = SkeletonAnimationGenerator(eg_motion, self.fps)
        sag.save(video_path)
        # Log.info("Save video to: {}".format(video_path))

        # load the video for tensorboardX
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        vlen = vr.__len__()
        selected_frames = range(0, vlen, 1)
        video = vr.get_batch(selected_frames)  # (frames, height, width, channels)
        video = rearrange(video, "t h w c -> 1 t c h w")  # (batch, frames, channels, height, width)

        # add to tensorbordX
        # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
        if trainer.logger is not None:
            trainer.logger.experiment.add_video("visual", video, global_step=trainer.global_step, fps=self.fps)

    # TODO: we need discuss when to use the callbacks when we actually start using this. @XY
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # only store the final batch
        if batch_idx == len(trainer.predict_dataloaders) - 1:
            self.vis_motion(trainer, outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # only store the final batch
        if batch_idx == len(trainer.val_dataloaders) - 1:
            self.vis_motion(trainer, outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # only store the final batch
        if batch_idx == len(trainer.test_dataloaders) - 1:
            self.vis_motion(trainer, outputs)
