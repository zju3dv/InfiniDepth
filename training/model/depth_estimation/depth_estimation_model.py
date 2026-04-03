import os
import time
from os.path import join
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from training.model.model import Model
from training.utils.base_utils import unproject_depthmap_focal
from training.utils.dpt.eval_utils import recover_metric_depth, recover_metric_depth_ransac
from training.utils.logger import Log
from training.utils.parallel_utils import async_call
from training.utils.sampling_np import _normalize01, _sobel_grad_mag, _laplacian_abs, _gaussian_blur, _avg_pool2d_same_exclude_pad, _geom_energy
from training.utils.vis_utils import colorize_depth_maps, visualize_depth


class DepthEstimationModel(Model):
    def __init__(
        self,
        pipeline,  # The pipeline is the model itself
        optimizer,  # The optimizer is the optimizer used to train the model
        lr_table,  # The lr_table is the learning rate table
        output_dir: str,
        output_tag: str = "default",
        clear_output_dir: bool = False,
        scheduler_cfg=None,  # The scheduler_cfg is the scheduler configuration
        ignored_weights_prefix=["pipeline.text_encoder", "pipeline.vae"],
        save_orig_pred=False,  # Whether to save the original prediction
        save_vis_depth=False,  # Whether to save the visualized depth
        save_vis_depth_and_concat_img=False,
        save_vis_depth_and_concat_gt=True,
        save_vis_depth_and_concat_lowres=True,
        save_metrics=False,
        vis_geometry_type="disparity",
        save_depth_mesh=False,
        save_lowres_depth_mesh=False,
        save_sparse_depth=False,
        save_and_concat_diff_map=True,
        log_scale=False,
        near_depth=-1.0,
        far_depth=-1.0,
        compute_abs_metric=False,
        compute_rel_metric=True,
        use_high_freq_mask=False,
        high_freq_sample_num=10000,
        ransac_align_depth=True,
        focal=1440.0,
        concat_axis=1,
        **kwargs,
    ):
        super().__init__(
            pipeline,
            optimizer,
            lr_table,
            output_dir,
            output_tag,
            clear_output_dir,
            scheduler_cfg,
            ignored_weights_prefix,
            **kwargs,
        )
        self._save_orig_pred = save_orig_pred
        self._concat_axis = concat_axis
        self._save_vis_depth = save_vis_depth
        self._save_vis_depth_and_concat_img = save_vis_depth_and_concat_img
        self._save_vis_depth_and_concat_gt = save_vis_depth_and_concat_gt
        self._save_depth_mesh = save_depth_mesh
        self._vis_geometry_type = vis_geometry_type
        self._save_lowres_depth_mesh = save_lowres_depth_mesh
        self._compute_abs_metric = compute_abs_metric
        self._compute_rel_metric = compute_rel_metric
        self._use_high_freq_mask = use_high_freq_mask
        self._high_freq_sample_num = high_freq_sample_num
        self._save_vis_depth_and_concat_lowres = save_vis_depth_and_concat_lowres
        self._save_and_concat_diff_map = save_and_concat_diff_map
        self._log_scale = log_scale
        self._focal = focal
        self._near_depth = near_depth
        self._far_depth = far_depth
        self._save_sparse_depth = save_sparse_depth
        self.align_depth_func = recover_metric_depth_ransac if ransac_align_depth else recover_metric_depth

        self.scene_metrics = []
        self.save_metrics = save_metrics
        Log.info(f"Results will be saved to: {self.output_dir}")

    def predict_step(self, batch, batch_idx, dataloader_idx=None, log_time=True):
        if log_time:
            cur_time = time.time()
        output = self.pipeline.forward_test(batch)
        if log_time:
            Log.info(f"Time taken for forward_test: {time.time() - cur_time} seconds")
        if self._save_orig_pred:
            output_dir = "temp"
            os.makedirs(output_dir, exist_ok=True)
            image_name = join(output_dir, batch["image_name"][0])
            np.savez_compressed(
                image_name + ".npz",
                ground_truth=batch["depth"][0][0].detach().cpu().numpy(),
                pred_depth=output["depth"][0][0].detach().cpu().numpy(),
                prompt_mask=batch["prompt_mask"][0][0].detach().cpu().numpy(),
            )
            self.save_depth(output["depth"], batch["image_name"], "orig_pred")

        if self._save_vis_depth:
            lowres_depth = None
            lowres_depth = (
                batch[f"prompt_{self._vis_geometry_type}"]
                if f"prompt_{self._vis_geometry_type}" in batch and lowres_depth is None
                else lowres_depth
            )
            self.save_vis_depth(
                output[f"{self._vis_geometry_type}"] if self._vis_geometry_type in output else output["depth"],
                batch["image"],
                batch["image_name"],
                "vis_depth",
                gt_depth=batch["depth"] if "depth" in batch else None,
                lowres_depth=lowres_depth,
                norm_depth=output["norm_depth"] if "norm_depth" in output else None,
                geometry_type=self._vis_geometry_type,
            )
        if self._save_depth_mesh and "disp" not in output:
            self.save_depth_mesh(output["depth"], batch["image"], batch["image_name"], "pointcloud")
        if self._save_lowres_depth_mesh:
            try:
                lowres_depth = output["lowres_depth"] if "lowres_depth" in output else batch["lowres_depth"]
                self.save_depth_mesh(
                    batch["lowres_depth"].detach().cpu().numpy(),
                    batch["image"],
                    batch["image_name"],
                    "pointcloud_lowres",
                )
            except Exception as e:
                Log.warn(f"Error in save_depth_mesh: {e}")
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        output = self.predict_step(batch, batch_idx, dataloader_idx, log_time=False)
        batch_size = batch["image"].shape[0]
        metrics_dict = self.compute_metrics(output, batch)
        for k, v in metrics_dict.items():
            if k in ["absolute_image_name", "relative_image_name"]:
                continue
            self.log(
                f"val/{k}",
                np.mean(v),
                on_step=False,
                on_epoch=True,
                prog_bar=True if "abs_rel" in k else False,
                logger=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def compute_metrics(self, output, batch):
        B = batch["image"].shape[0]
        metrics_dict = {}
        for b in range(B):
            image_name = batch["image_name"][b]
            if self._compute_abs_metric or self._compute_rel_metric:
                pred_depth = output["depth"][b][0].float().detach().cpu().numpy()
                if "mesh_depth" in batch:
                    Log.info("Using mesh depth for evaluation", tag="mesh_depth")
                    gt_depth = batch["mesh_depth"][b][0].float().detach().cpu().numpy()
                else:
                    gt_depth = batch["depth"][b][0].float().detach().cpu().numpy()
                msk = self.create_depth_mask(batch["dataset_name"], gt_depth)

            msk = msk & batch["mask"][b, 0].detach().cpu().numpy().astype(np.bool_)
            gt_depth[~msk] = 0.0

            if self._compute_abs_metric:
                if self._use_high_freq_mask:
                    assert "highfreq_mask" in batch, "highfreq_mask not in batch"
                    high_freq_msk = batch["highfreq_mask"][b].detach().cpu().numpy().astype(np.bool_)
                    metric_msk = msk & high_freq_msk
                else:
                    metric_msk = msk
                metrics_dict_item = self.compute_depth_metric(pred_depth, gt_depth, metric_msk)
                if metrics_dict_item is not None:  # fix bug here
                    metrics_dict_item["image_name"] = image_name
                    self.scene_metrics.append(metrics_dict_item)  
                    metrics_dict = self.update_metrics_dict(metrics_dict, metrics_dict_item, "absolute")

            if self._compute_rel_metric:
                pred_depth = self.align_depth_func(
                    pred_depth, gt_depth, msk, disp=("disp" in output and output["disp"]), log=self._log_scale
                )

                # ###### save aligned depth
                if self._use_high_freq_mask:
                    assert "highfreq_mask" in batch, "highfreq_mask not in batch"
                    high_freq_msk = batch["highfreq_mask"][b].detach().cpu().numpy().astype(np.bool_)
                    metric_msk = msk & high_freq_msk
                else:
                    metric_msk = msk
                metrics_dict_item = self.compute_depth_metric(pred_depth, gt_depth, metric_msk)
                if metrics_dict_item is not None:  # fix bug here
                    metrics_dict_item["image_name"] = image_name
                    self.scene_metrics.append(metrics_dict_item)  
                    metrics_dict = self.update_metrics_dict(metrics_dict, metrics_dict_item, "relative")

        return metrics_dict

    def update_metrics_dict(self, metrics_dict, metrics_dict_item, prefix):
        for k, v in metrics_dict_item.items():
            if f"{prefix}_{k}" not in metrics_dict:
                metrics_dict[f"{prefix}_{k}"] = []
            metrics_dict[f"{prefix}_{k}"].append(v)
        return metrics_dict

    def create_depth_mask(self, dataset_name, gt_depth):
        return gt_depth > 1e-3

    def create_high_freq_depth_mask(self, depth, mask=None, n=10000, temperature=0.8):
        H, W = depth.shape

        # valid region
        if mask is None:
            valid = np.ones((H, W), dtype=bool)
        else:
            valid = mask > 0

        # ---- energy maps ----
        score = _geom_energy(depth)

        # zero out invalid
        score = np.where(valid, score, 0.0).astype(np.float32)

        # temperature sharpening
        if temperature != 1.0:
            # avoid pow(0, negative) issues
            score = np.power(np.clip(score, 0.0, None), 1.0 / float(temperature))

        # ---- sampling probabilities ----
        probs = score.reshape(-1)
        sum_w = float(probs.sum())
        valid_num = int(valid.sum())

        if sum_w <= 0.0:
            # fallback: if there are valid pixels, uniform over valid; else uniform over all
            if valid_num > 0:
                probs = valid.reshape(-1).astype(np.float32)
                probs /= float(probs.sum())
            else:
                probs = np.full(H * W, 1.0 / (H * W), dtype=np.float32)
        else:
            probs = (probs / sum_w).astype(np.float32)

        replacement = True if valid_num == 0 else (n > valid_num)
        # replacement = True

        idx = np.random.choice(H * W, size=int(n), replace=replacement, p=probs)
        y = (idx // W).astype(np.int64)
        x = (idx % W).astype(np.int64)
        coords = np.stack([y, x], axis=-1)

        mask = np.zeros((H, W), dtype=np.uint8)
        y = np.clip(coords[:, 0].astype(int), 0, H - 1)
        x = np.clip(coords[:, 1].astype(int), 0, W - 1)
        mask[y, x] = 1

        return mask.astype(np.bool_)

    def compute_depth_metric(self, pred_depth, gt_depth, msk, save_path=None):
        gt = gt_depth[msk]
        pred = pred_depth[msk]
        gt_has_nan = np.isnan(gt).any()
        pred_ha_nan = np.isnan(pred).any()
        if gt_has_nan or pred_ha_nan:
            Log.warn(f"NaN values found in gt or pred depth: gt_nan={gt_has_nan}, pred_nan={pred_ha_nan}")
            return None

        thresh = np.maximum((gt / (pred + 1e-5)), (pred / (gt + 1e-5)))
        d01 = (thresh < 1.01).mean()
        d02 = (thresh < 1.02).mean()
        d04 = (thresh < 1.04).mean()
        d10 = (thresh < 1.1).mean()
        d25_05 = (thresh < 1.25 ** 0.5).mean()
        d25_1 = (thresh < 1.25).mean()
        d25_2 = (thresh < 1.25 ** 2).mean()
        d25_3 = (thresh < 1.25 ** 3).mean()

        l1 = np.mean(np.abs(gt - pred))

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        diff_inv = 1 / (pred + 1e-8) - 1 / (gt + 1e-8)
        irmse = np.sqrt((diff_inv**2).mean())
        imae = np.mean(np.abs(diff_inv))
        abs_rel = np.mean(np.abs(gt - pred) / (gt + 1e-5))

        return {
            "abs_rel": abs_rel,
            "mae_err": l1,
            "rmse_err": rmse,
            "irmse_err": irmse,
            "imae_err": imae,
            "delta_01": d01,
            "delta_02": d02,
            "delta_04": d04,
            "delta_10": d10,
            "delta_25_05": d25_05,
            "delta_25_1": d25_1,
            "delta_25_2": d25_2,
            "delta_25_3": d25_3,
        }

    @async_call
    def save_depth(self, depth, name, tag) -> None:
        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        for b in range(len(depth)):
            depth_np = depth[b][0].float().detach().cpu().numpy()
            last_split_len = len(name[b].split(".")[-1])
            save_name = name[b][: -(last_split_len + 1)] + ".npz"
            img_path = join(self.output_dir, f"{tag}/{save_name}")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            np.savez_compressed(img_path, data=np.round(depth_np, 3))

    # @async_call
    def save_depth_mesh(self, depth, rgb, name, tag) -> None:
        for b in range(len(depth)):
            if isinstance(depth, torch.Tensor):
                depth_np = depth[b][0].float().detach().cpu().numpy()
            else:
                depth_np = depth[b][0]
            self.save_depth_mesh_item(depth_np, rgb[b], name[b], tag, focal=self._focal)

    # @async_call
    def save_depth_mesh_item(self, depth_np, rgb, save_name, tag, focal) -> None:
        if "lowres" in tag:
            scale = depth_np.shape[0] / rgb.shape[1]
            focal = focal * scale
        import open3d as o3d

        rgb_np = rgb.detach().cpu().numpy().transpose((1, 2, 0))
        if rgb_np.shape[0] != depth_np.shape[0] or rgb_np.shape[1] != depth_np.shape[1]:
            rgb_np = cv2.resize(rgb_np, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_AREA)
        points = unproject_depthmap_focal(depth_np, focal=focal)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb_np.reshape(-1, 3))
        save_path = join(self.output_dir, f"{tag}/{save_name[:-4]}.ply")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)

    @async_call
    def save_vis_depth(
        self, depth, rgb, name, tag, gt_depth=None, lowres_depth=None, norm_depth=None, geometry_type="depth"
    ) -> None:
        # import ipdb; ipdb.set_trace()
        for b in range(len(depth)):
            depth_np = depth[b][0].float().detach().cpu().numpy()
            save_name = name[b]
            save_imgs = []
            save_img, depth_min, depth_max = visualize_depth(
                depth_np,
                ret_minmax=True,
                ret_type=np.float64,
            )
         
            save_imgs.append(save_img)

            if self._save_vis_depth_and_concat_img:
                rgb_np = rgb[b].float().detach().cpu().numpy().transpose((1, 2, 0))
                rgb_np = cv2.resize(rgb_np, (save_img.shape[1], save_img.shape[0]), interpolation=cv2.INTER_AREA)
                save_img = np.concatenate([rgb_np, save_img], axis=self._concat_axis)
                save_imgs.append(rgb_np)

            if gt_depth is not None and self._save_vis_depth_and_concat_gt:
                gt_depth_np = gt_depth[b][0].float().detach().cpu().numpy()
                if geometry_type == "disparity":
                    gt_depth_np = 1.0 / (gt_depth_np + 1e-6)
                gt_depth_vis = visualize_depth(
                    gt_depth_np,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    ret_type=np.float64,
                )
                save_img = np.concatenate([save_img, gt_depth_vis], axis=self._concat_axis)
                save_imgs.append(gt_depth_vis)

                if self._save_and_concat_diff_map:
                    pred_depth = depth_np
                    gt_depth = gt_depth_np
                    diff_map = self.compute_depth_difference(pred_depth, gt_depth, save_name)
                    diff_map_color = cv2.cvtColor(cv2.applyColorMap(diff_map, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0 
                    save_img = np.concatenate([save_img, diff_map_color], axis=self._concat_axis)
                    save_imgs.append(gt_depth_vis)
                    
            if lowres_depth is not None and self._save_vis_depth_and_concat_lowres:
                lowres_depth_np = lowres_depth[b][0].float().detach().cpu().numpy()
                tar_h, tar_w = depth_np.shape[0], depth_np.shape[1]
                if lowres_depth_np.shape[1] != tar_w or lowres_depth_np.shape[0] != tar_h:
                    lowres_depth_np = cv2.resize(lowres_depth_np, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
                # used in pdav2
                lowres_depth_np = visualize_depth(
                    lowres_depth_np,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    ret_type=np.float64,
                )
                lowres_depth_np = cv2.resize(
                    lowres_depth_np, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_LINEAR
                )
                save_img = np.concatenate([save_img, lowres_depth_np], axis=self._concat_axis)
                save_imgs.append(lowres_depth_np)
            if norm_depth is not None:
                raise NotImplementedError("Normalized depth visualization is not implemented")
                tar_h, tar_w = depth_np.shape[0], depth_np.shape[1]
                norm_depth_np = norm_depth[b][0].detach().cpu().numpy()
                # print(norm_depth_np)
                norm_depth_np = colorize_depth_maps(norm_depth_np, norm_depth_np.min(), norm_depth_np.max())[
                    0
                ].transpose((1, 2, 0))
                norm_depth_np = cv2.resize(norm_depth_np, (tar_w, tar_h), interpolation=cv2.INTER_LINEAR)
                save_img = np.concatenate([save_img, norm_depth_np], axis=self._concat_axis)
            if save_name.endswith(".h5"):
                save_name = save_name.replace(".h5", ".png")
            img_path = join(self.output_dir, f"{tag}/{save_name}")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            imageio.imwrite(img_path.replace(".png", ".jpg"), (save_img * 255.0).astype(np.uint8))

    def compute_depth_difference(self, pred_depth, gt_depth, save_name):
        assert pred_depth.ndim == 2 and gt_depth.ndim == 2, "Depth maps must be 2D arrays."

        diff_map = np.abs(pred_depth - gt_depth)

        diff_map_normalized = ((diff_map - diff_map.min()) / (diff_map.max() - diff_map.min()) * 255).astype(np.uint8)

        return diff_map_normalized 
