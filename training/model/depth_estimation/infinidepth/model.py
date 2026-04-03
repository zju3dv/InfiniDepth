import os
import hydra
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from InfiniDepth.model.block.config import model_configs, dinov3_model_configs
from InfiniDepth.model.block.implicit_decoder import ImplicitHead
from InfiniDepth.model.block.utils import make_coord
from InfiniDepth.model.block.convolution import BasicEncoder
from training.utils.logger import Log


EPS = 1e-6
InfiniDepthConfig = OmegaConf.create(
    {
        "encoder": "vitl",
        "use_dpt_implicit_head": True,
        "use_batch_infer": True,
        "ckpt_path": None,
        "load_pretrain_net": "",
        "warp_func": None,
        "use_basic_encoder": True,
        "basic_encoder_dim": 128,
        "hidden_list": [1024, 256, 32],
        "fusion_type": "concat",  # concat, gated
    }
)

def _get_acc_dtype():
    if not torch.cuda.is_available():
        return torch.float16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


acc_dtype = _get_acc_dtype()
degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
model = make_pipeline(poly_features, ransac)
TORCHHUB_ROOT = Path(__file__).resolve().parents[4] / "InfiniDepth" / "model" / "block" / "torchhub"


class InfiniDepth(nn.Module):
    use_bn = False
    use_clstoken = False

    def __init__(self, config: DictConfig):
        super().__init__()
        config = OmegaConf.merge(InfiniDepthConfig, config)
        self.config = config
        
        backbone = self.config.get("backbone", "dinov2")
        if backbone == "dinov2":
            model_config = model_configs[config.encoder]
        elif backbone == "dinov3":
            model_config = dinov3_model_configs[config.encoder]
        else:
            raise NotImplementedError
        self.model_config = model_config
        
        # patch size
        self.patch_size = config.get("patch_size", 14)
        # infer config
        self.use_batch_infer = config.get("use_batch_infer", True)
        self.enable_3d_uniform_sampling = config.get("enable_3d_uniform_sampling", False)

        # Learnable module definitions
        # backbone
        if backbone == "dinov2":
            self.pretrained = torch.hub.load(
                str(TORCHHUB_ROOT / "facebookresearch_dinov2_main"),
                f"dinov2_{config.encoder}14",
                source="local",
                pretrained=False,
                prompt_dino_config=config.get("prompt_dino", None),
            )
        elif backbone == "dinov3": # dinov3
             self.pretrained = torch.hub.load(
                str(TORCHHUB_ROOT / "dinov3"),
                f"dinov3_{config.encoder}",  # vitl16plus, vith16plus, vit7b16
                source="local",
                pretrained=False,
            )
        else: 
            raise NotImplementedError
        dim = self.pretrained.blocks[0].attn.qkv.in_features 


        if os.path.exists(self.config.dino_checkpoint):
            pretrained_model = torch.load(self.config.dino_checkpoint, map_location="cpu")
            Log.info(f"Loading DINO checkpoint from {self.config.dino_checkpoint}")
        else:
            print("No checkpoint found")
            print(self.config.dino_checkpoint)
            print("Trying:  wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")
            raise ValueError("No checkpoint found")
        self.pretrained.load_state_dict(pretrained_model, strict=True)

        # prompt
        self.use_prompt = self.config.get("use_prompt", True)
        if self.use_prompt:
            self.prompt_model = hydra.utils.instantiate(config.prompt, recursive=True)
        self.high_res_prompt = config.get("high_res", False)

        # normalization
        self.warp_type = self.config.get("warp_type", "minmax")  # minmax, median

        # BasicEncoder for low-level features
        self.use_basic_encoder = config.get("use_basic_encoder", False)
        if self.use_basic_encoder:
            self.basic_encoder_dim = config.get("basic_encoder_dim", 128)
            self.basic_encoder = BasicEncoder(
                input_dim=3,
                output_dim=self.basic_encoder_dim,
                stride=4
            )

        # implicit depth decoder head
        self.depth_implicit_head = ImplicitHead(
            hidden_dim=dim,
            basic_dim=config.get("basic_encoder_dim", 128),
            fusion_type="concat",
            out_dim=1,
            hidden_list=config.get("hidden_list", [1024, 256, 32]),
        )
        # Load warp function
        self.warp_func = hydra.utils.instantiate(config.warp_func)

        # Load pretrained model and checkpoint
        self._load_pretrain(config)
        self._load_checkpoint(config)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # training
        self.criterion = hydra.utils.instantiate(config.loss_cfg)
        self.geometry_type = config.get("geometry_type", "depth")  # disparity

    def _load_checkpoint(self, config):
        if config.get("ckpt_path", None) is None:
            return
        if os.path.exists(config.ckpt_path):
            Log.info(f"Loading checkpoint from {config.ckpt_path}")
            checkpoint = torch.load(config.ckpt_path, map_location="cpu")
            self.load_state_dict({k[9:]: v for k, v in checkpoint["state_dict"].items()})
        else:
            Log.warn(f"Checkpoint {config.ckpt_path} not found")

    def _load_pretrain(self, config):
        if config.get("load_pretrain_net", "") == "":
            return
        Log.info(f"Load pretrain network from {config.load_pretrain_net}")
        assert os.path.exists(config.load_pretrain_net), f"Pretrain network {config.load_pretrain_net} not found"
        model = torch.load(config.load_pretrain_net, "cpu")
        if "model" in model:
            model = model["model"]
            model = {k[7:]: model[k] for k in model}
            strict = False
        elif "state_dict" in model:
            model = model["state_dict"]
            model = {k[9:]: model[k] for k in model}
            strict = False
        elif list(model.keys())[0].startswith("net.pretrained"):
            model = {k.replace("net.pretrained", "pretrained"): v for k, v in model.items()}
            model = {k.replace("head", "depth_head"): v for k, v in model.items()}
            strict = False
        else:
            model = model
            strict = False
        for i, block in enumerate(self.pretrained.blocks):
            if hasattr(block, "copy_block") and f"pretrained.blocks.{i}.copy_block" not in model:
                block_params = {k: v for k, v in model.items() if k.startswith(f"pretrained.blocks.{i}.")}
                block_copy_params = {
                    k.replace(f"pretrained.blocks.{i}.", f"pretrained.blocks.{i}.copy_block."): v
                    for k, v in block_params.items()
                }
                model.update(block_copy_params)
        self.load_state_dict(model, strict=strict)

    def _get_gt(self, gt, gt_reference_meta, prompt_reference_meta):
        if self.use_prompt:
            gt_query_depth = gt / torch.clamp(prompt_reference_meta[...,0], min=1e-3)
        else:
            if self.warp_type == "minmax":
                gt_min = gt_reference_meta[f"{self.geometry_type}_min"][..., None, None]
                gt_max = gt_reference_meta[f"{self.geometry_type}_max"][..., None, None]
                gt_query_depth = (gt - gt_min) / torch.clamp((gt_max - gt_min), min=1e-3)
            elif self.warp_type == "median":
                gt_median = gt_reference_meta[f"{self.geometry_type}_median"][..., None, None]
                gt_query_depth = gt / torch.clamp(gt_median, min=1e-3)
            else:
                raise NotImplementedError
        return gt_query_depth

    def _ransac_align_depth(self, pred, gt, mask0):
        if type(pred).__module__ == torch.__name__:
            pred = pred.cpu().numpy()
        if type(gt).__module__ == torch.__name__:
            gt = gt.cpu().numpy()
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        gt = gt.squeeze()
        pred = pred.squeeze()
        mask = (gt > 1e-8) #& (pred > 1e-8)
        if mask0 is not None and mask0.sum() > 0:
            if type(mask0).__module__ == torch.__name__:
                mask0 = mask0.cpu().numpy()
            mask0 = mask0.squeeze()
            mask0 = mask0 > 0
            mask = mask & mask0
        gt_mask = gt[mask].astype(np.float32)
        pred_mask = pred[mask].astype(np.float32)

        try:
            model.fit(pred_mask[:, None], gt_mask[:, None])
            a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
            a = a.item()
            b = b.item()
        except:
            a, b = 1, 0
            
        if a > 0:
            pred_metric = a * pred + b
        else:
            pred_mean = np.mean(pred_mask)
            gt_mean = np.mean(gt_mask)
            pred_metric = pred * (gt_mean / pred_mean)

        return torch.from_numpy(pred_metric).unsqueeze(0).unsqueeze(0)
    
    def forward_train(self, batches):
        if not isinstance(batches, List):
            return self.forward_train_batch(batches)
        loss = 0.0
        outputs = []
        for batch in batches:
            outputs.append(self.forward_train_batch(batch))
            loss += outputs[-1]["loss"]
        return {"loss": loss}

    def forward_test(self, batch):
        prompt_depth, prompt_mask, reference_meta = self.warp_func.warp(
            batch[f"prompt_{self.geometry_type}"],
            ground_truth=batch[f"{self.geometry_type}"],
            ground_truth_mask=batch["mask"] if self.geometry_type == "depth" else batch["disparity_mask"],
            prompt_depth=batch[f"prompt_{self.geometry_type}"],
            prompt_mask=batch["prompt_mask"],
            batch=batch,
        )
        input_image = batch["image"]  # no downsampling here
        query_coord = batch[f"sampled_coord_{self.geometry_type}"]  # [B, N, 2]

        hr_depth = batch[f"{self.geometry_type}"]  # [B, 1, H, W]
        h, w = hr_depth.shape[-2:]

        if self.use_batch_infer:
            pred_query_depth = self.__batch_forward(
                input_image,     # [B, 3, H, W] in high resolution
                query_coord,     # [B, N, 2] sampled query coordinates in high resolution
                prompt_depth=prompt_depth,
                prompt_mask=prompt_mask,
                bsize=300000
            )
        else:
            pred_query_depth = self.__forward(
                input_image,      
                query_coord,         
                prompt_depth=prompt_depth,
                prompt_mask=prompt_mask,
            )
        
        depth = pred_query_depth.permute(0,2,1).view(1,1,h,w)  # [B,N,1] --> [B,1,H,W]
        if self.use_prompt:
            depth = self.warp_func.unwarp(
                depth,
                reference_meta=reference_meta,
                ground_truth=batch[f"{self.geometry_type}"],
                ground_truth_mask=batch["mask"] if self.geometry_type == "depth" else batch["disparity_mask"],
                prompt_depth=batch[f"prompt_{self.geometry_type}"],
                prompt_mask=batch["prompt_mask"],
                batch=batch,
            )
        else:
            if self.geometry_type == "disparity":
                gt = batch[f"{self.geometry_type}"]
                gt_msk = batch["mask"] if self.geometry_type == "depth" else batch["disparity_mask"]
                depth = self._ransac_align_depth(depth, gt, gt_msk)  # (1, 1, H, W)
                
        if self.geometry_type == "disparity":
            disparity = depth
            depth = 1.0 / torch.clamp(depth, min=5e-3)
            return {"depth": depth, "disparity": disparity}
        elif self.geometry_type == "depth":
            disparity = 1.0 / torch.clamp(depth, min=5e-3)
            return {"depth": depth, "disparity": disparity}

    def __batch_forward(self, x, coord, prompt_depth=None, prompt_mask=None, bsize=3000):
        """
        Forward pass with batching to avoid OOM.
        """
        h, w = x.shape[-2:]

        # DINOv3 branch (semantic features) - uses ImageNet normalization
        x_dino = (x - self._mean) / self._std
        with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=self.model_config["layer_idxs"],
                return_class_token=True,
            )
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        if self.use_prompt:
            features = self.prompt_model(features, prompt_depth, prompt_mask, patch_h, patch_w)

        # BasicEncoder branch (low-level features) - uses [-1, 1] normalization
        if self.use_basic_encoder:
            # Assuming x is in [0, 1] range, convert to [-1, 1]
            x_basic = 2.0 * x - 1.0
            # with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]
        else:
            basic_feat = None

        # generate feature map for learning implicit function
        feat = self.depth_implicit_head._encode_feat(features, patch_h, patch_w)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            # batch querying
            pred = self.depth_implicit_head._decode_dpt(feat, basic_feat, coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        return pred

    def __forward(self, x, coords, prompt_depth, prompt_mask):
        h, w = x.shape[-2:]

        # DINOv3 branch (semantic features) - uses ImageNet normalization
        x_dino = (x - self._mean) / self._std
        with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            features = self.pretrained.get_intermediate_layers(
                x_dino,
                n=self.model_config["layer_idxs"],
                return_class_token=True,
            )
        features = [list(feature) for feature in features]
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        if self.use_prompt:
            features = self.prompt_model(features, prompt_depth, prompt_mask, patch_h, patch_w)

        # BasicEncoder branch (low-level features) - uses [-1, 1] normalization
        if self.use_basic_encoder:
            # Assuming x is in [0, 1] range, convert to [-1, 1]
            x_basic = 2.0 * x - 1.0
            # with torch.autocast("cuda", enabled=True, dtype=acc_dtype):
            basic_feat = self.basic_encoder(x_basic)  # [B, 128, H/4, W/4]
        else:
            basic_feat = None

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            depth = self.depth_implicit_head(features, basic_feat, patch_h, patch_w, coords)

        return depth

    def forward_train_batch(self, batch):
        prompt_depth, prompt_mask, reference_meta = self.warp_func.warp(
            batch[f"prompt_{self.geometry_type}"],
            ground_truth=batch[f"sampled_{self.geometry_type}"],
            ground_truth_mask=torch.ones_like(batch[f"sampled_{self.geometry_type}"]), # all True
            prompt_depth=batch[f"prompt_{self.geometry_type}"],
            prompt_mask=batch["prompt_mask"],
            batch=batch,
        )

        query_coord = batch[f"sampled_coord_{self.geometry_type}"]  # [B, N, 2]
        input_image = batch["image"]
        pred_query_depth = self.__forward(
            input_image,     # [B, 3, h_lr, w_lr]
            query_coord,     # [B, N, 2] sampled query coordinates in high resolution
            prompt_depth=prompt_depth,
            prompt_mask=prompt_mask,  
        )

        gt_query_depth = self._get_gt(batch[f"sampled_{self.geometry_type}"], batch["reference_meta"], reference_meta) 

        ret_dict = {}
        loss, loss_item = self.criterion(
            pred_query_depth,
            gt_query_depth,
        )
        # Prepare return dict
        ret_dict = {}
        ret_dict.update(loss_item)
        ret_dict.update({"loss": loss})

        # ret_dict.update({"depth": pred_query_depth})  
        # Check for invalid loss values
        if loss.isnan().any() or loss.isinf().any():
            raise ValueError("loss is nan or inf")
        return ret_dict
