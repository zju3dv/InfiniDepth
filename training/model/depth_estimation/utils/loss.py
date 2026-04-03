import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from training.utils.logger import Log

EPS = 1e-6


def _smooth(err: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    if beta == 0:
        return err
    return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def compute_scale(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = b_0[valid] / (a_00[valid] + 1e-6)

    return x_0


def angle_diff_vec3(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute angular difference between 3D vectors using atan2(||a x b||, a · b).
    """
    cross_norm = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1)
    dot = (a * b).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.atan2(cross_norm + eps, dot)


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return divisor.to(dtype=image_loss.dtype)
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)



def trimmed_mae_loss(
    prediction, target, mask, trim=0.2, reduction=reduction_batch_based, scale=1.0, confidence=None
):
    M = torch.sum(mask, (-2, -1))
    res = prediction - target
    res = res * scale
    if len(res.shape) == 5 and res.shape[2] == 3:
        res = res.norm(dim=2)
    elif len(res.shape) == 5 and res.shape[2] == 1:
        res = res[:, :, 0]
    res = res[mask.bool()].abs()
    if confidence is not None:
        confidence = confidence[mask.bool()]
    else:
        confidence = torch.ones_like(res)
    res = res * confidence - torch.log(confidence)
    if trim == 0.0:  # no trim
        trimmed = res
    else:
        # trimmed, _ = torch.sort(res.view(-1), descending=False)[: int(len(res) * (1.0 - trim))]
        trimmed = torch.sort(res.view(-1), descending=False)[0][: int(len(res) * (1.0 - trim))]
    return reduction(trimmed, M)


def trimmed_mae_loss_no_mask(prediction, target, trim=0.2):
    res = prediction - target

    if trim == 0.0:  # no trim
        trimmed = res
    else:
        trimmed, _ = torch.sort(res.view(-1), descending=False)[: int(len(res) * (1.0 - trim))]
    return trimmed


def trimmed_rmse_loss_depth(prediction, target, mask, trim=0.2, reduction=reduction_batch_based, rmse=False):
    M = torch.sum(mask, (1, 2))
    res = ((prediction - target) ** 2) * target
    # if rmse:
    #     res = res.sqrt() * 0.02
    res = res[mask.bool()].abs()
    if trim == 0.0:  # no trim
        trimmed = res
    else:
        trimmed, _ = torch.sort(res.view(-1), descending=False)[: int(len(res) * (1.0 - trim))]
    loss = reduction(trimmed, M)
    if rmse:
        loss = loss.sqrt()
    return loss


def trimmed_rmse_loss(prediction, target, mask, trim=0.2, reduction=reduction_batch_based, rmse=False):
    M = torch.sum(mask, (1, 2))
    res = (prediction - target) ** 2
    # if rmse:
    #     res = res.sqrt() * 0.02
    res = res[mask.bool()].abs()
    if trim == 0.0:  # no trim
        trimmed = res
    else:
        trimmed, _ = torch.sort(res.view(-1), descending=False)[: int(len(res) * (1.0 - trim))]
    loss = reduction(trimmed, M)
    if rmse:
        loss = loss.sqrt()
    return loss


def normalize_prediction_robust(target, mask, detach=False):
    target = target.to(torch.float32)
    ssum = torch.sum(mask, (-1, -2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    _target = target.clone().detach() if detach else target.clone()
    _target[mask == 0] = torch.nan
    m[valid] = torch.nanmedian(_target[valid].view(valid.sum(), -1), dim=1).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (-1, -2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


def min_max_normalize(target, mask, detach=False):
    ssum = torch.sum(mask, (-1, -2))  # B
    valid = ssum > 0  # B
    B = target.shape[0]

    _target = target.clone().detach() if detach else target.clone()
    _target[mask == 0] = float("-inf")
    target_max = torch.max(_target.view(B, -1), -1).values
    _target = target.clone().detach() if detach else target.clone()
    _target[mask == 0] = float("inf")
    target_min = torch.min(_target.view(B, -1), -1).values

    target[valid] = (target[valid] - target_min[valid].view(-1, 1, 1)) / (
        target_max[valid].view(-1, 1, 1) - target_min[valid].view(-1, 1, 1)
    )
    return target


class TrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()

        self.trim = trim

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        elif reduction == "item-based":
            self.__reduction = reduction_item_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask, scale=1.0, confidence=None):
        # confidence = None
        return trimmed_mae_loss(
            prediction,
            target,
            mask,
            trim=self.trim,
            reduction=self.__reduction,
            scale=scale,
            confidence=confidence,
        )


class SILossWithDepthRMSE(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 0.5,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        rmse_weight: float = 0.0,
        depth_rmse_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile
        self.depth_rmse_weight = depth_rmse_weight
        Log.info(f"SILossWithDepthRMSE: {self.depth_rmse_weight}")

    def forward(self, pred, target, mask):
        if self.normalize == "median":
            target = normalize_median(target, mask)
        elif self.normalize == "percent":
            target = normalize_percentile(target, mask, self.normalize_percentile)
        elif self.normalize == "prewarp":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {self.normalize}")
        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred
        mae_loss = trimmed_mae_loss(pred, target, mask & (target < 10.0), trim=0.0)

        grad_loss = self.gradient_loss(pred, target, mask)
        total_loss = mae_loss + self.gradient_weight * grad_loss

        depth_mask = torch.logical_and(target > 1e-1, mask > 0)
        target_depth = torch.zeros_like(target)
        target_depth[depth_mask] = torch.tensor(1.0).to(dtype=target.dtype, device=target.device) / target[depth_mask]

        pred_depth = torch.zeros_like(pred)
        pred_mask = torch.logical_and(pred > 1e-1, depth_mask > 0)
        pred_depth[pred_mask] = torch.tensor(1.0).to(dtype=pred.dtype, device=pred.device) / pred[pred_mask]
        depth_rmse_loss = trimmed_rmse_loss(pred_depth, target_depth, pred_mask, trim=0.0, rmse=True)

        total_loss = total_loss + self.depth_rmse_weight * depth_rmse_loss

        loss_dict = {
            "mae_loss": mae_loss.detach(),
            "grad_loss": grad_loss.detach(),
            "depth_rmse_loss": depth_rmse_loss.detach(),
        }

        return total_loss, loss_dict


class SILoss(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 0.5,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        rmse_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile

    def forward(self, pred, target, mask):
        if self.normalize == "median":
            target = normalize_median(target, mask)
        elif self.normalize == "percent":
            target = normalize_percentile(target, mask, self.normalize_percentile)
        elif self.normalize == "prewarp":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {self.normalize}")
        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred

        # target[target > 10] = 10.
        # mask = torch.logical_or(mask, target > 10)

        mae_loss = trimmed_mae_loss(pred, target, mask & (target <= 10.0), trim=0.0)

        grad_loss = self.gradient_loss(pred, target, mask)

        total_loss = mae_loss + self.gradient_weight * grad_loss

        loss_dict = {
            "mae_loss": mae_loss.detach(),
            "grad_loss": grad_loss.detach(),
        }

        return total_loss, loss_dict


class SILoss_NoGrad(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 0.5,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        rmse_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile

    def forward(self, pred, target, mask):
        if self.normalize == "median":
            target = normalize_median(target, mask)
        elif self.normalize == "percent":
            target = normalize_percentile(target, mask, self.normalize_percentile)
        elif self.normalize == "prewarp":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {self.normalize}")
        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred

        # target[target > 10] = 10.
        # mask = torch.logical_or(mask, target > 10)

        mae_loss = trimmed_mae_loss(pred, target, mask & (target <= 10.0), trim=0.0)

        total_loss = mae_loss

        loss_dict = {
            "mae_loss": mae_loss.detach(),
        }

        return total_loss, loss_dict
    

class TrimmedProcrustesLoss(nn.Module):
    def __init__(
        self, trim=0.2, reduction="batch-based", normalize="None"
    ):
        super().__init__()

        self.__data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)

        self.__normalize = normalize

        self.__prediction_ssi = None

    def forward(
        self,
        prediction_dict: dict,
        target: torch.tensor,
        mask: torch.Tensor,
        batch: dict,
        **kwargs,
    ):
        assert "depth" in prediction_dict
        pred_depth = prediction_dict["depth"]
        assert pred_depth.ndim == 5 and pred_depth.shape[2] == 1

        pred_depth = pred_depth.flatten(0, 1).squeeze(1)
        target = target.flatten(0, 1).squeeze(1)
        mask = mask.flatten(0, 1).squeeze(1)

        if self.__normalize == "min_max":
            # NOTE: we only normalize target, we do NOT normalize prediction
            target = min_max_normalize(target, mask)

        self.__prediction_ssi = normalize_prediction_robust(pred_depth, mask, detach=False)
        target_ = normalize_prediction_robust(target, mask)

        info = {
            "norm_pred_min": self.__prediction_ssi.min().item(),
            "norm_pred_max": self.__prediction_ssi.max().item(),
            "norm_target_min": target_.min().item(),
            "norm_target_max": target_.max().item(),
        }

        total = self.__data_loss(self.__prediction_ssi, target_, mask)

        return total, info

    def __get_prediction_ssi(self):
        return self.__prediction_ssi


class MAELoss(nn.Module):
    def __init__(
        self,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile

    def forward(self, pred, target):
        mae_loss = trimmed_mae_loss(pred, target, target <= 10.0, trim=0.0)
        total_loss = mae_loss
        loss_dict = {
            "mae_loss": mae_loss.detach(),
        }

        return total_loss, loss_dict
    
    
class MAEPatchGradientLoss(nn.Module):
    def __init__(
        self,
        mae_weight: float = 1.0,
        grad_weight: float = 0.1,
        edge_aware_weight: float = 0.05,
        use_edge_aware: bool = True,
        edge_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.mae_weight = mae_weight
        self.grad_weight = grad_weight
        self.edge_aware_weight = edge_aware_weight
        self.use_edge_aware = use_edge_aware
        self.edge_threshold = edge_threshold
        Log.info(f"MAEPatchGradientLoss: mae_weight={mae_weight}, grad_weight={grad_weight}, "
                 f"edge_aware_weight={edge_aware_weight}, use_edge_aware={use_edge_aware}")

    def forward(self, pred, target, validity_mask=None, patch_info=None):
        B = pred.shape[0]

        if validity_mask is not None:
            valid_pred = pred[validity_mask.unsqueeze(-1).expand_as(pred)].view(-1)
            valid_target = target[validity_mask.unsqueeze(-1).expand_as(target)].view(-1)
            if valid_pred.numel() > 0:
                mae_loss = torch.abs(valid_pred - valid_target).mean()
            else:
                mae_loss = torch.tensor(0.0, device=pred.device)
        else:
            mae_loss = torch.abs(pred - target).mean()

        total_loss = self.mae_weight * mae_loss
        loss_dict = {"mae_loss": mae_loss.detach()}

        if patch_info is not None and len(patch_info.get('patch_indices', [])) > 0:
            grad_loss = self._compute_patch_gradient_loss(
                pred, target, validity_mask, patch_info, B
            )

            total_loss = total_loss + self.grad_weight * grad_loss
            loss_dict["grad_loss"] = grad_loss.detach()

        return total_loss, loss_dict

    def _compute_patch_gradient_loss(self, pred, target, validity_mask, patch_info, batch_size):
        num_patches_info = patch_info.get('num_patches', len(patch_info.get('patch_indices', [])))
        patch_pixel_count_info = patch_info.get('patch_pixel_count', None)

        is_num_patches_tensor = isinstance(num_patches_info, torch.Tensor)
        is_patch_pixel_count_tensor = isinstance(patch_pixel_count_info, torch.Tensor)

        if is_num_patches_tensor:
            num_patches = num_patches_info[0].item()
        else:
            num_patches = num_patches_info

        if is_patch_pixel_count_tensor:
            patch_pixel_count = patch_pixel_count_info[0].item()
        else:
            patch_pixel_count = patch_pixel_count_info

        patch_size = int(np.sqrt(patch_pixel_count))

        if pred.dim() == 3 and pred.shape[2] == 1:
            pred = pred.squeeze(-1)  # [B, N]
            target = target.squeeze(-1)  # [B, N]

        if validity_mask is None:
            validity_mask = torch.ones(pred.shape[0], pred.shape[1], dtype=torch.bool, device=pred.device)

        pred_patches = pred.reshape(batch_size, num_patches, patch_size, patch_size)
        target_patches = target.reshape(batch_size, num_patches, patch_size, patch_size)
        valid_patches = validity_mask.reshape(batch_size, num_patches, patch_size, patch_size)

        diff_patches = pred_patches - target_patches  # [B, num_patches, patch_size, patch_size]
        diff_patches = diff_patches * valid_patches.float()

        grad_x = torch.abs(diff_patches[:, :, :, 1:] - diff_patches[:, :, :, :-1])
        mask_x = valid_patches[:, :, :, 1:] & valid_patches[:, :, :, :-1]
        grad_x = grad_x * mask_x.float()

        grad_y = torch.abs(diff_patches[:, :, 1:, :] - diff_patches[:, :, :-1, :])
        mask_y = valid_patches[:, :, 1:, :] & valid_patches[:, :, :-1, :]
        grad_y = grad_y * mask_y.float()

        total_grad_sum = grad_x.sum() + grad_y.sum()
        total_valid_count = mask_x.float().sum() + mask_y.float().sum()

        # batch-based reduction
        if total_valid_count > 0:
            grad_loss = total_grad_sum / total_valid_count
        else:
            grad_loss = torch.tensor(0.0, device=pred.device)

        return grad_loss


class MAEHybridLoss(nn.Module):
    
    def __init__(
        self,
        mae_weight: float = 1.0,
        grad_weight: float = 0.5,
        normal_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.mae_weight = mae_weight
        self.grad_weight = grad_weight
        self.normal_weight = normal_weight
        Log.info(
            f"MAEHybridLoss: mae_weight={mae_weight}, grad_weight={grad_weight}, normal_weight={normal_weight}"
        )

    def forward(
        self,
        global_pred,
        global_target,
        global_mask=None,
        local_pred=None,
        local_target=None,
        local_mask=None,
        patch_info=None,
    ):
        B = global_pred.shape[0]
        device = global_pred.device

        if local_pred is not None and local_target is not None:
            all_pred = torch.cat([global_pred, local_pred], dim=1)  # [B, N_global + N_local, 1]
            all_target = torch.cat([global_target, local_target], dim=1)
            if global_mask is not None and local_mask is not None:
                all_mask = torch.cat([global_mask, local_mask], dim=1)
            elif global_mask is not None:
                all_mask = torch.cat([global_mask, torch.ones(B, local_pred.shape[1], dtype=torch.bool, device=device)], dim=1)
            elif local_mask is not None:
                all_mask = torch.cat([torch.ones(B, global_pred.shape[1], dtype=torch.bool, device=device), local_mask], dim=1)
            else:
                all_mask = None
        else:
            all_pred = global_pred
            all_target = global_target
            all_mask = global_mask

        mae_loss = self._compute_mae_loss(all_pred, all_target, all_mask)
        
        total_loss = self.mae_weight * mae_loss
        loss_dict = {"mae_loss": mae_loss.detach()}
        
        if self.grad_weight > 0 and local_pred is not None and local_target is not None and patch_info is not None:
            grad_loss = self._compute_patch_gradient_loss(
                local_pred, local_target, local_mask, patch_info, B
            )
            total_loss = total_loss + self.grad_weight * grad_loss
            loss_dict["grad_loss"] = grad_loss.detach()

        # 3. Patch Normal Loss (MoGe-style)
        if self.normal_weight > 0 and local_pred is not None and local_target is not None and patch_info is not None:
            normal_loss = self._compute_patch_normal_loss(
                local_pred, local_target, local_mask, patch_info, B
            )
            total_loss = total_loss + self.normal_weight * normal_loss
            loss_dict["normal_loss"] = normal_loss.detach()

        return total_loss, loss_dict

    def _parse_patch_layout(self, patch_info):
        """Parse patch layout and return (num_patches, patch_size)."""
        num_patches_info = patch_info.get("num_patches", 0)
        patch_pixel_count_info = patch_info.get("patch_pixel_count", None)
        patch_size_info = patch_info.get("patch_size", None)

        if isinstance(num_patches_info, torch.Tensor):
            num_patches = int(num_patches_info[0].item())
        else:
            num_patches = int(num_patches_info)

        if patch_size_info is not None:
            if isinstance(patch_size_info, torch.Tensor):
                patch_size = int(patch_size_info[0].item())
            else:
                patch_size = int(patch_size_info)
        elif patch_pixel_count_info is not None:
            if isinstance(patch_pixel_count_info, torch.Tensor):
                patch_pixel_count = int(patch_pixel_count_info[0].item())
            else:
                patch_pixel_count = int(patch_pixel_count_info)
            patch_size = int(np.sqrt(patch_pixel_count))
        else:
            return None, None

        if num_patches <= 0 or patch_size <= 1:
            return None, None
        return num_patches, patch_size

    def _compute_mae_loss(self, pred, target, mask):
        if mask is not None:
            valid_pred = pred[mask.unsqueeze(-1).expand_as(pred)].view(-1)
            valid_target = target[mask.unsqueeze(-1).expand_as(target)].view(-1)
            if valid_pred.numel() > 0:
                mae_loss = torch.abs(valid_pred - valid_target).mean()
            else:
                mae_loss = torch.tensor(0.0, device=pred.device)
        else:
            mae_loss = torch.abs(pred - target).mean()
        return mae_loss

    def _compute_patch_gradient_loss(self, pred, target, validity_mask, patch_info, batch_size):
        num_patches, patch_size = self._parse_patch_layout(patch_info)
        if num_patches is None or patch_size is None:
            return torch.tensor(0.0, device=pred.device)

        if pred.dim() == 3 and pred.shape[2] == 1:
            pred = pred.squeeze(-1)  # [B, N]
            target = target.squeeze(-1)  # [B, N]

        if validity_mask is None:
            validity_mask = torch.ones(pred.shape[0], pred.shape[1], dtype=torch.bool, device=pred.device)

        expected_total = num_patches * patch_size * patch_size
        actual_total = pred.shape[1]
        if actual_total != expected_total:
            Log.warn(f"MAEHybridLoss: local_pred shape {pred.shape} cannot be reshaped to "
                       f"[{batch_size}, {num_patches}, {patch_size}, {patch_size}]. "
                       f"Expected {expected_total} points, got {actual_total}. Skipping gradient loss.")
            return torch.tensor(0.0, device=pred.device)

        pred_patches = pred.reshape(batch_size, num_patches, patch_size, patch_size)
        target_patches = target.reshape(batch_size, num_patches, patch_size, patch_size)
        valid_patches = validity_mask.reshape(batch_size, num_patches, patch_size, patch_size)

        diff_patches = pred_patches - target_patches  # [B, num_patches, patch_size, patch_size]
        diff_patches = diff_patches * valid_patches.float()

        grad_x = torch.abs(diff_patches[:, :, :, 1:] - diff_patches[:, :, :, :-1])
        valid_grad_x = valid_patches[:, :, :, 1:] & valid_patches[:, :, :, :-1]
        
        grad_y = torch.abs(diff_patches[:, :, 1:, :] - diff_patches[:, :, :-1, :])
        valid_grad_y = valid_patches[:, :, 1:, :] & valid_patches[:, :, :-1, :]

        valid_grad_x_f = valid_grad_x.float()
        valid_grad_y_f = valid_grad_y.float()

        grad_x_masked = grad_x * valid_grad_x_f
        grad_y_masked = grad_y * valid_grad_y_f

        total_grad_sum = grad_x_masked.sum() + grad_y_masked.sum()
        total_valid_count = valid_grad_x_f.sum() + valid_grad_y_f.sum()

        if total_valid_count > 0:
            grad_loss = total_grad_sum / total_valid_count
        else:
            grad_loss = torch.tensor(0.0, device=pred.device)

        return grad_loss

    def _compute_patch_normal_loss(self, pred, target, validity_mask, patch_info, batch_size):
        """
        MoGe-style normal loss on local patches.
        Use pseudo 3D points: (x, y, depth), then compare patch-face normals.
        """
        num_patches, patch_size = self._parse_patch_layout(patch_info)
        if num_patches is None or patch_size is None:
            return torch.tensor(0.0, device=pred.device)

        if pred.dim() == 3 and pred.shape[2] == 1:
            pred = pred.squeeze(-1)
            target = target.squeeze(-1)

        if validity_mask is None:
            validity_mask = torch.ones(pred.shape[0], pred.shape[1], dtype=torch.bool, device=pred.device)
        else:
            validity_mask = validity_mask.bool()

        expected_total = num_patches * patch_size * patch_size
        actual_total = pred.shape[1]
        if actual_total != expected_total:
            Log.warn(
                f"MAEHybridLoss: local_pred shape {pred.shape} cannot be reshaped to "
                f"[{batch_size}, {num_patches}, {patch_size}, {patch_size}] for normal loss. "
                f"Expected {expected_total} points, got {actual_total}. Skipping normal loss."
            )
            return torch.tensor(0.0, device=pred.device)

        pred_patches = pred.reshape(batch_size, num_patches, patch_size, patch_size)
        target_patches = target.reshape(batch_size, num_patches, patch_size, patch_size)
        valid_patches = validity_mask.reshape(batch_size, num_patches, patch_size, patch_size)

        grid_1d = torch.linspace(-1.0, 1.0, steps=patch_size, device=pred.device, dtype=pred.dtype)
        grid_y, grid_x = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        grid_x = grid_x.reshape(1, 1, patch_size, patch_size).expand_as(pred_patches)
        grid_y = grid_y.reshape(1, 1, patch_size, patch_size).expand_as(pred_patches)

        pred_points = torch.stack([grid_x, grid_y, pred_patches], dim=-1).reshape(-1, patch_size, patch_size, 3)
        gt_points = torch.stack([grid_x, grid_y, target_patches], dim=-1).reshape(-1, patch_size, patch_size, 3)
        patch_mask = valid_patches.reshape(-1, patch_size, patch_size)

        leftup, rightup = pred_points[..., :-1, :-1, :], pred_points[..., :-1, 1:, :]
        leftdown, rightdown = pred_points[..., 1:, :-1, :], pred_points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :]
        gt_leftdown, gt_rightdown = gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup = patch_mask[..., :-1, :-1], patch_mask[..., :-1, 1:]
        mask_leftdown, mask_rightdown = patch_mask[..., 1:, :-1], patch_mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        min_angle = math.radians(1.0)
        max_angle = math.radians(90.0)
        beta_rad = math.radians(3.0)

        normal_err = (
            mask_upxleft.float()
            * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(min_angle, max_angle), beta=beta_rad)
            + mask_leftxdown.float()
            * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(min_angle, max_angle), beta=beta_rad)
            + mask_downxright.float()
            * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(min_angle, max_angle), beta=beta_rad)
            + mask_rightxup.float()
            * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(min_angle, max_angle), beta=beta_rad)
        )

        # Keep the same scaling style as MoGe normal loss.
        normal_loss = normal_err.mean() / (4 * patch_size)
        return normal_loss


class MAE_Gradient_Loss(nn.Module):
    def __init__(
        self,
        normalize: str = "median",
        align_pred: bool = True,
        normal_weight: float = 0.2,
        normalize_percentile: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normal_weight = normal_weight
        self.normalize_percentile = normalize_percentile

    def forward(
        self,
        pred,
        target,                       # same shape as pred
        n_pred=None,
        n_gt=None,
        normal_mask=None,
    ):
        mae_loss = trimmed_mae_loss(pred, target, target <= 10.0, trim=0.0)
        total_loss = mae_loss

        loss_dict = {
            "mae_loss": mae_loss.detach(),
        }

        if self.normal_weight > 0.0 and (n_pred is not None) and (n_gt is not None):
            cos_sim = (n_pred * n_gt).sum(dim=-1).clamp(-1.0, 1.0)   # [B, N]
            normal_loss = 1.0 - cos_sim.abs()                        # [B, N]

            if normal_mask is not None:
                valid = normal_mask.to(normal_loss.dtype)
                denom = valid.sum().clamp(min=1.0)
                normal_loss = (normal_loss * valid).sum() / denom
            else:
                normal_loss = normal_loss.mean()

            total_loss = total_loss + self.normal_weight * normal_loss
            loss_dict["normal_loss"] = normal_loss.detach()
            loss_dict["normal_w"] = torch.tensor(self.normal_weight, device=pred.device, dtype=pred.dtype)

        return total_loss, loss_dict


class MAE_Normal_Loss(nn.Module):
    def __init__(
        self,
        normalize: str = "median",
        align_pred: bool = True,
        normal_weight: float = 0.2,
        normalize_percentile: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normal_weight = normal_weight
        self.normalize_percentile = normalize_percentile

    def forward(
        self,
        pred,                         # [B, N, 1] or [B,1,H,W]
        target,                       # same shape as pred
        n_pred=None,                  # [B, N, 3]
        n_gt=None,                    # [B, N, 3]
        normal_mask=None,             # [B, N] (bool or 0/1)
    ):
        # ---- MAE loss ----
        mae_loss = trimmed_mae_loss(pred, target, target <= 10.0, trim=0.0)
        total_loss = mae_loss

        loss_dict = {
            "mae_loss": mae_loss.detach(),
        }

        if self.normal_weight > 0.0 and (n_pred is not None) and (n_gt is not None):
            eps = 1e-12
            max_deg = 90.0
            beta_deg = 3.0

            # atan2(||v×w||, v·w) 
            cross_norm = torch.linalg.norm(torch.cross(n_pred, n_gt, dim=-1), dim=-1)     # [B, N]
            dot        = (n_pred * n_gt).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)         # [B, N]
            angle = torch.atan2(cross_norm + eps, dot)                                    # [B, N], rad

            angle = torch.clamp_max(
                angle,
                torch.deg2rad(torch.tensor(max_deg, device=angle.device, dtype=angle.dtype))
            )

            beta = torch.deg2rad(torch.tensor(beta_deg, device=angle.device, dtype=angle.dtype))
            normal_loss = torch.sqrt(angle * angle + beta * beta) - beta                      # [B, N]

            if normal_mask is not None:
                valid = normal_mask.to(normal_loss.dtype)
                denom = valid.sum().clamp(min=1.0)
                normal_loss = (normal_loss * valid).sum() / denom
            else:
                normal_loss = normal_loss.mean()

            total_loss = total_loss + self.normal_weight * normal_loss
            loss_dict["normal_loss"] = normal_loss.detach()
            loss_dict["normal_w"] = torch.tensor(self.normal_weight, device=pred.device, dtype=pred.dtype)

        return total_loss, loss_dict


class SILossLog(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 0.5,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        rmse_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile

    def forward(self, pred, target, mask):
        if self.normalize == "median":
            target = normalize_median(target, mask)
        elif self.normalize == "percent":
            target = normalize_percentile(target, mask, self.normalize_percentile)
        elif self.normalize == "prewarp":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {self.normalize}")
        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred

        pred = torch.log(torch.clamp(1 + pred, min=1e-2))
        target = torch.log(torch.clamp(1 + target, min=1e-2))

        mae_loss = trimmed_mae_loss(pred, target, mask & (target < 10.0), trim=0.0)

        grad_loss = self.gradient_loss(pred, target, mask)

        total_loss = mae_loss + self.gradient_weight * grad_loss

        loss_dict = {
            "log_mae_loss": mae_loss.detach(),
            "log_grad_loss": grad_loss.detach(),
        }

        return total_loss, loss_dict


class SILossRMSE_depth(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 0.5,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        rmse_weight: float = 0.0,
        rmse: bool = False,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile
        self.rmse = rmse

    def forward(self, pred, target, mask):
        if self.normalize == "median":
            target = normalize_median(target, mask)
        elif self.normalize == "percent":
            target = normalize_percentile(target, mask, self.normalize_percentile)
        elif self.normalize == "prewarp":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {self.normalize}")
        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred

        mae_loss = trimmed_rmse_loss_depth(pred, target, mask & (target < 10.0), trim=0.0, rmse=self.rmse)

        grad_loss = self.gradient_loss(pred, target, mask)

        total_loss = mae_loss + self.gradient_weight * grad_loss

        loss_dict = {
            "rmse_loss": mae_loss.detach(),
            "grad_loss": grad_loss.detach(),
        }

        return total_loss, loss_dict


class SILossRMSE(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 0.5,
        normalize: str = "median",
        align_pred: bool = True,
        normalize_percentile: float = 0.02,
        rmse_weight: float = 0.0,
        rmse: bool = False,
    ):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred
        self.normalize_percentile = normalize_percentile
        self.rmse = rmse

    def forward(self, pred, target, mask):
        if self.normalize == "median":
            target = normalize_median(target, mask)
        elif self.normalize == "percent":
            target = normalize_percentile(target, mask, self.normalize_percentile)
        elif self.normalize == "prewarp":
            pass
        else:
            raise ValueError(f"Invalid normalize method: {self.normalize}")
        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred

        mae_loss = trimmed_rmse_loss(pred, target, mask & (target < 10.0), trim=0.0, rmse=self.rmse)

        grad_loss = self.gradient_loss(pred, target, mask)

        total_loss = mae_loss + self.gradient_weight * grad_loss

        loss_dict = {
            "rmse_loss": mae_loss.detach(),
            "grad_loss": grad_loss.detach(),
        }

        return total_loss, loss_dict


class SIDispLoss(nn.Module):
    def __init__(self, gradient_weight: float = 0.5, normalize: str = "median", align_pred: bool = True):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.gradient_loss = GradientLoss()
        self.normalize = normalize
        self.align_pred = align_pred

    def forward(self, pred, target, mask):

        # to disparity
        mask = mask & (target > 1e-3)
        target = 1.0 / target

        # normalize
        if self.normalize == "median":
            target = normalize_median(target, mask)


        # scale invariant loss
        if self.align_pred:
            scale = compute_scale(
                rearrange(pred, "b 1 h w -> b h w"),
                rearrange(target, "b 1 h w -> b h w"),
                rearrange(mask, "b 1 h w -> b h w"),
            )
            pred = scale.view(-1, 1, 1, 1) * pred

        mae_loss = trimmed_mae_loss(pred, target, mask & (target < 10.0), trim=0.0)

        grad_loss = self.gradient_loss(pred, target, mask)

        total_loss = mae_loss + self.gradient_weight * grad_loss

        loss_dict = {
            "mae_loss": mae_loss.detach(),
            "grad_loss": grad_loss.detach(),
        }

        return total_loss, loss_dict


def normalize_median(depth, mask):
    batch_size = depth.shape[0]
    medians = []
    for b in range(batch_size):
        median = torch.median(depth[b][mask[b] != 0])
        medians.append(median)
    medians = torch.stack(medians)
    depth = depth / rearrange(medians, "b -> b 1 1 1")
    return depth


def get_median_depth(depth, mask):
    batch_size = depth.shape[0]
    medians = []
    for b in range(batch_size):
        median = torch.median(depth[b][mask[b] != 0])
        medians.append(median)
    medians = torch.stack(medians)
    return medians


def normalize_percentile(depth, mask, percentile=0.02):
    batch_size = depth.shape[0]
    min_vals, max_vals = [], []
    for b in range(batch_size):
        min_val = torch.quantile(depth[b][mask[b] != 0], percentile)
        max_val = torch.quantile(depth[b][mask[b] != 0], 1 - percentile)
        if min_val == max_val:
            max_val = min_val + EPS
        min_vals.append(min_val)
        max_vals.append(max_val)
    min_vals = rearrange(torch.stack(min_vals), "b -> b 1 1 1")
    max_vals = rearrange(torch.stack(max_vals), "b -> b 1 1 1")
    depth = (depth - min_vals) / (max_vals - min_vals)
    return depth


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()

        if reduction == "batch-based":
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self.__reduction,
            )

        return total


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)
