import torch


# @torch.jit.script
def affine_padding(mat: torch.Tensor):
    """ 1. Pad the last row of a 3x4 mat or w2c matrix to 4x4.
        2. Pad the a 3x3 intrinsics matrix to 4x4.
    """
    # Already padded
    if mat.shape[-2] == 4:
        return mat

    # Batch agnostic padding for 3x3 intrinsics to 3x4
    if mat.shape[-1] == 3:
        pad = mat.new_zeros(mat.shape[:-1] + (1,))  # (..., 3, 1)
        mat = torch.cat([mat, pad], dim=-1)  # (..., 3, 4)

    # Batch agnostic padding
    sh = mat.shape
    pad0 = mat.new_zeros(sh[:-2] + (1, 3))  # (..., 1, 3)
    pad1 = mat.new_ones(sh[:-2] + (1, 1))  # (..., 1, 1)
    pad = torch.cat([pad0, pad1], dim=-1)  # (..., 1, 4)
    mat = torch.cat([mat, pad], dim=-2)  # (..., 4, 4)

    return mat


# @torch.jit.script
def affine_inverse(A: torch.Tensor):
    """ Pad and inverse the input matrix
    """
    R = A[..., :3, :3]  # (..., 3, 3)
    T = A[..., :3, 3:]  # (..., 3, 1)
    P = A[..., 3:, :]  # (..., 1, 4)

    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)  # (..., 4, 4)
