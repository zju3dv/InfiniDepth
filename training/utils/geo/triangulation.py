import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

# ========== Simple triangulation ==========


def triangulate_persp(Ts_w2c, c_p2d, **kwargs):
    """
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4)
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    # Move N to B
    B, V, N, _ = c_p2d.shape
    c_p2d = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)

    # Create A matrix
    A = []
    for v in range(V):
        P = Ts_w2c[:, v]  # (BN, 4, 4) Projection matrix for view v
        x = c_p2d[:, v]  # (BN, 2)
        # Construct the 2 rows per view
        row1 = x[:, [0]] * P[:, 2] - P[:, 0]
        row2 = x[:, [1]] * P[:, 2] - P[:, 1]
        A.append(row1)
        A.append(row2)
    A = torch.stack(A, dim=1)  # (B, 2*V, 4)

    # Solve for X
    _, _, Vh = torch.linalg.svd(A)
    X_homogeneous = Vh[:, -1, :]

    # Convert back to inhomogeneous coordinates
    w_p3d = X_homogeneous[:, :3] / X_homogeneous[:, 3:4]

    # Convert back to B, N, 3
    w_p3d = rearrange(w_p3d, "(b n) c -> b n c", b=B)

    return w_p3d


def triangulate_ortho(Ts_w2c, c_p2d, **kwargs):
    """
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4)
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    # Move N to B
    B, V, N, _ = c_p2d.shape
    c_p2d = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)

    # Create A matrix
    A = []
    b = []
    for v in range(V):
        P = Ts_w2c[:, v]  # (BN, 4, 4) Projection matrix for view v
        x = c_p2d[:, v]  # (BN, 2)
        # Construct the 2 rows per view
        row1 = P[:, 0, :-1]  # (BN, 3)
        row2 = P[:, 1, :-1]  # (BN, 3)
        A.append(row1)
        A.append(row2)
        b_ = -P[:, 0, -1:] + x[:, [0]]  # (BN, 1)
        b.append(b_)
        b_ = -P[:, 1, -1:] + x[:, [1]]  # (BN, 1)
        b.append(b_)
    A = torch.stack(A, dim=1)  # (BN, 2*V, 3)
    b = torch.stack(b, dim=1)  # (BN, 2*V, 1)

    # Solve for X
    w_p3d = torch.linalg.lstsq(A, b).solution  # (B*N, 3, 1)

    # Convert back to B, N, 3
    w_p3d = rearrange(w_p3d[..., 0], "(b n) c -> b n c", b=B)

    return w_p3d


def triangulate_ortho_3d(Ts_w2c, c_p2d, w_p3d_, **kwargs):
    """
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4)
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
        w_p3d torch.Tensor:  (B, V - 1, N, 3), N indicates different points
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    # Move N to B
    B, V, N, _ = c_p2d.shape
    c_p2d = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)
    w_p3d_ = rearrange(w_p3d_, "b v n c -> (b n) v c")

    # Create A matrix
    A = []
    b = []
    for v in range(V):
        P = Ts_w2c[:, v]  # (BN, 4, 4) Projection matrix for view v
        x = c_p2d[:, v]  # (BN, 2)
        # Construct the 2 rows per view
        row1 = P[:, 0, :-1]  # (BN, 3)
        row2 = P[:, 1, :-1]  # (BN, 3)
        A.append(row1)
        A.append(row2)
        b_ = -P[:, 0, -1:] + x[:, [0]]  # (BN, 1)
        b.append(b_)
        b_ = -P[:, 1, -1:] + x[:, [1]]  # (BN, 1)
        b.append(b_)
        if v == V - 1:
            # in LRM, condition view does not have predicted 3d points
            continue
        P_3d = w_p3d_[:, v]
        row1_3d = torch.zeros([B * N, 3], device=w_p3d_.device)
        row1_3d[:, 0] = 1.0
        row2_3d = torch.zeros([B * N, 3], device=w_p3d_.device)
        row2_3d[:, 1] = 1.0
        row3_3d = torch.zeros([B * N, 3], device=w_p3d_.device)
        row3_3d[:, 2] = 1.0
        A.append(row1_3d)
        A.append(row2_3d)
        A.append(row3_3d)
        b.append(P_3d[:, [0]])
        b.append(P_3d[:, [1]])
        b.append(P_3d[:, [2]])
    A = torch.stack(A, dim=1)  # (BN, 2*V + 3, 3)
    b = torch.stack(b, dim=1)  # (BN, 2*V + 3, 1)

    # Solve for X
    w_p3d = torch.linalg.lstsq(A, b).solution  # (B*N, 3, 1)

    # Convert back to B, N, 3
    w_p3d = rearrange(w_p3d[..., 0], "(b n) c -> b n c", b=B)

    return w_p3d


def triangulate_2d_3d(Ts_w2c, c_p2d, mode, w_p3d=None, weight_2d=1.0, weight_3d=1.0, **kwargs):
    """
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4), 2D
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points, 2D
        mode: "ortho" or "persp", it could also be a list of str that indicates the mode for each view
        w_p3d torch.Tensor: (B, W, N, 3), 3D. W is semantically identical to V, but it is not the same.
        weight_2d float or list of float
        weight_3d float or list of float
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    B, V, N, _ = c_p2d.shape

    # Set up 2D view projection mode
    if isinstance(mode, str):
        mode = [mode] * V
    assert len(mode) == V, "The length of mode should be the same as V."

    # Set up weight for each view
    if isinstance(weight_2d, float) or isinstance(weight_2d, int) or len(weight_2d.shape) == 0:
        weight_2d = [float(weight_2d)] * V
    assert len(weight_2d) == V, "The length of weight_2d should be the same as V."
    weights = []
    for w in weight_2d:
        weights.extend([w] * 2)

    # If w_p3d is also provided
    if w_p3d is not None:
        # Check
        B2, W, N2, _ = w_p3d.shape
        assert B == B2 and N == N2, "Batch size and number of points should be the same."

        # Set up weights
        if isinstance(weight_3d, float) or isinstance(weight_3d, int):
            weight_3d = [float(weight_3d)] * W
        assert len(weight_3d) == W, "The length of weight_3d should be the same as W."
        for w in weight_3d:
            weights.extend([w] * 3)

        # Move N to B as (BN)
        w_p3d = rearrange(w_p3d, "b v n c -> (b n) v c")

    # Move N to B as (BN)
    c_p2d = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)
    weights = torch.tensor(weights, device=c_p2d.device)

    # Create A b matrix
    A = []
    b = []

    # Add 2D observation
    for v in range(V):
        P = Ts_w2c[:, v]  # (BN, 4, 4) Projection matrix for view v
        x = c_p2d[:, v]  # (BN, 2)

        # Construct the 2 rows per view
        if mode[v] == "ortho":
            # R(1) * [X, Y, Z] = x - t(1)
            A.append(P[:, 0, :3])
            b.append(x[:, [0]] - P[:, 0, -1:])
            # R(2) * [X, Y, Z] = y - t(2)
            A.append(P[:, 1, :3])
            b.append(x[:, [1]] - P[:, 1, -1:])
        else:
            assert mode[v] == "persp"
            # (R(1) - x * R(3)) * [X, Y, Z] = x * t(3) - t(1)
            A.append(P[:, 0, :3] - x[:, [0]] * P[:, 2, :3])
            b.append(x[:, [0]] * P[:, 2, -1:] - P[:, 0, -1:])
            # (R(2) - x * R(3)) * [X, Y, Z] = x * t(3) - t(2)
            A.append(P[:, 1, :3] - x[:, [1]] * P[:, 2, :3])
            b.append(x[:, [1]] * P[:, 2, -1:] - P[:, 1, -1:])

    # Add 3D observation
    if w_p3d is not None:
        I = torch.eye(3, device=w_p3d.device).unsqueeze(0).repeat(B * N, 1, 1)
        for v in range(W):
            A.append(I[:, 0])
            A.append(I[:, 1])
            A.append(I[:, 2])
            b.append(w_p3d[:, v, [0]])
            b.append(w_p3d[:, v, [1]])
            b.append(w_p3d[:, v, [2]])

    A = torch.stack(A, dim=1)  # (BN, 2*V + 3*W, 3)
    b = torch.stack(b, dim=1)  # (BN, 2*V + 3*W, 1)

    A = A * weights[None:, None]
    b = b * weights[None:, None]

    # Solve for X
    w_p3d = torch.linalg.lstsq(A, b).solution  # (B*N, 3, 1)

    # Convert back to B, N, 3
    w_p3d = rearrange(w_p3d[..., 0], "(b n) c -> b n c", b=B)

    return w_p3d


# ========== Optimization-based triangulation ==========


@torch.inference_mode(mode=False)
def triangulate_optim(c_p2d, w_p3d, project_func, max_iter=100, lr=0.1, **kwargs):
    """
    Args:
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
        w_p3d torch.Tensor:  (B, N, 3), N indicates different points
        project_func func
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    optim_param = [w_p3d.clone().requires_grad_(True)]
    optimizer = torch.optim.AdamW(optim_param, lr=lr)
    for i in range(max_iter):
        c_p2d_ = project_func(optim_param[0])
        loss = F.mse_loss(c_p2d_, c_p2d.clone())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{i} loss: {loss.item()}")
    # use detach to release computation graph
    return optim_param[0].detach()


# ========== Special triangulation (model w3d from condition view as lp+a*ld) ========== #


def triangulate_ortho_c1v(Ts_w2c, c_p2d, controled_viewid=0):
    """
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4)
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
        controled_viewid: int, default=0
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    B, V, N, _ = c_p2d.shape

    # In the controled view: line_point + line_dir * x = point
    c_p2d_controled_view = c_p2d[:, controled_viewid]  # (B, N, 2)
    c_lp = F.pad(c_p2d_controled_view, (0, 1), value=1.0)  # (B, N, 3)
    c_ld = torch.zeros_like(c_lp)
    c_ld[..., 2] = 1  # (B, N, 3)

    Ts_c2w_ = torch.inverse(Ts_w2c[:, controled_viewid])  # (B, 4, 4)
    R_c2w_ = Ts_c2w_[:, :3, :3]  # (B, 3, 3)
    t_c2w_ = Ts_c2w_[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
    w_lp = einsum(R_c2w_, c_lp, "b c d, b n d -> b n c") + t_c2w_  # (B, N, 3)
    w_ld = einsum(R_c2w_, c_ld, "b c d, b n d -> b n c")

    # Move N to B as (BN)
    c_p2d_ = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c_ = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)
    w_lp_ = rearrange(w_lp, "b n c -> (b n) c")
    w_ld_ = rearrange(w_ld, "b n c -> (b n) c")

    # Create A matrix
    A = []
    b = []

    # We use one view as the control view, and triangulate the rest views
    rest_views = list(range(V))
    rest_views.remove(controled_viewid)
    for v in rest_views:
        Rld = einsum(Ts_w2c_[:, v, :3, :3], w_ld_, "b c d, b d -> b c")
        Rlp = einsum(Ts_w2c_[:, v, :3, :3], w_lp_, "b c d, b d -> b c")
        b_ = c_p2d_[:, v, :] - Rlp[:, :2] - Ts_w2c_[:, v, :2, 3]  # (BN, 2)
        A.append(Rld[:, [0]])
        A.append(Rld[:, [1]])
        b.append(b_[:, [0]])
        b.append(b_[:, [1]])

    A = torch.stack(A, dim=1)  # (BN, 2*(V-1), 1)
    b = torch.stack(b, dim=1)  # (BN, 2*(V-1), 1)

    # Solve for x
    x = torch.linalg.lstsq(A, b).solution.squeeze(2)  # (BN, 1)
    w_p3d = w_lp + w_ld * rearrange(x, "(b n) c -> b n c", b=B, n=N)  # (B, N, 3)
    return w_p3d


def triangulate_ortho_c1v(Ts_w2c, c_p2d, Ts_w2c_controled_view, c_p2d_controled_view):
    """The controled_view is provided explicitly as input
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4)
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
        Ts_w2c_controled_view torch.Tensor: (B, 4, 4)
        c_p2d_controled_view torch.Tensor: (B, N, 2)
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    B, V, N, _ = c_p2d.shape

    # In the controled view: line_point + line_dir * x = point
    c_lp = F.pad(c_p2d_controled_view, (0, 1), value=1.0)  # (B, N, 3)
    c_ld = torch.zeros_like(c_lp)
    c_ld[..., 2] = 1  # (B, N, 3)

    Ts_c2w_ = torch.inverse(Ts_w2c_controled_view)  # (B, 4, 4)
    R_c2w_ = Ts_c2w_[:, :3, :3]  # (B, 3, 3)
    t_c2w_ = Ts_c2w_[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
    w_lp = einsum(R_c2w_, c_lp, "b c d, b n d -> b n c") + t_c2w_  # (B, N, 3)
    w_ld = einsum(R_c2w_, c_ld, "b c d, b n d -> b n c")

    # Move N to B as (BN)
    c_p2d_ = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c_ = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)
    w_lp_ = rearrange(w_lp, "b n c -> (b n) c")
    w_ld_ = rearrange(w_ld, "b n c -> (b n) c")

    # Create A matrix
    A = []
    b = []

    # We use one view as the control view, and triangulate the rest views
    for v in range(V):
        Rld = einsum(Ts_w2c_[:, v, :3, :3], w_ld_, "b c d, b d -> b c")
        Rlp = einsum(Ts_w2c_[:, v, :3, :3], w_lp_, "b c d, b d -> b c")
        b_ = c_p2d_[:, v, :] - Rlp[:, :2] - Ts_w2c_[:, v, :2, 3]  # (BN, 2)
        A.append(Rld[:, [0]])
        A.append(Rld[:, [1]])
        b.append(b_[:, [0]])
        b.append(b_[:, [1]])

    A = torch.stack(A, dim=1)  # (BN, 2*(V-1), 1)
    b = torch.stack(b, dim=1)  # (BN, 2*(V-1), 1)

    # Solve for x
    x = torch.linalg.lstsq(A, b).solution.squeeze(2)  # (BN, 1)
    w_p3d = w_lp + w_ld * rearrange(x, "(b n) c -> b n c", b=B, n=N)  # (B, N, 3)
    return w_p3d


def triangulate_c1v(Ts_w2c, c_p2d, T_w2c_cv, c_p2d_cv, mode="ortho", mode_cv="persp"):
    """The controled_view is provided explicitly as input
    Args:
        Ts_w2c torch.Tensor: (B, V, 4, 4)
        c_p2d torch.Tensor:  (B, V, N, 2), N indicates different points
        mode: "ortho" or "persp"
        T_w2c_cv torch.Tensor: (B, 4, 4), controled view
        c_p2d_cv torch.Tensor: (B, N, 2), controled view
        mode_cv: "ortho" or "persp"
    Returns:
        w_p3d torch.Tensor: (B, N, 3)
    """
    B, V, N, _ = c_p2d.shape

    # In the controled view: line_point + line_dir * x = point
    c_lp = F.pad(c_p2d_cv, (0, 1), value=1.0)  # (B, N, 3)
    if mode_cv == "ortho":
        c_ld = torch.zeros_like(c_lp)
        c_ld[..., 2] = 1  # (B, N, 3)
    else:
        assert mode_cv == "persp"
        c_ld = c_lp.clone()  # (B, N, 3)

    T_c2w_ = torch.inverse(T_w2c_cv)  # (B, 4, 4)
    R_c2w_ = T_c2w_[:, :3, :3]  # (B, 3, 3)
    t_c2w_ = T_c2w_[:, :3, 3].unsqueeze(1)  # (B, 1, 3)
    w_lp = einsum(R_c2w_, c_lp, "b c d, b n d -> b n c") + t_c2w_  # (B, N, 3)
    w_ld = einsum(R_c2w_, c_ld, "b c d, b n d -> b n c")

    # Move N to B as (BN)
    c_p2d_ = rearrange(c_p2d, "b v n c -> (b n) v c")
    Ts_w2c_ = repeat(Ts_w2c, "b v c d -> (b n) v c d", n=N)
    w_lp_ = rearrange(w_lp, "b n c -> (b n) c")
    w_ld_ = rearrange(w_ld, "b n c -> (b n) c")

    # Create A matrix
    A = []
    b = []

    # We use one view as the control view, and triangulate the rest views
    for v in range(V):
        Rld = einsum(Ts_w2c_[:, v, :3, :3], w_ld_, "b c d, b d -> b c")
        Rlp = einsum(Ts_w2c_[:, v, :3, :3], w_lp_, "b c d, b d -> b c")

        if mode == "ortho":
            b_ = c_p2d_[:, v, :] - Rlp[:, :2] - Ts_w2c_[:, v, :2, 3]  # (BN, 2)
        else:
            assert mode == "persp"
            raise NotImplementedError

        A.append(Rld[:, [0]])
        A.append(Rld[:, [1]])
        b.append(b_[:, [0]])
        b.append(b_[:, [1]])

    A = torch.stack(A, dim=1)  # (BN, 2*(V-1), 1)
    b = torch.stack(b, dim=1)  # (BN, 2*(V-1), 1)

    # Solve for x
    x = torch.linalg.lstsq(A, b).solution.squeeze(2)  # (BN, 1)
    w_p3d = w_lp + w_ld * rearrange(x, "(b n) c -> b n c", b=B, n=N)  # (B, N, 3)
    return w_p3d
