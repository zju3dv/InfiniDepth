from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import re
import shutil
import tempfile
import uuid

import cv2
import numpy as np
import open3d as o3d
import trimesh

from .io_utils import read_depth_array, save_sampled_point_clouds
from .vis_utils import colorize_depth_maps


_DEPTH_EXTENSIONS = {".png", ".npz", ".npy", ".h5", ".hdf5", ".exr"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class ExampleCase:
    name: str
    image_path: str
    depth_path: Optional[str] = None

    @property
    def has_depth(self) -> bool:
        return self.depth_path is not None


@dataclass(frozen=True)
class DemoArtifacts:
    comparison_path: str
    color_depth_path: str
    gray_depth_path: str
    raw_depth_path: str
    ply_path: str
    glb_path: str

    def download_files(self) -> list[str]:
        paths = [
            self.comparison_path,
            self.color_depth_path,
            self.gray_depth_path,
            self.raw_depth_path,
            self.ply_path,
            self.glb_path,
        ]
        return [path for path in paths if os.path.exists(path)]


def scan_example_cases(example_dir: str | Path) -> list[ExampleCase]:
    root = Path(example_dir)
    if not root.exists():
        return []

    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in _DEPTH_EXTENSIONS and ext not in _IMAGE_EXTENSIONS:
            continue

        match = re.match(r"(.+?)_(rgb|depth)$", path.stem)
        if match:
            name, kind = match.groups()
        else:
            name = path.stem
            kind = "image" if ext in _IMAGE_EXTENSIONS else "depth"

        grouped.setdefault(name, {})[kind] = path

    cases: list[ExampleCase] = []
    for name, files in sorted(grouped.items()):
        image_path = files.get("rgb") or files.get("image")
        if image_path is None:
            continue
        depth_path = files.get("depth")
        cases.append(
            ExampleCase(
                name=name,
                image_path=str(image_path.resolve()),
                depth_path=str(depth_path.resolve()) if depth_path is not None else None,
            )
        )
    return cases


def ensure_session_output_dir(app_name: str, session_hash: Optional[str]) -> Path:
    safe_hash = session_hash or f"session-{uuid.uuid4().hex[:10]}"
    output_dir = Path(tempfile.gettempdir()) / app_name / safe_hash
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def _depth_range(depth: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    if not np.any(valid_mask):
        return 0.0, 1.0

    valid_depth = depth[valid_mask]
    lo = float(np.percentile(valid_depth, 1.0))
    hi = float(np.percentile(valid_depth, 99.0))
    if not np.isfinite(lo):
        lo = float(np.min(valid_depth))
    if not np.isfinite(hi):
        hi = float(np.max(valid_depth))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def colorize_depth_array(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = _valid_depth_mask(depth)
    lo, hi = _depth_range(depth, valid_mask)
    return colorize_depth_maps(depth, min_depth=lo, max_depth=hi, cmap="Spectral", valid_mask=valid_mask)


def grayscale_depth_array(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = _valid_depth_mask(depth)
    lo, hi = _depth_range(depth, valid_mask)

    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid_mask):
        scaled = ((depth - lo) / (hi - lo)).clip(0, 1)
        gray = (scaled * 255.0).astype(np.uint8)
        gray[~valid_mask] = 0
    return gray


def build_depth_comparison(image_rgb: np.ndarray, color_depth: np.ndarray, spacer_px: int = 32) -> np.ndarray:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    target_h, target_w = color_depth.shape[:2]
    image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    spacer = np.full((target_h, spacer_px, 3), 255, dtype=np.uint8)
    return np.concatenate([image_rgb, spacer, color_depth], axis=1)


def preview_depth_file(depth_path: Optional[str]) -> tuple[Optional[np.ndarray], str]:
    if not depth_path:
        return None, "No input depth loaded."

    depth = read_depth_array(depth_path)
    preview = colorize_depth_array(depth)
    valid_mask = _valid_depth_mask(depth)
    valid_count = int(valid_mask.sum())
    message = f"Loaded depth: {Path(depth_path).name}"
    if valid_count > 0:
        valid_depth = depth[valid_mask]
        message += (
            f" | shape={depth.shape[0]}x{depth.shape[1]}"
            f" | valid={valid_count}"
            f" | range=[{float(valid_depth.min()):.3f}, {float(valid_depth.max()):.3f}]"
        )
    else:
        message += f" | shape={depth.shape[0]}x{depth.shape[1]} | no valid positive depth values"
    return preview, message


def export_point_cloud_assets(
    sampled_coord,
    sampled_depth,
    rgb_image,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    output_dir: str | Path,
    filter_flying_points: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> tuple[str, str]:
    output_dir = Path(output_dir)
    ply_path = output_dir / "pointcloud.ply"
    glb_path = output_dir / "pointcloud.glb"

    save_sampled_point_clouds(
        sampled_coord=sampled_coord,
        sampled_depth=sampled_depth,
        rgb_image=rgb_image,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        output_path=str(ply_path),
        filter_flying_points=filter_flying_points,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError("Generated point cloud is empty.")

    points = points * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    colors = np.asarray(pcd.colors)

    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(str(ply_path), pcd)

    vertex_colors = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    trimesh.PointCloud(vertices=points, colors=vertex_colors).export(str(glb_path))

    return str(ply_path), str(glb_path)


def save_demo_artifacts(
    image_rgb: np.ndarray,
    pred_depth: np.ndarray,
    output_dir: str | Path,
) -> DemoArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_depth = colorize_depth_array(pred_depth)
    gray_depth = grayscale_depth_array(pred_depth)
    comparison = build_depth_comparison(image_rgb=image_rgb, color_depth=color_depth)

    comparison_path = str(output_dir / "depth_comparison.png")
    color_depth_path = str(output_dir / "depth_color.png")
    gray_depth_path = str(output_dir / "depth_gray.png")
    raw_depth_path = str(output_dir / "raw_depth.npy")

    cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    cv2.imwrite(color_depth_path, cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR))
    cv2.imwrite(gray_depth_path, gray_depth)
    np.save(raw_depth_path, pred_depth.astype(np.float32))

    return DemoArtifacts(
        comparison_path=comparison_path,
        color_depth_path=color_depth_path,
        gray_depth_path=gray_depth_path,
        raw_depth_path=raw_depth_path,
        ply_path=str(output_dir / "pointcloud.ply"),
        glb_path=str(output_dir / "pointcloud.glb"),
    )
