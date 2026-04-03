import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from pathlib import Path
from training.dataset.depth_estimation.depth_estimation_dataset import Dataset as BaseDataset


class Dataset(BaseDataset):
    def read_depth(self, index, depth=None, highfreq_mask_suffix="_highfreq_mask"):
        depth_path = self.depth_files[index]
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) 
        depth, mask, disparity, disparity_mask = super().read_depth(index, depth)
        
        mask[depth > 150] = 0
        mask[depth < 3] = 0   # avoid ood
        disparity_mask[depth > 150] = 0  # mask sky when inference
        disparity_mask[depth < 3] = 0    # avoid ood

        scene_dir = Path(depth_path).parent.parent
        if self.num_pts == None:
            highfreq_mask_dir = scene_dir / "highfreq_mask"
        else:
            highfreq_mask_dir = scene_dir / f"highfreq_mask_pts_{self.num_pts}"
        highfreq_mask_path = str(highfreq_mask_dir / f"{Path(depth_path).stem}{highfreq_mask_suffix}.png")

        if not os.path.exists(highfreq_mask_path):
            highfreq_mask = None
        else:
            highfreq_mask = cv2.imread(highfreq_mask_path, cv2.IMREAD_GRAYSCALE)

        return depth, mask, disparity, disparity_mask, highfreq_mask