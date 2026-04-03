import numpy as np
from training.dataset.depth_estimation.depth_estimation_dataset import Dataset as BaseDataset
from scipy import ndimage


class Dataset(BaseDataset):
    def read_depth(self, index, depth=None):
        depth_path = self.depth_files[index]
        depth = np.load(depth_path)[:, :, 0]
        valid_mask = np.load(depth_path.replace(".npy", "_mask.npy"))
        valid_mask = (valid_mask == 1) & (depth >= 0.6) & (depth <= 350)
        dx = ndimage.sobel(depth, 0)  # horizontal derivative
        dy = ndimage.sobel(depth, 1)  # vertical derivative
        grad = np.abs(dx) + np.abs(dy)
        valid_mask[grad > 0.3] = 0
        depth[valid_mask == 0] = 0
        return super().read_depth(index, depth)
