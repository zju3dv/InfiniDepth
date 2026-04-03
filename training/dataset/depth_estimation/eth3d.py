import numpy as np
from training.dataset.depth_estimation.depth_estimation_dataset import Dataset as BaseDataset


class Dataset(BaseDataset):

    def read_depth(self, index):
        depth_path = self.depth_files[index]
        if depth_path.endswith(".npz"):
            depth = np.load(depth_path)["data"]
        elif depth_path.endswith(".JPG"):
            rgb = self.read_rgb(index)
            depth = np.fromfile(depth_path, dtype=np.float32).reshape(rgb.shape[0], rgb.shape[1])
        return super().read_depth(index, depth)
