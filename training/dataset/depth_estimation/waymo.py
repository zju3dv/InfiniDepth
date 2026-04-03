import numpy as np
from training.dataset.depth_estimation.depth_estimation_dataset import Dataset as BaseDataset


class Dataset(BaseDataset):

    def read_depth(self, index, depth=None):
        depth_dict = np.load(self.depth_files[index], allow_pickle=True).item()
        depth = np.zeros_like(depth_dict["mask"]).astype(np.float32)
        depth[depth_dict["mask"]] = depth_dict["value"]
        return super().read_depth(index, depth)
