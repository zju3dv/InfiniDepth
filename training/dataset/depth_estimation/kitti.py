import cv2
from training.dataset.depth_estimation.depth_estimation_dataset import Dataset as BaseDataset


class Dataset(BaseDataset):
    def read_depth(self, index, depth=None):
        depth = cv2.imread(self.depth_files[index], cv2.IMREAD_ANYDEPTH) / 256.0
        depth, mask, disparity, disparity_mask = super().read_depth(index, depth)
        return depth, mask, disparity, disparity_mask