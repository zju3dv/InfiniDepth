import cv2
from training.dataset.depth_estimation.depth_estimation_dataset import Dataset as BaseDataset


class Dataset(BaseDataset):
    def read_rgb(self, index):
        rgb = super().read_rgb(index)
        return rgb[9:-9, 12:-12]

    def read_depth(self, index):
        depth_path = self.depth_files[index]
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 1000.0
        depth = depth[9:-9, 12:-12]
        return super().read_depth(index, depth)
