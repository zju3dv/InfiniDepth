import random
from torch.utils.data import BatchSampler


class ParamBatchSampler(BatchSampler):
    """
    Poetry in code: A sampler that yields batches with dynamic image and view sizes.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last: bool = False,
        shuffle: bool = True,
        landscape_prob: float = 0.8,
        fix_size: int = 504,
        image_size_range: tuple = (504, 909),
        min_view_size: int = 2,
        constant_size: int = 504 * 504 * 6,
    ):
        super().__init__(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
        self.dataset = dataset
        self.landscape_prob = landscape_prob  # Probability to sample landscape orientation
        self.fix_size = fix_size  # Fixed size for one dimension
        self.image_size_range = image_size_range  # Range for random image size
        self.min_view_size = min_view_size  # Minimum number of views
        self.constant_size = constant_size  # Constant for controlling view/scene size

    def set_epoch(self, epoch: int):
        # Set random seed for reproducibility per epoch
        self.dist_sampler.set_epoch(epoch)
        random.seed(epoch)

    def __iter__(self):
        """
        For each batch, randomly determine image size and view/scene size,
        yielding a poetic batch of indices and their parameters.
        """
        for batch_indices in super().__iter__():
            # Randomly select a size, aligned to 14
            size = random.randint(self.image_size_range[0], self.image_size_range[1]) // 14 * 14
            # Decide orientation: landscape or portrait
            if random.random() < self.landscape_prob:
                height, width = min(size, self.fix_size), max(size, self.fix_size)
            else:
                height, width = max(size, self.fix_size), min(size, self.fix_size)
            # Calculate max view size and sample view size
            max_view = self.constant_size // (height * width)
            view_size = random.randint(self.min_view_size, max(max_view, self.min_view_size))
            # Calculate scene size
            scene_size = self.constant_size // (view_size * height * width)
            # Yield poetic batch: index and its parameters
            yield [
                (idx, {"height": height, "width": width, "view_size": view_size, "scene_size": scene_size})
                for idx in batch_indices
            ]
