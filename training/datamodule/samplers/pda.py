import random
from torch.utils.data import BatchSampler


class ParamBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        drop_last: bool = False,
        shuffle: bool = True,
        image_size_range: tuple = (504, 896 + 13),
        patch_size: int = 14,  # 14 or 16
    ):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size=batch_size, drop_last=drop_last)
        self.image_size_range = image_size_range
        self.patch_size = patch_size
        self.anchor_size = 504 if patch_size == 14 else 512

    def set_epoch(self, epoch: int):
        self.dist_sampler.set_epoch(epoch)
        random.seed(epoch)

    def __iter__(self):
        for batch_indices in super().__iter__():
            size = random.randint(self.image_size_range[0], self.image_size_range[1]) // self.patch_size * self.patch_size
            if random.random() < 0.8:
                height = min(size, self.anchor_size)
                width = max(size, self.anchor_size)
            else:
                height = max(size, self.anchor_size)
                width = min(size, self.anchor_size)
            yield [(idx, {"height": height, "width": width}) for idx in batch_indices]

