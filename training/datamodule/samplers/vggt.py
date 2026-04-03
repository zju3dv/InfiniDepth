import random
import torch.distributed as dist
from torch.utils.data import Sampler


class ParamBatchSampler(Sampler):
    def __init__(self, dataset, world_size, rank):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.batch_idx = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        random.shuffle(self.indices)
        i = 0
        while i < len(self.indices):
            if self.rank == 0:
                view_size = random.randint(2, 24)
                batch_size = 48 // view_size
                image_size = random.randint(400, 600)
                params = {"view_size": view_size, "batch_size": batch_size, "image_size": image_size}
            else:
                params = None

            obj_list = [params]
            dist.broadcast_object_list(obj_list, src=0)
            params = obj_list[0]

            self.dataset.update_params(params)

            batch_indices = self.indices[i : i + params["batch_size"]]
            i += params["batch_size"]

            yield batch_indices

    def __len__(self):
        return len(self.indices)
