import torch
from torch import Tensor
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.utilities.distributed import gather_all_tensors
from torch.distributed import is_initialized


class SumAggregator:
    def __init__(self):
        super().__init__()
        self.sum_value = torch.tensor(0.0)
        self.sum_count = torch.tensor(0.0)

    def update(self, value, count=1.0):
        """
        Args:
            value: an averaged-across-batch scalar value
            count: usually the batch size
        """
        # Ensure the value and weight are scalar tensors on cpu
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        value = value.cpu().float().reshape([])
        if not isinstance(count, Tensor):
            count = torch.as_tensor(count, dtype=torch.float32, device="cpu")
        count = count.cpu().float().reshape([])

        # records
        self.sum_value += value * count
        self.sum_count += count

    def compute(self):  # across processes
        return gather_and_sum(self.sum_value) / self.reduce_sum_count()

    def reduce_sum_count(self):
        return gather_and_sum(self.sum_count)

    def reset(self):
        self.sum_value = torch.tensor(0.0)
        self.sum_count = torch.tensor(0.0)


def gather_and_sum(x):
    """
    Gather all tensors and sum them up.
    If not in distributed mode, return x directly.
    """
    if not is_initialized():
        return x
    # By default the group is NCCL, therefore send to cuda first.
    sum_x = sum(gather_all_tensors(x.cuda())).cpu()
    return sum_x


class ListAggregator:
    def __init__(self):
        super().__init__()
        self.tensor_list = []

    @rank_zero_only
    def update(self, tensor):
        """
        Args:
            tensor:
        """
        if not is_initialized():
            all_gpu_tensor = [tensor.detach().cpu()]
        else:
            all_gpu_tensor = gather_all_tensors(tensor)
            all_gpu_tensor = [t.detach().cpu() for t in all_gpu_tensor]
        self.tensor_list.extend(all_gpu_tensor)

    @rank_zero_only
    def get_tensor(self):
        return torch.cat(self.tensor_list, dim=0)

    @rank_zero_only
    def length(self):
        return len(self.get_tensor())

    def reset(self):
        self.tensor_list = []
