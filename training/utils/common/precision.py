from typing import Optional
import torch
from omegaconf import DictConfig

from common.logger import get_logger

logger = get_logger(__name__)
_PRECISION: Optional[str] = None


def init_precision(config: DictConfig) -> None:
    global _PRECISION
    _PRECISION = config.training.get("precision", "bf16_amp")
    assert _PRECISION in ["fp32", "tf32", "bf16_amp", "bf16_fsdp"]
    logger.info(f"Running with precision: {_PRECISION}.")
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled()
    torch.backends.cudnn.allow_tf32 = tf32_enabled()


def tf32_enabled() -> bool:
    assert _PRECISION is not None
    return _PRECISION in ["tf32", "bf16_amp", "bf16_fsdp"]


def bf16_amp_enabled() -> bool:
    assert _PRECISION is not None
    return _PRECISION == "bf16_amp"


def bf16_fsdp_enabled() -> bool:
    assert _PRECISION is not None
    return _PRECISION == "bf16_fsdp"


def bf16_enabled() -> bool:
    assert _PRECISION is not None
    return _PRECISION in ["bf16_fsdp", "bf16_amp"]


class autocast(torch.autocast):
    """
    Serve as context managers that
    allow regions of your script to run in mixed precision.
    Currently, only BF16 is supported.
    """

    def __init__(self) -> None:
        super().__init__(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    def __enter__(self):
        self._enabled &= bf16_enabled()
        super().__enter__()


class autocast_fsdp(torch.autocast):
    """
    Serve as context managers.
    When your module is wrapped with FSDP, you should
    run module.fwd/bwd under this instance other than autocast.
    Run FSDP module with bf16_fsdp_enabled under bf16_amp context
    will cause a decrease in training efficiency.
    Currently, only BF16 is supported.
    """

    def __init__(self) -> None:
        super().__init__(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    def __enter__(self):
        self._enabled &= bf16_amp_enabled()
        super().__enter__()
