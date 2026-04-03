"""
Platform utility methods.
"""

import os
import subprocess
from typing import Optional

from common.logger import get_logger

logger = get_logger(__name__)


def get_region() -> Optional[str]:
    """
    Get the region of the current machine.
    Returns: "cn" | "us"
    """
    # ARNOLD_REGION is used by merlin trials.
    # BYTE_REGION is used by merlin devbox.
    region = os.environ.get("ARNOLD_REGION") or os.environ.get("BYTE_REGION")
    return region.lower() if region else None


def get_task_id() -> Optional[str]:
    """
    Get merlin task id.
    Example task id: "b35eea67ca5f3962"
    """
    return os.environ.get("MERLIN_JOB_ID")


def upload_asset(src: str, msg: str):
    """
    Upload asset under src dir to merlin.
    """
    trial_id = os.getenv("ARNOLD_TRIAL_ID")
    msg = "_".join([f"trial_{trial_id}_trace", msg])
    res = subprocess.run(
        ["/opt/tiger/mlx_deploy/bin/mlx", "asset", "upload", "-n", msg, src],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    if res.returncode != 0:
        logger.warning("Fail to upload!")
    else:
        logger.info("Successfully uploaded.")
    logger.info(res.stdout)
