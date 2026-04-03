import os
import socket
from training.entrys import *  # noqa: F403, F401, F405
from training.utils.logger import Log


def main():
    from training.config.config import cfg

    if Log.is_main_process():
        Log.info("InfiniDepth")
        separator = "\033[91m" + "-" * 80 + "\033[0m"
        experiment_info = f"\033[92mExperiment: \033[0m\033[94m{cfg.exp_name}\033[0m"
        print(f"{separator}\n{experiment_info}\n{separator}")
    globals()[cfg.entry](cfg)


def check_port(port):

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False  # if no error, port is not occupied
        except OSError:
            return True  # if error, port is occupied


if __name__ == "__main__":
    assert (
        "workspace" in os.environ
    ), """
    workspace is not set, please set it by:
    export workspace=/path/to/your/workspace
    The workspace environment variable can be used for different users
    to store their experiment data.
    """
    assert (
        "commonspace" in os.environ
    ), """
    commonspace is not set, please set it by:
    export commonspace=/path/to/your/commonspace
    The commonspace environment variable is used to store datasets and pretrained checkpoints.
    Please do not modify contents in commonspace during experiment runtime.
    """
    if "HTCODE_DEBUG_DDP" in os.environ:
        Log.info("DEBUG DDP")
    elif "ARNOLD_WORKER_0_HOST" in os.environ:
        os.environ["MASTER_IP"] = os.environ["ARNOLD_WORKER_0_HOST"]
        os.environ["MASTER_ADDR"] = os.environ["ARNOLD_WORKER_0_HOST"]
        if check_port(int(os.environ["ARNOLD_WORKER_0_PORT"])) is False:
            os.environ["MASTER_PORT"] = os.environ["ARNOLD_WORKER_0_PORT"]
        os.environ["NODE_SIZE"] = os.environ["ARNOLD_WORKER_NUM"]
        os.environ["NODE_RANK"] = os.environ["ARNOLD_ID"]
        os.environ["NCCL_DEBUG"] = "WARN"
    main()
