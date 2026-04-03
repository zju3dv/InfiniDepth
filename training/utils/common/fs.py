"""
File system operations. Currently supports local and hadoop file systems.
"""

import hashlib
import os
import pickle
import shutil
import subprocess
import tarfile
import tempfile
from typing import List, Optional

from common.distributed import barrier_if_distributed, get_global_rank, get_local_rank
from common.logger import get_logger

logger = get_logger(__name__)


def is_hdfs_path(path: str) -> bool:
    """
    Detects whether a path is an hdfs path.
    A hdfs path must startswith "hdfs://" protocol prefix.
    """
    return path.lower().startswith("hdfs://")


def download(
    path: str,
    dirname: Optional[str] = None,
    filename: Optional[str] = None,
    add_hash_suffix: bool = True,
    distributed: bool = True,
    overwrite: bool = False,
) -> str:
    """
    Download a file to a local location. Returns the local path.
    This function avoids repeated download if it has already been downloaded before.
    Under distributed context, only local rank zero will download and the rest will wait.
    Args:
        path: source file path.
        dirname: destination directory, or None for auto.
        filename: destination file name, or None for auto.
        add_hash_suffix: whether to add a hash suffix to distinguish
                         between files with same name but different paths.
        distributed: True if this method is called by all ranks. False if called by a single rank.
        overwrite: whether to overwrite a downloaded file.
    """
    # If local path and no destination specification, directly return.
    if not is_hdfs_path(path) and dirname is None and filename is None:
        return path

    # Compute a local filename.
    if dirname is None:
        dirname = "downloads"
    if filename is None:
        filename = os.path.split(path)[-1]
        if add_hash_suffix:
            hashname = hashlib.md5(path.encode("utf-8")).hexdigest()
            filename += "." + hashname

    pathname = os.path.join(dirname, filename)
    to_bytenas = os.path.abspath(dirname).startswith("/mnt/bn")

    # If distributed, only local rank zero performs download.
    # If the destination is on bytenas, only global rank zero performs download.
    if (not distributed) or (get_global_rank() == 0) or (get_local_rank() == 0 and not to_bytenas):
        # Download if the file doesn't exist.
        if overwrite and os.path.exists(pathname):
            remove(pathname)
        if not os.path.exists(pathname):
            os.makedirs(dirname, exist_ok=True)
            logger.info(f"Downloading {path} to {pathname}")
            copy(path, pathname)

    # If distributed, all ranks must wait.
    if distributed:
        barrier_if_distributed()
    return pathname


def download_and_extract(path: str) -> str:
    """
    Download from hdfs if needed and extract tarball if needed.
    Do nothing if the file has already been downloaded and extracted locally.
    Returns the extracted local path.
    Under distributed context, only local rank zero will do work and the rest will wait.
    """
    # Download from hdfs if needed.
    path = download(path)
    # If the path is a file instead of directory,
    # assume it is a tarball and try extract it.
    if os.path.isfile(path):
        with tarfile.open(path) as tar:
            # Assume the tarball's first entry as the directory name.
            folder_name = tar.next().name
            # If distributed, only local rank zero performs the extraction.
            if get_local_rank() == 0:
                # Extract only if it hasn't been extracted before.
                if not os.path.exists(folder_name):
                    tar.extractall(".")
            # If distributed, all ranks must wait.
            barrier_if_distributed()
            path = folder_name
    return path


def listdir(path: str) -> List[str]:
    """
    List directory. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if is_hdfs_path(path):
        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE,
        )

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]

    return files


def listdir_with_metafile(path: str) -> List[str]:
    """
    Create a metafile caching the list directory result.
    Read from metafile for all other ranks and all future list operations.
    Same behavior as listdir(path).
    """
    # Local directory should directly return.
    if not is_hdfs_path(path):
        return listdir(path)

    # Define metafile path.
    metafile = os.path.join(path, "metafile.pkl")

    # Write metafile if not exists, only by global rank zero.
    if get_global_rank() == 0 and not exists(metafile):
        files = listdir(path)
        with tempfile.NamedTemporaryFile("wb", delete=True) as f:
            f.write(pickle.dumps(files))
            f.flush()
            copy(f.name, metafile, blocking=True)
        logger.info(f"Created metafile for {path}")

    # All other ranks wait.
    barrier_if_distributed()

    # All ranks read from metafile.
    with open(download(metafile), "rb") as f:
        files = pickle.loads(f.read())

    # Assert to prevent directory move.
    assert all(
        file.startswith(path) for file in files
    ), f"metafile for path: {path} is outdated. The directory likely has been moved."

    # Return the list of files.
    return files


def exists(path: str) -> bool:
    """
    Check whether a path exists.
    Returns True if exists, False otherwise.
    """
    if is_hdfs_path(path):
        process = subprocess.run(["hdfs", "dfs", "-test", "-e", path], capture_output=True)
        return process.returncode == 0
    return os.path.exists(path)


def mkdir(path: str):
    """
    Create a directory.
    Create all parent directory if not present. No-op if directory already present.
    """
    if is_hdfs_path(path):
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    else:
        os.makedirs(path, exist_ok=True)


def copy(src: str, tgt: str, blocking: bool = True):
    """
    Copy a file.
    """
    if src == tgt:
        return

    src_hdfs = is_hdfs_path(src)
    tgt_hdfs = is_hdfs_path(tgt)

    if not src_hdfs and not tgt_hdfs:
        shutil.copy(src, tgt)
        return

    if src_hdfs and tgt_hdfs:
        process = subprocess.Popen(["hdfs", "dfs", "-cp", "-f", src, tgt])
    elif src_hdfs and not tgt_hdfs:
        process = subprocess.Popen(["hdfs", "dfs", "-get", "-c", "64", src, tgt])
    elif not src_hdfs and tgt_hdfs:
        process = subprocess.Popen(["hdfs", "dfs", "-put", "-f", src, tgt])

    if blocking:
        process.wait()


def move(src: str, tgt: str):
    """
    Move a file.
    """
    if src == tgt:
        return

    src_hdfs = is_hdfs_path(src)
    tgt_hdfs = is_hdfs_path(tgt)

    if src_hdfs and tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-mv", src, tgt])
    elif not src_hdfs and not tgt_hdfs:
        shutil.move(src, tgt)
    else:
        copy(src, tgt)
        remove(src)


def remove(path: str):
    if is_hdfs_path(path):
        subprocess.run(["hdfs", "dfs", "-rm", "-r", path])
    elif os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)

