# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from pathlib import Path
import re
from typing import List, Tuple

from setuptools import setup, find_packages


NAME = "dinov3"
DESCRIPTION = ""

URL = "https://github.com/facebookresearch/dinov3"
AUTHOR = "Meta AI"
REQUIRES_PYTHON = ">=3.11"
HERE = Path(__file__).parent


try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


def get_requirements(path: str = HERE / "requirements.txt") -> Tuple[List[str], List[str]]:
    requirements = []
    extra_indices = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip("\r\n")
            if line.startswith("--extra-index-url "):
                extra_indices.append(line[18:])
                continue
            requirements.append(line)
    return requirements, extra_indices


def get_package_version() -> str:
    with open(HERE / "dinov3/__init__.py") as f:
        result = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if result:
            return result.group(1)
    raise RuntimeError("Can't get package version")


requirements, extra_indices = get_requirements()
version = get_package_version()
dev_requirements, _ = get_requirements(HERE / "requirements-dev.txt")


setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={
        "": ["*.yaml"],
    },
    install_requires=requirements,
    dependency_links=extra_indices,
    extras_require={
        "dev": dev_requirements,
    },
    install_package_data=True,
    license="DINOv3 LIcense",
    license_files=("LICENSE.md",),
    classifiers=[
        # Trove classifiers: https://github.com/pypa/trove-classifiers/blob/main/src/trove_classifiers/__init__.py
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
