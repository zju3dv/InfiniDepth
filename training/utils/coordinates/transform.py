from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Final, Any, Iterator, Callable

import numpy as np
from numpy import typing as npt
from enum import Enum

class Direction(Enum):
    UP = 0b001
    DOWN = 0b010
    LEFT = 0b011
    RIGHT = 0b100
    FORWARD = 0b101
    BACKWARD = 0b110

_DIRECTION_TRANSFORM_ROWS: Final[dict[Direction, npt.NDArray]] = {
    Direction.RIGHT: np.array([1, 0, 0]),
    Direction.LEFT: np.array([-1, 0, 0]),
    Direction.UP: np.array([0, 1, 0]),
    Direction.DOWN: np.array([0, -1, 0]),
    Direction.FORWARD: np.array([0, 0, 1]),
    Direction.BACKWARD: np.array([0, 0, -1])
}

class Handedness(Enum):
    """
    Enum expressing the handedness of a coordinate system.
    """
    Undefined = 0b00
    Left = 0b01
    Right = 0b10


@dataclass(frozen=True)
class CoordinateSystem:
    """
    Describes a 3-dimensional coordinate system via the positive direction of the three axes.
    """
    x_direction: Direction
    y_direction: Direction
    z_direction: Direction

    @staticmethod
    def open_gl() -> "CoordinateSystem":
        """
        :return: The OpenGL coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.UP, Direction.BACKWARD)

    @staticmethod
    def open_cv() -> "CoordinateSystem":
        """
        :return: The OpenCV coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.DOWN, Direction.FORWARD)

    @staticmethod
    def colmap() -> "CoordinateSystem":
        """
        :return: The Colmap coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.DOWN, Direction.FORWARD)

    @staticmethod
    def nerfstudio_camera() -> "CoordinateSystem":
        """
        :return: The Nerfstudio camera coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.UP, Direction.BACKWARD)

    @staticmethod
    def nerfstudio_world() -> "CoordinateSystem":
        """
        :return: The Nerfstudio world coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.FORWARD, Direction.UP)

    @staticmethod
    def pytorch_3d() -> "CoordinateSystem":
        """
        :return: The PyTorch 3D coordinate system.
        """
        return CoordinateSystem(Direction.LEFT, Direction.UP, Direction.FORWARD)

    @staticmethod
    def blender() -> "CoordinateSystem":
        """
        :return: The Blender coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.FORWARD, Direction.UP)

    @staticmethod
    def unity() -> "CoordinateSystem":
        """
        :return: The Unity engine coordinate system.
        """
        return CoordinateSystem(Direction.RIGHT, Direction.UP, Direction.FORWARD)

    @staticmethod
    def unreal() -> "CoordinateSystem":
        """
        :return: The Unreal engine coordinate system.
        """
        return CoordinateSystem(Direction.FORWARD, Direction.RIGHT, Direction.UP)

    def __iter__(self) -> Iterator[Direction]:
        return iter((self.x_direction, self.y_direction, self.z_direction))

    @property
    @cache
    def handedness(self) -> Handedness:
        """
        :return: The handedness of the coordinate system.
        """
        t_mat = self.mat.T
        determinant = np.linalg.det(t_mat)

        if determinant > 0:
            return Handedness.Right
        elif determinant < 0:
            return Handedness.Left
        else:
            return Handedness.Undefined

    @property
    def is_left_handed(self) -> bool:
        """
        :return: Whether this coordinate system is left-handed.
        """
        return self.handedness == Handedness.Left

    @property
    def is_right_handed(self) -> bool:
        """
        :return: Whether this coordinate system is right-handed.
        """
        return self.handedness == Handedness.Right

    @property
    @cache
    def mat(self, dtype: Any = None) -> npt.NDArray:
        """
        :return: The transformation matrix representing this coordinate system (row major).
        """
        return np.array([_DIRECTION_TRANSFORM_ROWS[direction] for direction in self], dtype=dtype)

    def convert(self, mat: npt.NDArray, target_system: 'CoordinateSystem') -> npt.NDArray:
        """
        Converts the given matrix from this coordinate system to the target coordinate system.
        :param mat: The matrix to convert in row major. Its shape must match 3xN.
        :param target_system: The target coordinate system to convert to.
        :return: The converted matrix.
        """
        from_mat = self.mat
        to_mat = target_system.mat

        conv_mat = to_mat @ np.linalg.inv(from_mat) @ mat
        return conv_mat

    def convert_func(self, target_system: 'CoordinateSystem') -> Callable[[npt.NDArray], npt.NDArray]:
        """
        Creates a function that converts matrices from this coordinate system to the target coordinate system.
        :param target_system: The target coordinate system to convert to.
        :return: A function converting matrices from this coordinate system to the target coordinate system.
        """
        from_mat = self.mat
        inv_from_mat = np.linalg.inv(from_mat)
        to_mat = target_system.mat

        def inner(mat: npt.NDArray) -> npt.NDArray:
            return to_mat @ inv_from_mat @ mat

        return inner