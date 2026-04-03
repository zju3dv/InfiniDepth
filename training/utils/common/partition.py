"""
Partition utility functions.
"""

from typing import Any, List


def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    """
    Partition a list by size.
    When indivisible, the last group contains fewer items than the target size.

    Examples:
        - data: [1,2,3,4,5]
        - size: 2
        - return: [[1,2], [3,4], [5]]
    """
    assert size > 0
    return [data[i : (i + size)] for i in range(0, len(data), size)]


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    assert groups > 0
    return [data[i::groups] for i in range(groups)]


def shift_list(data: List[Any], n: int) -> List[Any]:
    """
    Rotate a list by n elements.

    Examples:
        - data: [1,2,3,4,5]
        - n: 3
        - return: [4,5,1,2,3]
    """
    return data[(n % len(data)) :] + data[: (n % len(data))]
