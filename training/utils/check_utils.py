import torch


def check_equal_get_one(x, name="input"):
    """Check if all elements in x are equal"""
    if len(x) == 1 or isinstance(x, str):
        return x
    if isinstance(x[0], str):
        assert sum([x_ != x[0] for x_ in x]) == 0, f"All elements of {name} should be equal."
    else:  # tensor
        assert (x[0] == x).all(), f"All elements of {name} should be equal."
    return x[0]
