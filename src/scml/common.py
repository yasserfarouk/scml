"""Common functions used in all modules"""
from typing import Iterable, List, Union, Tuple
import numpy as np
import random

__all__ = ["integer_cut", "intin", "realin", "make_array"]


def integer_cut(n: int, l: int, l_m: Union[int, List[int]]) -> List[int]:
    """
    Generates l random integers that sum to n where each of them is at least l_m
    Args:
        n: total
        l: number of levels
        l_m: minimum per level

    Returns:

    """
    if not isinstance(l_m, Iterable):
        l_m = [l_m] * l
    sizes = np.asarray(l_m)
    if n < sizes.sum():
        raise ValueError(
            f"Cannot generate {l} numbers summing to {n}  with a minimum summing to {sizes.sum()}"
        )
    while sizes.sum() < n:
        sizes[random.randint(0, l - 1)] += 1
    return sizes.tolist()


def realin(rng: Union[Tuple[float, float], float]) -> float:
    """
    Selects a random number within a range if given or the input if it was a float

    Args:
        rng: Range or single value

    Returns:

        the real within the given range
    """
    if isinstance(rng, float) or isinstance(rng, int):
        return float(rng)
    if abs(rng[1] - rng[0]) < 1e-8:
        return rng[0]
    return rng[0] + random.random() * (rng[1] - rng[0])


def intin(rng: Union[Tuple[int, int], int]) -> int:
    """
    Selects a random number within a range if given or the input if it was an int

    Args:
        rng: Range or single value

    Returns:

        the int within the given range
    """
    if isinstance(rng, int):
        return rng
    if rng[0] == rng[1]:
        return rng[0]
    return random.randint(rng[0], rng[1])


def make_array(x: Union[np.ndarray, Tuple[int, int], int], n, dtype=int) -> np.ndarray:
    """Creates an array with the given choices"""
    if not isinstance(x, Iterable):
        return np.ones(n, dtype=dtype) * x
    if isinstance(x, tuple) and len(x) == 2:
        if dtype == int:
            return np.random.randint(x[0], x[1] + 1, n, dtype=dtype)
        return x[0] + np.random.rand(n) * (x[1] - x[0])
    x = list(x)
    if len(x) == n:
        return np.array(x)
    return np.array(list(random.choices(x, k=n)))
