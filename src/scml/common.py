"""Common functions used in all modules"""
from dataclasses import dataclass
import random
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

__all__ = [
    "integer_cut",
    "intin",
    "realin",
    "strin",
    "make_array",
    "distribute_quantities",
]


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


def strin(lst: Union[List[str], str]) -> str:
    """
    Selects a random string from a list (or just returns the string if no list
    is given)

    Args:
        lst: list of value

    Returns:

        the real within the given range
    """
    if isinstance(lst, str):
        return lst
    return random.choice(lst)


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


def distribute_quantities(
    equal: bool, predictability: float, q: List[int], a: int, n_steps: int
):
    """Used internally by generate() methods to distribute exogenous contracts

    Args:
        equal: whether the quantities are to be distributed equally
        predictability: how much are quantities for the same agent at different
                        times are similar
        q: The quantity per step to be distributed
        a: The number of agents to distribute over.

    Returns:
        an n_steps * a list of lists giving the distributed quantities where
        sum[s, :] ~= q[s]

    """
    if sum(q) == 0:
        return [np.asarray([0] * a, dtype=int) for _ in range(n_steps)]
    if equal:
        values = np.maximum(1, np.round(q / a).astype(int)).tolist()
        return [np.asarray([values[p]] * a, dtype=int) for _ in range(n_steps)]
    if predictability < 0.01:
        values = []
        for s in range(n_steps):
            values.append(integer_cut(q[s], a, 0))
            assert sum(values[-1]) == q[s]
        return values
    values = []
    qz = int(0.5 + sum(q) / len(q))
    base_cut = integer_cut(qz, a, 0)
    for s in range(0, n_steps):
        if qz == 0 or q[s] == 0:
            values.append([0] * a)
            continue
        values.append([int(0.5 + _ * q[s] / qz) for _ in base_cut])
        n_changes = int(0.5 + (1.0 - predictability) * q[s])
        if not n_changes:
            continue
        added = integer_cut(n_changes, a, 0)
        subtracted = integer_cut(n_changes, a, 0)
        assert isinstance(added[0], int) and isinstance(subtracted[0], int)
        for i in range(len(values[-1])):
            values[-1][i] += added[i] - subtracted[i]
            if values[-1][i] >= 0:
                continue
            errs = -values[-1][i]
            values[-1][i] = 0
            while errs > 0:
                diffs = integer_cut(errs, a - 1, 0)
                diffs = diffs[:i] + [0] + diffs[i:]
                for j in range(len(values[-1])):
                    if j == i:
                        continue
                    if values[-1][j] >= diffs[j]:
                        values[-1][j] -= diffs[j]
                        errs -= diffs[j]
        assert (
            abs(sum(values[-1]) - q[s]) < 3
        ), f"Failed to distribute: expected {q[s]} but got {sum(values[-1])}: {values[-1]}"
        assert (
            min(values[-1]) >= 0
        ), f"Negative  value {min(values[-1])} in quantities!\n{values}"
    return values
