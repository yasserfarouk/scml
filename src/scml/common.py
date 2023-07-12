"""Common functions used in all modules"""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "fraction_cut",
    "integer_cut",
    "intin",
    "realin",
    "strin",
    "make_array",
    "distribute_quantities",
]


def fraction_cut(n: int, p: np.ndarray) -> np.ndarray:
    """Distributes n items on boxes with probabilities relative to p"""
    l = len(p) - 1
    x = (np.round(100 * n * p).astype(np.int64) // 100).astype(int)

    total = x.sum()

    while total > n:
        i = random.randint(0, l)
        if x[i] > 0:
            x[i] -= 1
            total -= 1

    while total < n:
        x[random.randint(0, l)] += 1
        total += 1
    return x


def integer_cut(
    n: int,
    l: int,
    l_m: int | list[int],
    l_x: int | list[int] | None = None,
    randomize: bool = True,
) -> list[int]:
    """
    Generates l random integers that sum to n where each of them is at least l_m
    Args:
        n: total
        l: number of levels
        l_m: minimum per level
        l_x: maximum per level
        randomize: If true, the integers resulting are randomized otherwise they will always be in the same order

    Returns:

    """
    if l_x is None:
        l_x = [float("inf")] * l  # type: ignore
    if not isinstance(l_x, Iterable):
        l_x = [l_x] * l  # type: ignore
    if not isinstance(l_m, Iterable):
        l_m = [l_m] * l  # type: ignore
    sizes = np.asarray(l_m)
    if n < sizes.sum():
        raise ValueError(
            f"Cannot generate {l} numbers summing to {n}  with a minimum summing to {sizes.sum()}"
        )
    if n > sum(l_x):  # type: ignore
        raise ValueError(
            f"Cannot generate {l} numbers summing to {n}  with a maximum summing to {sum(l_x)}"  # type: ignore
        )
    valid = [i for i, s in enumerate(sizes) if l_x[i] > s]  # type: ignore
    k = 0
    while sizes.sum() < n:
        if randomize:
            j, k = valid[k], k + 1
            if k >= len(valid):
                k = 0
        else:
            j = random.choice(valid)
        sizes[j] += 1
        if sizes[j] >= l_x[j]:  # type: ignore
            valid.remove(j)
            if not randomize:
                k = max(k - 1, 0)
    return sizes.tolist()


def realin(rng: tuple[float, float] | float) -> float:
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


def strin(lst: list[str] | str) -> str:
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


def intin(rng: tuple[int, int] | int) -> int:
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


def make_array(
    x: np.ndarray | tuple[int | float, int | float] | int | float,
    n,
    dtype: type[float] | type[int] = int,
) -> np.ndarray:
    """Creates an array with the given choices"""
    if not isinstance(x, Iterable):
        return np.ones(n, dtype=dtype) * x
    if isinstance(x, tuple) and len(x) == 2:
        if dtype == int:
            return np.random.randint(x[0], x[1] + 1, n, dtype=dtype)  # type: ignore
        return x[0] + np.random.rand(n) * (x[1] - x[0])
    xlst = list(x)
    if len(xlst) == n:
        return np.array(xlst)
    return np.array(list(random.choices(xlst, k=n)))


def distribute_quantities(
    equal: bool,
    predictability: float,
    q: list[int] | NDArray,
    a: int,
    n_steps: int,
    limit: list[int] | None = None,
):
    """Used internally by generate() methods to distribute exogenous contracts

    Args:
        equal: whether the quantities are to be distributed equally
        predictability: how much are quantities for the same agent at different
                        times are similar
        q: The quantity per step to be distributed
        a: The number of agents to distribute over.
        limit: The maximum quantity per step for each agent (len(limit) == a). Only used if `equal==False`

    Returns:
        an n_steps * a list of lists giving the distributed quantities where
        sum[s, :] ~= q[s]. The error can be up to 2*a per step

    """
    q = np.asarray(q)
    if limit is not None and not isinstance(limit, Iterable):
        limit = [limit] * a  # type: ignore
    # if we do not distribute anything just return all zeros
    if sum(q) == 0:
        return [np.asarray([0] * a, dtype=int) for _ in range(n_steps)]
    # if all quantities are to be distributed equally, just do that directly
    # ensuring each agent gets at least one item.
    # what happens if q does not divide a? q/a is rounded.
    if equal:
        values = np.maximum(np.round(q / a).astype(int), 1).tolist()
        # if limit is not None:
        #     values = [a if a < b else b for a, b in zip(values, limit)]
        return [np.asarray([values[_]] * a, dtype=int) for _ in range(n_steps)]
    if predictability < 0.01:
        values = []
        for s in range(n_steps):
            values.append(integer_cut(q[s], a, 0, limit))
            assert sum(values[-1]) == q[s]
        return values
    values = []
    assert all(_ >= 0 for _ in q), f"We have some negative quantities! {q}"
    qz = int(0.5 + sum(q) / len(q))
    base_cut = integer_cut(qz, a, 0, limit)
    limit_sum = sum(limit) if limit is not None else float("inf")
    if limit is not None:
        assert all(
            [a <= b for a, b in zip(base_cut, limit)]
        ), f"base_cut above limit:\nbase_cut: {base_cut}\nLimit: {limit}"
    assert min(base_cut) >= 0, f"base cut has negative value {base_cut}"

    def adjust_values(v, limit):
        for i in range(len(v)):
            if v[i] >= 0:
                continue
            # we have too few at index i
            errs = -v[i]
            v[i] = 0
            while errs > 0:
                diffs = integer_cut(errs, a - 1, 0)
                diffs = diffs[:i] + [0] + diffs[i:]
                for j in range(len(v)):
                    if j == i:
                        continue
                    if v[j] >= diffs[j]:
                        v[j] -= diffs[j]
                        errs -= diffs[j]
            continue
        if limit is None:
            return v
        for i in range(len(v)):
            if v[i] <= limit[i]:
                continue
            # we have too many at index i
            errs = v[i] - limit[i]
            v[i] = limit[i]
            available = [x - y for x, y in zip(limit, v)]
            available = available[:i] + available[i + 1 :]
            if sum(available) < errs:
                errs = sum(available)
            while errs > 0:
                diffs = integer_cut(errs, a - 1, 0, available)
                diffs = diffs[:i] + [0] + diffs[i:]
                for j in range(len(v)):
                    if j == i:
                        continue
                    if limit[j] - v[j] >= diffs[j]:
                        v[j] += diffs[j]
                        errs -= diffs[j]
        return v

    for s in range(0, n_steps):
        assert (
            limit is None or sum(limit) >= q[s]
        ), f"Sum of limits is {limit_sum} but we need to distribute {q[s]} at step {s}"
        if qz == 0 or q[s] == 0:
            values.append([0] * a)
            continue

        v = [int(0.5 + _ * q[s] / qz) for _ in base_cut]
        n_changes = max(0, min(q[s], int(0.5 + (1.0 - predictability) * q[s])))
        if limit is not None:
            n_changes = min(n_changes, sum(l - x for l, x in zip(limit, v)))
        if n_changes <= 0:
            values.append(adjust_values(v, limit))
            continue
        subtracted = integer_cut(n_changes, a, 0)
        upper = (
            [l + s - c for l, c, s in zip(limit, v, subtracted)]
            if limit is not None
            else None
        )
        added = integer_cut(n_changes, a, 0, upper)
        # assert isinstance(added[0], int) and isinstance(subtracted[0], int)
        for i in range(len(v)):
            v[i] += added[i] - subtracted[i]
        values.append(adjust_values(v, limit))

    for s, v in enumerate(values):
        if limit is not None:
            assert all(
                a >= b for a, b in zip(limit, v)
            ), f"Some values are above limit\n Limit: {limit}\nValues: {v}"
        assert (
            abs(sum(v) - q[s]) < 2 * a
        ), f"Failed to distribute: expected {q[s]} but got {sum(v)}: {values[-1]}"
        assert min(v) >= 0, f"Negative  value {min(v)} in quantities!\n{v}"
    return values
