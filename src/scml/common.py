"""Common functions used in all modules"""

from __future__ import annotations

import random
from typing import Any, Iterable

import numpy as np
from negmas.helpers import distribute_integer_randomly, get_class
from numpy.typing import NDArray

__all__ = [
    "fraction_cut",
    "integer_cut",
    "intin",
    "realin",
    "strin",
    "isin",
    "isinfloat",
    "isinclass",
    "isinobject",
    "make_array",
    "distribute_quantities",
    "IterableOrInt",
    "IterableOrFloat",
    "IterableOrClass",
    "IterableOrObject",
    "distribute",
    "EPSILON",
]


IterableOrInt = tuple[int, int] | set[int] | list[int] | int | np.ndarray
IterableOrFloat = tuple[float, float] | set[float] | list[float] | float | np.ndarray
IterableOrClass = Iterable[str | type] | type | str
IterableOrObject = Iterable[str | Any] | Any
EPSILON = 1e-5


def isinobject(x: IterableOrObject, y: IterableOrClass):
    return isinclass(
        type(x) if not isinstance(x, Iterable) else [type(_) for _ in x], y
    )


def isinclass(x: IterableOrClass, y: IterableOrClass):
    """Checks that x is within the range specified by y. Ugly but works"""
    if not isinstance(x, Iterable) and not isinstance(y, Iterable):
        return issubclass(get_class(x), get_class(y))
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(y, Iterable):
        y = [y]
    x = [get_class(_) for _ in x]
    y = [get_class(_) for _ in y]
    for a in x:
        for b in y:
            if issubclass(a, b):  # type: ignore
                break
        else:
            return False
    return True


def isin(x: IterableOrInt, y: IterableOrInt):
    """Checks that x is within the range specified by y. Ugly but works"""
    if not isinstance(x, Iterable) and not isinstance(y, Iterable):
        return x == y
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()
    if isinstance(x, list):
        x = {_ for _ in x}
    if isinstance(y, list):
        y = {_ for _ in y}
    if isinstance(x, tuple):
        if isinstance(y, tuple):
            return y[0] <= x[0] <= y[-1]
        if not isinstance(y, Iterable):
            return x[0] == y == x[-1]
        x = set(list(range(x[0], x[-1])))
    if isinstance(y, tuple):
        if not isinstance(x, Iterable):
            return y[0] <= x <= y[-1]
        y = set(list(range(y[0], y[-1])))
    if not isinstance(x, Iterable):
        x = {x}
    if not isinstance(y, Iterable):
        y = {y}
    assert isinstance(x, set) and isinstance(
        y, set
    ), f"{x=} ({type(x)=}), {y=} ({type(y)})"
    return not x.difference(y)


def isinfloat(x: IterableOrFloat, y: IterableOrFloat):
    """Checks that x is within the range specified by y. Ugly but works"""
    if not isinstance(x, Iterable) and not isinstance(y, Iterable):
        return abs(x - y) < EPSILON
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()
    if isinstance(x, tuple):
        if isinstance(y, tuple):
            return y[0] - EPSILON <= x[0] <= y[-1] + EPSILON
        if not isinstance(y, Iterable):
            return abs(x[0] - y) < EPSILON and abs(y - x[-1]) < EPSILON
    if isinstance(y, tuple):
        if not isinstance(x, Iterable):
            return y[0] - EPSILON <= x <= y[-1] + EPSILON
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(y, Iterable):
        y = [y]
    for a in x:
        for b in y:
            if abs(a - b) < EPSILON:
                break
        else:
            return False
    return True


def fraction_cut(n: int, p: np.ndarray) -> np.ndarray:
    """Distributes n items on boxes with probabilities relative to p"""
    mx = len(p) - 1
    x = (np.round(100 * n * p).astype(np.int64) // 100).astype(int)

    total = x.sum()

    while total > n:
        i = random.randint(0, mx)
        if x[i] > 0:
            x[i] -= 1
            total -= 1

    while total < n:
        x[random.randint(0, mx)] += 1
        total += 1
    return x


def integer_cut(
    total: int,
    n: int,
    mx: int | list[int],
    mn: int | list[int] | None = None,
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
    if mn is None:
        mn = [float("inf")] * n  # type: ignore
    if not isinstance(mn, Iterable):
        mn = [mn] * n  # type: ignore
    if not isinstance(mx, Iterable):
        mx = [mx] * n  # type: ignore
    sizes = np.asarray(mx)
    if total < sizes.sum():
        raise ValueError(
            f"Cannot generate {n} numbers summing to {total}  with a minimum summing to {sizes.sum()}"
        )
    if total > sum(mn):  # type: ignore
        raise ValueError(
            f"Cannot generate {n} numbers summing to {total}  with a maximum summing to {sum(mn)}"  # type: ignore
        )
    valid = [i for i, s in enumerate(sizes) if mn[i] > s]  # type: ignore
    k = 0
    while sizes.sum() < total:
        if not randomize:
            j, k = valid[k], k + 1
            if k >= len(valid):
                k = 0
        else:
            j = random.choice(valid)
        sizes[j] += 1
        if sizes[j] >= mn[j]:  # type: ignore
            valid.remove(j)
            if not randomize:
                k = max(k - 1, 0)
    return sizes.tolist()


def realin(rng: tuple[float, float] | float | list[float] | np.ndarray) -> float:
    """
    Selects a random number within a range if given or the input if it was a float

    Args:
        rng: Range or single value

    Returns:

        the real within the given range
    """
    if isinstance(rng, np.ndarray):
        rng = rng.tolist()
    if isinstance(rng, list):
        rng = random.choice(rng)
    if isinstance(rng, float) or isinstance(rng, int):
        return float(rng)
    if abs(rng[-1] - rng[0]) < 1e-8:
        return rng[0]
    return rng[0] + random.random() * (rng[-1] - rng[0])


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


def intin(rng: tuple[int, int] | int | list[int] | np.ndarray) -> int:
    """
    Selects a random number within a range if given or the input if it was an int

    Args:
        rng: Range or single value

    Returns:

        the int within the given range
    """
    if isinstance(rng, np.ndarray):
        rng = rng.tolist()
    if isinstance(rng, list):
        rng = random.choice(rng)
    if isinstance(rng, int):
        return rng
    if rng[0] == rng[1]:
        return rng[0]
    return random.randint(rng[0], rng[1])


def make_array(
    x: np.ndarray | list[int] | tuple[int | float, int | float] | int | float,
    n: int,
    dtype: type[float] | type[int] = int,
    min_total: int = 0,
) -> np.ndarray:
    """Creates an array with the given choices"""
    if not isinstance(x, Iterable):
        assert (
            x * n >= min_total
        ), f"You are asking to make an array with {x} values that is at least {min_total} in length!!"
        return np.ones(n, dtype=dtype) * int(x)
    if isinstance(x, tuple) and len(x) == 2:
        assert (
            min_total < 1 or n * x[-1] >= min_total
        ), f"Cannot generate an array with choices{x=} and a minimum total of {min_total}"
        if dtype == int:
            lst = np.random.randint(x[0], x[1] + 1, n, dtype=dtype)  # type: ignore
        else:
            lst = x[0] + np.random.rand(n) * (x[1] - x[0])
        n_total = lst.sum()
        if n_total >= min_total:
            return lst
        missing = min_total - n_total
        for _ in range(100):
            lst = lst + np.asarray(
                distribute_integer_randomly(missing, len(lst), min_per_bin=0)
            )
            if lst.max() <= x[-1]:
                break
            lst = np.asarray([min(_, x[-1]) for _ in lst])
        else:
            lst = np.asarray(
                distribute_integer_randomly(
                    min_total, n, min_per_bin=int(x[0]) if x[0] else 0
                )
            )
        return lst
    # we have a list. Return it as it is if it has the correct length else sample from it
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
    q = np.asarray(q).flatten()
    if limit is not None and not isinstance(limit, Iterable):
        limit = [limit] * a  # type: ignore
    # if we do not distribute anything just return all zeros
    if q.sum() == 0:
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
    assert np.all(q >= 0), f"We have some negative quantities! {q}"
    qz = int(0.5 + q.sum() / len(q))
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

    q = q.flatten().tolist()
    assert len(q) == n_steps
    for s in range(n_steps):
        assert (
            limit is None or sum(limit) >= q[s]
        ), f"Sum of limits is {limit_sum} but we need to distribute {q[s]} at step {s}"
        if qz == 0 or q[s] == 0:
            values.append([0] * a)
            continue

        v = [int(0.5 + _ * float(q[s] / qz)) for _ in base_cut]
        n_changes = max(0, min(q[s], int(0.5 + (1.0 - predictability) * q[s])))
        if limit is not None:
            n_changes = min(n_changes, sum(k - x for k, x in zip(limit, v)))
        if n_changes <= 0:
            values.append(adjust_values(v, limit))
            continue
        subtracted = integer_cut(n_changes, a, 0)
        upper = (
            [lmt + s - c for lmt, c, s in zip(limit, v, subtracted)]
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


def distribute(
    q: int,
    n: int,
    *,
    mx: int | None = None,
    equal=False,
    concentrated=False,
    allow_zero=False,
) -> list[int]:
    """Distributes q values over n bins.

    Args:
        q: Quantity to distribute
        n: number of bins to distribute q over
        mx: Maximum allowed per bin. `None` for no limit
        equal: Try to make the values in each bins as equal as possible
        concentrated: If true, will try to concentrate offers in few bins. `mx` must be passed in this case
        allow_zero: Allow some bins to be zero even if that is not necessary
    """
    from collections import Counter

    from numpy.random import choice

    q, n = int(q), int(n)

    if mx is not None and q > mx * n:
        q = mx * n

    if concentrated:
        assert mx is not None
        lst = [0] * n
        if not allow_zero:
            for i in range(min(q, n)):
                lst[i] = 1
        q -= sum(lst)
        if q == 0:
            random.shuffle(lst)
            return lst
        for i in range(n):
            q += lst[i]
            lst[i] = min(mx, q)
            q -= lst[i]
        random.shuffle(lst)
        return lst

    if q < n:
        lst = [0] * (n - q) + [1] * q
        random.shuffle(lst)
        return lst

    if q == n:
        return [1] * n
    if allow_zero:
        per = 0
    else:
        per = (q // n) if equal else 1
    q -= per * n
    r = Counter(choice(n, q))
    return [r.get(_, 0) + per for _ in range(n)]
