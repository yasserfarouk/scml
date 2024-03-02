import itertools
import random
import sys
from typing import Callable

import numpy as np
from negmas import ResponseType
from negmas.sao import SAOResponse

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import QUANTITY, UNIT_PRICE
from scml.oneshot.context import ANACOneShotContext
from scml.oneshot.rl.action import ActionManager, FlexibleActionManager
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.observation import ObservationManager

__all__ = ["random_action", "random_policy", "greedy_policy"]


def random_action(obs: np.ndarray, env: OneShotEnv) -> np.ndarray:
    """Samples a random action from the action space of the"""
    _ = obs
    return env.action_space.sample()


def random_policy(
    obs: np.ndarray, env: OneShotEnv, pend: float = 0.05, paccept: float = 0.15
) -> np.ndarray:
    """
    Ends the negotiation or accepts with a predefined probability or samples a random response.
    """
    _ = obs
    r = random.random()
    action = env.action_space.sample()
    if r < pend:
        i = random.randint(0, len(action) // 2)
        action[i : i + 2] = 0
    elif r < pend + paccept:
        i = random.randint(0, len(action) // 2)
        action[i : i + 2] = (0, 1)
    return action


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def all_but_concentrated(q, n) -> list[int]:
    """Distributes q over n so that as many values as possible are nonzero with one value being as large as possible"""
    q = int(q)
    if n < 1:
        return []
    if q <= n:
        return [1] * q + [0] * (n - q)
    lst = [1] * n
    if n == 1:
        lst[0] += q - n
    else:
        lst[random.randint(0, n - 1)] += q - n
    return lst


def greedy_policy(
    obs: np.ndarray,
    awi: OneShotAWI,
    obs_manager: ObservationManager,
    action_manager: ActionManager = FlexibleActionManager(ANACOneShotContext()),
    debug=False,
    distributor: Callable[[int, int], list[int]] = all_but_concentrated,
) -> np.ndarray:
    """
    A simple greedy policy.

    Args:
        obs: The current observation
        awi: The AWI of the agent running the policy
        obs_manager: The observation manager used to encode the observation
        action_manager: The action manager to be used to encode the action
        debug: If True, extra assertions are tested
        distributor: A callable that receives a total quantity to be distributed
                     over n partners and returns a list of n values that sum to this total quantity

    Remarks:
        - Accepts the subset of offers with maximum total quantity under current needs.
        - The remaining quantity is distributed over the remaining partners using the distributor function
        - Prices are set to the worst for the agent if the price range is small else they are set randomly

    """
    assert awi is not None and (awi.is_first_level or awi.is_last_level), f"{awi=}"
    offers = obs_manager.get_offers(awi, obs)

    if debug:
        received_offers = {k: o for k, o in offers.items() if o is not None}
        assert isinstance(awi, OneShotAWI)
        awi_offers = awi.current_offers
        received_keys = []
        for k in received_offers.keys():
            if "+" in k:
                received_keys += [_ for _ in k.split("+") if _ in awi_offers.keys()]
            else:
                received_keys.append(k)
        if set(awi_offers.keys()) != set(received_keys):
            raise AssertionError(
                f"AWI keys do not match received keys\n"
                f"{awi_offers=}\n{offers=}\n{received_offers=}\n{received_keys=}"
            )
        for k, v in received_offers.items():
            if "+" in k:
                q, t, p = v
                assert q == sum(awi_offers.get(_, (0, 0, 0))[0] for _ in k.split("+"))
                assert (
                    abs(
                        p * q
                        - sum(
                            awi_offers.get(_, (0, 0, 0))[-1]
                            * awi_offers.get(_, (0, 0, 0))[0]
                            for _ in k.split("+")
                        )
                    )
                    < 1e-5
                )
                assert all(
                    t == awi_offers[_][1] for _ in k.split("+") if _ in awi_offers
                )
                continue
            assert awi_offers[k] == v, (
                f"AWI values do not match received values\n"
                f"{awi_offers[k]=} != {offers[k]}"
            )
    needed = awi.needed_supplies if not awi.is_first_level else awi.needed_sales
    all_offers = list(offers.values())
    all_partners = list(offers.keys())
    n_partners = len(all_partners)
    all_indices = list(range(len(offers)))
    best, diff = None, sys.maxsize
    for indices in powerset(all_indices):
        q = sum(
            r[QUANTITY] if r is not None else 0
            for r in [all_offers[_] for _ in indices]
        )
        d = needed - q
        if d < 0:
            continue
        if d < diff:
            best, diff = indices, d
        if d == 0:
            break
    os = (
        awi.current_input_outcome_space
        if not awi.is_first_level
        else awi.current_output_outcome_space
    )
    t = awi.current_step
    mn = os.issues[UNIT_PRICE].min_value
    mx = os.issues[UNIT_PRICE].max_value
    if mx - mn < 3:
        prices = [mn if awi.is_first_level else mx] * n_partners
    else:
        prices = [os.issues[UNIT_PRICE].rand() for _ in range(n_partners)]
    if not best:
        # there are no acceptable offers
        quantities = distributor(needed, n_partners)
        response = dict(
            zip(
                all_partners,
                [
                    SAOResponse(ResponseType.REJECT_OFFER, (q, t, p))
                    for q, p in zip(quantities, prices)
                ],
            )
        )
    else:
        # we should accept the indices in best
        best = set(best)
        quantities = distributor(diff, n_partners - len(best))
        j, response = (
            0,
            dict(
                zip(
                    all_partners,
                    itertools.repeat(SAOResponse(ResponseType.END_NEGOTIATION, None)),
                )
            ),
        )
        for i, p in enumerate(all_partners):
            if i in best:
                response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, all_offers[i])
                continue
            if debug:
                assert quantities[j] >= 0
            response[p] = (
                SAOResponse(ResponseType.REJECT_OFFER, (quantities[j], t, prices[j]))
                if quantities[j] > 0
                else SAOResponse(ResponseType.END_NEGOTIATION, None)
            )
            j += 1
            if j >= len(quantities):
                break
    return action_manager.encode(awi, response)
