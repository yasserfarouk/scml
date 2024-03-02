from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
import random

from scml.oneshot.ufun import OneShotUFun
from negmas import Outcome, ResponseType, SAOResponse, SAOState
from negmas.sao.negotiators.controlled import ControlledSAONegotiator

from scml.oneshot import OneShotAgent, OneShotSyncAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

from ..switches import DefaultOneShotWorld

DEFAULT_SEED = 1


class SimpleAgent(OneShotAgent):
    """A greedy agent based on OneShotAgent"""

    def before_step(self):
        self.secured = 0

    def on_negotiation_success(self, contract, mechanism):
        self.secured += contract.agreement["quantity"]

    def propose(self, negotiator_id: str, state) -> Outcome | None:
        return self.best_offer(negotiator_id)

    def respond(self, negotiator_id, state: SAOState, source=None):  # type: ignore
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return ResponseType.END_NEGOTIATION
        return (
            ResponseType.ACCEPT_OFFER
            if offer[QUANTITY] <= my_needs
            else ResponseType.REJECT_OFFER
        )

    def best_offer(self, negotiator_id) -> Outcome | None:
        my_needs = self._needed(negotiator_id)
        if my_needs <= 0:
            return None
        nmi = self.get_nmi(negotiator_id)
        if not nmi:
            return None
        quantity_issue = nmi.issues[QUANTITY]
        unit_price_issue = nmi.issues[UNIT_PRICE]
        offer = [-1] * 3
        offer[QUANTITY] = max(
            min(my_needs, quantity_issue.max_value), quantity_issue.min_value
        )
        offer[TIME] = self.awi.current_step
        if self._is_selling(nmi):
            offer[UNIT_PRICE] = unit_price_issue.max_value
        else:
            offer[UNIT_PRICE] = unit_price_issue.min_value
        return tuple(offer)

    def _needed(self, negotiator_id=None):
        return (
            self.awi.current_exogenous_input_quantity
            + self.awi.current_exogenous_output_quantity
            - self.secured
        )

    def _is_selling(self, nmi):
        return nmi.annotation["product"] == self.awi.my_output_product


class BetterAgent(SimpleAgent):
    """A greedy agent based on OneShotAgent with more sane strategy"""

    def __init__(self, *args, concession_exponent=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self._e = concession_exponent

    def propose(self, negotiator_id: str, state) -> Outcome | None:
        offer = super().propose(negotiator_id, state)
        if not offer:
            return None
        offer = list(offer)
        offer[UNIT_PRICE] = int(
            round(self._find_good_price(self.get_nmi(negotiator_id), state))
        )
        return tuple(offer)

    def respond(self, negotiator_id, state, source=None):
        offer = state.current_offer
        response = super().respond(negotiator_id, state, source=source)
        if response != ResponseType.ACCEPT_OFFER:
            return response
        nmi = self.get_nmi(negotiator_id)
        assert offer is not None
        return (
            response
            if self._is_good_price(nmi, state, offer[UNIT_PRICE])
            else ResponseType.REJECT_OFFER
        )

    def _is_good_price(self, nmi, state, price):
        """Checks if a given price is good enough at this stage"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # a good price is one better than the threshold
        if self._is_selling(nmi):
            return (price - mn) >= th * (mx - mn)
        else:
            return (mx - price) >= th * (mx - mn)

    def _find_good_price(self, nmi, state):
        """Finds a good-enough price conceding linearly over time"""
        mn, mx = self._price_range(nmi)
        th = self._th(state.step, nmi.n_steps)
        # offer a price that is around th of your best possible price
        if self._is_selling(nmi):
            return mn + th * (mx - mn)
        else:
            return mx - th * (mx - mn)

    def _price_range(self, nmi):
        """Finds the minimum and maximum prices"""
        mn = nmi.issues[UNIT_PRICE].min_value
        mx = nmi.issues[UNIT_PRICE].max_value
        return mn, mx

    def _th(self, step, n_steps):
        """calculates a descending threshold (0 <= th <= 1)"""
        return ((n_steps - step - 1) / (n_steps - 1)) ** self._e


class SyncAgent(OneShotSyncAgent, BetterAgent):  # type: ignore
    """A greedy agent based on OneShotSyncAgent"""

    def __init__(self, *args, threshold=0.5, max_round_diff=2, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        self.max_round_diff = max_round_diff
        self.ufun: OneShotUFun  # type: ignore

    def before_step(self):
        super().before_step()
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)

    def first_proposals(self):
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.best_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        steps = [_.step for _ in states.values()]
        missing = set(self.negotiators.keys()).difference(set(states.keys()))
        d = max(steps) - min(steps)
        if missing:
            for n in missing:
                assert not self.negotiators[
                    n
                ][
                    0
                ].nmi.state.running, f"{n} is not present in the state and the negotiation is still running (Max round diff is {d})."
        assert d <= self.max_round_diff, f"Max round diff is {d}\n\t{states}"
        responses = {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in self.first_proposals().items()
        }
        my_needs = self._needed()
        is_selling = (self._is_selling(self.get_nmi(_)) for _ in offers.keys())
        sorted_offers = sorted(
            zip(offers.values(), is_selling),
            key=lambda x: (-x[0][UNIT_PRICE]) if x[1] else x[0][UNIT_PRICE],
        )
        secured, outputs, chosen = 0, [], dict()
        for i, k in enumerate(offers.keys()):
            offer, is_output = sorted_offers[i]
            secured += offer[QUANTITY]
            if secured >= my_needs:
                break
            chosen[k] = offer
            outputs.append(is_output)

        u = self.ufun.from_offers(tuple(chosen.values()), tuple(outputs))
        rng = self.ufun.max_utility - self.ufun.min_utility
        threshold = self._threshold * rng + self.ufun.min_utility
        if u >= threshold:
            for k in chosen.keys():
                responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
        return responses


class ReporterAgent(BetterAgent):
    def __init__(self, *args, max_round_diff=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_nums = defaultdict(int)
        self.max_round_diff = max_round_diff

    def step(self):
        self.round_nums = defaultdict(int)

    def respond(self, negotiator_id, state, source=None):
        offer = state.current_offer
        assert state.running, (
            f"{self.id} called to respond in a negotiation that "
            f"is no longer running\n{state}\noffer:{offer}\npartner:{negotiator_id}"
        )
        self.round_nums[negotiator_id] += 1
        max_diff = max(self.round_nums.values()) - min(self.round_nums.values())
        if max_diff > self.max_round_diff:
            mx, mx_id = float("-inf"), None
            mn, mn_id = float("inf"), None
            for k, v in self.round_nums.items():
                if v > mx:
                    mx, mx_id = v, k
                if v < mn:
                    mn, mn_id = v, k
            assert mn_id is not None and mx_id is not None
            mnsteps = (self.negotiators[mn_id][0].nmi.state.step) * 2
            if max_diff > (self.max_round_diff * mnsteps + 1):
                msg = (
                    f"{self.negotiators[negotiator_id][0].owner.id}: Max round diff is "
                    f"{max_diff} (allowed up to {self.max_round_diff * mnsteps + 1} with"
                    f" mnsteps {mnsteps} and mn {mn}) which happens in negotiations with {mx_id} ({mx}) and "
                    f"{mn_id} {mn}\n{mn_id} state: {self.negotiators[mn_id][0].nmi.state}"
                    f"\n{mx_id} state: {self.negotiators[mx_id][0].nmi.state}"
                )
                if isinstance(self.negotiators[mn_id][0], ControlledSAONegotiator):
                    sync_partner = [
                        _
                        for _ in self.negotiators[mn_id][0].nmi._mechanism.negotiators
                        if _.owner.id == mn_id
                    ][0].parent
                    for k, v in vars(sync_partner).items():
                        if k.startswith("_SAOSync"):
                            msg += (
                                f'\n{k.replace("_SAOSyncController__", "")}:{str(v)}\n'
                            )
                raise AssertionError(msg)
                # print(msg)
                # log_str = f"{datetime.now()}: on round {self.round_nums[negotiator_id]} with opp {negotiator_id}"
                # self.awi.logdebug_agent(log_str)
        return super().respond(negotiator_id, state, source=source)

    def on_negotiation_success(self, contract, mechanism):
        partners = [_ for _ in contract.partners if _ != self.id]
        assert (
            len(partners) == 1
        ), f"id={self.id}, partners={contract.partners}\n{contract}\n{mechanism.state}"
        if partners[0] in self.round_nums.keys():
            del self.round_nums[partners[0]]

    def on_negotiation_failure(
        self,
        partners,
        annotation,
        mechanism,
        state,
    ) -> None:
        npartners = [_ for _ in partners if _ != self.id]
        assert (
            len(npartners) == 1
        ), f"id={self.id}, partners={partners}\n{annotation}\n{state}"
        if npartners[0] in self.round_nums.keys():
            del self.round_nums[npartners[0]]


def run_experiment(
    n_sync_suppliers: int = 0,
    n_sync_consumers: int = 0,
    consumer_reporter: bool = True,
    supplier_reporter: bool = True,
    seed: int = DEFAULT_SEED,
    n_steps: int = 5,
    name: str = "experiment",
    enable_time_limit=True,
    shuffle_negotiations=False,
):
    suppliers = (
        ([ReporterAgent] if supplier_reporter else [])
        + [SyncAgent] * n_sync_suppliers
        + [BetterAgent] * (2 - n_sync_suppliers)
    )  # type: ignore
    consumers = (
        ([ReporterAgent] if consumer_reporter else [])
        + [SyncAgent] * n_sync_consumers
        + [BetterAgent] * (2 - n_sync_consumers)
    )  # type: ignore
    max_round_diff = (
        len(suppliers) * n_sync_consumers + len(consumers) * n_sync_suppliers + 1
    )
    supplier_params = [
        (
            dict(controller_params=dict(max_round_diff=max_round_diff))
            if supplier_reporter
            else dict()
        )
    ] + [dict() for _ in range(len(suppliers) - 1)]
    consumer_params = [
        (
            dict(controller_params=dict(max_round_diff=max_round_diff))
            if consumer_reporter
            else dict()
        )
    ] + [dict() for _ in range(len(consumers) - 1)]
    types = suppliers + consumers
    params = supplier_params + consumer_params
    if n_steps <= 2:
        raise ValueError("Negmas can't handle less than 3 steps")

    random.seed(seed)
    np.random.seed(seed)

    time_limit_args = (
        {}
        if enable_time_limit
        else {
            "time_limit": math.inf,
            "neg_time_limit": math.inf,
            "neg_step_time_limit": math.inf,
        }
    )

    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
            agent_types=tuple(types),
            agent_params=params,
            n_agents_per_process=len(consumers) + len(suppliers),
            n_processes=2,
            n_steps=n_steps,
            construct_graphs=True,
            shuffle_negotiations=shuffle_negotiations,
            name=name,
            **time_limit_args,  # type: ignore
        )
    )

    # print(f"Running experiment {name}...")
    world.run()


@given(
    n_sync_suppliers=st.integers(0, 2),
    n_sync_consumers=st.integers(0, 2),
    supplier_reporter=st.booleans(),
    consumer_reporter=st.booleans(),
    seed=st.integers(0, 1000),
)
@settings(deadline=600_000, max_examples=200)
def test_sync_experiment_multiple_seeds_2022_conditions(
    n_sync_suppliers,
    n_sync_consumers,
    supplier_reporter,
    consumer_reporter,
    seed,
):
    if not consumer_reporter and not supplier_reporter:
        return
    run_experiment(
        n_sync_suppliers=n_sync_suppliers,
        n_sync_consumers=n_sync_consumers,
        supplier_reporter=supplier_reporter,
        consumer_reporter=consumer_reporter,
        name=f"out_of_sync_bug_s{seed}",
        seed=seed,
        enable_time_limit=False,
        shuffle_negotiations=False,
    )


@pytest.mark.parametrize(
    [
        "n_sync_suppliers",
        "n_sync_consumers",
        "consumer_reporter",
        "supplier_reporter",
        "shuffle_negotiations",
    ],
    list(
        itertools.product(
            range(2),
            range(2),
            (True, False),
            (False, True),
            (False, True),
        )
    ),
)
def test_sync_experiment(
    n_sync_suppliers,
    n_sync_consumers,
    supplier_reporter,
    consumer_reporter,
    shuffle_negotiations,
):
    if not consumer_reporter and not supplier_reporter:
        return
    seed = DEFAULT_SEED
    try:
        run_experiment(
            n_sync_suppliers=n_sync_suppliers,
            n_sync_consumers=n_sync_consumers,
            supplier_reporter=supplier_reporter,
            consumer_reporter=consumer_reporter,
            name=f"out_of_sync_bug_s{seed}",
            seed=seed,
            enable_time_limit=False,
            shuffle_negotiations=shuffle_negotiations,
        )
    except AssertionError as e:
        # we expect errors when negotiations are shuffledd or when we are avoiding ultimatum
        if not shuffle_negotiations:
            raise e


@pytest.mark.skip("May not raise. Check later")
def test_should_raise_in_2020_conditions():
    n_sync_suppliers = 2
    n_sync_consumers = 2
    supplier_reporter = True
    consumer_reporter = True
    shuffle_negotiations = True
    seed = DEFAULT_SEED
    with pytest.raises(AssertionError):
        run_experiment(
            n_sync_suppliers=n_sync_suppliers,
            n_sync_consumers=n_sync_consumers,
            supplier_reporter=supplier_reporter,
            consumer_reporter=consumer_reporter,
            name=f"out_of_sync_bug_s{seed}",
            seed=seed,
            enable_time_limit=False,
            shuffle_negotiations=shuffle_negotiations,
        )


def test_one_sync_supplier_and_one_sync_consumer_active_negotiations_are_synced():
    n_sync_suppliers = 1
    n_sync_consumers = 1
    supplier_reporter = True
    consumer_reporter = True
    shuffle_negotiations = False
    seed = DEFAULT_SEED
    run_experiment(
        n_sync_suppliers=n_sync_suppliers,
        n_sync_consumers=n_sync_consumers,
        supplier_reporter=supplier_reporter,
        consumer_reporter=consumer_reporter,
        name=f"out_of_sync_bug_s{seed}",
        seed=seed,
        enable_time_limit=False,
        shuffle_negotiations=shuffle_negotiations,
    )


def test_one_sync_agents_active_negotiations_are_synced():
    n_sync_suppliers = 1
    n_sync_consumers = 0
    supplier_reporter = True
    consumer_reporter = False
    shuffle_negotiations = False
    seed = DEFAULT_SEED
    run_experiment(
        n_sync_suppliers=n_sync_suppliers,
        n_sync_consumers=n_sync_consumers,
        supplier_reporter=supplier_reporter,
        consumer_reporter=consumer_reporter,
        name=f"out_of_sync_bug_s{seed}",
        seed=seed,
        enable_time_limit=False,
        shuffle_negotiations=shuffle_negotiations,
    )


def test_no_sync_agents_active_negotiations_are_synced():
    n_sync_suppliers = 0
    n_sync_consumers = 0
    supplier_reporter = True
    consumer_reporter = False
    shuffle_negotiations = False
    seed = DEFAULT_SEED
    run_experiment(
        n_sync_suppliers=n_sync_suppliers,
        n_sync_consumers=n_sync_consumers,
        supplier_reporter=supplier_reporter,
        consumer_reporter=consumer_reporter,
        name=f"out_of_sync_bug_s{seed}",
        seed=seed,
        enable_time_limit=False,
        shuffle_negotiations=shuffle_negotiations,
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["--show-capture=no", __file__]))
