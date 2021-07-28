import os
from typing import Tuple, Dict
import time
from pytest import raises, mark
from contextlib import contextmanager

from negmas import ResponseType
from negmas.sao import SAOResponse

from scml.oneshot.agent import (
    OneShotSyncAgent,
)
from scml.oneshot.agents import RandomOneShotAgent
from scml.oneshot.common import QUANTITY
from scml.oneshot.common import TIME
from scml.oneshot.common import UNIT_PRICE
from negmas.helpers import force_single_thread


class MySyncAgent(OneShotSyncAgent):
    in_counter_all = False

    def __init__(self, use_sleep=False, check_negs=False, **kwargs):
        super().__init__(**kwargs)
        self.offers: Dict[str, Tuple[int]] = {}
        self._delay_using_sleep = use_sleep
        self._check_negs = check_negs
        self._countering_set = set()

    def delay(self):
        """Tune this to take ~3s"""
        if self._delay_using_sleep:
            time.sleep(3.0)
        else:
            a = 0
            for i in range(100000000):
                a *= i

    def counter_all(self, offers, states):
        s = set(self.get_ami(_) for _ in offers.keys())
        if self.in_counter_all and (
            not self._check_negs
            or (self._check_negs and len(self.countering_set.intersection(s)))
        ):
            raise RuntimeError(
                "uh-oh! new offers: {}, previous offers: {}".format(offers, self.offers)
            )

        self.in_counter_all = True
        self.countering_set = s
        self.offers = offers
        self.delay()
        self.in_counter_all = False

        return {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in self.first_proposals().items()
        }

    def first_proposals(self):
        return dict(
            zip(
                self.negotiators.keys(),
                (self.get_offer(neg_id) for neg_id in self.negotiators.keys()),
            )
        )

    def get_offer(self, negotiator_id: str):
        ami = self.get_ami(negotiator_id)
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]

        offer = [-1] * 3
        offer[QUANTITY] = quantity_issue.max_value
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = unit_price_issue.max_value
        return tuple(offer)


class SleepingNotChecking(MySyncAgent):
    def __init__(self, *args, **kwargs):
        kwargs["use_sleep"], kwargs["check_negs"] = True, False
        super().__init__(*args, **kwargs)


class SleepingChecking(MySyncAgent):
    def __init__(self, *args, **kwargs):
        kwargs["use_sleep"], kwargs["check_negs"] = True, True
        super().__init__(*args, **kwargs)


class NotSleepingChecking(MySyncAgent):
    def __init__(self, *args, **kwargs):
        kwargs["use_sleep"], kwargs["check_negs"] = False, True
        super().__init__(*args, **kwargs)


class NotSleepingNotChecking(MySyncAgent):
    def __init__(self, *args, **kwargs):
        kwargs["use_sleep"], kwargs["check_negs"] = False, False
        super().__init__(*args, **kwargs)


@contextmanager
def does_not_raise(err):
    yield None

@mark.skipif(os.environ["GITHUB_ACTIONS"])
@mark.parametrize(
    ["use_sleep", "check_negs", "single_thread", "raise_expected"],
    [
        (False, False, True, False),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, True, False),
        (False, True, False, False),
        (True, True, False, False),
        (True, False, False, False),
        (False, False, False, True),
    ],
)
def test_sync_counter_all_reenters_as_expected(
    use_sleep, check_negs, single_thread, raise_expected
):
    from scml.oneshot import SCML2020OneShotWorld

    types = {
        (False, False): NotSleepingNotChecking,
        (False, True): NotSleepingChecking,
        (True, False): SleepingNotChecking,
        (True, True): SleepingChecking,
    }

    n_steps = 5

    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            [types[(use_sleep, check_negs)], RandomOneShotAgent],
            n_agents_per_process=2,
            n_processes=2,
            n_steps=n_steps,
        ),
        compact=True,
        no_logs=True,
    )
    force_single_thread(single_thread)
    with (raises if raise_expected else does_not_raise)(RuntimeError):
        world.run()
    force_single_thread(False)
