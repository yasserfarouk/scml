import sys
import time
from contextlib import contextmanager

import pytest
from negmas import ResponseType
from negmas.helpers import force_single_thread, humanize_time
from negmas.sao import SAOResponse
from pytest import mark, raises

from scml.oneshot.agent import OneShotSyncAgent
from scml.oneshot.agents import RandomOneShotAgent
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE


class MySyncAgent(OneShotSyncAgent):
    in_counter_all = False

    def __init__(self, use_sleep=False, check_negs=False, **kwargs):
        super().__init__(**kwargs)
        self.offers: dict[str, tuple[int, ...]] = {}
        self._delay_using_sleep = use_sleep
        self._check_negs = check_negs
        self._countering_set = set()

    def delay(self):
        """Tune this to take ~3s"""
        if self._delay_using_sleep:
            time.sleep(3.0)
        else:
            a = 0
            for i in range(100000):
                a *= i

    def counter_all(self, offers, states):
        _ = states
        s = {self.get_nmi(_) for _ in offers.keys()}
        if self.in_counter_all and (
            not self._check_negs
            or (self._check_negs and len(self.countering_set.intersection(s)))
        ):
            raise RuntimeError(
                f"uh-oh! new offers: {offers}, previous offers: {self.offers}"
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

    def first_proposals(self) -> dict:
        return dict(
            zip(
                self.negotiators.keys(),
                (self.get_offer(neg_id) for neg_id in self.negotiators.keys()),
            )
        )

    def get_offer(self, negotiator_id: str):
        ami = self.get_nmi(negotiator_id)
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
    _ = err
    yield None


def sync_counter_all_reenters_as_expected(
    use_sleep, check_negs, single_thread, raise_expected
):
    from ..switches import DefaultOneShotWorld

    _strt = time.perf_counter()
    print(
        f"Running with {use_sleep=}, {check_negs=}, {single_thread=}, {raise_expected=} ... ",
        end="",
        flush=True,
    )

    types = {
        (False, False): NotSleepingNotChecking,
        (False, True): NotSleepingChecking,
        (True, False): SleepingNotChecking,
        (True, True): SleepingChecking,
    }

    n_steps = 5

    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
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
    print(f"DONE in {humanize_time(time.perf_counter() - _strt)}", flush=True)


CONDITIONS = (
    (True, True, False, False),
    (True, True, True, False),
    (True, False, True, False),
    (False, True, False, False),
    (False, True, True, False),
    (False, False, True, False),
    (True, False, False, True),
    (False, False, False, True),
)


if pytest in sys.modules:
    from ..switches import SCML_RUN2021_ONESHOT_SYNC, SCML_RUN_TEMP_FAILING

    @mark.parametrize(
        ["use_sleep", "check_negs", "single_thread", "raise_expected"],
        CONDITIONS,
    )
    @pytest.mark.skipif(
        condition=not SCML_RUN2021_ONESHOT_SYNC or not SCML_RUN_TEMP_FAILING,
        reason="Environment set to ignore running oneshot tests. See switches.py",
    )
    def test_sync_counter_all_reenters_as_expected(
        use_sleep, check_negs, single_thread, raise_expected
    ):
        sync_counter_all_reenters_as_expected(
            use_sleep, check_negs, single_thread, raise_expected
        )


if __name__ == "__main__":
    for args in CONDITIONS:
        sync_counter_all_reenters_as_expected(*args)
