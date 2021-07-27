from typing import Tuple, Dict
import datetime
import random
import time
from pytest import raises

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offers: Dict[str, Tuple[int]] = {}


    def init(self):
        self.step_time = datetime.datetime.now()

    def step(self):
        self.step_time = datetime.datetime.now()

    @staticmethod
    def delay():
        """Tune this to take ~3s"""
        before = datetime.datetime.now()
        time.sleep(3.0)
        # a = 0
        # for i in range(100000000):
        #     a *= i
        after = datetime.datetime.now()
        print("delay time:", after - before)

    def counter_all(self, offers, states):
        if self.in_counter_all:
            raise RuntimeError("uh-oh! new offers: {}, previous offers: {}"
                .format(offers, self.offers))

        self.in_counter_all = True
        self.offers = offers
        self.delay()
        self.in_counter_all = False

        return {
            k: SAOResponse(ResponseType.REJECT_OFFER, v)
            for k, v in self.first_proposals().items()
        }

    def first_proposals(self):
        return  dict(zip(
                self.negotiators.keys(),
                (self.get_offer(neg_id) for neg_id in self.negotiators.keys())
        ))

    def get_offer(self, negotiator_id: str):
        ami = self.get_ami(negotiator_id)
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]

        offer = [-1] * 3
        offer[QUANTITY] = quantity_issue.max_value
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = unit_price_issue.max_value
        return tuple(offer)


def test_sync_counter_all_reenters():
    from scml.oneshot import SCML2020OneShotWorld

    n_steps = 5

    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            [MySyncAgent, RandomOneShotAgent],
            n_agents_per_process=(2, 4),
            n_processes=2,
            n_steps=n_steps,
        ),
        compact=True,
        no_logs=True,
    )
    with raises(RuntimeError):
        world.run()

def test_sync_counter_all_doesnot_reenter_in_a_thread():
    from scml.oneshot import SCML2020OneShotWorld

    n_steps = 5

    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            [MySyncAgent, RandomOneShotAgent],
            n_agents_per_process=(2, 4),
            n_processes=2,
            n_steps=n_steps,
        ),
        compact=True,
        no_logs=True,
    )
    force_single_thread(True)
    world.run()
    force_single_thread(False)
