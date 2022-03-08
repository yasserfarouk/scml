from pprint import pformat

import pytest
from negmas import ResponseType
from negmas.helpers import single_thread
from negmas.sao import SAOResponse

from scml.oneshot import OneShotSyncAgent
from scml.oneshot import SCML2020OneShotWorld
from scml.oneshot.agents import RandomOneShotAgent
from scml.scml2020.common import QUANTITY
from scml.scml2020.common import TIME
from scml.scml2020.common import UNIT_PRICE


class MyExogAgent(OneShotSyncAgent):
    def step(self):

        super().step()
        assert self.awi.current_exogenous_input_quantity == self.ufun.ex_qin
        assert abs(self.awi.current_exogenous_input_price - self.ufun.ex_pin) < 1e-3
        assert self.awi.current_exogenous_output_quantity == self.ufun.ex_qout
        assert abs(self.awi.current_exogenous_output_price - self.ufun.ex_pout) < 1e-3

    def before_step(self):
        assert self.awi.current_exogenous_input_quantity == self.ufun.ex_qin
        assert abs(self.awi.current_exogenous_input_price - self.ufun.ex_pin) < 1e-3
        assert self.awi.current_exogenous_output_quantity == self.ufun.ex_qout
        assert abs(self.awi.current_exogenous_output_price - self.ufun.ex_pout) < 1e-3

    def counter_all(self, offers, states):
        ex_quant = (
            self.awi.current_exogenous_input_quantity
            if self.awi.is_first_level
            else self.awi.current_exogenous_output_quantity
        )
        is_selling = self.awi.is_first_level

        assert self.awi.current_exogenous_input_quantity == self.ufun.ex_qin
        assert abs(self.awi.current_exogenous_input_price - self.ufun.ex_pin) < 1e-3
        assert self.awi.current_exogenous_output_quantity == self.ufun.ex_qout
        assert abs(self.awi.current_exogenous_output_price - self.ufun.ex_pout) < 1e-3

        utils = []
        price = self.awi.current_output_issues[UNIT_PRICE].max_value
        for i in range(11):
            o = [-1, -1, -1]
            o[UNIT_PRICE] = price
            o[QUANTITY] = i
            o[TIME] = self.awi.current_step
            utils.append(
                self.ufun.from_offers(
                    (
                        tuple(
                            o,
                        ),
                    ),
                    (is_selling,),
                )
            )
        assumed_best = [-1, -1, -1]
        assumed_best[UNIT_PRICE] = price
        assumed_best[QUANTITY] = ex_quant
        assumed_best[TIME] = self.awi.current_step
        assumed_best_u = self.ufun.from_offers((tuple(assumed_best),), (is_selling,))

        best_u, best_quant = max([(u, idx) for idx, u in enumerate(utils)])

        assert (
            (not is_selling)
            or (not self.ufun.ok_to_sell_at(price))
            or (best_u == assumed_best_u)
            or best_quant >= ex_quant
        ), f"best: {best_quant}, exog: {ex_quant}\nu(best)={best_u}, u(exog) = {assumed_best_u}\nprices: {self.awi.trading_prices}\nufun:{pformat(vars(self.ufun))}"

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
        ami = self.get_nmi(negotiator_id)
        quantity_issue = ami.issues[QUANTITY]
        unit_price_issue = ami.issues[UNIT_PRICE]

        offer = [-1] * 3
        offer[QUANTITY] = quantity_issue.max_value
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = unit_price_issue.max_value
        return offer


def test_underfullfilment_is_irrational():
    agent_types = [RandomOneShotAgent, MyExogAgent] * 2
    for _ in range(10):
        world = SCML2020OneShotWorld(
            **SCML2020OneShotWorld.generate(
                agent_types,
                n_steps=100,
                n_processes=2,
                n_agents_per_process=2,
                random_agent_types=False,
            ),
        )
        with single_thread():
            world.run()
