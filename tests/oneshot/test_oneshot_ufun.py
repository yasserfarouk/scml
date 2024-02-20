from pprint import pformat

from negmas import ResponseType
from negmas.helpers import single_thread
from negmas.sao import SAOResponse

from scml.oneshot import OneShotSyncAgent
from scml.oneshot.agents import RandomOneShotAgent
from scml.oneshot.ufun import OneShotUFun
from scml.scml2020.common import QUANTITY, TIME, UNIT_PRICE

from ..switches import DefaultOneShotWorld


class MyExogAgent(OneShotSyncAgent):
    def step(self):
        super().step()
        assert isinstance(self.ufun, OneShotUFun)
        assert self.awi.current_exogenous_input_quantity == self.ufun.ex_qin
        assert abs(self.awi.current_exogenous_input_price - self.ufun.ex_pin) < 1e-3
        assert self.awi.current_exogenous_output_quantity == self.ufun.ex_qout
        assert abs(self.awi.current_exogenous_output_price - self.ufun.ex_pout) < 1e-3

    def before_step(self):
        assert isinstance(self.ufun, OneShotUFun)
        assert self.awi.current_exogenous_input_quantity == self.ufun.ex_qin
        assert abs(self.awi.current_exogenous_input_price - self.ufun.ex_pin) < 1e-3
        assert self.awi.current_exogenous_output_quantity == self.ufun.ex_qout
        assert abs(self.awi.current_exogenous_output_price - self.ufun.ex_pout) < 1e-3

    def counter_all(self, offers, states):
        assert isinstance(self.ufun, OneShotUFun)
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
            o = tuple(o)
            utils.append(
                self.ufun.from_offers(
                    (o,),  # type: ignore
                    (is_selling,),
                )
            )
        assumed_best = [-1, -1, -1]
        assumed_best[UNIT_PRICE] = price
        assumed_best[QUANTITY] = ex_quant
        assumed_best[TIME] = self.awi.current_step
        assumed_best_u = self.ufun.from_offers((tuple(assumed_best),), (is_selling,))  # type: ignore

        best_u, best_quant = max((u, idx) for idx, u in enumerate(utils))

        assert (
            (not is_selling)
            or (not self.ufun.ok_to_sell_at(price))
            or (best_u == assumed_best_u)
            or best_quant >= ex_quant
        ), f"best: {best_quant}, exog: {ex_quant}\nu(best)={best_u}, u(exog) = {assumed_best_u}\nprices: {self.awi.trading_prices}\nufun:{pformat(vars(self.ufun))}"

        return {
            k: SAOResponse(ResponseType.REJECT_OFFER, tuple(v))
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
        return offer


def test_underfullfilment_is_irrational():
    agent_types = [RandomOneShotAgent, MyExogAgent] * 2
    for _ in range(10):
        world = DefaultOneShotWorld(
            **DefaultOneShotWorld.generate(
                agent_types,
                n_steps=100,
                n_processes=2,
                n_agents_per_process=2,
                random_agent_types=False,
            ),
        )
        with single_thread():
            world.run()


def test_uses_registered_sales_and_supplies():
    u = OneShotUFun(
        ex_pin=10 * 10,
        ex_qin=10,
        ex_pout=15 * 5,
        ex_qout=5,
        input_product=0,
        input_agent=True,
        output_agent=False,
        production_cost=2,
        disposal_cost=0.1,
        storage_cost=0.0,
        shortfall_penalty=0.3,
        input_penalty_scale=None,
        output_penalty_scale=None,
        storage_penalty_scale=None,
        n_input_negs=4,
        n_output_negs=5,
        current_step=6,
        time_range=(0, 0),
        input_qrange=(1, 10),
        input_prange=(11, 12),
        output_qrange=(1, 10),
        output_prange=(14, 15),
        suppliers={"a", "b", "c"},
        consumers={"d", "e", "f"},
        agent_id="",
    )
    info = u.from_offers(
        tuple(), tuple(), return_info=True, ignore_signed_contracts=True
    )
    p = info.producible
    # assert correct production amount (what can and needs to be produced)
    assert p == 5
    # assert correct estimate for no agreements
    nosale = 15 * 5 - 10 * 10 - 5 * 2 - 0.1 * 5 * 10
    assert (
        u.from_offers(tuple(), tuple(), return_info=False, ignore_signed_contracts=True)
        == nosale
    )
    # assert best possible agreement
    optim = u.from_offers(
        ((5, 6, 15),),
        (True,),
        return_info=False,
        ignore_signed_contracts=True,
    )
    assert optim == (15 * 10 - 10 * 10 - 10 * 2)
    # ignoring signed contracts changes nothing when there are no signed contracts
    assert optim == u.from_offers(
        ((5, 6, 15),),
        # (True,),
        return_info=False,
        ignore_signed_contracts=False,
    )
    # distributing agreements changes nothing
    for ignore in (True, False):
        assert optim == u.from_offers(
            (
                (2, 6, 14),
                (2, 6, 16),
                (1, 6, 15),
            ),
            # (True, True, True),
            return_info=False,
            ignore_signed_contracts=ignore,
        )
    expected = 15 * 5 + 14 * 2 - 10 * 10 - 7 * 2 - 0.1 * 3 * 10
    for ignore in (True, False):
        assert expected == u.from_offers(
            ((2, 6, 14),),
            # (True,),
            return_info=False,
            ignore_signed_contracts=ignore,
        )
    u.register_sale(2, 14, u.current_step)
    assert u._signed_is_output[0] is True
    assert u._signed_agreements[0] == (2, 6, 14)
    assert len(u._signed_agreements) == len(u._signed_is_output) == 1
    assert (
        u.from_offers(
            tuple(), tuple(), return_info=False, ignore_signed_contracts=False
        )
        == expected
    )
    assert (
        u.from_offers(tuple(), tuple(), return_info=False, ignore_signed_contracts=True)
        == nosale
    )

    assert optim == u.from_offers(
        (
            (2, 6, 16),
            (1, 6, 15),
        ),
        # (True, True),
        return_info=False,
        ignore_signed_contracts=False,
    )

    expected = 15 * 5 + 16 * 2 + 1 * 15 - 10 * 10 - 8 * 2 - 0.1 * 2 * 10
    assert expected == u.from_offers(
        (
            (2, 6, 16),
            (1, 6, 15),
        ),
        # (True, True),
        return_info=False,
        ignore_signed_contracts=True,
    )
    u.register_supply(2, 8, u.current_step)
    assert u._signed_is_output[1] is False
    assert u._signed_agreements[1] == (2, 6, 8)
    assert len(u._signed_agreements) == len(u._signed_is_output) == 2
    assert expected == u.from_offers(
        (
            (2, 6, 16),
            (1, 6, 15),
        ),
        # (True, True),
        return_info=False,
        ignore_signed_contracts=True,
    )
    u.n_lines = 12
    expected = 15 * 10 + 13 * 2 - 10 * 10 - 2 * 8 - 12 * 2
    assert expected == u.from_offers(
        (
            (2, 6, 16),
            (1, 6, 15),
            (2, 6, 13),
        ),
        # (True, True, True),
        return_info=False,
        ignore_signed_contracts=False,
    )
    # remove last supply contract
    u._signed_is_output = u._signed_is_output[:1]
    u._signed_agreements = u._signed_agreements[:1]
    assert expected == u.from_offers(
        (
            (2, 6, 16),
            (1, 6, 15),
            (2, 6, 13),
            (2, 6, 8),
        ),
        (True, True, True, False),
        return_info=False,
        ignore_signed_contracts=False,
    )

    assert expected == u.from_offers(
        dict(
            d=(2, 6, 16),
            e=(1, 6, 15),
            f=(2, 6, 13),
            a=(2, 6, 8),
        ),
        return_info=False,
        ignore_signed_contracts=False,
    )


def test_find_limit():
    u = OneShotUFun(
        ex_pin=10 * 10,
        ex_qin=10,
        ex_pout=0,
        ex_qout=0,
        input_product=0,
        input_agent=True,
        output_agent=False,
        production_cost=2,
        disposal_cost=0.1,
        shortfall_penalty=0.3,
        storage_cost=0,
        input_penalty_scale=None,
        output_penalty_scale=None,
        storage_penalty_scale=None,
        n_input_negs=0,
        n_output_negs=5,
        current_step=6,
        time_range=(0, 0),
        input_qrange=(1, 10),
        input_prange=(11, 12),
        output_qrange=(1, 10),
        output_prange=(14, 15),
        consumers={"d", "e", "f", "g", "h"},
        agent_id="",
    )
    u.find_limit(True, ignore_signed_contracts=True)
    u.find_limit(False, ignore_signed_contracts=True)
    assert u.max_utility == (10 * 15 - 10 * 12)
    assert int(u.min_utility) == -148
    assert u._best is not None
    assert u._worst is not None
    assert u.best_option.input_price == 11
    assert u.best_option.input_quantity == 0
    assert u.best_option.output_quantity == 10
    assert u.best_option.output_price == 15
    assert u.worst_option.input_price == 12
    assert u.worst_option.input_quantity == 0
    assert u.worst_option.output_quantity == 50
    assert u.worst_option.output_price == 14

    u.register_sale_failure("d")
    u.find_limit(True, ignore_signed_contracts=True)
    u.find_limit(False, ignore_signed_contracts=True)
    assert u.max_utility == (10 * 15 - 10 * 12)
    assert int(u.min_utility) == -148
    assert u.best_option.input_price == 11
    assert u.best_option.input_quantity == 0
    assert u.best_option.output_quantity == 10
    assert u.best_option.output_price == 15
    assert u.worst_option.input_price == 12
    assert u.worst_option.input_quantity == 0
    assert u.worst_option.output_quantity == 50
    assert u.worst_option.output_price == 14

    u.find_limit(True, ignore_signed_contracts=False)
    u.find_limit(False, ignore_signed_contracts=False)
    assert u.max_utility == (10 * 15 - 10 * 12)
    assert int(u.min_utility) == -110
    assert u.best_option.input_price == 11
    assert u.best_option.input_quantity == 0
    assert u.best_option.output_quantity == 10
    assert u.best_option.output_price == 15
    assert u.worst_option.input_price == 12
    assert u.worst_option.input_quantity == 0
    assert u.worst_option.output_quantity == 0
    assert u.worst_option.output_price == 14

    assert u.from_offers(
        tuple(), tuple(), ignore_signed_contracts=False
    ) <= u.from_offers(((40, 5, 14),), (True,), ignore_signed_contracts=False)

    u.register_sale(3, 14, u.current_step)
    u.register_sale(1, 14, u.current_step)
    u.find_limit(True, ignore_signed_contracts=True)
    u.find_limit(False, ignore_signed_contracts=True)
    assert u.max_utility == (10 * 15 - 10 * 12)
    assert int(u.min_utility) == -148
    assert u.best_option.input_price == 11
    assert u.best_option.input_quantity == 0
    assert u.best_option.output_quantity == 10
    assert u.best_option.output_price == 15
    assert u.worst_option.input_price == 12
    assert u.worst_option.input_quantity == 0
    assert u.worst_option.output_quantity == 50
    assert u.worst_option.output_price == 14

    u.find_limit(True, ignore_signed_contracts=False)
    u.find_limit(False, ignore_signed_contracts=False)
    assert u.max_utility == (6 * 15 + 4 * 14 - 10 * 12)
    assert int(u.min_utility) == -58
    assert u.best_option.input_price == 11
    assert u.best_option.input_quantity == 0
    assert u.best_option.output_quantity == 6
    assert u.best_option.output_price == 15
    assert u.worst_option.input_price == 12
    assert u.worst_option.input_quantity == 0
    assert u.worst_option.output_quantity == 0
    assert u.worst_option.output_price == 14

    assert u.from_offers(
        tuple(), tuple(), ignore_signed_contracts=False
    ) <= u.from_offers(((20, 5, 14),), (True,), ignore_signed_contracts=False)
