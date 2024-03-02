import random
from collections import defaultdict

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from negmas import ResponseType
from negmas.genius.bridge import genius_bridge_is_running
from negmas.genius.gnegotiators import NiceTitForTat
from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, ConstFun, IdentityFun, LinearFun
from negmas.sao import SAOResponse
from numpy.testing import assert_allclose
from pytest import mark, raises

import scml
from scml.oneshot import (
    OneShotSingleAgreementAgent,
    builtin_agent_types,
)
from scml.oneshot.agent import (
    OneShotAgent,
    OneShotIndNegotiatorsAgent,
    OneShotSyncAgent,
)
from scml.oneshot.agents import GreedySyncAgent, RandomOneShotAgent
from scml.oneshot.agents.rand import RandDistOneShotAgent
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE, is_system_agent
from scml.oneshot.sysagents import DefaultOneShotAdapter
from scml.oneshot.ufun import OneShotUFun
from scml.oneshot import (
    SCML2020OneShotWorld,
    SCML2021OneShotWorld,
    SCML2022OneShotWorld,
    SCML2023OneShotWorld,
    SCML2024OneShotWorld,
)

from ..switches import (
    SCML_ON_GITHUB,
    DefaultOneShotWorld,
    SCML_TRY2020,
    SCML_TRY2021,
    SCML_TRY2022,
    SCML_TRY2023,
    SCML_TRY2024,
)

random.seed(0)

COMPACT = True
NOLOGS = True
# agent types to be tested
types = builtin_agent_types(False)

active_types = types
std_types = scml.scml2020.builtin_agent_types(as_str=False)
# try:
#     from scml_agents import get_agents
#
#     std_types += list(get_agents(2020, as_class=True, winners_only=True))
# except ImportError:
#     pass


class MyOneShotAgent(RandomOneShotAgent):
    def respond(self, negotiator_id, state, source=None) -> ResponseType:
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        assert (
            negotiator_id in self.awi.my_consumers
            or negotiator_id in self.awi.my_suppliers
        ), (self.id, self.name, negotiator_id)
        return super().respond(negotiator_id, state, source)

    def propose(self, negotiator_id, state):
        assert (
            negotiator_id in self.awi.my_consumers
            or negotiator_id in self.awi.my_suppliers
        ), (self.id, self.name, negotiator_id)
        return super().propose(negotiator_id, state)


def generate_world(
    agent_types,
    world_type=DefaultOneShotWorld,
    n_processes=2,
    n_steps=10,
    n_agents_per_process=2,
    n_lines=10,
    **kwargs,
):
    kwargs["no_logs"] = True
    kwargs["compact"] = True
    kwargs["fast"] = True
    world = world_type(
        **world_type.generate(
            agent_types,
            n_processes=n_processes,
            n_steps=n_steps,
            n_lines=n_lines,
            n_agents_per_process=n_agents_per_process,
            **kwargs,
        )
    )
    for s1, s2 in zip(world.suppliers[:-1], world.suppliers[1:]):
        assert len(set(s1).intersection(set(s2))) == 0
    for s1, s2 in zip(world.consumers[:-1], world.consumers[1:]):
        assert len(set(s1).intersection(set(s2))) == 0
    for p in range(n_processes):
        assert len(world.suppliers[p + 1]) == n_agents_per_process
        assert len(world.consumers[p]) == n_agents_per_process
    for a in world.agents.keys():
        if is_system_agent(a):
            continue
        assert len(world.agent_inputs[a]) == 1
        assert len(world.agent_outputs[a]) == 1
        assert len(world.agent_processes[a]) == 1
        assert len(world.agent_suppliers[a]) == (
            n_agents_per_process if world.agent_inputs[a][0] != 0 else 1
        )
        assert len(world.agent_consumers[a]) == (
            n_agents_per_process if world.agent_outputs[a][0] != n_processes else 1
        )
    return world


# def test_negotiator_ids_are_partner_ids():
#     n_processes = 5
#     world = generate_world(
#         [MyOneShotAgent],
#         n_processes=n_processes,
#         fast=True,
#     )
#     world.run()


@given(n_processes=st.integers(2, 6))
@settings(deadline=300_000, max_examples=20)
def test_quantity_distribution(n_processes):
    from pprint import pformat

    for _ in range(20):
        world = generate_world(
            [MyOneShotAgent],
            n_processes=n_processes,
            compact=True,
            no_logs=True,
        )
        for contracts in world.exogenous_contracts.values():
            for c in contracts:
                for p in c.partners:
                    if is_system_agent(p):
                        continue
                    lines = world.agent_profiles[p].n_lines
                    assert (
                        lines >= c.agreement["quantity"] >= 0
                    ), f"Contract: {str(c)} has negative or more quantity than n. lines {lines}\n{pformat(world.info)}"


world_types = []
for op, w in zip(
    (SCML_TRY2020, SCML_TRY2021, SCML_TRY2022, SCML_TRY2023, SCML_TRY2024),
    (
        SCML2020OneShotWorld,
        SCML2021OneShotWorld,
        SCML2022OneShotWorld,
        SCML2023OneShotWorld,
        SCML2024OneShotWorld,
    ),
):
    if op:
        world_types.append(w)


@pytest.mark.skip("Too slow. Investigate later")
@given(
    agent_type=st.sampled_from(types),
    n_processes=st.integers(2, 4),
    world_type=st.sampled_from(world_types),
)
@settings(deadline=30_000, max_examples=10)
def test_can_run_with_a_single_agent_type(agent_type, world_type, n_processes):
    if issubclass(agent_type, OneShotSingleAgreementAgent) and n_processes > 2:
        return
    world = generate_world(
        [agent_type],
        world_type=world_type,
        n_processes=n_processes,
        compact=COMPACT,
        no_logs=NOLOGS,
        fast=True,
    )
    world.run()


@pytest.mark.skip("Too slow. Investigate later")
@given(
    agent_types=st.lists(
        st.sampled_from(active_types),
        min_size=1,
        max_size=len(active_types),
        unique=True,
    ),
    n_processes=st.integers(2, 4),
)
@settings(deadline=300_000, max_examples=20)
def test_can_run_with_a_multiple_agent_types(agent_types, n_processes):
    if (
        any(issubclass(_, OneShotSingleAgreementAgent) for _ in agent_types)
        and n_processes > 2
    ):
        return
    world = generate_world(
        agent_types,
        n_processes=n_processes,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()


@pytest.mark.skipif(SCML_ON_GITHUB, reason="Known to timeout on CI")
@given(n_processes=st.integers(2, 2))
@settings(deadline=300_000, max_examples=20)
def test_something_happens_with_random_agents(n_processes):
    world = generate_world(
        [RandomOneShotAgent],
        n_processes=n_processes,
        compact=COMPACT,
        no_logs=NOLOGS,
        n_steps=15,
    )
    world.run()
    assert len(world.signed_contracts) + len(world.cancelled_contracts) != 0


def test_basic_awi_info_suppliers_consumers():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
            agent_types=RandomOneShotAgent,
            n_steps=10,
            n_processes=2,
            compact=True,
            no_logs=True,
        )
    )
    for aid in world.agents:
        if is_system_agent(aid):
            continue
        a = world.agents[aid]
        awi: OneShotAWI = a.awi  # type: ignore
        assert a.id in awi.all_suppliers[awi.my_output_product]
        assert a.id in awi.all_consumers[awi.my_input_product]
        assert awi.my_consumers == world.agent_consumers[aid]
        assert awi.my_suppliers == world.agent_suppliers[aid]
        input_product = awi.my_input_product
        assert all(
            _.endswith(str(input_product - 1)) or awi.is_system(_)
            for _ in awi.my_suppliers
        )
        assert all(
            _.endswith(str(input_product + 1)) or awi.is_system(_)
            for _ in awi.my_consumers
        )


def test_generate():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
            agent_types=RandomOneShotAgent,
            n_steps=10,
            n_processes=2,
            compact=True,
            no_logs=True,
        )
    )
    world.run()
    assert True


def test_a_tiny_world():
    world = generate_world(
        [RandomOneShotAgent],
        n_processes=2,
        n_steps=5,
        n_agents_per_process=2,
        n_lines=5,
    )
    world.run()
    assert True


def test_graph():
    world = generate_world(
        [RandomOneShotAgent],
        n_processes=2,
        n_steps=10,
        n_agents_per_process=2,
        n_lines=5,
    )
    world.graph(together=True)
    world.step()
    world.graph(steps=None, together=True)
    world.graph(steps=None, together=False)
    world.run()
    world.graph((0, world.n_steps), together=False)
    world.graph((0, world.n_steps), together=True)


def test_graphs_lead_to_no_unknown_nodes():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(agent_types=[RandomOneShotAgent], n_steps=10),
        construct_graphs=True,
    )
    world.graph((0, world.n_steps))


def test_ufun_min_max_in_world():
    for _ in range(20):
        world = DefaultOneShotWorld(
            **DefaultOneShotWorld.generate(
                agent_types=[RandomOneShotAgent], n_steps=10
            ),
            construct_graphs=False,
            compact=True,
            no_logs=True,
        )
        world.step()
        for aid, agent in world.agents.items():  # type: ignore
            if is_system_agent(aid):
                continue
            agent: DefaultOneShotAdapter
            ufun = agent.make_ufun(add_exogenous=True)
            ufun.find_limit(True)
            ufun.find_limit(False)
            mn, mx = ufun.min_utility, ufun.max_utility
            assert mx >= mn


@given(
    ex_qin=st.integers(0, 3),
    ex_qout=st.integers(0, 3),
    ex_pin=st.integers(2, 10),
    ex_pout=st.integers(2, 10),
    production_cost=st.integers(0, 2),
    disposal_cost=st.floats(0.5, 1.5),
    shortfall_penalty=st.floats(1.5, 2.5),
    level=st.integers(0, 2),
    force_exogenous=st.booleans(),
    lines=st.integers(1, 3),
    balance=st.integers(0, 100),
    input_penalty_scale=st.floats(0.1, 2),
    output_penalty_scale=st.floats(0.1, 4),
    inegs=st.integers(0, 3),
    onegs=st.integers(0, 3),
    perishable=st.booleans(),
)
@settings(deadline=None)
def test_ufun_limits(
    ex_qin,
    ex_qout,
    ex_pin,
    ex_pout,
    production_cost,
    disposal_cost,
    shortfall_penalty,
    level,
    force_exogenous,
    lines,
    balance,
    input_penalty_scale,
    output_penalty_scale,
    inegs,
    onegs,
    perishable,
):
    if perishable:
        storage_cost = 0
    else:
        storage_cost, disposal_cost = disposal_cost / 10, 0
    # these cases do not happen in 2020. May be we still need to test them
    if inegs < 1 and onegs < 1:
        return
    if inegs > 0 and ex_qin > 0:
        return
    if onegs > 0 and ex_qout > 0:
        return
    _ufun_unit2(
        ex_qin,
        ex_qout,
        ex_pin,
        ex_pout,
        production_cost,
        disposal_cost,
        shortfall_penalty,
        level,
        force_exogenous,
        lines,
        balance,
        input_penalty_scale,
        output_penalty_scale,
        inegs,
        onegs,
        storage_cost,
        perishable,
    )


def _ufun_unit2(
    ex_qin,
    ex_qout,
    ex_pin,
    ex_pout,
    production_cost,
    disposal_cost,
    shortfall_penalty,
    level,
    force_exogenous,
    lines,
    balance,
    input_penalty_scale,
    output_penalty_scale,
    inegs,
    onegs,
    storage_cost=0,
    perishable=True,
):
    if level == 0:
        input_agent, output_agent = True, False
    elif level == 1:
        input_agent, output_agent = False, False
    else:
        input_agent, output_agent = False, True

    ufun = OneShotUFun(
        agent_id="",
        perishable=perishable,
        ex_qin=ex_qin,
        ex_qout=ex_qout,
        ex_pin=ex_pin,
        ex_pout=ex_pout,
        production_cost=production_cost,
        disposal_cost=disposal_cost,
        storage_cost=storage_cost,
        shortfall_penalty=shortfall_penalty,
        input_agent=input_agent,
        output_agent=output_agent,
        n_lines=lines,
        force_exogenous=force_exogenous,
        input_product=0 if input_agent else 2,
        time_range=(0, 0),
        input_qrange=(1, 15),
        input_prange=(1, 15),
        output_qrange=(1, 15),
        output_prange=(1, 15),
        n_input_negs=inegs,
        n_output_negs=onegs,
        current_step=0,
        input_penalty_scale=input_penalty_scale,
        output_penalty_scale=output_penalty_scale,
        storage_penalty_scale=input_penalty_scale,
        current_balance=balance,
    )
    worst_gt, best_gt = (
        ufun.find_limit_brute_force(False),
        ufun.find_limit_brute_force(True),
    )
    mn, mx = worst_gt.utility, best_gt.utility
    assert mx >= mn, f"Worst: {worst_gt}\nBest : {best_gt}"
    # if force_exogenous:
    #     best_optimal = ufun.find_limit_optimal(True)
    #     worst_optimal = ufun.find_limit_optimal(False)
    #     assert abs(mx - best_optimal.utility) < 1e-1, f"{best_gt}\n{best_optimal}"
    #     assert (mn - worst_optimal.utility) < 1e-1, f"{worst_gt}\n{worst_optimal}"
    #     best = ufun.find_limit(True)
    #     worst = ufun.find_limit(False)
    #     assert abs(best_gt.utility - mx) < 1e-1, f"{best_gt}\n{best}"
    #     assert abs(worst_gt.utility - mn) < 1e-1, f"{worst_gt}\n{worst}"
    #     # best_greedy = ufun.find_limit_greedy(True)
    #     # worst_greedy = ufun.find_limit_greedy(False)
    #     # assert best_gt == best_greedy
    #     # assert worst_gt == worst_greedy


def test_ufun_limits_example():
    _ufun_unit2(
        ex_qin=0,
        ex_qout=0,
        ex_pin=2,
        ex_pout=2,
        production_cost=0,
        disposal_cost=0.5,
        shortfall_penalty=1.5,
        level=0,
        force_exogenous=True,
        lines=1,
        balance=0,
        input_penalty_scale=0.1,
        output_penalty_scale=0.1,
        inegs=1,
        onegs=1,
    )


@given(
    ex_qin=st.integers(0, 10),
    ex_qout=st.integers(0, 10),
    ex_pin=st.integers(2, 10),
    ex_pout=st.integers(2, 10),
    production_cost=st.integers(1, 5),
    disposal_cost=st.floats(0.5, 1.5),
    shortfall_penalty=st.floats(1.5, 2.5),
    perishable=st.booleans(),
    level=st.integers(0, 2),
    force_exogenous=st.booleans(),
    qin=st.integers(0, 10),
    qout=st.integers(0, 10),
    pin=st.integers(2, 10),
    pout=st.integers(2, 10),
    lines=st.integers(1, 15),
    balance=st.integers(0, 1000),
)
@settings(deadline=None)
def test_ufun_unit(
    ex_qin,
    ex_qout,
    ex_pin,
    ex_pout,
    production_cost,
    disposal_cost,
    shortfall_penalty,
    level,
    force_exogenous,
    qin,
    qout,
    pin,
    pout,
    lines,
    balance,
    perishable,
):
    if perishable:
        storage_cost = 0
    else:
        storage_cost, disposal_cost = disposal_cost / 10, 0

    _ufun_unit(
        ex_qin,
        ex_qout,
        ex_pin,
        ex_pout,
        production_cost,
        disposal_cost,
        shortfall_penalty,
        level,
        force_exogenous,
        qin,
        qout,
        pin,
        pout,
        lines,
        balance,
        storage_cost=storage_cost,
        perishable=perishable,
    )


def _ufun_unit(
    ex_qin,
    ex_qout,
    ex_pin,
    ex_pout,
    production_cost,
    disposal_cost,
    shortfall_penalty,
    level,
    force_exogenous,
    qin,
    qout,
    pin,
    pout,
    lines,
    balance,
    storage_cost,
    perishable,
):
    if level == 0:
        input_agent, output_agent = True, False
    elif level == 1:
        input_agent, output_agent = False, False
    else:
        input_agent, output_agent = False, True

    ufun = OneShotUFun(
        agent_id="",
        ex_qin=ex_qin,
        ex_qout=ex_qout,
        ex_pin=ex_pin,
        ex_pout=ex_pout,
        production_cost=production_cost,
        disposal_cost=disposal_cost,
        storage_cost=storage_cost,
        perishable=perishable,
        shortfall_penalty=shortfall_penalty,
        input_agent=input_agent,
        output_agent=output_agent,
        n_lines=lines,
        force_exogenous=force_exogenous,
        input_product=0 if input_agent else 2,
        time_range=(0, 0),
        input_qrange=(1, 15),
        input_prange=(1, 15),
        output_qrange=(1, 15),
        output_prange=(1, 15),
        n_input_negs=5,
        n_output_negs=5,
        current_step=0,
        input_penalty_scale=1,
        output_penalty_scale=3,
        storage_penalty_scale=1,
        current_balance=balance,
    )
    # if force_exogenous:
    # for v in (True, False):
    # a = ufun.find_limit_greedy(v)
    # try:
    #     b = ufun.find_limit_optimal(v, check=True)
    # except:
    #     pass
    # else:
    #     # assert a == b, f"Failed for {v} Greedy gave {a}\nOptimal gave {b}"
    #     assert (v and b >= a) or (
    #         not v and b <= a
    #     ), f"Failed for {v} Greedy gave {a}\nOptimal gave {b}"
    # ufun.best = ufun.find_limit(True)
    # ufun.worst = ufun.find_limit(False)
    #
    mn, mx = ufun.min_utility, ufun.max_utility
    if mx is None:
        mx = float("inf")
    if mn is None:
        mn = float("-inf")

    assert mx >= mn or mx == mn == 0
    ufun.from_offers(
        ((qin, 0, pin / qin if qin else 0), (qout, 0, pout / qout if qout else 0)),
        (False, True),
    )
    # u = ufun.from_aggregates(qin, qout, pin, pout)
    # assert mn <= u <= mx, f"{mn}, {u}, {mx}\nworst: {ufun.worst}\nbest: {ufun.best}"


def test_ufun_unit_example():
    _ufun_unit(
        ex_qin=0,
        ex_qout=1,
        ex_pin=2,
        ex_pout=2,
        production_cost=1,
        disposal_cost=0.5,
        shortfall_penalty=1.5,
        level=0,
        force_exogenous=True,
        qin=0,
        qout=0,
        pin=2,
        pout=2,
        lines=10,
        balance=float("inf"),
        perishable=True,
        storage_cost=0.0,
    )


def test_ufun_example():
    _ufun_unit(
        ex_qin=0,
        ex_qout=0,
        ex_pin=0,
        ex_pout=0,
        production_cost=1,
        disposal_cost=0.5,
        shortfall_penalty=1.5,
        level=0,
        force_exogenous=False,
        qin=1,
        qout=1,
        pin=2,
        pout=4,
        lines=10,
        balance=float("inf"),
        perishable=True,
        storage_cost=0.0,
    )


def test_builtin_agent_types():
    from negmas.helpers import get_full_type_name

    strs = scml.oneshot.builtin_agent_types(True)
    types = scml.oneshot.builtin_agent_types(False)
    assert len(strs) == len(types)
    assert len(strs) > 0
    assert all(
        [
            get_full_type_name(a).split(".")[-1] == b.split(".")[-1]
            for a, b in zip(types, strs)
        ]
    )


def test_builtin_aspiration():
    from scml.oneshot import SingleAgreementAspirationAgent

    n_processes = 2
    world = generate_world(
        [SingleAgreementAspirationAgent],
        n_processes=n_processes,
        compact=True,
        no_logs=True,
    )
    world.run()


# @pytest.mark.skipif(SCML_ON_GITHUB, reason="Known to timeout on CI")
@pytest.mark.skip(reason="Not sure why but it is failing.")
@given(
    atype=st.lists(
        st.sampled_from(std_types + types),
        unique=True,
        min_size=2,
        max_size=6,  # type: ignore
    )
)
@settings(deadline=900_000, max_examples=10)
def test_adapter(atype):
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(agent_types=atype, n_steps=10),
        construct_graphs=False,
        compact=True,
        no_logs=True,
    )
    world.run()


class MyIndNeg(OneShotIndNegotiatorsAgent):
    def generate_ufuns(self) -> dict:
        return defaultdict(lambda: self.ufun)


class MyGeniusIndNeg(OneShotIndNegotiatorsAgent):
    def __init__(self, *args, **kwargs):
        kwargs["default_negotiator_type"] = NiceTitForTat
        kwargs["normalize_ufuns"] = True
        kwargs["set_reservation"] = True
        super().__init__(*args, **kwargs)

    def generate_ufuns(self):
        d = dict()
        # generate ufuns that prever higher prices when selling
        for partner_id in self.awi.my_consumers:
            if self.awi.is_system(partner_id):
                continue
            d[partner_id] = LinearAdditiveUtilityFunction(
                dict(
                    quantity=LinearFun(0.1),
                    time=ConstFun(0),
                    unit_price=IdentityFun(),
                ),
                weights=dict(
                    quantity=0.1,
                    time=0.0,
                    unit_price=0.9,
                ),
                issues=self.awi.current_output_issues,
                reserved_value=0.0,
            )
        # generate ufuns that prever lower prices when selling
        for partner_id in self.awi.my_suppliers:
            issues = self.awi.current_input_issues
            if self.awi.is_system(partner_id):
                continue
            d[partner_id] = LinearAdditiveUtilityFunction(
                dict(
                    quantity=IdentityFun(),
                    time=ConstFun(0.0),
                    unit_price=AffineFun(-1, issues[UNIT_PRICE].max_value),
                ),
                weights=dict(
                    quantity=0.1,
                    time=0.0,
                    unit_price=0.9,
                ),
                issues=self.awi.current_input_issues,
                reserved_value=0.0,
            )
        return d


def test_ind_negotiators():
    n_processes = 2
    world = generate_world(
        [MyIndNeg],
        n_processes=n_processes,
        compact=True,
        no_logs=True,
    )
    world.run()


@pytest.mark.skipif(
    condition=not genius_bridge_is_running(),
    reason="No Genius Bridge, skipping genius-agent tests",
)
def test_ind_negotiators_genius():
    n_processes = 5
    world = generate_world(
        [MyGeniusIndNeg],
        n_processes=n_processes,
        compact=True,
        no_logs=True,
    )
    world.run()


def test_production_cost_increase():
    from scml.oneshot.agents import GreedyOneShotAgent

    NPROCESSES = 5
    costs = [[] for _ in range(NPROCESSES)]
    for _ in range(100):
        world = DefaultOneShotWorld(
            **DefaultOneShotWorld.generate(
                GreedyOneShotAgent,
                n_agents_per_process=10,
                n_processes=NPROCESSES,
            ),
            compact=True,
            no_logs=True,
        )
        for aid in world.agent_profiles.keys():
            if is_system_agent(aid):
                continue
            profile = world.agent_profiles[aid]
            costs[profile.input_product].append(profile.cost)
    mean_costs = [sum(_) / len(_) for _ in costs]
    assert all(
        [
            b > (0.5 * (i + 2) / (i + 1)) * a
            for i, (a, b) in enumerate(zip(mean_costs[:-1], mean_costs[1:]))
        ]
    ), f"non-ascending costs {mean_costs}"


@mark.parametrize("penalties_scale", ["trading", "catalog"])
def test_ufun_penalty_scales_are_correct(penalties_scale):
    from scml.oneshot.ufun import OneShotUFun

    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
            MyIndNeg,
            n_agents_per_process=3,
            n_processes=2,
            n_steps=30,
            penalties_scale=penalties_scale,
        ),
        compact=True,
        no_logs=True,
    )
    for _ in range(30):
        old_trading = world.trading_prices.copy()
        world.step()
        for _, a in world.agents.items():
            awi = a.awi  # type: ignore
            awi: OneShotAWI
            if is_system_agent(a.id):
                continue
            u: OneShotUFun = a.ufun  # type: ignore
            if penalties_scale == "trading":
                assert (
                    u.output_penalty_scale
                    # == awi.trading_prices[awi.my_output_product]
                    == old_trading[awi.my_output_product]
                )
                assert (
                    u.input_penalty_scale
                    # == awi.trading_prices[awi.my_input_product]
                    == old_trading[awi.my_input_product]
                )
            else:
                assert (
                    u.output_penalty_scale == awi.catalog_prices[awi.my_output_product]
                )
                assert u.input_penalty_scale == awi.catalog_prices[awi.my_input_product]


class MySyncAgent(OneShotSyncAgent):
    new_step = False

    def step(self):
        self.new_step = True

    def counter_all(self, offers, states):
        """Respond to a set of offers given the negotiation state of each."""
        if self.new_step:
            raise RuntimeError("counter_all before first_proposals")
        return dict(
            zip(
                self.negotiators.keys(),
                [SAOResponse(ResponseType.END_NEGOTIATION, None)]
                * len(self.negotiators),
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

    def first_proposals(self) -> dict:
        """Decide a first proposal on every negotiation.
        Returning None for a negotiation means ending it."""
        self.new_step = False
        return dict(
            zip(
                self.negotiators.keys(),
                (self.get_offer(neg_id) for neg_id in self.negotiators.keys()),
            )
        )


@mark.parametrize(
    ("agent_types", "n_agents_per_process", "n_processes"),
    [
        ([], 2, 2),
        ([], 3, 2),
        ([], 4, 2),
        ([], 5, 2),
        ([], 6, 2),
        ([RandomOneShotAgent], 2, 2),
        ([RandomOneShotAgent], 3, 2),
        ([RandomOneShotAgent], 4, 2),
        ([RandomOneShotAgent], 5, 2),
        ([RandomOneShotAgent], 6, 2),
        ([GreedySyncAgent], 2, 2),
        ([GreedySyncAgent], 3, 2),
        ([GreedySyncAgent], 4, 2),
        ([GreedySyncAgent], 5, 2),
        ([GreedySyncAgent], 6, 2),
        ([RandDistOneShotAgent], 2, 2),
        ([RandDistOneShotAgent], 3, 2),
        ([RandDistOneShotAgent], 4, 2),
        ([RandDistOneShotAgent], 5, 2),
        ([RandDistOneShotAgent], 6, 2),
    ],
)
def test_sync_agent_receives_first_proposals_before_counter_all(
    agent_types, n_agents_per_process, n_processes
):
    n_steps = 50
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
            [MySyncAgent] + agent_types,
            n_processes=n_processes,
            n_steps=n_steps,
            n_lines=10,
            n_agents_per_process=n_agents_per_process,
            no_logs=True,
            compact=True,
            ignore_agent_exceptions=False,
            ignore_negotiation_exceptions=False,
            ignore_contract_execution_exceptions=False,
            ignore_simulation_exceptions=False,
        )
    )
    world.run()
    assert world.current_step >= n_steps - 1


class MyRandomAgent(RandomOneShotAgent):
    def has_trade(self, s=None):
        if s is None:
            s = self.awi.current_step
        return (self.awi._world._sold_quantity[:, s] > 0) & (
            np.abs(self.awi._world._real_price[:, s] - self.catalog) >= 1
        )

    def trading_(self, s=None):
        if s is None:
            s = self.awi.current_step
        return self.awi._world._trading_price[:, s]

    def init(self):
        super().init()
        self.trading, self.trading_before = [], []
        self.catalog = self.awi.catalog_prices.copy()
        self.n_products = self.awi.n_products
        assert_allclose(self.catalog, self.trading_())
        assert_allclose(self.catalog, self.awi.trading_prices)

    def before_step(self):
        self.trading_before.append(self.awi.trading_prices.tolist())

        assert_allclose(self.trading_before[-1], self.trading_())
        assert_allclose(self.catalog, self.awi.catalog_prices)

    def step(self):
        super().step()
        self.trading.append(self.awi.trading_prices.tolist())

        assert_allclose(self.catalog, self.awi.catalog_prices)
        assert_allclose(self.trading[-1], self.trading_())
        with_trade = self.has_trade()

        if not np.any(with_trade):
            return
        with raises(AssertionError):
            assert_allclose(self.trading[-1], self.catalog)


@mark.parametrize(
    ["n_agents", "n_processes", "n_steps"],
    [
        (1, 2, 16),
        (2, 2, 16),
        (5, 2, 16),
        (1, 3, 16),
        (2, 3, 16),
        (5, 3, 16),
    ],
)
def test_trading_prices_updated(n_agents, n_processes, n_steps):
    from negmas.helpers import force_single_thread

    eps = 1e-3

    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            [MyRandomAgent] * n_processes,
            n_agents_per_process=n_agents,
            n_processes=n_processes,
            n_steps=n_steps,
        ),
        compact=True,
        no_logs=True,
    )
    catalog_prices = world.catalog_prices
    diffs = np.zeros_like(catalog_prices)

    # we start at catlaog prices
    for aid, agent in world.agents.items():
        assert np.abs(agent.awi.trading_prices - catalog_prices).max() < eps  # type: ignore

    force_single_thread(True)
    for _ in range(n_steps):
        world.step()
        trading_prices = None
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
            trading_prices = agent.awi.trading_prices.copy()  # type: ignore
            break
        diffs = np.maximum(diffs, np.abs(trading_prices - catalog_prices))

    assert diffs.max() > eps
    force_single_thread(False)


class MyOneShotDoNothing(OneShotAgent):
    def propose(self, negotiator_id, state) -> Outcome | None:
        return None

    def respond(self, negotiator_id, state, source=None):
        return ResponseType.END_NEGOTIATION


def test_do_nothing_goes_bankrupt():
    world = generate_world(
        [MyOneShotDoNothing], DefaultOneShotWorld, 2, 1000, 4, cash_availability=0.001
    )
    world.run()
    for aid, agent in world.agents.items():
        if is_system_agent(aid):
            continue
        assert world.is_bankrupt[
            aid
        ], f"Agent {aid} is not bankrupt with balance {world.current_balance(aid)} and initial_balance of {world.initial_balances[aid]}"  # type: ignore
        assert agent.awi.is_bankrupt()  # type: ignore


class PricePumpingAgent(OneShotAgent):
    """An agent that causes the intermediate price to go up over time"""

    def top_outcome(self, negotiator_id):
        return tuple(_.max_value for _ in self.get_nmi(negotiator_id).issues)

    def propose(self, negotiator_id, state):
        return self.top_outcome(negotiator_id)

    def respond(self, negotiator_id, state, source=None):
        offer = state.current_offer  # type: ignore
        return (
            ResponseType.ACCEPT_OFFER
            if self.top_outcome(negotiator_id) == offer
            else ResponseType.REJECT_OFFER
        )


def check_trading_explosion(world, checked_types=(PricePumpingAgent,)):
    stats = world.stats_df
    contracts = pd.DataFrame(world.saved_contracts)
    negotiations = pd.DataFrame(world.saved_negotiations)
    for aid, agent in world.agents.items():
        if is_system_agent(aid):
            continue
        if not agent.awi.is_last_level or not any(
            issubclass(agent.__class__, _) for _ in checked_types
        ):
            continue
        # all sellers should go bankrupt
        assert world.is_bankrupt[aid]
        assert agent.awi.is_bankrupt()
        # bankrupt agents remain bankrupt
        bankrupt = stats[f"bankrupt_{aid}"].values.astype(bool)
        bankrupt_first_index = np.where(bankrupt is True)[0][0]
        assert not np.any(bankrupt[:bankrupt_first_index])
        assert np.all(bankrupt[bankrupt_first_index:])

        # cannot become bankrupt without contracts
        assert (
            len(
                contracts.loc[
                    (contracts.signed_at <= bankrupt_first_index)
                    & (contracts.buyer_name == aid),
                    :,
                ]
            )
            > 0
        )

        # no contracts after becoming bankrupt
        assert (
            len(
                contracts.loc[
                    (contracts.signed_at > bankrupt_first_index)
                    & (contracts.buyer_name == aid),
                    :,
                ]
            )
            == 0
        )

        # no negotiations after becoming bankrupt
        assert (
            len(
                negotiations.loc[
                    (negotiations.requested_at > bankrupt_first_index)
                    & (negotiations.partners.apply(lambda x: aid in x)),
                    :,
                ]
            )
            == 0
        )

        # some negotiations before becoming bankrupt
        assert (
            len(
                negotiations.loc[
                    (negotiations.requested_at <= bankrupt_first_index)
                    & (negotiations.partners.apply(lambda x: aid in x)),
                    :,
                ]
            )
            > 0
        )


def test_price_pumping_happen():
    world = generate_world([PricePumpingAgent], DefaultOneShotWorld, 2, 300, 4)
    world.run()
    check_trading_explosion(world)


def test_price_pumping_happen_with_random_included():
    world = generate_world(
        [PricePumpingAgent, RandomOneShotAgent], DefaultOneShotWorld, 2, 300, 4
    )
    world.run()
    check_trading_explosion(world)


def test_production_capacity():
    world = generate_world(
        [PricePumpingAgent, RandomOneShotAgent], DefaultOneShotWorld, 2, 300, 4
    )
    agent = list(world.agents.values())[0]
    assert agent._obj is not None
    world.step()
    awi: OneShotAWI = agent._obj.awi  # type: ignore
    assert awi is not None
    consumers = awi.all_consumers
    for a, b in zip(consumers[:-1], awi.production_capacities, strict=True):
        assert b == len(a) * awi.profile.n_lines


def test_price_pumping_bankrupts_random_agents():
    types = [PricePumpingAgent, RandomOneShotAgent]
    world = generate_world(types, DefaultOneShotWorld, 2, 300, 4)
    world.run()
    check_trading_explosion(world, types)
