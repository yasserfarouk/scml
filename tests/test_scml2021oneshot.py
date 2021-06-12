import random
from collections import defaultdict

import hypothesis.strategies as st
import pytest
from hypothesis import given
from hypothesis import settings
from negmas import genius_bridge_is_running
from negmas import save_stats
from negmas.genius import Atlas3
from negmas.genius import GeniusNegotiator
from negmas.genius import NiceTitForTat
from negmas.genius import YXAgent
from negmas.genius.ginfo import ALL_PASSING_NEGOTIATORS
from negmas.helpers import unique_name
from negmas.utilities import LinearUtilityAggregationFunction
from negmas.utilities import LinearUtilityFunction
from pytest import mark

import scml
from scml.oneshot import SCML2020OneShotWorld
from scml.oneshot import builtin_agent_types
from scml.oneshot.agent import OneShotIndNegotiatorsAgent
from scml.oneshot.agents import RandomOneShotAgent
from scml.oneshot.common import QUANTITY
from scml.oneshot.common import TIME
from scml.oneshot.common import UNIT_PRICE
from scml.oneshot.ufun import OneShotUFun
from scml.scml2020 import is_system_agent
from scml.scml2020.components import production

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
    def respond(self, negotiator_id, state, offer):
        if not (
            negotiator_id in self.awi.my_consumers
            or negotiator_id in self.awi.my_suppliers
        ):
            breakpoint()
        assert (
            negotiator_id in self.awi.my_consumers
            or negotiator_id in self.awi.my_suppliers
        ), (self.id, self.name, negotiator_id)
        return super().respond(negotiator_id, state, offer)

    def propose(self, negotiator_id, state):
        if not (
            negotiator_id in self.awi.my_consumers
            or negotiator_id in self.awi.my_suppliers
        ):
            breakpoint()
        assert (
            negotiator_id in self.awi.my_consumers
            or negotiator_id in self.awi.my_suppliers
        ), (self.id, self.name, negotiator_id)
        return super().propose(negotiator_id, state)


def generate_world(
    agent_types,
    n_processes=3,
    n_steps=10,
    n_agents_per_process=2,
    n_lines=10,
    **kwargs,
):
    kwargs["no_logs"] = True
    kwargs["compact"] = True
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
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


def test_negotiator_ids_are_partner_ids():
    n_processes = 5
    world = generate_world(
        [MyOneShotAgent],
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/{MyOneShotAgent.__name__}" f"Fine{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        compact=True,
        no_logs=True,
    )
    world.run()
    # save_stats(world, world.log_folder)


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


@mark.parametrize("agent_type", types)
@given(n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=20)
def test_can_run_with_a_single_agent_type(agent_type, n_processes):
    world = generate_world(
        [agent_type],
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/{agent_type.__name__}" f"Fine{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    save_stats(world, world.log_folder)


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
    world = generate_world(
        agent_types,
        name=unique_name(
            f"scml2020tests/multi/{'-'.join(_.__name__[:3] for _ in agent_types)}/"
            f"Fine_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        n_processes=n_processes,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    save_stats(world, world.log_folder)


@given(n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=20)
def test_something_happens_with_random_agents(n_processes):
    world = generate_world(
        [RandomOneShotAgent],
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/do_something/" f"Fine_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        compact=COMPACT,
        no_logs=NOLOGS,
        n_steps=15,
    )
    world.run()
    assert len(world.signed_contracts) + len(world.cancelled_contracts) != 0


def test_basic_awi_info_suppliers_consumers():
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
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
        assert a.id in a.awi.all_suppliers[a.awi.my_output_product]
        assert a.id in a.awi.all_consumers[a.awi.my_input_product]
        assert a.awi.my_consumers == world.agent_consumers[aid]
        assert a.awi.my_suppliers == world.agent_suppliers[aid]
        l = a.awi.my_input_product
        assert all(
            _.endswith(str(l - 1)) or a.awi.is_system(_) for _ in a.awi.my_suppliers
        )
        assert all(
            _.endswith(str(l + 1)) or a.awi.is_system(_) for _ in a.awi.my_consumers
        )


def test_generate():
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
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
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(agent_types=[RandomOneShotAgent], n_steps=10),
        construct_graphs=True,
    )
    world.graph((0, world.n_steps))


def test_ufun_min_max_in_world():
    for _ in range(20):
        world = SCML2020OneShotWorld(
            **SCML2020OneShotWorld.generate(
                agent_types=[RandomOneShotAgent], n_steps=10
            ),
            construct_graphs=False,
            compact=True,
            no_logs=True,
        )
        world.step()
        for aid, agent in world.agents.items():
            if is_system_agent(aid):
                continue
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
):
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
):
    if level == 0:
        input_agent, output_agent = True, False
    elif level == 1:
        input_agent, output_agent = False, False
    else:
        input_agent, output_agent = False, True

    ufun = OneShotUFun(
        ex_qin=ex_qin,
        ex_qout=ex_qout,
        ex_pin=ex_pin,
        ex_pout=ex_pout,
        production_cost=production_cost,
        disposal_cost=disposal_cost,
        shortfall_penalty=shortfall_penalty,
        input_agent=input_agent,
        output_agent=output_agent,
        n_lines=lines,
        force_exogenous=force_exogenous,
        input_product=0 if input_agent else 2,
        input_qrange=(1, 15),
        input_prange=(1, 15),
        output_qrange=(1, 15),
        output_prange=(1, 15),
        n_input_negs=inegs,
        n_output_negs=onegs,
        current_step=0,
        input_penalty_scale=input_penalty_scale,
        output_penalty_scale=output_penalty_scale,
        current_balance=balance,
    )
    worst_gt, best_gt = ufun.find_limit_brute_force(False), ufun.find_limit_brute_force(
        True
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
):
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
):
    if level == 0:
        input_agent, output_agent = True, False
    elif level == 1:
        input_agent, output_agent = False, False
    else:
        input_agent, output_agent = False, True

    ufun = OneShotUFun(
        ex_qin=ex_qin,
        ex_qout=ex_qout,
        ex_pin=ex_pin,
        ex_pout=ex_pout,
        production_cost=production_cost,
        disposal_cost=disposal_cost,
        shortfall_penalty=shortfall_penalty,
        input_agent=input_agent,
        output_agent=output_agent,
        n_lines=lines,
        force_exogenous=force_exogenous,
        input_product=0 if input_agent else 2,
        input_qrange=(1, 15),
        input_prange=(1, 15),
        output_qrange=(1, 15),
        output_prange=(1, 15),
        n_input_negs=5,
        n_output_negs=5,
        current_step=0,
        input_penalty_scale=1,
        output_penalty_scale=3,
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
    ufun.best = ufun.find_limit(True)
    ufun.worst = ufun.find_limit(False)

    mn, mx = ufun.min_utility, ufun.max_utility
    if mx is None:
        mx = float("inf")
    if mn is None:
        mn = float("-inf")

    assert mx >= mn or mx == mn == 0
    u = ufun.from_offers(
        [(qin, 0, pin / qin if qin else 0), (qout, 0, pout / qout if qout else 0)],
        [False, True],
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
    from negmas.helpers import get_full_type_name

    from scml.oneshot import SingleAgreementAspirationAgent

    n_processes = 2
    world = generate_world(
        [SingleAgreementAspirationAgent],
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/{SingleAgreementAspirationAgent.__name__}Fine{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        compact=True,
        no_logs=True,
    )
    world.run()


@given(
    atype=st.lists(
        st.sampled_from(std_types + types), unique=True, min_size=2, max_size=6
    )
)
@settings(deadline=900_000, max_examples=10)
def test_adapter(atype):
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(agent_types=atype, n_steps=10),
        construct_graphs=False,
        compact=True,
        no_logs=True,
    )
    world.run()


class MyIndNeg(OneShotIndNegotiatorsAgent):
    def generate_ufuns(self):
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
            d[partner_id] = LinearUtilityAggregationFunction(
                dict(
                    quantity=lambda x: 0.1 * x,
                    time=lambda x: 0.0,
                    unit_price=lambda x: x,
                ),
                weights=dict(
                    quantity=0.1,
                    time=0.0,
                    unit_price=0.9,
                ),
                outcome_type=tuple,
                reserved_value=0.0,
            )
        # generate ufuns that prever lower prices when selling
        for partner_id in self.awi.my_suppliers:
            issues = self.awi.current_input_issues
            if self.awi.is_system(partner_id):
                continue
            d[partner_id] = LinearUtilityAggregationFunction(
                dict(
                    quantity=lambda x: x,
                    time=lambda x: 0.0,
                    unit_price=lambda x: issues[UNIT_PRICE].max_value - x,
                ),
                weights=dict(
                    quantity=0.1,
                    time=0.0,
                    unit_price=0.9,
                ),
                outcome_type=tuple,
                reserved_value=0.0,
            )
        return d


def test_ind_negotiators():
    n_processes = 5
    world = generate_world(
        [MyIndNeg],
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/{MyIndNeg.__name__}" f"Fine{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
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
        name=unique_name(
            f"scml2020tests/single/{MyIndNeg.__name__}" f"Fine{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        compact=True,
        no_logs=True,
    )
    world.run()


def test_production_cost_increase():
    from scml.oneshot.agents import GreedyOneShotAgent
    from scml.oneshot.world import SCML2020OneShotWorld

    NPROCESSES = 5
    costs = [[] for _ in range(NPROCESSES)]
    for _ in range(100):
        world = SCML2020OneShotWorld(
            **SCML2020OneShotWorld.generate(
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
