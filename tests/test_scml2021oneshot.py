import random

import hypothesis.strategies as st
from hypothesis import given
from hypothesis import settings
from negmas import save_stats
from negmas.helpers import unique_name
from negmas.utilities.ops import normalize
from pytest import mark
import pytest

import scml
from scml.oneshot import SCML2020OneShotWorld, builtin_agent_types
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent
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
        assert all(_.endswith(str(l - 1)) or a.awi.is_system(_) for _ in a.awi.my_suppliers)
        assert all(_.endswith(str(l + 1)) or a.awi.is_system(_) for _ in a.awi.my_consumers)



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
            ufun = agent.make_ufun()
            ufun.find_limit(True)
            ufun.find_limit(False)
            mn, mx = ufun.min_utility, ufun.max_utility
            assert mx >= mn


@given(
    ex_qin=st.integers(0, 10),
    ex_qout=st.integers(0, 10),
    ex_pin=st.integers(2, 10),
    ex_pout=st.integers(2, 10),
    production_cost=st.integers(1, 5),
    storage_cost=st.floats(0.5, 1.5),
    delivery_penalty=st.floats(1.5, 2.5),
    level=st.integers(0, 2),
    force_exogenous=st.booleans(),
    qin=st.integers(0, 10),
    qout=st.integers(0, 10),
    pin=st.integers(2, 10),
    pout=st.integers(2, 10),
    lines=st.integers(2, 15),
)
def test_ufun_unit(
    ex_qin,
    ex_qout,
    ex_pin,
    ex_pout,
    production_cost,
    storage_cost,
    delivery_penalty,
    level,
    force_exogenous,
    qin,
    qout,
    pin,
    pout,
    lines,
):
    _ufun_unit(
        ex_qin,
        ex_qout,
        ex_pin,
        ex_pout,
        production_cost,
        storage_cost,
        delivery_penalty,
        level,
        force_exogenous,
        qin,
        qout,
        pin,
        pout,
        lines,
    )


def _ufun_unit(
    ex_qin,
    ex_qout,
    ex_pin,
    ex_pout,
    production_cost,
    storage_cost,
    delivery_penalty,
    level,
    force_exogenous,
    qin,
    qout,
    pin,
    pout,
    nlines,
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
        storage_cost=storage_cost,
        delivery_penalty=delivery_penalty,
        input_agent=input_agent,
        output_agent=output_agent,
        n_lines=nlines,
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
        output_penalty_scale=1,
    )
    ufun.best = ufun.find_limit(True)
    ufun.worst = ufun.find_limit(False)
    mn, mx = ufun.min_utility, ufun.max_utility
    assert mx >= mn or mx == mn == 0
    u = ufun.from_aggregates(qin, qout, pin, pout)
    assert mn <= u <= mx


def test_ufun_example():
    _ufun_unit(
        ex_qin=0,
        ex_qout=0,
        ex_pin=0,
        ex_pout=0,
        production_cost=1,
        storage_cost=0.5,
        delivery_penalty=1.5,
        level=0,
        force_exogenous=False,
        qin=1,
        qout=1,
        pin=2,
        pout=4,
        nlines=10
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


@given(
    atype=st.lists(
        st.sampled_from(std_types + types), unique=True, min_size=2, max_size=6
    )
)
@settings(deadline=900_000, max_examples=10)
def test_adapter(atype):
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(agent_types=atype, n_steps=20),
        construct_graphs=False,
        compact=True,
        no_logs=True,
    )
    world.run()


def test_adapter_example():
    atype = [
        scml.scml2020.agents.random.RandomAgent,
        scml.oneshot.agents.nothing.OneshotDoNothingAgent,
        scml.scml2020.agents.do_nothing.DoNothingAgent,
        scml.scml2020.agents.indneg.IndependentNegotiationsAgent,
        scml.scml2020.agents.decentralizing.DecentralizingAgent,
        scml.scml2020.agents.decentralizing.DecentralizingAgentWithLogging,
    ]
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(agent_types=atype, n_steps=10),
        construct_graphs=False,
        compact=True,
        no_logs=True,
    )
    world.run()
