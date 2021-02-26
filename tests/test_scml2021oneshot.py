import random

import hypothesis.strategies as st
from hypothesis import given
from hypothesis import settings
from negmas import save_stats
from negmas.helpers import unique_name
from pytest import mark
import pytest

import scml
from scml.oneshot import SCML2020OneShotWorld, builtin_agent_types
from scml.oneshot.agent import OneShotAgent
from scml.oneshot.agents import RandomOneShotAgent
from scml.scml2020 import is_system_agent

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


def test_generate():
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            agent_types=RandomOneShotAgent,
            n_steps=10,
            n_processes=2,
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


def test_ufun_min_max():
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(agent_types=[RandomOneShotAgent], n_steps=10),
        construct_graphs=True,
    )
    world.step()
    for aid, agent in world.agents.items():
        if is_system_agent(aid):
            continue
        ufun = agent.make_ufun()
        mn, mx = ufun.min_utility, ufun.max_utility
        assert mx > mn or mx == mn == 0


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
    )
    world.run()
