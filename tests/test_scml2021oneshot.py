from negmas import save_stats
from negmas.helpers import unique_name
from pytest import mark
from hypothesis import given, settings
import hypothesis.strategies as st

from scml.scml2020 import is_system_agent
from scml.oneshot import SCML2020OneShotWorld
from scml.oneshot.builtin import RandomOneShotAgent

import random

random.seed(0)

COMPACT = True
NOLOGS = True
# agent types to be tested
types = [RandomOneShotAgent]
active_types = types


def generate_world(
    agent_types,
    n_processes=3,
    n_steps=10,
    n_agents_per_process=2,
    n_lines=10,
    **kwargs,
):

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
            agent_types=RandomOneShotAgent, n_steps=10, n_processes=2,
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
