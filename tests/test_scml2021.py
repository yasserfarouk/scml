import random

import hypothesis.strategies as st
from hypothesis import example
from hypothesis import given
from hypothesis import reproduce_failure
from hypothesis import settings
from negmas import save_stats
from negmas.helpers import unique_name
from pytest import mark

import scml
from scml.oneshot.agents import RandomOneShotAgent
from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.scml2020 import BuyCheapSellExpensiveAgent
from scml.scml2020 import DoNothingAgent
from scml.scml2020 import RandomAgent
from scml.scml2020 import SCML2021World
from scml.scml2020 import is_system_agent
from scml.scml2020.agents.decentralizing import DecentralizingAgent

random.seed(0)

COMPACT = True
NOLOGS = True
# agent types to be tested
types = scml.scml2020.builtin_agent_types(as_str=False)
oneshot_types = scml.oneshot.builtin_agent_types(as_str=False)
active_types = [_ for _ in types if _ != DoNothingAgent]


def generate_world(
    agent_types,
    n_processes=3,
    n_steps=10,
    n_agents_per_process=2,
    n_lines=10,
    initial_balance=10_000,
    buy_missing_products=True,
    **kwargs,
):
    kwargs["no_logs"] = True
    kwargs["compact"] = True
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types,
            n_processes=n_processes,
            n_steps=n_steps,
            n_lines=n_lines,
            n_agents_per_process=n_agents_per_process,
            initial_balance=initial_balance,
            buy_missing_products=buy_missing_products,
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
@given(buy_missing=st.booleans(), n_processes=st.integers(2, 4))
@settings(deadline=500_000, max_examples=20)
def test_can_run_with_a_single_agent_type(agent_type, buy_missing, n_processes):
    world = generate_world(
        [agent_type],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/{agent_type.__name__}"
            f"{'Buy' if buy_missing else 'Fine'}{n_processes}",
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
    buy_missing=st.booleans(),
    n_processes=st.integers(2, 4),
)
@settings(deadline=300_000, max_examples=20)
@example(
    [RandomAgent, SyncRandomOneShotAgent],
    False,
    2,
)
def test_can_run_with_multiple_agent_types(agent_types, buy_missing, n_processes):
    world = generate_world(
        agent_types,
        buy_missing_products=buy_missing,
        name=unique_name(
            f"scml2020tests/multi/{'-'.join(_.__name__[:3] for _ in agent_types)}/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        n_processes=n_processes,
        initial_balance=10_000,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    save_stats(world, world.log_folder)


@given(
    buy_missing=st.booleans(),
    n_processes=st.integers(2, 4),
    initial_balance=st.sampled_from([50, 10_000, 10_000_000]),
)
@settings(deadline=500_000)
def test_nothing_happens_with_do_nothing(buy_missing, n_processes, initial_balance):
    world = generate_world(
        [DoNothingAgent],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/doing_nothing/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}_b{initial_balance}",
            add_time=True,
            rand_digits=4,
        ),
        initial_balance=initial_balance,
        bankruptcy_limit=initial_balance,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    assert len(world.contracts_per_step) == 0
    for a, f, p in world.afp:
        if is_system_agent(a.id):
            continue
        if (
            a.awi.my_input_product == 0
            or a.awi.my_input_product == a.awi.n_processes - 1
        ):
            assert f.current_balance <= initial_balance, (
                f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
                f"should go down"
            )
        # else:
        #     assert f.current_balance == initial_balance, (
        #         f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
        #         f"should not change"
        #     )


@given(buy_missing=st.booleans(), n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=20)
def test_something_happens_with_random_agents(buy_missing, n_processes):
    world = generate_world(
        [RandomAgent],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/do_something/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        initial_balance=10_000,
        bankruptcy_limit=10_000,
        compact=COMPACT,
        no_logs=NOLOGS,
        n_steps=15,
    )
    world.run()
    assert len(world.signed_contracts) + len(world.cancelled_contracts) != 0


@given(n_processes=st.integers(2, 4))
@settings(deadline=300_000, max_examples=3)
def test_agents_go_bankrupt(n_processes):
    buy_missing = True
    world = generate_world(
        [RandomAgent],
        buy_missing_products=buy_missing,
        n_processes=n_processes,
        name=unique_name(
            f"scml2020tests/single/bankrupt/"
            f"{'Buy' if buy_missing else 'Fine'}_p{n_processes}",
            add_time=True,
            rand_digits=4,
        ),
        initial_balance=0,
        bankruptcy_limit=0,
        n_steps=10,
        compact=COMPACT,
        no_logs=NOLOGS,
    )
    world.run()
    #    assert len(world.signed_contracts) + len(world.cancelled_contracts) == 0
    for a, f, p in world.afp:
        if is_system_agent(a.id):
            continue
        if (
            a.awi.my_input_product == 0
            or a.awi.my_input_product == a.awi.n_processes - 1
        ):
            assert f.current_balance <= 0, (
                f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
                f"should go down"
            )
            assert f.is_bankrupt, (
                f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes}) should "
                f"be bankrupt (balance = {f.current_balance}, inventory={f.current_inventory})"
            )
        # else:
        #     assert f.current_balance == 0, (
        #         f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes})'s balance "
        #         f"should not change"
        #     )
        #     assert not f.is_bankrupt, (
        #         f"{a.name} (process {a.awi.my_input_product} of {a.awi.n_processes}) should "
        #         f"NOT be bankrupt (balance = {f.current_balance}, "
        #         f"inventory={f.current_inventory})"
        #     )


def test_generate():
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=DoNothingAgent, n_steps=10, n_processes=4, initial_balance=None
        )
    )
    world.run()
    assert True


def test_a_tiny_world():
    world = generate_world(
        [DecentralizingAgent],
        n_processes=2,
        n_steps=5,
        n_agents_per_process=2,
        n_lines=5,
        initial_balance=10_000,
        buy_missing_products=True,
    )
    world.run()
    assert True


def test_graph():
    world = generate_world(
        [DecentralizingAgent],
        n_processes=2,
        n_steps=10,
        n_agents_per_process=2,
        n_lines=5,
        initial_balance=10_000,
        buy_missing_products=True,
    )
    world.graph(together=True)
    world.step()
    world.graph(steps=None, together=True)
    world.graph(steps=None, together=False)
    world.run()
    world.graph((0, world.n_steps), together=False)
    world.graph((0, world.n_steps), together=True)


def test_graphs_lead_to_no_unknown_nodes():
    world = SCML2021World(
        **SCML2021World.generate(
            agent_types=[DecentralizingAgent, BuyCheapSellExpensiveAgent], n_steps=10
        ),
        construct_graphs=True,
        no_logs=True,
    )
    world.graph((0, world.n_steps))


@given(
    atype=st.lists(
        st.sampled_from(oneshot_types + types), unique=True, min_size=2, max_size=6
    )
)
@example(
    atype=[
        scml.oneshot.agents.greedy.GreedySyncAgent,
        scml.scml2020.agents.random.RandomAgent,
    ],
)
@settings(deadline=300_000, max_examples=30)
def test_adapter(atype):
    world = SCML2021World(
        **SCML2021World.generate(agent_types=atype, n_steps=10),
        construct_graphs=False,
        no_logs=True,
        compact=True,
    )
    world.run()


def test_production_cost_increase():
    from scml.scml2020 import SCML2021World
    from scml.scml2020.agents import DecentralizingAgent

    NPROCESSES = 5
    costs = [[] for _ in range(NPROCESSES)]
    for _ in range(100):
        world = SCML2021World(
            **SCML2021World.generate(
                DecentralizingAgent,
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
            costs[profile.input_products[0]].append(
                profile.costs[:, profile.input_products[0]].mean()
            )
    mean_costs = [sum(_) / len(_) for _ in costs]
    assert all(
        [
            b > (0.5 * (i + 2) / (i + 1)) * a
            for i, (a, b) in enumerate(zip(mean_costs[:-1], mean_costs[1:]))
        ]
    ), f"non-ascending costs {mean_costs}"
