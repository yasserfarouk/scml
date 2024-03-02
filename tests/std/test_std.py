from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.oneshot.agents.nothing import Placeholder
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.world import PLACEHOLDER_AGENT_PREFIX
from scml.std.agents import GreedyStdAgent, SyncRandomStdAgent
from scml.std.world import SCML2024StdWorld

from tests.switches import DefaultStdWorld


def test_single_run():
    agent_types = [GreedyStdAgent, SyncRandomStdAgent, SyncRandomOneShotAgent]

    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(agent_types=agent_types, n_steps=50),
        construct_graphs=True,
        debug=True,
    )
    world.run()


def test_run_random():
    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(agent_types=[SyncRandomStdAgent], n_steps=50),
        construct_graphs=True,
        debug=True,
    )
    world.run()


def make_configs(n, n_trials, n_processes):
    types = [f"{PLACEHOLDER_AGENT_PREFIX}{i}" for i in range(n)]
    return [
        (
            tuple(types),
            SCML2024StdWorld.generate(
                types,
                agent_params=None,
                n_steps=10,
                n_processes=n_processes,
                random_agent_types=True,
                construct_graphs=True,
            ),
        )
        for _ in range(n_trials)
    ]


def test_replace_agents():
    n = 4
    configs = make_configs(n, 2, 5)
    for old, c in configs:
        d = SCML2024StdWorld.replace_agents(c, old, tuple(f"A{i}" for i in range(n)))
        new_types = [_["controller_type"] for _ in d["agent_params"]]
        old_types = [_["controller_type"] for _ in d["agent_params"]]
        assert all(a[-1] == b[-1] for a, b in zip(new_types, old_types))


def sumall(d):
    s = 0
    for v in d.values():
        s += sum(v.values())
    return s


def test_awi_needed_level0():
    class MyAWI(OneShotAWI):
        def __init__(self):
            super().__init__(None, Placeholder())  # type: ignore

        @property
        def is_perishable(self):
            return False

        @property
        def current_step(self):
            return 10

        @property
        def current_exogenous_input_quantity(self):
            return 20

        @property
        def current_exogenous_output_quantity(self):
            return 0

        @property
        def current_inventory_input(self):
            return 5

        @property
        def current_inventory_output(self):
            return 0

        @property
        def is_first_level(self):
            return True

        @property
        def is_last_level(self):
            return False

        @property
        def is_middle_level(self):
            return False

        @property
        def current_exogenous_output_price(self):
            return 0

        @property
        def current_exogenous_input_price(self):
            return 2

    awi = MyAWI()
    sales = [
        ("C0", 9, 50, 9),
        ("C0", 10, 50, 10),
        ("C1", 5, 25, 10),
        ("C2", 3, 75, 10),
        ("C3", 2, 75, 10),
        ("C1", 5, 25, 11),
        ("C2", 3, 75, 11),
        ("C0", 5, 25, 20),
        ("C3", 2, 75, 20),
    ]
    supplies = []
    for sale in sales:
        awi._register_sale(*sale)
    for supply in supplies:
        awi._register_supply(*supply)

    assert awi.total_sales == 20
    assert awi.total_supplies == 0
    assert awi.needed_supplies == 0
    assert awi.needed_sales == 5
    assert sumall(awi.future_supplies) == 0
    assert sumall(awi.future_sales) == 15
    assert awi.total_future_sales == 15
    assert awi.total_future_supplies == 0
    for t, v in [(9, 0), (10, 20), (11, 8), (20, 7)]:
        assert awi.total_sales_at(t) == v
        assert awi.total_supplies_at(t) == 0
    for t, v in [(9, 0), (10, 20), (11, 28), (20, 35)]:
        assert awi.total_sales_until(t) == v
        assert awi.total_supplies_until(t) == 0
    for (t1, t2), v in [
        ((9, 7), 0),
        ((10, 10), 20),
        ((9, 10), 20),
        ((9, 13), 28),
        ((10, 12), 28),
        ((10, 11), 28),
        ((10, 19), 28),
        ((10, 20), 35),
        ((11, 20), 15),
        ((11, 15), 8),
        ((15, 20), 7),
        ((19, 20), 7),
    ]:
        assert awi.total_sales_between(t1, t2) == v, f"{t1=}, {t2=}, {v=}"
        assert awi.total_supplies_between(t1, t2) == 0


def test_awi_needed_level_last():
    class MyAWI(OneShotAWI):
        def __init__(self):
            super().__init__(None, Placeholder())  # type: ignore

        @property
        def is_perishable(self):
            return False

        @property
        def current_step(self):
            return 10

        @property
        def current_exogenous_input_quantity(self):
            return 0

        @property
        def current_exogenous_output_quantity(self):
            return 20

        @property
        def current_inventory_input(self):
            return 0

        @property
        def current_inventory_output(self):
            return 5

        @property
        def is_first_level(self):
            return False

        @property
        def is_last_level(self):
            return True

        @property
        def is_middle_level(self):
            return False

        @property
        def current_exogenous_output_price(self):
            return 0

        @property
        def current_exogenous_input_price(self):
            return 2

    awi = MyAWI()
    supplies = [
        ("S0", 9, 50, 9),
        ("S0", 10, 50, 10),
        ("S1", 5, 25, 10),
        ("S2", 3, 75, 10),
        ("S3", 2, 75, 10),
        ("S1", 5, 25, 11),
        ("S2", 3, 75, 11),
        ("S0", 5, 25, 20),
        ("S3", 2, 75, 20),
    ]
    sales = []
    for supply in supplies:
        awi._register_supply(*supply)
    for sale in sales:
        awi._register_sale(*sale)

    assert awi.total_supplies == 20
    assert awi.total_sales == 0
    assert awi.needed_supplies == 5
    assert awi.needed_sales == 0
    assert sumall(awi.future_supplies) == 15
    assert sumall(awi.future_sales) == 0
    assert awi.total_future_supplies == 15
    assert awi.total_future_sales == 0
    for t, v in [(9, 0), (10, 20), (11, 8), (20, 7)]:
        assert awi.total_supplies_at(t) == v
        assert awi.total_sales_at(t) == 0
    for t, v in [(9, 0), (10, 20), (11, 28), (20, 35)]:
        assert awi.total_supplies_until(t) == v
        assert awi.total_sales_until(t) == 0
    for (t1, t2), v in [
        ((9, 7), 0),
        ((10, 10), 20),
        ((9, 10), 20),
        ((9, 13), 28),
        ((10, 12), 28),
        ((10, 11), 28),
        ((10, 19), 28),
        ((10, 20), 35),
        ((11, 20), 15),
        ((11, 15), 8),
        ((15, 20), 7),
        ((19, 20), 7),
    ]:
        assert awi.total_supplies_between(t1, t2) == v, f"{t1=}, {t2=}, {v=}"


def test_awi_needed_level_middle():
    class MyAWI(OneShotAWI):
        def __init__(self):
            super().__init__(None, Placeholder())  # type: ignore

        @property
        def is_perishable(self):
            return False

        @property
        def current_step(self):
            return 10

        @property
        def current_exogenous_input_quantity(self):
            return 0

        @property
        def current_exogenous_output_quantity(self):
            return 0

        @property
        def current_inventory_input(self):
            return 5

        @property
        def current_inventory_output(self):
            return 0

        @property
        def is_first_level(self):
            return False

        @property
        def is_last_level(self):
            return False

        @property
        def is_middle_level(self):
            return True

        @property
        def current_exogenous_output_price(self):
            return 0

        @property
        def current_exogenous_input_price(self):
            return 2

    awi = MyAWI()
    sales = [
        ("C0", 9, 50, 9),
        ("C0", 10, 50, 10),
        ("C1", 1, 25, 10),
        ("C2", 2, 75, 10),
        ("C3", 2, 75, 10),
        ("C1", 5, 25, 11),
        ("C2", 3, 75, 11),
        ("C0", 5, 25, 20),
        ("C3", 2, 75, 20),
    ]
    supplies = [
        ("S0", 6, 300, 9),
        ("S0", 4, 500, 10),
        ("S1", 4, 250, 10),
        ("S2", 3, 750, 10),
        ("S3", 7, 750, 10),
        ("S1", 3, 250, 11),
        ("S2", 3, 750, 11),
        ("S0", 0, 250, 20),
        ("S3", 4, 750, 20),
    ]
    for sale in sales:
        awi._register_sale(*sale)
    for supply in supplies:
        awi._register_supply(*supply)

    assert awi.total_sales == 15
    assert awi.total_supplies == 18
    assert awi.needed_supplies == -8
    assert awi.needed_sales == 8
    assert sumall(awi.future_supplies) == 10
    assert sumall(awi.future_sales) == 15
    assert awi.total_future_sales == 15
    assert awi.total_future_supplies == 10
    for t, vo, vi in [(9, 0, 0), (10, 15, 18), (11, 8, 6), (20, 7, 4)]:
        assert awi.total_sales_at(t) == vo
        assert awi.total_supplies_at(t) == vi
    for t, vo, vi in [(9, 0, 0), (10, 15, 18), (11, 23, 24), (20, 30, 28)]:
        assert awi.total_sales_until(t) == vo
        assert awi.total_supplies_until(t) == vi


def test_can_run_single_agent():
    # agent_types = [RandDistOneShotAgent, RandomOneShotAgent, EqualDistOneShotAgent, GreedyOneShotAgent]
    agent_types = [SyncRandomStdAgent]
    world = DefaultStdWorld(
        **DefaultStdWorld.generate(agent_types=agent_types, n_steps=50),
        construct_graphs=True,
    )
    world.run()
