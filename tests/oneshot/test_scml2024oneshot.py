import pytest

from scml.oneshot.agents.greedy import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    GreedySyncAgent,
)
from scml.oneshot.agents.random import RandomOneShotAgent, SyncRandomOneShotAgent
from scml.oneshot.common import is_system_agent
from scml.oneshot.world import SCML2024OneShotWorld


@pytest.mark.parametrize(
    "atype,no_bankrupt,some_profits",
    [
        (SyncRandomOneShotAgent, True, True),
        (RandomOneShotAgent, False, False),
        (GreedySingleAgreementAgent, False, False),
        (GreedyOneShotAgent, False, False),
        (GreedySyncAgent, True, False),
    ],
)
def test_run_single_agent(atype, no_bankrupt, some_profits):
    world = SCML2024OneShotWorld(**SCML2024OneShotWorld.generate(atype, n_steps=50))
    types = [
        a._obj.__class__  # type: ignore
        for k, a in world.agents.items()
        if not is_system_agent(k)
    ]
    assert all(
        [issubclass(_, atype) for _ in types]
    ), f"Not all types are {atype.__name__}: {types}"
    world.run()
    d = world.scores()
    scores = list(d.values())
    if no_bankrupt:
        assert min(scores) >= 0, (
            f"Some negative scores (i.e. some agents went bankrupt)!!\n"
            f"{{k:v for k, v in d.items() if v < 0}}"
        )
    if some_profits:
        assert max(scores) >= 1, f"No agents got a any profits:\n{d}"
