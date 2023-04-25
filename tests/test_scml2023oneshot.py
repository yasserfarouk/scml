from scml.oneshot.agents import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    GreedySyncAgent,
    RandomOneShotAgent,
)
from scml.oneshot.world import SCML2023OneShotWorld


def test_equal_exogenous_supply():
    world = SCML2023OneShotWorld(
        **SCML2023OneShotWorld.generate(
            agent_types=[
                GreedySyncAgent,
                GreedyOneShotAgent,
                GreedySingleAgreementAgent,
                RandomOneShotAgent,
            ],
            agent_processes=None,
            n_processes=2,
            n_steps=10,
            random_agent_types=False,
            production_costs=2,
            exogenous_price_dev=0.0,
            equal_exogenous_sales=True,
            equal_exogenous_supply=True,
        )
    )
    world.run()
