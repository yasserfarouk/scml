from typing import List, Union

from scml.scml2020.agent import SCML2020Agent
from scml.scml2020.agents import (
    BuyCheapSellExpensiveAgent,
    DecentralizingAgent,
    RandomAgent,
)
from scml.utils import (
    anac2020_world_generator,
    anac_assigner_std,
    anac_config_generator_std,
)

COMPETITORS = [DecentralizingAgent, BuyCheapSellExpensiveAgent, RandomAgent]


def generate_world(
    n_steps: int, competitors: List[Union[str, SCML2020Agent]], n_agents_per_competitor
):
    config = anac_config_generator_std(
        n_competitors=len(competitors),
        n_agents_per_competitor=n_agents_per_competitor,
        n_steps=n_steps,
    )
    assigned = anac_assigner_std(
        config,
        max_n_worlds=None,
        n_agents_per_competitor=n_agents_per_competitor,
        competitors=competitors,
        params=[dict() for _ in competitors],
    )
    return [anac2020_world_generator(**(a[0])) for a in assigned]


if __name__ == "__main__":
    worlds = generate_world(10, COMPETITORS, 1)
    for world in worlds:
        world.run()
        print(world.stats_df.head())
