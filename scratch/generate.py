from typing import List
from typing import Union

from scml.scml2020 import SCML2020Agent
from scml.scml2020.agents import BuyCheapSellExpensiveAgent
from scml.scml2020.agents import DecentralizingAgent
from scml.scml2020.agents import RandomAgent
from scml.scml2020.utils import anac2020_assigner
from scml.scml2020.utils import anac2020_config_generator
from scml.scml2020.utils import anac2020_world_generator

COMPETITORS = [DecentralizingAgent, BuyCheapSellExpensiveAgent, RandomAgent]


def generate_world(
    n_steps: int, competitors: List[Union[str, SCML2020Agent]], n_agents_per_competitor
):
    config = anac2020_config_generator(
        n_competitors=len(competitors),
        n_agents_per_competitor=n_agents_per_competitor,
        n_steps=n_steps,
    )
    assigned = anac2020_assigner(
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
