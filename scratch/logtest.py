from scml import SCML2020World
from scml import DecentralizingAgent, RandomAgent, DoNothingAgent

# create and run the world
world = SCML2020World(
    **SCML2020World.generate(
        agent_types=[DoNothingAgent], n_steps=10, n_processes=2, name="test_world"
    )
)
world.run()
