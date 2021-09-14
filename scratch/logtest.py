from scml import DecentralizingAgent
from scml import DoNothingAgent
from scml import RandomAgent
from scml import SCML2020World

# create and run the world
world = SCML2020World(
    **SCML2020World.generate(
        agent_types=[DoNothingAgent], n_steps=10, n_processes=2, name="test_world"
    )
)
world.run()
