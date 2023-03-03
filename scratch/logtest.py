from scml import DecentralizingAgent, DoNothingAgent, RandomAgent, SCML2020World

# create and run the world
world = SCML2020World(
    **SCML2020World.generate(
        agent_types=[DoNothingAgent], n_steps=10, n_processes=2, name="test_world"
    )
)
world.run()
