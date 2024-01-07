from scml.oneshot.agents import SyncRandomOneShotAgent
from scml.std.agents import GreedyStdAgent, SyncRandomStdAgent
from scml.std.world import SCML2024StdWorld


def test_single_run():
    agent_types = [GreedyStdAgent, SyncRandomStdAgent, SyncRandomOneShotAgent]

    world = SCML2024StdWorld(
        **SCML2024StdWorld.generate(agent_types=agent_types, n_steps=50),
        construct_graphs=True,
    )
    world.run()
