import pytest
from negmas import ResponseType

from scml import OneShotAgent
from scml import SCML2020OneShotWorld
from scml.oneshot.agents.random import RandomOneShotAgent


def run_simulation(agent_types):
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            agent_types,
            n_steps=10,
            random_agent_types=True,
        ),
        construct_graphs=True,
    )
    world.run()
    return world


class MyOneShotDoNothing(OneShotAgent):
    """My Agent that does nothing"""

    def propose(self, negotiator_id, state):
        # print(f"proposing to {negotiator_id} at {state.step}")
        return None

    def respond(self, negotiator_id, state, offer):
        # print(f"proposing to {negotiator_id} for {offer} at {state.step}")
        return ResponseType.REJECT_OFFER

    def on_negotiation_end(self, negotiator_id: str, state) -> None:
        assert state.agreement is None


def test_do_nothing_gets_no_contracts():
    run_simulation([MyOneShotDoNothing, RandomOneShotAgent])
