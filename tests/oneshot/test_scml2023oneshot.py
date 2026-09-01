import random

import pytest
from negmas import ResponseType, SAOResponse

from scml.oneshot.agents import (
    GreedyOneShotAgent,
    GreedySingleAgreementAgent,
    GreedySyncAgent,
    RandomOneShotAgent,
)
from scml.oneshot.world import is_system_agent
from scml.utils import (
    anac2023_oneshot_world_generator,
    anac_assigner_oneshot,
    anac_config_generator_oneshot,
)

from ..switches import DefaultOneShotWorld, DefaultStdWorld

# LOG_PARAMS = dict(
#     no_logs=False,
#     log_stats_every=1,
#     log_file_level=logging.DEBUG,
#     log_screen_level=logging.ERROR,
#     save_signed_contracts=True,
#     save_cancelled_contracts=True,
#     save_negotiations=True,
#     save_resolved_breaches=True,
#     save_unresolved_breaches=True,
# )
LOG_PARAMAS = {"no_logs": True}


def test_equal_exogenous_supply():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
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
        ),
        **LOG_PARAMAS,
    )
    world.run()


def test_equal_exogenous_supply_stepping():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
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
            one_offer_per_step=True,
        ),
        **LOG_PARAMAS,
    )
    while world.step():
        pass
    assert len(world.contracts_executed) > 0


def test_equal_exogenous_supply_stepping_with_no_action():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
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
            one_offer_per_step=True,
        ),
        **LOG_PARAMAS,
    )
    world.step_with(actions={}, init=True)
    while world.step_with(actions={}):
        pass
    assert len(world.contracts_executed) > 0


def test_equal_exogenous_supply_stepping_with_random_action():
    world = DefaultOneShotWorld(
        **DefaultOneShotWorld.generate(
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
            one_offer_per_step=True,
        ),
        **LOG_PARAMAS,
    )
    agents = list(random.choices(list(world.agents.values()), k=1))
    world.step_with(actions={}, init=True)

    def make_actions():
        actions = {}
        for agent in agents:
            negotiator, responses = None, {}
            for t in ["buy", "sell"]:
                for partner, neg in agent.awi.current_negotiation_details[t].items():  # type: ignore
                    assert agent.id in (neg.buyer, neg.seller)
                    partner2 = neg.buyer if agent.id == neg.seller else neg.seller
                    assert partner2 == partner
                    negotiator = [_.id for _ in neg.nmi._mechanism.negotiators if _.owner.id == agent.id][0]
                    partner = [_.id for _ in neg.nmi._mechanism.negotiators if _.owner.id != agent.id][0]
                    if random.random() > 0.5:
                        responses[neg.nmi.mechanism_id] = {negotiator: SAOResponse(ResponseType.REJECT_OFFER, neg.nmi.random_outcome())}
                    elif random.random() < 0.1:
                        responses[neg.nmi.mechanism_id] = {negotiator: SAOResponse(ResponseType.END_NEGOTIATION, None)}
                    else:
                        responses[neg.nmi.mechanism_id] = {negotiator: SAOResponse(ResponseType.ACCEPT_OFFER, neg.nmi.state.current_offer)}

            actions[agent.id] = responses
        return actions

    actions = make_actions()
    while world.step_with(actions=actions):
        actions = make_actions()
    assert len(world.contracts_executed) > 0


@pytest.mark.parametrize("world_type", [DefaultOneShotWorld, DefaultStdWorld])
def test_every_negotiation_has_both_partners(world_type):
    """Every mechanism must be created with both of its partners in it.

    ``_request_negotiations`` handles the buying and the selling side in one
    loop. It used to assign the freshly created negotiators to the parameter it
    was reading, so the selling side reused the buying side's negotiators. A
    negotiator that already joined a mechanism cannot join another one, so
    ``Mechanism.add`` dropped it and the selling mechanism was registered with a
    single negotiator, which never starts and never produces an agreement.

    Only agents that both buy from and sell to non-system partners are affected,
    and only those that initiate negotiations, so this needs at least four
    processes to show up.
    """
    world = world_type(
        **world_type.generate(
            agent_types=[RandomOneShotAgent],
            n_processes=4,
            n_agents_per_process=2,
            n_steps=5,
            random_agent_types=False,
        ),
        **LOG_PARAMAS,
    )
    world.step_with(actions={}, init=True)

    negotiations = list(world._negotiations.values())
    assert negotiations, "No negotiations were started, the test would pass vacuously"
    bad = [(neg.partners, len(neg.mechanism.negotiators)) for neg in negotiations if len(neg.mechanism.negotiators) != 2]
    assert not bad, f"Mechanisms registered without both partners: {bad}"

    # the middle agents are the ones that exercise both sides of the loop
    middles = [
        aid
        for aid in world.agents
        if not is_system_agent(aid)
        and any(not is_system_agent(_) for _ in world.agent_suppliers[aid])
        and any(not is_system_agent(_) for _ in world.agent_consumers[aid])
    ]
    assert middles, "No agent buys and sells to non-system partners, the test would pass vacuously"
    for aid in middles:
        details = world.agents[aid].awi.current_negotiation_details  # type: ignore
        assert details["buy"], f"{aid} has no buying negotiations"
        assert details["sell"], f"{aid} has no selling negotiations"


@pytest.mark.parametrize("year", [2023])
def test_anac_single_world(year):
    configs = anac_config_generator_oneshot(year, n_competitors=1, n_agents_per_competitor=1)
    assigned = anac_assigner_oneshot(configs, 1, competitors=[RandomOneShotAgent], params=None, fair=False)
    assert len(assigned) == 1
    assigned = assigned[0][0]
    world = anac2023_oneshot_world_generator(year=year, **assigned)
    world.run()
    assert len(world.contracts_executed) > 0
