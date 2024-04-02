import pytest
from scml.oneshot.agents.rand import NiceAgent, RandomOneShotAgent
from scml.oneshot.common import is_system_agent
from negmas import ResponseType, SAOResponse
from negmas.negotiators.modular import itertools

from scml.common import isinobject
from scml.oneshot.context import (
    DEFAULT_PLACEHOLDER_AGENT_TYPES,
    ANACContext,
    ANACOneShotContext,
    BalancedConsumerContext,
    BalancedSupplierContext,
    ConsumerContext,
    FixedPartnerNumbersContext,
    FixedPartnerNumbersOneShotContext,
    GeneralContext,
    LimitedPartnerNumbersContext,
    LimitedPartnerNumbersOneShotContext,
    RepeatingContext,
    StrongConsumerContext,
    StrongSupplierContext,
    SupplierContext,
    WeakConsumerContext,
    WeakSupplierContext,
    MonopolicContext,
    EutopiaContext,
    EutopiaConsumerContext,
    EutopiaSupplierContext,
    SingleAgentPerLevelSupplierContext,
    SingleAgentPerLevelConsumerContext,
)

context_types = (
    GeneralContext,
    RepeatingContext,
    ConsumerContext,
    SupplierContext,
    StrongConsumerContext,
    StrongSupplierContext,
    WeakConsumerContext,
    WeakSupplierContext,
    BalancedConsumerContext,
    BalancedSupplierContext,
    ANACContext,
    ANACOneShotContext,
    FixedPartnerNumbersContext,
    FixedPartnerNumbersOneShotContext,
    LimitedPartnerNumbersContext,
    LimitedPartnerNumbersOneShotContext,
    MonopolicContext,
    SingleAgentPerLevelSupplierContext,
    EutopiaContext,
    EutopiaConsumerContext,
    EutopiaSupplierContext,
    SingleAgentPerLevelConsumerContext,
)


@pytest.mark.parametrize("context_type", context_types)
def test_context_can_generate_and_run(context_type):
    context = context_type()
    config = context.make_config()
    assert isinstance(config, dict)
    for i in range(10):
        world, agents = context.generate()
        world.init()
        try:
            world.step(
                neg_actions=dict(
                    zip(
                        [_.id for _ in agents],
                        [
                            dict(
                                zip(
                                    world.agents.keys(),
                                    itertools.repeat(
                                        SAOResponse(ResponseType.END_NEGOTIATION, None)
                                    ),
                                )
                            )
                            for _ in agents
                        ],
                    )
                )
            )
        except Exception:
            pass
        assert context.is_valid_world(
            world, raise_on_failure=True
        ), f"world {i} does not belong to the context"
        assert len(agents) == 1, f"world {i} has incorrect agents {agents}"
        for a in agents:
            assert isinobject(
                a,
                DEFAULT_PLACEHOLDER_AGENT_TYPES,  # type: ignore
            ), f"world {i} has incorrect agent type for agent {a.id} ({type(a)=}) {a}"
            for b in world.agents.values():
                # if isinobject(b, DEFAULT_DUMMY_AGENT_TYPES):  # type: ignore
                #     continue
                if not b._obj:  # type: ignore
                    continue
                assert context.is_valid_awi(
                    b._obj.awi  # type: ignore
                ), f"world {i} has incorrect AWI for agent {b.id}"

    if issubclass(context_type, RepeatingContext):
        c2 = context_type(configs=context.configs)
    else:
        c2 = context_type()
    assert context.contains_context(
        c2, raise_on_failure=True
    ), "Identical contexts do not match"
    assert c2.contains_context(
        context, raise_on_failure=True
    ), "Identical contexts do not match"


def test_monoplic_context():
    context = MonopolicContext()
    for _ in range(10):
        world, a = context((MyAgent,))
        world.step()
        assert len(a) == 1
        agent = a[0]
        assert isinstance(agent, MyAgent)
        assert agent.awi.n_competitors == 0
        assert len(agent.awi.my_competitors) == 0


@pytest.mark.parametrize("level", [0, 1])
def test_single_agent_context(level):
    context = (
        SingleAgentPerLevelSupplierContext()
        if level == 0
        else SingleAgentPerLevelConsumerContext()
    )
    for _ in range(10):
        world, a = context((MyAgent,))
        world.step()
        assert all(len(_) == 1 for _ in a[0].awi.all_suppliers)
        for agent in world.agents.values():
            if is_system_agent(agent.id):
                continue
            assert agent.awi.n_competitors == 0
            assert not agent.awi.my_competitors


class MyAgent(RandomOneShotAgent):
    ...


@pytest.mark.parametrize(
    ["context_type", "level"],
    ((EutopiaConsumerContext, -1), (EutopiaSupplierContext, 0), (EutopiaContext, None)),
)
def test_eutopia_contexts(context_type, level):
    context = context_type()
    for _ in range(10):
        world, a = context((MyAgent,))
        world.step()
        assert len(a) == 1
        agent = a[0]
        assert not agent.awi.is_middle_level
        assert isinstance(agent, MyAgent)
        assert agent.awi.n_competitors == 0
        if level is None:
            pass
        elif level == -1:
            assert agent.awi.is_last_level
        else:
            assert agent.awi.is_first_level

        for aa in world.agents.values():
            if is_system_agent(aa.id):
                continue
            assert agent.id == aa.id or isinstance(
                aa._obj, NiceAgent
            ), f"{aa.id=} != {agent.id} AND {type(aa)} is not NiceAgent"
