import pytest
from negmas import ResponseType, SAOResponse
from negmas.negotiators.modular import itertools

from scml.oneshot.agents.nothing import OneShotDummyAgent
from scml.oneshot.rl.common import isinclass, isinobject
from scml.oneshot.rl.context import (
    DEFAULT_DUMMY_AGENT_TYPES,
    ANACContext,
    ANACOneShotContext,
    ConsumerContext,
    FixedPartnerNumbersContext,
    FixedPartnerNumbersOneShotContext,
    GeneralContext,
    LimitedPartnerNumbersContext,
    LimitedPartnerNumbersOneShotContext,
    RepeatingContext,
    SupplierContext,
)


@pytest.mark.parametrize(
    "context_type",
    (
        GeneralContext,
        RepeatingContext,
        ConsumerContext,
        SupplierContext,
        ANACContext,
        ANACOneShotContext,
        FixedPartnerNumbersContext,
        FixedPartnerNumbersOneShotContext,
        LimitedPartnerNumbersContext,
        LimitedPartnerNumbersOneShotContext,
    ),
)
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
        except:
            pass
        assert context.is_valid_world(
            world, raise_on_failure=True
        ), f"world {i} does not belong to the context"
        assert len(agents) == 1, f"world {i} has incorrect agents {agents}"
        for a in agents:
            assert isinobject(
                a, DEFAULT_DUMMY_AGENT_TYPES  # type: ignore
            ), f"world {i} has incorrect agent type for agent {a.id} ({type(a)=}) {a}"
            for b in world.agents.values():
                # if isinobject(b, DEFAULT_DUMMY_AGENT_TYPES):  # type: ignore
                #     continue
                if not b._obj:  # type: ignore
                    continue
                assert context.is_valid_awi(
                    b._obj.awi  # type: ignore
                ), f"world {i} has incorrect AWI for agent {b.id}"

    c2 = context_type()
    assert context.contains_context(
        c2, raise_on_failure=True
    ), f"Identical contexts do not match"
    assert c2.contains_context(
        context, raise_on_failure=True
    ), f"Identical contexts do not match"
