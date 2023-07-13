import random

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse
from pytest import mark

from scml.common import intin
from scml.oneshot.common import QUANTITY
from scml.oneshot.rl.action import (
    ActionManager,
    FixedPartnerNumbersActionManager,
    LimitedPartnerNumbersActionManager,
    UnconstrainedActionManager,
)
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.factory import (
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
)
from scml.oneshot.rl.observation import (
    FixedPartnerNumbersObservationManager,
    LimitedPartnerNumbersObservationManager,
)


def random_policy(env):
    return lambda _: env.action_space.sample()


def make_env(
    level=0,
    n_consumers=(4, 8),
    n_suppliers=(0, 0),
    n_lines=(10, 10),
    extra_checks=False,
    type="fixed",
) -> OneShotEnv:
    if type == "fixed":
        n_consumers = intin(n_consumers)
        n_suppliers = intin(n_suppliers)
        n_lines = intin(n_lines)
    factory_type, obs_type, act_type = dict(
        fixed=(
            FixedPartnerNumbersOneShotFactory,
            FixedPartnerNumbersObservationManager,
            FixedPartnerNumbersActionManager,
        ),
        limited=(
            LimitedPartnerNumbersOneShotFactory,
            LimitedPartnerNumbersObservationManager,
            LimitedPartnerNumbersActionManager,
        ),
        unlimited=(
            LimitedPartnerNumbersOneShotFactory,
            LimitedPartnerNumbersObservationManager,
            UnconstrainedActionManager,
        ),
    )[type]
    factory = factory_type(
        n_suppliers=n_suppliers,  # type: ignore
        n_consumers=n_consumers,  # type: ignore
        level=level,
        n_lines=n_lines,
    )
    return OneShotEnv(
        action_manager=act_type(factory=factory),
        observation_manager=obs_type(factory=factory, extra_checks=extra_checks),
        factory=factory,
        extra_checks=False,
    )


@mark.parametrize("type_", ["unlimited", "fixed", "limited"])
def test_env_runs(type_):
    env = make_env(type=type_)

    obs, info = env.reset()
    for _ in range(500):
        action = random_policy(env)(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


@mark.parametrize("type_", ["unlimited", "fixed", "limited"])
def test_training(type_):
    from stable_baselines3 import A2C

    env = make_env(extra_checks=False, type=type_)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    assert vec_env is not None
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")
        #
        #


def test_rl_agent_fallback():
    factory = FixedPartnerNumbersOneShotFactory()
    action, obs = (
        FixedPartnerNumbersActionManager(factory),
        FixedPartnerNumbersObservationManager(factory),
    )
    world, agents = factory(types=(OneShotRLAgent,))
    assert len(agents) == 1
    assert isinstance(agents[0]._obj, OneShotRLAgent), agent.type_name  # type: ignore
    world.run()


def test_rl_agent_with_a_trained_model():
    from stable_baselines3 import A2C

    env = make_env(extra_checks=False, type="unlimited")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10)

    factory = LimitedPartnerNumbersOneShotFactory()
    obs = LimitedPartnerNumbersObservationManager(factory)
    world, agents = factory(
        types=(OneShotRLAgent,),
        params=(dict(models=[model_wrapper(model)], observation_managers=[obs]),),
    )
    assert len(agents) == 1
    agent = agents[0]
    assert isinstance(agent._obj, OneShotRLAgent), agent.type_name  # type: ignore
    world.step()
    assert agent._valid_index == 0  # type: ignore
    world.run()


# @mark.parametrize(
#     "type_",
#     [
#         LimitedPartnerNumbersActionManager,
#         FixedPartnerNumbersActionManager,
#         UnconstrainedActionManager,
#     ],
# )
# def test_action_manager(type_: type[ActionManager]):
#     factory = FixedPartnerNumbersOneShotFactory()
#     manager = type_(factory)
#     space = manager.make_space()
#     world, agents = factory()
#     for _ in range(100):
#         agent = agents[0]
#         # action = space.sample()
#         responses = dict()
#         awi = agent.awi
#         for aid, nmi in awi.state.running_sell_nmis.items():
#             mine_indx = [i for i, x in enumerate(nmi.agent_ids) if x == agent.id][0]
#             partner_indx = [i for i, x in enumerate(nmi.agent_ids) if x != agent.id][0]
#             partner = [x for i, x in enumerate(nmi.agent_ids) if x != agent.id][0]
#             resp = random.choice(
#                 [
#                     ResponseType.REJECT_OFFER,
#                     ResponseType.END_NEGOTIATION,
#                     ResponseType.ACCEPT_OFFER,
#                 ]
#             )
#             responses[partner] = SAOResponse(
#                 resp,
#                 awi.current_output_outcome_space.random_outcome()
#                 if resp != ResponseType.END_NEGOTIATION
#                 else None,
#             )
#         world.step(1, neg_actions={agent.id: responses})
#         action = manager.encode(awi, responses)
#         decoded = manager.decode(awi, action)
#         encoded = manager.encode(awi, decoded)
#         assert np.all(np.isclose(action, encoded)), f"{action=}\n{decoded=}\n{encoded=}"
