import logging
import random
from functools import partial
from typing import Any

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse
from pytest import mark
from stable_baselines3 import A2C

from scml.common import intin
from scml.oneshot.rl.action import ActionManager, UnconstrainedActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.context import (
    FixedPartnerNumbersOneShotContext,
    LimitedPartnerNumbersOneShotContext,
)
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.observation import (
    FixedPartnerNumbersObservationManager,
    LimitedPartnerNumbersObservationManager,
)
from scml.oneshot.rl.policies import greedy_policy, random_action, random_policy
from scml.oneshot.rl.reward import RewardFunction
from scml.std.rl.context import (
    FixedPartnerNumbersStdContext,
    LimitedPartnerNumbersStdContext,
)

NTRAINING = 100


def make_env(
    level=0,
    n_consumers=(4, 8),
    n_suppliers=(0, 0),
    n_lines=(10, 10),
    extra_checks=False,
    type="fixed",
    log=False,
    oneshot=True,
) -> OneShotEnv:
    log_params: dict[str, Any] = (
        dict(
            no_logs=False,
            log_stats_every=1,
            log_file_level=logging.DEBUG,
            log_screen_level=logging.ERROR,
            save_signed_contracts=True,
            save_cancelled_contracts=True,
            save_negotiations=True,
            save_resolved_breaches=True,
            save_unresolved_breaches=True,
            debug=True,
        )
        if log
        else dict(debug=True)
    )
    log_params.update(
        dict(
            ignore_agent_exceptions=False,
            ignore_negotiation_exceptions=False,
            ignore_contract_execution_exceptions=False,
            ignore_simulation_exceptions=False,
        )
    )
    if type == "fixed":
        n_consumers = intin(n_consumers)
        n_suppliers = intin(n_suppliers)
        n_lines = intin(n_lines)
    if oneshot:
        context_type, obs_type, act_type = dict(
            fixed=(
                FixedPartnerNumbersOneShotContext,
                FixedPartnerNumbersObservationManager,
                UnconstrainedActionManager,
            ),
            limited=(
                LimitedPartnerNumbersOneShotContext,
                LimitedPartnerNumbersObservationManager,
                UnconstrainedActionManager,
            ),
            unlimited=(
                LimitedPartnerNumbersOneShotContext,
                LimitedPartnerNumbersObservationManager,
                UnconstrainedActionManager,
            ),
        )[type]
    else:
        context_type, obs_type, act_type = dict(
            fixed=(
                FixedPartnerNumbersStdContext,
                FixedPartnerNumbersObservationManager,
                UnconstrainedActionManager,
            ),
            limited=(
                LimitedPartnerNumbersStdContext,
                LimitedPartnerNumbersObservationManager,
                UnconstrainedActionManager,
            ),
            unlimited=(
                LimitedPartnerNumbersStdContext,
                LimitedPartnerNumbersObservationManager,
                UnconstrainedActionManager,
            ),
        )[type]
    context = context_type(
        n_suppliers=n_suppliers,  # type: ignore
        n_consumers=n_consumers,  # type: ignore
        level=level,
        n_lines=n_lines,
        world_params=log_params,
    )
    return OneShotEnv(
        action_manager=act_type(context=context),
        observation_manager=obs_type(context=context, extra_checks=extra_checks),
        context=context,
        extra_checks=False,
    )


@mark.parametrize("type_", ["unlimited", "fixed", "limited"])
def test_env_runs(type_):
    env = make_env(type=type_)

    obs, info = env.reset()
    for _ in range(500):
        action = partial(random_action, env=env)(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


@mark.parametrize("type_", ["unlimited", "fixed", "limited"])
def test_training(type_):
    env = make_env(extra_checks=True, type=type_)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=NTRAINING)

    vec_env = model.get_env()
    assert vec_env is not None
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


def test_rl_agent_fallback():
    context = FixedPartnerNumbersOneShotContext()
    world, agents = context.generate(types=(OneShotRLAgent,))
    assert len(agents) == 1
    assert isinstance(agents[0]._obj, OneShotRLAgent), agent.type_name  # type: ignore
    world.run()


def test_rl_agent_with_a_trained_model():
    env = make_env(extra_checks=False, type="unlimited")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=NTRAINING)

    context = LimitedPartnerNumbersOneShotContext()
    obs = LimitedPartnerNumbersObservationManager(context, extra_checks=False)
    world, agents = context.generate(
        types=(OneShotRLAgent,),
        params=(dict(models=[model_wrapper(model)], observation_managers=[obs]),),
    )
    assert len(agents) == 1
    agent = agents[0]
    assert isinstance(agent._obj, OneShotRLAgent), agent.type_name  # type: ignore
    world.step()
    assert agent._valid_index == 0  # type: ignore
    world.run()


def test_env_runs_one_world():
    env = make_env(type="unlimited")

    obs, info = env.reset()
    for _ in range(env._world.n_steps):
        action = partial(random_action, env=env)(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


@mark.skip(reason=f"Known to fail.")
@mark.parametrize(
    "type_",
    [
        # LimitedPartnerNumbersActionManager,
        # FixedPartnerNumbersActionManager,
        UnconstrainedActionManager,
    ],
)
def test_action_manager(type_: type[ActionManager]):
    context = FixedPartnerNumbersOneShotContext()
    manager = type_(context)
    world, agents = context.generate()
    for _ in range(100):
        agent = agents[0]
        # action = space.sample()
        responses = dict()
        awi = agent.awi
        for nmi in awi.state.current_sell_nmis.values():
            partner = [x for x in nmi.agent_ids if x != agent.id][0]
            resp = random.choice(
                [
                    ResponseType.REJECT_OFFER,
                    ResponseType.END_NEGOTIATION,
                    ResponseType.ACCEPT_OFFER,
                ]
            )
            responses[partner] = SAOResponse(
                resp,
                awi.current_output_outcome_space.random_outcome()
                if resp != ResponseType.END_NEGOTIATION
                else None,
            )
        world.step(1, neg_actions={agent.id: responses})
        action = manager.encode(awi, responses)
        decoded = manager.decode(awi, action)
        encoded = manager.encode(awi, decoded)
        assert np.all(np.isclose(action, encoded)), f"{action=}\n{decoded=}\n{encoded=}"


def test_env_random_policy():
    env = make_env(log=True)

    obs, info = env.reset()
    world = env._world
    assert world.n_steps is not None and world.neg_n_steps is not None
    for _ in range(world.n_steps * world.neg_n_steps):
        action = partial(random_action, env=env)(obs)
        assert env.action_space.contains(
            action
        ), "f{action} not contained in the action space"
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert world.current_step == world.n_steps - 1
    assert len(world.saved_contracts) > 0


def test_env_greedy_policy_no_end():
    env = make_env(log=True)

    obs, _ = env.reset()
    world = env._world
    accepted_sometime = False
    ended_everything = True
    greedy = partial(
        greedy_policy,
        action_manager=env._action_manager,
        obs_manager=env._obs_manager,
        awi=env._agent.awi,
        debug=True,
    )
    for _ in range(world.n_steps * world.neg_n_steps):  # type: ignore
        # decoded_action = greedy(obs)
        # action = env._action_manager.encode(env._agent.awi, decoded_action)
        action = greedy(obs)
        decoded_action = env._action_manager.decode(env._agent.awi, action)  # type: ignore
        if not all(
            _.response == ResponseType.END_NEGOTIATION for _ in decoded_action.values()
        ):
            ended_everything = False
        if any(
            _.response == ResponseType.ACCEPT_OFFER for _ in decoded_action.values()
        ):
            accepted_sometime = True

        # assert env.action_space.contains(
        #     action
        # ), f"{action} not contained in the action space"
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert world.current_step == world.n_steps - 1
    assert len(world.saved_contracts) > 0
    assert accepted_sometime
    assert not ended_everything


def test_env_random_policy_no_end():
    env = make_env(log=False)

    obs, info = env.reset()
    world = env._world
    accepted_sometime = False
    ended_everything = True
    assert world.n_steps is not None and world.neg_n_steps is not None
    for _ in range(world.n_steps * world.neg_n_steps):
        action = partial(random_policy, env=env)(obs)
        decoded_action = env._action_manager.decode(env._agent.awi, action)
        if not all(
            _.response == ResponseType.END_NEGOTIATION for _ in decoded_action.values()
        ):
            ended_everything = False
        if any(
            _.response == ResponseType.ACCEPT_OFFER for _ in decoded_action.values()
        ):
            accepted_sometime = True

        assert env.action_space.contains(
            action
        ), "f{action} not contained in the action space"
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert world.current_step == world.n_steps - 1
    assert len(world.saved_contracts) > 0
    assert accepted_sometime
    assert not ended_everything


class TestReducingNeedsReward(RewardFunction):
    def before_action(self, awi):
        needs = awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        return needs

    def __call__(self, awi, action, info):
        current_needs = (
            awi.state.needed_sales if awi.level == 0 else awi.state.needed_supplies
        )
        return info - current_needs


def test_reward_reception():
    context = FixedPartnerNumbersOneShotContext(
        n_suppliers=0,
        n_consumers=4,
        level=0,
    )

    env = OneShotEnv(
        action_manager=UnconstrainedActionManager(context=context),
        observation_manager=FixedPartnerNumbersObservationManager(
            context=context, n_bins=20, extra_checks=False
        ),
        context=context,
        reward_function=TestReducingNeedsReward(),
    )

    obs, _ = env.reset()

    results = []
    for _ in range(100):
        action = obs[:8]
        obs, reward, terminated, truncated, _ = env.step(action)
        results.append([obs[-5], obs[-4], reward])
        if terminated or truncated:
            obs, _ = env.reset()
    assert len(env._world.saved_contracts) > 0
    assert (
        len([c for c in env._world.saved_contracts if env._agent_id in c["signatures"]])
        > 0
    )
    results = np.asarray(results)
    # check that not all rewards are received at the beginning of a new step
    assert np.sum(results[:, 0] * results[:, -1]) > 0
    # assert False, f"{results}"


def test_relative_times_make_sense():
    context = FixedPartnerNumbersOneShotContext(
        n_suppliers=0,
        n_consumers=4,
        level=0,
    )

    env = OneShotEnv(
        action_manager=UnconstrainedActionManager(context=context, extra_checks=True),
        observation_manager=FixedPartnerNumbersObservationManager(
            context=context, n_bins=20
        ),
        context=context,
    )

    obs, _ = env.reset()

    results = []
    policy = partial(
        greedy_policy,
        action_manager=env._action_manager,
        obs_manager=env._obs_manager,
        awi=env._agent.awi,
        debug=True,
    )
    for _ in range(60):
        results.append([obs[-5], obs[-4]])
        action = policy(obs)
        decoded = env._action_manager.decode(env._agent.awi, action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    assert results[-1][-1] > 0, f"{results}"