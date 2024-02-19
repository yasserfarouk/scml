import logging
import random
from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse
from pytest import mark

try:
    from stable_baselines3 import A2C
except ImportError:
    A2C = None

from scml.common import intin
from scml.oneshot.agents.rand import RandDistOneShotAgent
from scml.oneshot.context import (
    ConsumerContext,
    FixedPartnerNumbersOneShotContext,
    LimitedPartnerNumbersContext,
    LimitedPartnerNumbersOneShotContext,
    SupplierContext,
)
from scml.oneshot.rl.action import ActionManager, FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.observation import FlexibleObservationManager, ObservationManager
from scml.oneshot.rl.policies import greedy_policy, random_action, random_policy
from scml.oneshot.rl.reward import RewardFunction
from scml.std.context import (
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
                FlexibleObservationManager,
                FlexibleActionManager,
            ),
            limited=(
                LimitedPartnerNumbersOneShotContext,
                FlexibleObservationManager,
                FlexibleActionManager,
            ),
            unlimited=(
                LimitedPartnerNumbersOneShotContext,
                FlexibleObservationManager,
                FlexibleActionManager,
            ),
        )[type]
    else:
        context_type, obs_type, act_type = dict(
            fixed=(
                FixedPartnerNumbersStdContext,
                FlexibleObservationManager,
                FlexibleActionManager,
            ),
            limited=(
                LimitedPartnerNumbersStdContext,
                FlexibleObservationManager,
                FlexibleActionManager,
            ),
            unlimited=(
                LimitedPartnerNumbersStdContext,
                FlexibleObservationManager,
                FlexibleActionManager,
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


@mark.skipif(
    A2C is None,
    "Skipped because you do not have stable-baselines3 installed.Try: python -m pip install stable-baselines3",
)
@mark.parametrize("type_", ["unlimited", "fixed", "limited"])
def test_training(type_):
    if A2C is None:
        return
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


@mark.skipif(
    A2C is None,
    "Skipped because you do not have stable-baselines3 installed.Try: python -m pip install stable-baselines3",
)
def test_rl_agent_with_a_trained_model():
    if A2C is None:
        return
    env = make_env(extra_checks=False, type="unlimited")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=NTRAINING)

    context = LimitedPartnerNumbersOneShotContext()
    obs = FlexibleObservationManager(context, extra_checks=False)
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


# @mark.skip(reason=f"Known to fail.")
@mark.parametrize(
    "type_",
    [
        # LimitedPartnerNumbersActionManager,
        # FixedPartnerNumbersActionManager,
        FlexibleActionManager,
    ],
)
def test_action_manager(type_: type[ActionManager]):
    context = LimitedPartnerNumbersContext()
    manager = type_(context)
    world, agents = context.generate()
    agent = agents[0]
    assert world.neg_n_steps is not None
    for _ in range(min(100, world.neg_n_steps * (world.n_steps - 1))):
        # action = space.sample()
        responses = dict()
        awi = None
        if not agent._awi:
            responses = defaultdict(
                lambda: SAOResponse(ResponseType.REJECT_OFFER, (1, 0, 22))
            )
        else:
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
        if awi is not None:
            action = manager.encode(awi, responses)
            decoded = manager.decode(awi, action)
            encoded = manager.encode(awi, decoded)
            assert np.all(
                np.isclose(action, encoded)
            ), f"{action=}\n{decoded=}\n{encoded=}"


class RecorderAgent(RandDistOneShotAgent):
    def __init__(self, *args, obs, **kwargs):
        self.__obs = obs
        super().__init__(*args, **kwargs)

    def counter_all(self, offers, states):
        r = super().counter_all(offers, states)
        self._saved_offers = offers
        self._saved_states = states
        self._saved_response = r
        obs = self.__obs
        encoded = obs.encode(self.awi.state)
        decoded = obs.get_offers(self.awi, encoded)
        expected = offers
        current = self.awi.current_offers
        for k, v in current.items():
            assert expected.get(k, None) == v, f"{current=}\n{expected=}"
        for k, v in decoded.items():
            assert "+" in k or expected.get(k, None) == v, f"{decoded=}\n{expected=}"
        for k, v in expected.items():
            assert (
                decoded.get(k, None) is None or decoded.get(k, None) == v
            ), f"{decoded=}\n{expected=}"
            assert current.get(k, None) == v, f"{current=}\n{expected=}"
        assert sum(_[0] for _ in expected.values() if _) == sum(
            _[0] for _ in decoded.values() if _
        )
        assert sum(_[0] * _[-1] for _ in expected.values() if _) == sum(
            _[0] * _[-1] for _ in decoded.values() if _
        )
        return r

    def first_proposals(self):
        d = super().first_proposals()
        self._my_first_proposals = d
        obs = self.__obs
        state = self.awi.state
        expected = state.current_offers
        encoded = obs.encode(state)
        decoded = obs.get_offers(self.awi, encoded)
        for k, v in decoded.items():
            assert (
                "+" in k or expected.get(k, None) == v
            ), f"{decoded=}\n{expected=}\n{encoded=}"
        for k, v in expected.items():
            assert (
                decoded.get(k, None) is None or decoded.get(k, None) == v
            ), f"{decoded=}\n{expected=}\n{encoded=}"
        assert sum(_[0] for _ in expected.values() if _) == sum(
            _[0] for _ in decoded.values() if _
        )
        assert sum(_[0] * _[-1] for _ in expected.values() if _) == sum(
            _[0] * _[-1] for _ in decoded.values() if _
        )
        return d


# @mark.skip(reason=f"Known to fail.")
@mark.parametrize(
    "obs_type",
    [
        # FixedPartnerNumbersObservationManager,
        FlexibleObservationManager,
    ],
)
def test_obs_manager(obs_type: type[ObservationManager]):
    n_worlds = 2
    level = random.randint(0, 1)
    context = SupplierContext() if level == 0 else ConsumerContext()
    obs = obs_type(context)
    for _ in range(n_worlds):
        world, agents = context.generate(
            types=(RecorderAgent,), params=(dict(obs=obs),)
        )
        world.init()
        agent = agents[0]
        assert isinstance(agent, RecorderAgent)
        for _ in range(world.n_steps):
            world.step()


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
        action_manager=FlexibleActionManager(context=context),
        observation_manager=FlexibleObservationManager(
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
        action_manager=FlexibleActionManager(context=context, extra_checks=True),
        observation_manager=FlexibleObservationManager(context=context, n_bins=20),
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
        return_decoded=True,
    )
    for _ in range(60):
        results.append([obs[-5], obs[-4]])
        decoded = policy(obs)
        assert isinstance(decoded, np.ndarray)
        # decoded = env._action_manager.decode(env._agent.awi, encoded)
        obs, _, terminated, truncated, _ = env.step(decoded)
        if terminated or truncated:
            obs, _ = env.reset()
    assert results[-1][-1] > 0, f"{results}"
