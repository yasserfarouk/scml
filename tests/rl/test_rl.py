import logging
import random
from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
from scml.oneshot.agents.greedy import GreedyOneShotAgent
from negmas.gb.common import ResponseType
from negmas.outcomes.issue_ops import itertools
from negmas.sao.common import SAOResponse
from pytest import mark

try:
    from stable_baselines3 import A2C, SAC
except ImportError:
    A2C, SAC = None, None

from scml.oneshot.agents.rand import RandDistOneShotAgent
from scml.oneshot.context import (
    BaseContext,
    ConsumerContext,
    FixedPartnerNumbersOneShotContext,
    LimitedPartnerNumbersContext,
    LimitedPartnerNumbersOneShotContext,
    SupplierContext,
)
from scml.oneshot.rl.action import ActionManager, FlexibleActionManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper, group_partners
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.observation import FlexibleObservationManager, ObservationManager
from scml.oneshot.rl.policies import greedy_policy, random_action, random_policy
from scml.oneshot.rl.reward import RewardFunction
from scml.oneshot.context import (
    GeneralContext,
    ANACContext,
    FixedPartnerNumbersContext,
    ANACOneShotContext,
    StrongSupplierContext,
    StrongConsumerContext,
    WeakSupplierContext,
    WeakConsumerContext,
    BalancedSupplierContext,
    BalancedConsumerContext,
    RepeatingContext,
)
from scml.oneshot.rl.helpers import (
    clip,
    discretize_and_clip,
    normalize_and_clip,
    encode_offers_no_time,
    decode_offers_no_time,
)
from ..switches import SCML_RUNALL_TESTS

SCML_TEST_BRITTLE_ISSUES = False
NTRAINING = 100
TEST_ALG = {True: SAC, False: A2C}

all_context_types = [
    GeneralContext,
    ANACContext,
    LimitedPartnerNumbersContext,
    FixedPartnerNumbersContext,
    ANACOneShotContext,
    LimitedPartnerNumbersOneShotContext,
    FixedPartnerNumbersOneShotContext,
    SupplierContext,
    ConsumerContext,
    StrongSupplierContext,
    StrongConsumerContext,
    WeakSupplierContext,
    WeakConsumerContext,
    BalancedSupplierContext,
    BalancedConsumerContext,
    RepeatingContext,
]
few_context_types = [
    ANACContext,
    RepeatingContext,
    SupplierContext,
    ConsumerContext,
]
context_types = all_context_types if SCML_RUNALL_TESTS else few_context_types
DefaultContext = ANACContext


def make_env(
    context_type: type[BaseContext] = DefaultContext,
    oneshot=True,
    extra_checks=False,
    log=False,
    confirm_context_match=True,
    continuous=False,
) -> OneShotEnv:
    world_params: dict[str, Any] = (
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
        )
        if log
        else dict()
    )
    world_params |= dict(
        ignore_agent_exceptions=False,
        ignore_negotiation_exceptions=False,
        ignore_contract_execution_exceptions=False,
        ignore_simulation_exceptions=False,
        debug=True,
        sync_calls=True,
    )
    if oneshot:
        obs_type, act_type = (FlexibleObservationManager, FlexibleActionManager)
    else:
        obs_type, act_type = (FlexibleObservationManager, FlexibleActionManager)
    context = context_type(
        # n_suppliers=n_suppliers,  #
        # n_consumers=n_consumers,  #
        # level=level,
        # n_lines=n_lines,
        world_params=world_params,
    )
    return OneShotEnv(
        action_manager=act_type(context=context, continuous=continuous),
        observation_manager=obs_type(
            context=context, extra_checks=extra_checks, continuous=continuous
        ),
        context=context,
        extra_checks=extra_checks,
        debug=confirm_context_match,
    )


@mark.parametrize(
    ["type_", "continuous"], list(itertools.product(context_types, [False, True]))
)
def test_env_runs(type_, continuous):
    env = make_env(type_, continuous=continuous)

    obs, _ = env.reset()
    for _ in range(500):
        action = partial(random_action, env=env)(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


@mark.skipif(
    A2C is None,
    reason="Skipped because you do not have stable-baselines3 installed.Try: python -m pip install stable-baselines3",
)
@mark.parametrize(
    ["type_", "continuous"], list(itertools.product(context_types, [False, True]))
)
def test_training(type_, continuous):
    alg = TEST_ALG[continuous]
    if alg is None:
        return
    env = make_env(type_, extra_checks=True, continuous=continuous)

    model = alg("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=NTRAINING)

    vec_env = model.get_env()
    assert vec_env is not None
    obs = vec_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)  # type: ignore
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


def test_rl_agent_fallback():
    context = DefaultContext()
    world, agents = context.generate(types=(OneShotRLAgent,))
    assert len(agents) == 1
    assert isinstance(agents[0], OneShotRLAgent), agents[0].type_name
    world.run()


@mark.skipif(
    A2C is None,
    reason="Skipped because you do not have stable-baselines3 installed.Try: python -m pip install stable-baselines3",
)
@mark.parametrize(
    ["type_", "continuous"], list(itertools.product(context_types, [False, True]))
)
def test_rl_agent_with_a_trained_model_in_memory(type_, continuous):
    alg = TEST_ALG[continuous]
    if alg is None:
        return
    env = make_env(type_, extra_checks=False, continuous=continuous)

    model = alg("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=NTRAINING)

    context = env._context
    obs = env._obs_manager
    act = env._action_manager
    world, agents = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[obs],
                action_managers=[act],
            ),
        ),
    )
    assert len(agents) == 1
    agent = agents[0]
    assert isinstance(agent, OneShotRLAgent), agent.type_name
    world.step()
    assert agent._valid_index == 0
    world.run()


@mark.skipif(
    A2C is None,
    reason="Skipped because you do not have stable-baselines3 installed.Try: python -m pip install stable-baselines3",
)
@mark.parametrize(
    ["type_", "continuous"], list(itertools.product(context_types, [False, True]))
)
def test_rl_agent_with_a_trained_model(type_, continuous):
    alg = TEST_ALG[continuous]
    if alg is None:
        return
    env = make_env(type_, extra_checks=False, continuous=continuous)

    model = alg("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=NTRAINING)

    if issubclass(type_, RepeatingContext):
        context = type_(configs=env._context.configs)  # type: ignore
    else:
        context = type_()
    obs = FlexibleObservationManager(context, continuous=continuous)
    act = FlexibleActionManager(context, continuous=continuous)
    world, agents = context.generate(
        types=(OneShotRLAgent,),
        params=(  # type: ignore
            dict(
                models=[model_wrapper(model)],
                observation_managers=[obs],
                action_managers=[act],
            )
        ),
    )
    assert len(agents) == 1
    agent = agents[0]
    assert isinstance(agent, OneShotRLAgent), agent.type_name
    world.step()
    assert agent._valid_index == 0  #
    world.run()


def test_env_runs_one_world():
    env = make_env()

    obs, _ = env.reset()
    for _ in range(env._world.n_steps):
        action = partial(random_action, env=env)(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


class MyOneShotAgent(GreedyOneShotAgent):
    ...


# TODO: do not skip this test
@mark.skip("Known to fail but maybe it does not affect real operation of the RL agent.")
@mark.parametrize(
    ["context_type", "continuous", "type_"],
    list(itertools.product(context_types, [False, True], [FlexibleActionManager])),
)
def test_action_manager(context_type, continuous, type_: type[ActionManager]):
    context = context_type(placeholder_types=(MyOneShotAgent,))
    manager = type_(context=context, continuous=continuous)
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
            for nmi in awi.current_nmis.values():
                partner = [x for x in nmi.agent_ids if x != agent.id][0]
                resp = random.choice(
                    [
                        ResponseType.REJECT_OFFER,
                        ResponseType.END_NEGOTIATION,
                        ResponseType.ACCEPT_OFFER,
                    ]
                )
                outcome = (
                    awi.current_output_outcome_space.random_outcome()
                    if nmi.annotation["seller"] == agent.id
                    else awi.current_input_outcome_space.random_outcome()
                )
                responses[partner] = SAOResponse(
                    resp,
                    outcome if resp != ResponseType.END_NEGOTIATION else None,
                )
        world.step(1, neg_actions={agent.id: responses})
        if awi is not None:
            action = manager.encode(awi, responses)
            decoded = manager.decode(awi, action)
            encoded = manager.encode(awi, decoded)
            if not np.all(np.isclose(action, encoded)):
                raise AssertionError(
                    f"Unexpected decoding on step {awi.current_step}\n{responses=}\n{action=}\n{decoded=}\n{encoded=}"
                )


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
        encoded = obs.encode(self.awi)
        decoded = obs.get_offers(self.awi, encoded)
        expected = offers
        current = self.awi.current_offers
        for k, v in current.items():
            assert (
                expected.get(k, None) == v
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}\n{encoded=}"
        for k, v in decoded.items():
            assert (
                "+" in k or expected.get(k, None) == v
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}\n{encoded=}"
        for k, v in expected.items():
            assert (
                decoded.get(k, None) is None or decoded.get(k, None) == v
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}\n{encoded=}"
            assert (
                current.get(k, None) == v
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}\n{encoded=}"
        if SCML_TEST_BRITTLE_ISSUES:
            assert (
                sum(_[0] for _ in expected.values() if _)
                == sum(_[0] for _ in decoded.values() if _)
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}\n{encoded=}"
            # note that decoding may not be perfect for discrete obs managers because encoding converts to int which loses fractions for groups
            if not (
                abs(
                    sum(_[0] * _[-1] for _ in expected.values() if _)
                    - sum(_[0] * _[-1] for _ in decoded.values() if _)
                )
                <= sum(_[0] for _ in expected.values() if _)
            ):
                raise AssertionError(
                    f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}"
                )
        return r

    def first_proposals(self):
        d = super().first_proposals()
        self._my_first_proposals = d
        obs = self.__obs
        state = self.awi
        expected = state.current_offers
        encoded = obs.encode(state)
        decoded = obs.get_offers(self.awi, encoded)
        for k, v in decoded.items():
            assert (
                "+" in k or expected.get(k, None) == v
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}"
        for k, v in expected.items():
            assert (
                decoded.get(k, None) is None or decoded.get(k, None) == v
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}"
        if SCML_TEST_BRITTLE_ISSUES:
            assert (
                sum(_[0] for _ in expected.values() if _)
                == sum(_[0] for _ in decoded.values() if _)
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}"
            assert (
                sum(_[0] * _[-1] for _ in expected.values() if _)
                == sum(_[0] * _[-1] for _ in decoded.values() if _)
            ), f"{self.awi.current_step=}\n{self.awi.current_input_issues}\n{self.awi.current_output_issues}\n{decoded=}\n{expected=}"
        return d


# @mark.skip(reason=f"Known to fail.")
@mark.parametrize(
    ["context_type", "continuous", "type_"],
    list(itertools.product(context_types, [False, True], [FlexibleObservationManager])),
)
def test_obs_manager(context_type, continuous, type_: type[ObservationManager]):
    n_worlds = 2
    context = context_type()
    obs = type_(context, continuous=continuous)
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

    obs, _ = env.reset()
    world = env._world
    assert world.n_steps is not None and world.neg_n_steps is not None
    for _ in range(world.n_steps * world.neg_n_steps):
        action = partial(random_action, env=env)(obs)
        assert env.action_space.contains(
            action
        ), "f{action} not contained in the action space"
        obs, _, terminated, truncated, _ = env.step(action)
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
        debug=SCML_TEST_BRITTLE_ISSUES,
    )
    assert isinstance(world.neg_n_steps, int)
    for _ in range(world.n_steps * world.neg_n_steps):  #
        # decoded_action = greedy(obs)
        # action = env._action_manager.encode(env._agent.awi, decoded_action)
        action = greedy(obs)
        decoded_action = env._action_manager.decode(env._agent.awi, action)  #
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
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert world.current_step == world.n_steps - 1
    assert len(world.saved_contracts) > 0
    if SCML_TEST_BRITTLE_ISSUES:
        assert accepted_sometime
        assert not ended_everything


def test_env_random_policy_no_end():
    env = make_env(log=False)

    obs, _ = env.reset()
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
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    assert world.current_step == world.n_steps - 1
    assert len(world.saved_contracts) > 0
    assert accepted_sometime
    assert not ended_everything


class RewardNeedsReduction(RewardFunction):
    n_before = 0
    n_call = 0

    def before_action(self, awi):
        self.n_before += 1
        needs = awi.needed_sales if awi.level == 0 else awi.needed_supplies
        return needs

    def __call__(self, awi, action, info):
        self.n_call += 1
        _ = action
        current_needs = awi.needed_sales if awi.level == 0 else awi.needed_supplies
        return info - current_needs


def test_reward_reception():
    context = DefaultContext()

    rewardfun = RewardNeedsReduction()
    env = OneShotEnv(
        action_manager=FlexibleActionManager(context=context),
        observation_manager=FlexibleObservationManager(context=context),
        context=context,
        reward_function=rewardfun,
    )

    obs, _ = env.reset()

    results, n_steps = [], 100
    policy = partial(
        greedy_policy,
        action_manager=env._action_manager,
        obs_manager=env._obs_manager,
        awi=env._agent.awi,
        debug=SCML_TEST_BRITTLE_ISSUES,
    )
    for _ in range(n_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        results.append([obs, reward])
        if terminated or truncated:
            obs, _ = env.reset()
    assert rewardfun.n_call == rewardfun.n_before
    assert rewardfun.n_call >= n_steps
    assert rewardfun.n_before >= n_steps
    assert len(env._world.saved_contracts) > 0, "There are no contracts"
    assert (
        len([c for c in env._world.saved_contracts if env._agent_id in c["signatures"]])
        > 0
    ), "No contracts from the RL agent"


def test_relative_times_make_sense():
    context = DefaultContext()

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
        debug=SCML_TEST_BRITTLE_ISSUES,
    )
    for _ in range(60):
        results.append([obs[-5], obs[-4]])
        action = policy(obs)
        assert isinstance(action, np.ndarray)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    assert results[-1][-1] >= 0, f"{results}"


def test_clip():
    assert clip(0) == 0
    assert clip(1) == 1
    assert clip(2) == 1
    assert clip(-2) == 0
    assert clip(0.0) == 0.0
    assert clip(0.5) == 0.5
    assert clip(1.3) == 1.0
    assert clip(-1.3) == 0.0


def test_discretize():
    assert discretize_and_clip(0.0, 11) == 0
    assert discretize_and_clip(1.0, 11) == 10
    assert discretize_and_clip(0.25, 11) == 3
    assert discretize_and_clip(0.245, 11) == 2
    assert discretize_and_clip(0.255, 11) == 3
    assert discretize_and_clip(0.3, 11) == 3


def test_normalize():
    assert normalize_and_clip(3, 0, 6) == 0.5
    assert normalize_and_clip(0, 0, 6) == 0.0
    assert normalize_and_clip(6, 0, 6) == 1.0
    assert normalize_and_clip(7, 0, 6) == 1.0


def test_encode_offers_no_time_supplier():
    minp, maxp = 3, 20
    offer_map = dict(
        a11=None, a12=(4, 0, 10), a13=(2, 0, 4), a21=(3, 0, 6), a31=None, a41=(0, 0, 9)
    )
    groups = [["a11", "a12", "a13"], ["a21"], ["a31"], ["a44", "z1"], ["a41"], []]
    offers = encode_offers_no_time(offer_map, groups, minp, maxp)
    expected = [
        (6, (40 + 8) / 6 - minp),
        (3, 6 - minp),
        (0, 0),
        (0, 0),
        (0, maxp - minp),
    ]
    assert all(
        len(a) == len(b) == 2 and a[0] == b[0] and abs(a[1] - b[1]) < 1e-8
        for a, b in zip(offers, expected)
    ), f"{offers=}\n{expected=}"
    # check decoding is correct when no n_prices is passed
    decoded = decode_offers_no_time(
        offers, len(groups), 0, groups, [], 0, False, minp, -1, maxp
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, (40 + 8) / 6)
    assert decoded["a21"] == (3, 0, 6)
    # check decoding is correct when n_prices equal the true number of prices
    decoded = decode_offers_no_time(
        offers,
        len(groups),
        0,
        groups,
        [],
        0,
        False,
        minp,
        -1,
        maxp,
        n_prices=maxp - minp + 1,
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, (40 + 8) / 6)
    assert decoded["a21"] == (3, 0, 6)
    # check decoding is correct when n_prices is np
    np = 2
    decoded = decode_offers_no_time(
        offers, len(groups), 0, groups, [], 0, False, minp, -1, maxp, n_prices=np
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, ((40 + 8) / 6) * (np / (maxp - minp + 1)))
    assert decoded["a21"] == (3, 0, 6 * (np / (maxp - minp + 1)))


def test_encode_offers_no_time_consumer():
    minp, maxp = 3, 20
    offer_map = dict(
        a11=None, a12=(4, 0, 10), a13=(2, 0, 4), a21=(3, 0, 6), a31=None, a41=(0, 0, 9)
    )
    groups = [["a11", "a12", "a13"], ["a21"], ["a31"], ["a44", "z1"], ["a41"], []]
    offers = encode_offers_no_time(offer_map, groups, minp, maxp)
    expected = [
        (6, (40 + 8) / 6 - minp),
        (3, 6 - minp),
        (0, 0),
        (0, 0),
        (0, maxp - minp),
    ]
    assert all(
        len(a) == len(b) == 2 and a[0] == b[0] and abs(a[1] - b[1]) < 1e-8
        for a, b in zip(offers, expected)
    ), f"{offers=}\n{expected=}"
    # check decoding is correct when no n_prices is passed
    decoded = decode_offers_no_time(
        offers, 0, len(groups), [], groups, 0, False, -1, minp, -1, maxp
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, (40 + 8) / 6)
    assert decoded["a21"] == (3, 0, 6)
    # check decoding is correct when n_prices equal the true number of prices
    decoded = decode_offers_no_time(
        offers,
        0,
        len(groups),
        [],
        groups,
        0,
        False,
        -1,
        minp,
        -1,
        maxp,
        n_prices=maxp - minp + 1,
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, (40 + 8) / 6)
    assert decoded["a21"] == (3, 0, 6)
    # check decoding is correct when n_prices is np
    np = 2
    decoded = decode_offers_no_time(
        offers, len(groups), 0, groups, [], 0, False, minp, -1, maxp, n_prices=np
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, ((40 + 8) / 6) * (np / (maxp - minp + 1)))
    assert decoded["a21"] == (3, 0, 6 * (np / (maxp - minp + 1)))


def test_encode_offers_no_time_both():
    minip, maxip = 3, 20
    minop, maxop = 5, 30
    offer_map = dict(
        a11=None,
        a12=(4, 0, 10),
        a13=(2, 0, 4),
        a21=(3, 0, 6),
        a31=None,
        a41=(0, 0, 9),
        b11=(2, 0, 9),
        b12=(3, 0, 22),
        b22=(8, 0, 11),
        b21=(3, 0, 13),
        b31=(7, 0, 9),
    )
    groupsi = [["a11", "a12", "a13"], ["a21"], ["a31"], ["a44", "z1"], ["a41"], []]
    groupso = [["b11", "b12"], ["b21", "b22"], ["b31"], ["b44", "z1"], []]
    groups = groupsi + groupso
    offers = encode_offers_no_time(offer_map, groupsi, minip, maxip)
    offers += encode_offers_no_time(offer_map, groupso, minop, maxop)
    expected = [
        (6, (40 + 8) / 6 - minip),
        (3, 6 - minip),
        (0, 0),
        (0, 0),
        (0, maxip - minip),
        (0, 0),
        (5, (18 + 66) / 5 - minop),
        (11, (88 + 39) / 11 - minop),
        (7, 9 - minop),
        (0, 0),
        (0, 0),
    ]
    assert all(
        len(a) == len(b) == 2 and a[0] == b[0] and abs(a[1] - b[1]) < 1e-8
        for a, b in zip(offers, expected)
    ), f"{offers=}\n{expected=}\n{minip=}, {maxip=}, {minop=}, {maxop=}"
    # check decoding is correct when no n_prices is passed
    decoded = decode_offers_no_time(
        offers,
        len(groupsi),
        len(groupso),
        groupsi,
        groupso,
        0,
        False,
        minip,
        minop,
        maxip,
        maxop,
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, (40 + 8) / 6)
    assert decoded["a21"] == (3, 0, 6)
    assert decoded["b11+b12"] == (5, 0, (18 + 66) / 5)
    assert decoded["b21+b22"] == (11, 0, (88 + 39) / 11)

    # check decoding is correct when n_prices is np
    np = 2
    decoded = decode_offers_no_time(
        offers,
        len(groupsi),
        len(groupso),
        groupsi,
        groupso,
        0,
        False,
        minip,
        minop,
        maxip,
        maxop,
        n_prices=np,
    )
    for group in groups:
        if not group:
            continue
        assert "+".join(group) in decoded
    assert decoded["a11+a12+a13"] == (6, 0, (np / (maxip - minip + 1)) * (40 + 8) / 6)
    assert decoded["a21"] == (3, 0, (np / (maxip - minip + 1)) * 6)
    assert decoded["b11+b12"] == (5, 0, (np / (maxop - minop + 1)) * (18 + 66) / 5)
    assert (
        decoded["b21+b22"] is not None
        and sum(
            abs(a - b)
            for a, b in zip(
                decoded["b21+b22"], (11, 0, (np / (maxop - minop + 1)) * (88 + 39) / 11)
            )
        )
        < 1e-4
    )


@mark.parametrize("extend", [False, True])
def test_grouping_too_few_partners(extend):
    partners = ["a11", "a21", "a31", "a41", "a12", "a22"]
    n = 9
    groups = group_partners(partners, n, 2, extend=extend)
    assert len(groups) == n
    for i in range(len(partners)):
        assert len(groups[i]) == 1
        assert groups[i][0] == partners[i]
    for i in range(len(partners), n):
        if extend:
            assert (
                len(groups[i]) == 1
            ), f"{i=}: {groups[i]=}, {n=}\n{partners=}\n{groups=}"
            assert (
                groups[i][0] == partners[i - len(partners)]
            ), f"{i=}: {groups[i]=}, {n=}\n{partners=}\n{groups=}"
        else:
            assert (
                len(groups[i]) == 0
            ), f"{i=}: {groups[i]=}, {n=}\n{partners=}\n{groups=}"


@mark.parametrize("extend", [False, True])
def test_grouping_exact_number(extend):
    _ = extend
    partners = ["a11", "a21", "a31", "a41", "a12", "a22"]
    n = len(partners)
    groups = group_partners(partners, n, 2, extend=extend)
    assert len(groups) == n
    assert all(a == b[0] for a, b in zip(partners, groups, strict=True))


@mark.parametrize("extend", [False, True])
def test_grouping_too_many_partners(extend):
    _ = extend
    partners = ["a11", "a21", "a31", "a41", "a12", "a22"]
    n = 4
    groups = group_partners(partners, n, 2, extend=extend)
    assert len(groups) == n
    assert tuple(groups[0]) == ("a11", "a12")
    assert tuple(groups[1]) == ("a21", "a22")
    assert tuple(groups[2]) == ("a31",)
    assert tuple(groups[3]) == ("a41",)
