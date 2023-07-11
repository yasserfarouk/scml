from pytest import mark

from scml.common import intin
from scml.oneshot.rl.action import (
    FixedPartnerNumbersActionManager,
    LimitedPartnerNumbersActionManager,
)
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.factory import (
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
)
from scml.oneshot.rl.observation import (
    FixedPartnerNumbersObservationManager,
    LimitedPartnerNumbersObservationManager,
)


def random_policy(env, obs):
    return env.action_space.sample()


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
    )[type]
    n_quantities = n_lines
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


@mark.parametrize("type_", ["fixed", "limited"])
def test_env_runs(type_):
    env = make_env(type=type_)

    obs, info = env.reset()
    for _ in range(500):
        action = random_policy(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


@mark.parametrize("type_", ["fixed", "limited"])
def test_training(type_):
    import gymnasium as gym
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
