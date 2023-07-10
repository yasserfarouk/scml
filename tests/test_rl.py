from scml.oneshot.rl.action import DefaultActionManager
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.observation import DefaultObservationManager


def test_env_runs():
    level = 0
    n_consumers = 4
    n_suppliers = 0
    n_lines = 10
    n_quantities = n_lines

    env = OneShotEnv(
        DefaultActionManager(
            n_suppliers=n_suppliers, n_consumers=n_consumers, n_quantities=n_lines
        ),
        DefaultObservationManager(
            n_suppliers=n_suppliers,
            n_consumers=n_consumers,
            n_lines=n_lines,
            n_quantities=n_quantities,
        ),
        n_suppliers=n_suppliers,
        n_consumers=n_consumers,
        level=level,
        n_lines=n_lines,
    )
    obs, info = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
