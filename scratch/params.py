import itertools
import pandas as pd
import scml
from scml.oneshot import SCML2020OneShotWorld
from scml.scml2020.common import is_system_agent
from scml.oneshot.agents import RandomOneShotAgent
import seaborn as sns
import matplotlib.pyplot as plt

METHOD = "bruteforce"
NSTEPS = 10
NWORLDS = 20
data = []
storage_cost = [0.1, 0.2, 0.5, 1.0, 2.0]
delivery_penalty = [0.1, 0.2, 0.4, 1.0, 2.0]
storage_cost = [0.01, 0.1, 1]
delivery_penalty = [0.01, 0.1, 1]


class Recorder(SCML2020OneShotWorld):
    def __init__(self, *args, storage_cost, delivery_penalty, **kwargs):
        super().__init__(*args, **kwargs)
        self.__storage = storage_cost
        self.__penalty = delivery_penalty

    def simulation_step(self, stage):
        global data
        result = super().simulation_step(stage)
        for aid, a in self.agents.items():
            if is_system_agent(aid):
                continue
            if a.ufun is None:
                continue
            if METHOD.startswith("b"):
                a.ufun.worst, a.ufun.best = a.ufun.find_limits_brute_force()
            elif METHOD.startswith("o")
                a.ufun.best = a.ufun.find_limit_optimal(True)
                a.ufun.worst = a.ufun.find_limit_optimal(False)
            elif METHOD.startswith("g"):
                a.ufun.best = a.ufun.find_limit_greedy(True)
                a.ufun.worst = a.ufun.find_limit_greedy(False)
            else:
                a.ufun.best = a.ufun.find_limit(True)
                a.ufun.worst = a.ufun.find_limit(False)

            mx, mn = a.ufun.max_utility, a.ufun.min_utility
            state = a.awi.state()
            profile = a.awi.profile
            d = dict(
                storage=self.__storage,
                delivery=self.__penalty,
                exogenous_input_quantity=state.exogenous_input_quantity,
                exogenous_input_price=state.exogenous_input_price,
                exogenous_output_quantity=state.exogenous_output_quantity,
                exogenous_output_price=state.exogenous_output_price,
                storage_cost=state.storage_cost,
                delivery_penalty=state.delivery_penalty,
                current_balance=state.current_balance,
                cost=profile.cost,
                input_product=profile.input_product,
                n_lines=profile.n_lines,
                delivery_penalty_mean=profile.delivery_penalty_mean,
                storage_cost_mean=profile.storage_cost_mean,
                delivery_penalty_dev=profile.delivery_penalty_dev,
                storage_cost_dev=profile.storage_cost_dev,
                world=self.id,
                agent=aid,
                input=a.awi.my_input_product,
                output=a.awi.my_output_product,
                level=a.awi.level,
                max_util=mx,
                min_util=mn,
            )
            data.append(d)
        return result


for s, d in itertools.product(storage_cost, delivery_penalty):
    for _ in range(NWORLDS):
        world = Recorder(
            **Recorder.generate(
                agent_types=RandomOneShotAgent,
                n_steps=NSTEPS,
                storage_cost=s,
                delivery_penalty=d,
            ),
            storage_cost=s,
            delivery_penalty=d,
        )
        world.run()
        print(f"{_+1} of {NWORLDS} completed", flush=True)
df = pd.DataFrame.from_records(data)
df.to_csv("limits.csv", index=False)
mxstats = df.groupby(["storage", "delivery", "level"])[["max_util"]].describe()
mxstats.to_csv("max_stats.csv")
mnstats = df.groupby(["storage", "delivery", "level"])[["min_util"]].describe()
mnstats.to_csv("min_stats.csv")
print(mxstats)
print(mnstats)

for i in range(2):
    fig, axs = plt.subplots(2, 2)
    sns.violinplot(
        data=df.loc[df.level == i, :], x="storage", y="max_util", ax=axs[0, 0]
    )
    sns.violinplot(
        data=df.loc[df.level == i, :], x="storage", y="min_util", ax=axs[0, 1]
    )
    sns.violinplot(
        data=df.loc[df.level == i, :], x="delivery", y="max_util", ax=axs[1, 0]
    )
    sns.violinplot(
        data=df.loc[df.level == i, :], x="delivery", y="min_util", ax=axs[1, 1]
    )
    plt.suptitle(f"Level {i}")
    fig.show()
