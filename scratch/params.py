#!/usr/bin/env python
import itertools
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
import typer
from negmas.helpers import add_records

from scml.oneshot import SCML2020OneShotWorld
from scml.oneshot.agents import GreedyOneShotAgent
from scml.oneshot.agents import RandomOneShotAgent
from scml.scml2020.common import is_system_agent

app = typer.Typer()
# disposal_cost = [0.1, 0.2, 0.5, 1.0, 2.0]
# shortfall_penalty = [0.1, 0.2, 0.4, 1.0, 2.0]

BASE_PATH = Path(".")

PARAMS = dict(
    disposal_cost=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1),
    shortfall_penalty=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1),
    max_productivity=(1.0, 0.9, 0.8, 0.6, 0.5),
    profit_means=((0.0, 0.1), (0.1, 0.1), (0.1, 0.2), (0.0, 0.2)),
)


def save_data(data, dir_name, post):
    dir_name = Path(dir_name)
    file_name = dir_name / f"limits_{post}.csv"
    add_records(file_name, data)
    df = pd.read_csv(file_name, index_col=None)
    mxstats = (
        df.groupby(["p_disposal_cost", "p_shortfall_penalty", "level"])[["max_util"]]
        .describe()
        .reset_index()
    )
    mxstats.to_csv(dir_name / f"max_stats_{post}.csv", index=False)
    mnstats = (
        df.groupby(["p_disposal_cost", "p_shortfall_penalty", "level"])[["min_util"]]
        .describe()
        .reset_index()
    )
    mnstats.to_csv(dir_name / f"min_stats_{post}.csv", index=False)
    # print(mnstats)
    # print(mxstats)


class Recorder(SCML2020OneShotWorld):
    """Records UFun ranges"""

    def __init__(
        self, *args, params, util_eval_method="bruteforce", dir_name=BASE_PATH, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__util_eval_method = util_eval_method
        self.__dir_name = dir_name
        self.__params = params

    def simulation_step(self, stage):
        result = super().simulation_step(stage)
        for aid, a in self.agents.items():
            if is_system_agent(aid):
                continue
            if a.ufun is None:
                continue
            if self.__util_eval_method.startswith("b"):
                a.ufun.worst = a.ufun.find_limit_brute_force(False)
                a.ufun.best = a.ufun.find_limit_brute_force(True)
            elif self.__util_eval_method.startswith("o"):
                a.ufun.best = a.ufun.find_limit_optimal(True)
                a.ufun.worst = a.ufun.find_limit_optimal(False)
            elif self.__util_eval_method.startswith("g"):
                a.ufun.best = a.ufun.find_limit_greedy(True)
                a.ufun.worst = a.ufun.find_limit_greedy(False)
            else:
                a.ufun.best = a.ufun.find_limit(True)
                a.ufun.worst = a.ufun.find_limit(False)

            mx, mn = a.ufun.max_utility, a.ufun.min_utility
            max_producible = a.ufun.best.producible
            min_producible = a.ufun.worst.producible
            state = a.awi.state()
            profile = a.awi.profile
            d = {f"p_{k}": v for k, v in self.__params.items()}
            d.update({f"f_{k}": v for k, v in vars(a.ufun).items()})
            d.update(
                dict(
                    step=self.current_step,
                    exogenous_input_quantity=state.exogenous_input_quantity,
                    exogenous_input_price=state.exogenous_input_price,
                    exogenous_output_quantity=state.exogenous_output_quantity,
                    exogenous_output_price=state.exogenous_output_price,
                    disposal_cost=state.disposal_cost,
                    shortfall_penalty=state.shortfall_penalty,
                    production_cost=profile.cost,
                    current_balance=state.current_balance,
                    input_product=profile.input_product,
                    n_lines=profile.n_lines,
                    shortfall_penalty_mean=profile.shortfall_penalty_mean,
                    disposal_cost_mean=profile.disposal_cost_mean,
                    shortfall_penalty_dev=profile.shortfall_penalty_dev,
                    disposal_cost_dev=profile.disposal_cost_dev,
                    world=self.id,
                    agent=aid,
                    input=a.awi.my_input_product,
                    output=a.awi.my_output_product,
                    level=a.awi.level,
                    max_producible=max_producible,
                    min_producible=min_producible,
                    max_util=mx,
                    min_util=mn,
                    limit_method=self.__util_eval_method,
                )
            )
            # print(d)
            save_data([d], self.__dir_name, self.__util_eval_method)
        return result


def run_once(params, n_steps, method):
    world = Recorder(
        **Recorder.generate(
            agent_types=[GreedyOneShotAgent, RandomOneShotAgent],
            n_steps=n_steps,
            **params,
        ),
        dir_name=BASE_PATH,
        util_eval_method=method,
        params=params,
    )
    world.run()


@app.command()
def run(
    steps: int = 10,
    worlds: int = 10,
    method: str = "bruteforce",
    serial: bool = False,
    parallelism: float = 1.0,
    vars: str = "",
    fast: bool = False,
):
    if parallelism > 1.5:
        workers = int(parallelism)
    elif 0 < parallelism < 1:
        workers = int(parallelism * cpu_count())
    else:
        workers = int(0.5 * cpu_count())
    if workers < 1:
        workers = 2

    if len(vars) < 1:
        params = {k: v for k, v in PARAMS.items()}
    else:
        keys = set(vars.split(";"))
        params = {k: v for k, v in PARAMS.items() if k in keys}
    if fast:
        params = {k: v if len(v) < 2 else (v[0], v[-1]) for k, v in params.items()}

    methods = method.split(";")
    print(f"Starting run with {steps} steps and {worlds} worlds: Method={method}")
    param_names = tuple(params.keys())
    param_values = list(itertools.product(*tuple(params.values())))
    print(
        f"Will run {len(methods) * len(param_values) * steps * worlds} simulation steps.",
        flush=True,
    )
    futures = []
    if serial:
        for method in methods:
            for v in tqdm.tqdm(param_values):
                run_once(dict(zip(param_names, v)), steps, method)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for method in methods:
                for v in param_values:
                    futures.append(
                        pool.submit(run_once, dict(zip(param_names, v)), steps, method)
                    )
        print("RUNNING ...", flush=True)
        for f in tqdm.tqdm(as_completed(futures), total=len(param_values)):
            pass
    print("DONE")


@app.command()
def plot(method: str = "bruteforce"):
    methods = method.split(";")
    dfs = []
    dir_name = BASE_PATH
    for method in methods:
        dfs.append(pd.read_csv(BASE_PATH / f"limits_{method}.csv"))
    df = pd.concat(dfs, ignore_index=True)
    if len(df) < 1:
        print("No data")
        return
    print(f"Found {len(df)} data records")
    mxstats = (
        df.groupby(["p_disposal_cost", "p_shortfall_penalty", "level"])[["max_util"]]
        .describe()
        .reset_index()
    )
    mxstats.to_csv(dir_name / f"max_stats_{method.replace(';', '_')}.csv", index=False)
    mnstats = (
        df.groupby(["p_disposal_cost", "p_shortfall_penalty", "level"])[["min_util"]]
        .describe()
        .reset_index()
    )
    mnstats.to_csv(dir_name / f"min_stats_{method.replace(';', '_')}.csv", index=False)
    print(mnstats)
    print(mxstats)
    for k in PARAMS.keys():
        if len(PARAMS[k]) < 2:
            continue
        df[f"p_{k}"] = df[f"p_{k}"].apply(lambda x: str(x))
        print(f"Plotting {k}")
        fig, axs = plt.subplots(2, 2)
        sns.barplot(
            data=df.loc[df.level == 1, :], x=f"p_{k}", y="max_util", ax=axs[0, 0]
        )
        sns.barplot(
            data=df.loc[df.level == 1, :], x=f"p_{k}", y="min_util", ax=axs[0, 1]
        )
        sns.barplot(
            data=df.loc[df.level == 2, :], x=f"p_{k}", y="max_util", ax=axs[1, 0]
        )
        sns.barplot(
            data=df.loc[df.level == 2, :], x=f"p_{k}", y="min_util", ax=axs[1, 1]
        )
        plt.suptitle(f"{k}")
        fig.show()
        fig, axs = plt.subplots(2, 2)
        sns.barplot(
            data=df.loc[df.level == 1, :], x=f"p_{k}", y="max_producible", ax=axs[0, 0]
        )
        sns.barplot(
            data=df.loc[df.level == 1, :], x=f"p_{k}", y="min_producible", ax=axs[0, 1]
        )
        sns.barplot(
            data=df.loc[df.level == 2, :], x=f"p_{k}", y="max_producible", ax=axs[1, 0]
        )
        sns.barplot(
            data=df.loc[df.level == 2, :], x=f"p_{k}", y="min_producible", ax=axs[1, 1]
        )
        plt.suptitle(f"{k}")
        fig.show()


if __name__ == "__main__":
    app()
