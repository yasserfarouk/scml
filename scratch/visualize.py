import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from negmas.helpers import dump, load, unique_name

BASE = Path.home() / "negmas" / "tournaments"
DST_FOLDER = Path.home() / "negmas" / "visdata"
DST_FOLDER.mkdir(parents=True, exist_ok=True)

# tournament files
SCORES_FILE = "scores.csv"
CONFIGS_FILE = "assigned_configs.csv"  # has n_steps, __dir_name
# WORLD_STATS_FILE = "stats.csv" # has path and world

TOURNAMENT_REQUIRED = [SCORES_FILE, CONFIGS_FILE]

# single world files
CONTRACTS_FILE = "contracts_full_info.csv"
NEGOTIATIONS_FILE = "negotiations.csv"
BREACHES_FILE = "breaches.csv"
AGENTS_FILE = "agents.csv"
PARAMS_FILE = "params.csv"
STATS_FILE = "stats.csv"
WORLD_REQUIRED = [STATS_FILE]

path_map = {"/export/home": "/Users"}


def nonzero(f):
    return f.exists() and f.stat().st_size > 0


def get_folders(base_folder, main_file, required):
    return [
        _.parent
        for _ in base_folder.glob(f"**/{main_file}")
        if all(nonzero(f) for f in [_.parent / n for n in required])
    ]


def get_torunaments(base_folder):
    return get_folders(
        base_folder, main_file=CONFIGS_FILE, required=TOURNAMENT_REQUIRED
    )


def get_worlds(base_folder):
    return get_folders(base_folder, main_file=AGENTS_FILE, required=WORLD_REQUIRED)


def parse_tournament(path, t_indx, base_indx):
    configs = load(path / CONFIGS_FILE)
    if not configs:
        return None, None, None
    scores = load(path / SCORES_FILE).to_dict(orient="records")
    if not scores:
        return None, None, None
    worlds = []
    world_indx = dict()
    for i, c in enumerate(configs):
        worlds.append(
            dict(
                indx=i + base_indx,
                path=c["__dir_name"],
                name=c["world_params"]["name"],
                n_steps=c["world_params"]["n_steps"],
                n_processes=c["world_params"]["n_processes"],
                n_agents=len(c["is_default"]),
                tournament=path.name,
                tournament_indx=t_indx,
            )
        )
        world_indx[worlds[-1]["indx"]] = worlds[-1]["name"]

    agents = []

    for i, s in enumerate(scores):
        agents.append(
            dict(
                indx=i + base_indx,
                name=s["agent_id"],
                type=s["agent_type"],
                # id=s["agent_id"],
                score=s["score"],
                world=s["world"],
                world_indx=world_indx.get(s["world"], None),
            )
        )
    return worlds, pd.DataFrame.from_records(agents)


def parse_world(path, wname, agents, w_indx, base_indx):
    stats = pd.read_csv(path / STATS_FILE, index_col=0)
    agent_names = agents["name"].unique()
    # inventory_{}_input, output
    stat_names = [
        "inventory_input",
        "inventory_output",
        "balance",
        "productivity",
        "spot_market_loss",
        "spot_market_quantity",
        "assets",
        "bankrupt",
    ]
    results = []
    for aname in agent_names:
        colnames = []
        for n in stat_names:
            ns = n.split("_")
            if len(ns) > 1:
                col_name = f"{ns[0]}_{aname}_{ns[-1]}"
            else:
                col_name = f"{ns[0]}_{aname}"
            colnames.append(col_name)
        x = stats.loc[:, colnames].reset_index().rename(columns=dict(index="step"))
        x.columns = [
            _ if aname not in _ else _.replace(f"_{aname}", "") for _ in x.columns
        ]
        if len(x):
            x["relative_time"] = x["step"] / len(x)
        else:
            x["relative_time"] = 0.0
        x["agent"] = aname
        x["world"] = wname
        results.append(x)
    return pd.concat(results, ignore_index=True)


def get_data(base_folder):
    paths = get_torunaments(base_folder)
    tournaments, worlds, agents, stats = [], [], [], []
    for i, t in enumerate(paths):
        print(f"Processing {t.name} [{i} of {len(tournaments)}]", flush=True)
        indx = i + 1
        base_indx = (i + 1) * 1_000_000
        tournaments.append(dict(indx=indx, path=t, name=t.name))
        w, a = parse_tournament(t, indx, base_indx)
        for j, world in enumerate(w):
            print(f"\tWorld {world['name']} [{j} of {len(w)}]", flush=True)
            wagents = a.loc[a.world == world["name"]]
            stats.append(
                parse_world(
                    world["path"],
                    world["name"],
                    wagents,
                    base_indx + j + 1,
                    base_indx + j + 1,
                )
            )
        worlds.append(pd.DataFrame.from_records(w))
        agents.append(a)

    tournaments = pd.DataFrame.from_records(tournaments)
    worlds = pd.concat(worlds, ignore_index=True)
    agents = pd.concat(agents, ignore_index=True)
    return tournaments, worlds, agents, stats


def main(base_folder):
    tournaments, worlds, agents, stats = get_data(base_folder)
    dst_folder = base_folder / unique_name(base_folder.name, rand_digits=3)
    dst_folder.mkdir(parents=True, exist_ok=True)
    for df, name in zip(
        (tournaments, worlds, agents, stats),
        ("tournaments", "worlds", "agents", "stats"),
    ):
        df.to_csv(dst_folder / name, index=False)


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))
