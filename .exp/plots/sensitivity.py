import argparse
import json
import math
import pathlib

import matplotlib.pyplot as plt


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]


def dict_match(partial, actual):
    for key, value in partial.items():
        if key not in actual:
            return False
        if isinstance(value, dict):
            if not isinstance(actual[key], dict) or not dict_match(value, actual[key]):
                return False
        elif actual[key] != value:
            return False
    return True


def get_path(cfg, path):
    for key in path:
        cfg = cfg[key]
    return cfg


def without_path(partial, path):
    partial = json.loads(json.dumps(partial))  # deep copy
    cur = partial
    for key in path[:-1]:
        cur = cur[key]
    cur.pop(path[-1], None)
    return partial


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def plot_sensitivity(results: list[dict]):

    if not results:
        raise ValueError("no results to plot")

    # Reference configuration: A run contributes to a sensitivity curve if its 
    # config matches this baseline on every field, except the parameter varied
    baseline = {
        "problem": {
            "max_time": 31_536_000.0,
            "max_fuel": 25_000.0,
            "max_load": 10,
            "chaser_load": 1.5,
            "factor_time": 0.0,
            "factor_fuel": 1.0,
        },
    }

    # Parameters varied
    varied = [
        {"label": "max time", "path": ["problem", "max_time"], "xscale": "linear"},
        {"label": "max fuel", "path": ["problem", "max_fuel"], "xscale": "linear"},
        {"label": "max load", "path": ["problem", "max_load"], "xscale": "linear"},
    ]

    # Per-run metrics on the plots
    metrics = [
        {
            "yvalue": lambda r: r["solution"]["cost"],
            "ylabel": "objective", "yscale": "linear",
        },
        {
            "yvalue": lambda r: int(r["solution"]["feasible"]),
            "ylabel": "feasibility", "yscale": "linear",
        },
    ]

    nrows, ncols = len(varied), len(metrics)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for ri, param in enumerate(varied):

        # Select runs matching the baseline on all but the varied parameter
        reduced = without_path(baseline, param["path"])
        matching = [r for r in results if dict_match(reduced, r["config"])]

        # Group the matching runs by the varied parameter value
        groups: dict = {}
        for r in matching:
            groups.setdefault(get_path(r["config"], param["path"]), []).append(r)
        values = sorted(groups)

        print(f"{param['label']}: {[(v, len(groups[v])) for v in values]}")

        for ci, metric in enumerate(metrics):
            ax = axs[ri][ci]
            means, stds = [], []
            for v in values:
                ys = [metric["yvalue"](r) for r in groups[v]]
                mean, std = mean_std(ys)
                means.append(mean)
                stds.append(std)

            ax.errorbar(values, means, yerr=stds, marker="o", capsize=3)
            ax.set_xlabel(param["label"])
            ax.set_ylabel(metric["ylabel"])
            ax.set_xscale(param["xscale"])
            ax.set_yscale(metric["yscale"])
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sensitivity for dataset {results[0]['config']['problem']['path']}")
    fig.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    results = []
    for file in (ROOT_DIR / "exp" / "results").glob("sensitivity_*.json"):
        with open(file) as f:
            result = json.load(f)
            if result["config"]["problem"]["path"] == args.dataset:
                results.append(result)

    plot_sensitivity(results)


if __name__ == "__main__":
    main()
