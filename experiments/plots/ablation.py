import argparse
import csv
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


def step_interp(xs, ys, grid):
    out, j, last = [], 0, ys[0]
    for g in grid:
        while j < len(xs) and xs[j] <= g:
            last = ys[j]
            j += 1
        out.append(last)
    return out


def plot_ablation(results: list[dict]):

    if not results:
        raise ValueError("no results to plot")
    
    match_configs = [
        {
            "config": {
                "solver": {
                    "args": {
                        "middle": { "args": { "allow_waiting": True, "departure_hints": True, }},
                        "outer": { "args": { "generation": "greedy", "raan_ordering": True, }}
                    }
                }
            },
            "label": "Baseline",
        },
        {
            "config": { "solver": { "args": { "middle": { "args": { "allow_waiting": False }}}}},
            "label": "No waiting",
        },
        {
            "config": { "solver": { "args": { "middle": { "args": { "departure_hints": False }}}}},
            "label": "No departure hints",
        },
        {
            "config": { "solver": { "args": { "outer": { "args": { "generation": "random" }}}}},
            "label": "No greedy generation",
        },
        {
            "config": { "solver": { "args": { "outer": { "args": { "raan_ordering": False }}}}},
            "label": "No RAAN ordering",
        },
    ]

    axes_spec = [
        {
            "x": lambda tr: [p[1] for p in tr],   # iter
            "y": lambda tr: [p[2] for p in tr],   # objective
            "xlabel": "iter", "ylabel": "objective",
            "xscale": "linear", "yscale": "linear",
        },
        {
            "x": lambda tr: [p[0] for p in tr],   # time
            "y": lambda tr: [p[2] for p in tr],   # objective
            "xlabel": "time", "ylabel": "objective",
            "xscale": "linear", "yscale": "linear",
        },
    ]

    # Assign each run to the first matching config
    groups: list[list[dict]] = [[] for _ in match_configs]
    for result in results:
        for gi, mc in enumerate(match_configs):
            if dict_match(mc["config"], result["config"]):
                groups[gi].append(result)
                break
        else:
            print(f"warning: no matching config for: {result['config']}")

    for gi, runs in enumerate(groups):
        print(f"{match_configs[gi]['label']}: {len(runs)} results")
    

    fig, axs = plt.subplots(1, len(axes_spec), figsize=(6 * len(axes_spec), 5), squeeze=False)
    axs = axs.ravel()
    cmap = plt.get_cmap("tab10")

    for gi, (mc, runs) in enumerate(zip(match_configs, groups)):

        if not runs:
            continue

        color = cmap(gi % 10)
        for ax, spec in zip(axs, axes_spec):
            series = [(spec["x"](r["trace"]), spec["y"](r["trace"])) for r in runs]

            # Individual runs with same color, low opacity
            for xs, ys in series:
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1)

            # Mean across runs on a shared grid with full opacity
            grid = sorted({x for xs, _ in series for x in xs})
            pack = [step_interp(xs, ys, grid) for xs, ys in series]
            mean = [sum(col) / len(col) for col in zip(*pack)]
            ax.plot(grid, mean, color=color, alpha=1.0, linewidth=2, label=mc["label"])

    for ax, spec in zip(axs, axes_spec):
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.set_xscale(spec["xscale"])
        ax.set_yscale(spec["yscale"])
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    fig.suptitle(f"Ablation for dataset {results[0]['config']['problem']['path']}")
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    results = []
    for file in (ROOT_DIR / "exp" / "results").glob("ablation_*.json"):
        with open(file) as f:
            result = json.load(f)
            if result["config"]["problem"]["path"] == args.dataset:
                results.append(result)

    plot_ablation(results)


if __name__ == "__main__":
    main()
