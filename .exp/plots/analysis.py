import argparse
import csv
import json
import math
import pathlib

import matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]


def plot_conv(result: dict):
    trace = result["trace"]
    times = [point[0] for point in trace]
    iters = [point[1] for point in trace]
    costs = [point[2] for point in trace]
    feasibles = [point[3] for point in trace]

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    for ax, xv, yv, xl, yl, xs, ys in zip(
        axs, 
        (iters, iters, times), 
        (feasibles, costs, costs), 
        ("iter", "iter", "time"),
        ("feasible", "objective", "objective"),
        ("linear", "linear", "linear"),
        ("linear", "linear", "linear"),
    ):
        ax.plot(xv, yv)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xscale(xs)
        ax.set_yscale(ys)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Analysis for dataset {result['config']['problem']['path']}")
    fig.tight_layout()

    plt.show()


def plot_plan(result: dict):

    DAY = 86_400.0
    MU = 3.986_004e14
    J2 = 1.082_626e-3
    RE = 6.378_137e6

    def propagate(s, dt):
        n = math.sqrt(MU / s["a"] ** 3)
        p = s["a"] * (1.0 - s["e"] ** 2)
        f = 1.5 * J2 * (RE / p) ** 2 * n
        r_dt = -f * math.cos(s["i"])
        o_dt = f * (2.0 - 2.5 * math.sin(s["i"]) ** 2)
        out = {
            "a": s["a"],
            "e": s["e"],
            "i": s["i"],
            "r": s["r"] + r_dt * dt,
            "o": s["o"] + o_dt * dt,
            "t": s["t"]
        }
        return out
    
    def circular_mean(vals):
        s = sum(math.sin(v) for v in vals)
        c = sum(math.cos(v) for v in vals)
        mean = math.atan2(s, c)
        return mean + 2.0 * math.pi if mean < 0.0 else mean
    
    # Recover debris initial state
    data_path = ROOT_DIR / result["config"]["problem"]["path"]
    debris: list[dict] = []
    with open(data_path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            debris.append({
                "a": float(row["a[m]"]),
                "e": float(row["e[prop]"]),
                "i": float(row["i[rad]"]),
                "r": float(row["r[rad]"]),
                "o": float(row["o[rad]"]),
                "t": float(row["t[rad]"]),
            })

    # Recover chasers initial state
    assert(result["config"]["problem"]["chaser_init"] == "centroid")
    chasers = {
        "a":sum(d["a"] for d in debris) / len(debris),
        "e":sum(d["e"] for d in debris) / len(debris),
        "i":sum(d["i"] for d in debris) / len(debris),
        "r":circular_mean([d["r"] for d in debris]),
        "o":circular_mean([d["o"] for d in debris]),
        "t":circular_mean([d["t"] for d in debris]),
    }

    # Keplerian elements to plot and their labels
    elements = [
        # ("a", "semi-major axis [m]"),
        # ("e", "eccentricity [-]"),
        # ("i", "inclination [rad]"),
        ("r", "RAAN [rad]"),
        ("o", "arg. of perigee [rad]"),
        # ("t", "true anomaly [rad]"),
    ]


    solution = result["solution"]

    fig, axs = plt.subplots(1, 2, figsize=(15, 8), sharex=True)
    axs = axs.ravel()
    cmap = plt.get_cmap("tab10")

    for si, (seq, meets) in enumerate(zip(solution["sequences"], solution["meets"])):

        if len(seq) < 2:
            continue

        color = cmap(si % 10)
        hosts = [chasers] + [debris[seq[i]] for i in range(1, len(seq))]
        times = solution["times"][si]

        # Path between debris rendezvous
        nsteps = 40
        for i in range(len(hosts) - 1):
            depart = meets[i + 1] - times[i + 1]   # end of waiting on host i

            # Waiting phase
            wait_ts = [meets[i] + (depart - meets[i]) * k / nsteps for k in range(nsteps + 1)]
            wait_pts = [propagate(hosts[i], t) for t in wait_ts]

            # Transfer phase
            src = propagate(hosts[i], depart)
            dst = propagate(hosts[i + 1], meets[i + 1])
            trans_ts = [depart + (meets[i + 1] - depart) * k / nsteps for k in range(nsteps + 1)]
            trans_pts = [
                {key: src[key] + (dst[key] - src[key]) * k / nsteps for key in src}
                for k in range(nsteps + 1)
            ]

            days = [t / DAY for t in wait_ts + trans_ts]
            pnts = wait_pts + trans_pts
            for j, (ax, (key, _)) in enumerate(zip(axs, elements)):
                label = f"chaser {si}" if (i == 0 and j == 0) else None
                ax.plot(days, [p[key] for p in pnts], color=color, label=label)

        # Highlight debris rendezvous
        for i in range(1, len(seq)):
            pts = propagate(hosts[i], meets[i])
            day = meets[i] / DAY
            for j, (ax, (key, _)) in enumerate(zip(axs, elements)):
                ax.plot(day, pts[key], "o", color=color, markersize=3)

    for ax, (_, ylabel) in zip(axs, elements):
        ax.set_xlabel("time [days]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    fig.suptitle(f"Mission plan for dataset {result['config']['problem']['path']}")
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot", type=str, choices=["conv", "plan"])
    parser.add_argument("result", type=pathlib.Path)
    args = parser.parse_args()

    with open(args.result) as file:
        result = json.load(file)
        if result["config"]["experiment"]["name"] != "analysis":
            raise ValueError(f"expected 'analysis' experiment result'")

    if args.plot == "conv":
        plot_conv(result)
    elif args.plot == "plan":
        plot_plan(result)
    else:
        raise ValueError(f"unknown plot type '{args.plot}'")


if __name__ == "__main__":
    main()
