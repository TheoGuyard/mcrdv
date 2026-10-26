# Experiments

Numerical experiments for the `mcrdv` library.

## Install

```bash
pip install -e ".[experiments]"
```

## Running one experiment

```bash
cd experiments
./<experiment>/run.py          # one run from the configuration in ./<experiment>/run.yaml 
./<experiment>/plot.py         # aggregate runs and draw figures on screen
./<experiment>/plot.py --save  # write figures data to ./<experiment>/figures/
```

One configuration file `./<experiment>/run.yaml` describes exactly one experiment run. Sweeping
across many runs is a separate concern, and lives in `sweeps.yaml`. Each run writes one
self-contained record holding the configuration it used *and* its results, so a
result never depends on a file that might have been edited since.

```
experiments/
  _common/         shared core (see "How it fits together" below)
  sweeps.yaml      every campaign, and what running it costs on the cluster
  script.py        send, install, make, run, receive, clean
  tmp/             generated campaigns; disposable

  convergence/     incumbent cost against elapsed time
  sequence/        the search's presets: what each level costs and buys
  sensitivity/     one instance parameter at a time, on generated clouds
  showcase/        the structure of a produced mission plan
  schedule/        the schedule layer alone: gap to a fine grid, time per call
  optimality/      the search against the exact optimum of small instances
  transfer/        how the transfer layer's definition reshapes a plan
  comparison/      mcrdv beside planners of the literature, on their instances
```

Every experiment folder has the same four parts:

| file | what it is |
|---|---|
| `run.yaml` | **one** run, by hand: an instance, a seed, a solver |
| `run.py` | `execute(config) -> result`; everything else is delegated |
| `plot.yaml` | which results each figure is about |
| `plot.py` | how to draw them, and what to export for tikz |

Plotting **displays** by default and writes nothing; `--save` writes instead.

## How it fits together

`_common/` holds only what several experiments share, in three modules:

| module | what it does | knows about mcrdv? |
|---|---|---|
| `runner` | configurations, `sweeps.yaml` campaigns, the `run.py` loop, records | no |
| `planners` | `instance:` → `Problem`, `solver:` → mcrdv stack, solving, plan metrics | yes |
| `plotter` | `plot.yaml` → figures, and their export for pgfplots | no |

`runner` and `plotter` know nothing of mcrdv: **an experiment need not solve
with mcrdv at all.**

Code used by a single experiment lives in that experiment's `run.py` (or
`plot.py`): the exact optimum (the `optimum` sequence layer) in
`optimality/run.py`, the schedule bench in
`schedule/run.py`, the synthetic clouds in `sensitivity/run.py`, the
plan-structure readings in `showcase/run.py`, the cost oracles in
`transfer/run.py`, the literature's planners and transfer models in
`comparison/run.py`, and the numbers those papers publish in
`comparison/plot.py`. A layer written there registers itself by name in
`planners.TRANSFERS`, `SCHEDULES` or `SEQUENCES` (a generated instance in
`planners.GENERATORS`), so a configuration reaches it like any shipped one.
