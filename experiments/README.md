# Experiments

Numerical experiments for the `mcrdv` paper.

**One configuration file describes exactly one run.** Sweeping across many runs
is a separate concern, and lives in `sweeps/`. Each run writes one self-contained
record holding the configuration it used *and* its results, so a result never
depends on a file that might have been edited since.

```
experiments/
  common/          shared core (see "How it fits together" below)
  sweeps/          how to batch each experiment on a cluster
  tmp/             generated campaign configs; disposable
  collect.py       status of an experiment's records, and a flat index
  submit.sh        submit a campaign as a Slurm array job
  slurm/           the array-job template

  smoke/           pipeline check, about a minute, not a paper result
  scaling/         cost and running time against catalogue size
  baseline/        the search against the RAAN-sweep construction
  convergence/     incumbent cost against elapsed time
  sensitivity/     one-at-a-time sweeps over the parameters
  showcase/        the structure of a produced mission plan
```

Every experiment folder has the same four parts:

| file | what it is |
|---|---|
| `run.yaml` | **one** run: an instance, a seed, a layer stack |
| `run.py` | `execute(config) -> result`; everything else is delegated |
| `plot.yaml` | which results each figure is about |
| `plot.py` | how to draw them, and what to export for tikz |

Plotting **displays** by default and writes nothing; `--save` writes instead.

## Install

```bash
pip install -e ".[experiments]"     # adds pyyaml and matplotlib
```

The solver itself still needs only numpy and numba — compute nodes never install
matplotlib.

## Running one experiment

```bash
cd experiments
./smoke/run.py          # one run, from run.yaml
./smoke/plot.py         # draws them on screen
./smoke/plot.py --save  # writes them to figures/ instead
```

That is the whole one-shot path: no sweep script, no cluster, nothing generated.
`run.py` writes one `YYYYMMDD_HHMMSS_xxxxxxxx.pkl` into `results/` and touches
nothing else.

## Running a campaign

```bash
SWEEP=$(./sweeps/scaling.py --quiet)    # -> tmp/scaling_20260917_184012
./scaling/run.py "$SWEEP"               # locally, everything in one process
```

or on the cluster — **edit `slurm/array.sbatch` once** to set `--account`:

```bash
SWEEP=$(./sweeps/scaling.py --quiet)
./submit.sh scaling "$SWEEP" --array 0-9
# ... jobs run, adding .pkl files to scaling/results/
./collect.py scaling --campaign "$SWEEP"   # did every config produce a record?
rm -rf "$SWEEP"                            # done with it
```

Run `collect.py --campaign` *before* clearing the campaign: comparing records
against the manifest is the one check that catches a task that died, and a job
array otherwise leaves a perfectly healthy-looking results directory with holes
in it. Nothing else is lost by deleting a campaign — every configuration is
copied verbatim into the record of the run it produced.

Useful flags:

```bash
./sweeps/scaling.py --dry-run          # list the configs, write nothing
./scaling/run.py "$SWEEP" --dry-run    # list this task's share
./scaling/run.py "$SWEEP/run-0007.yaml"   # reproduce exactly one
./scaling/plot.py --results $SCRATCH/mcrdv/scaling --only cost
./scaling/plot.py --save --out ~/paper/figures     # write somewhere else
```

### Sizing the array

A task runs its share **sequentially in one process**, which is deliberate:
numba compiles the solver in eight to twelve seconds and then never again in
that process, so one task per run would pay that toll on every run. Size the
walltime as

```
configs_per_task x (search budget + setup) + ~15 s of compilation
```

Configs are spread across tasks by stride rather than in contiguous blocks,
since runs differ in cost by orders of magnitude and a stride keeps tasks
balanced.

## How it fits together

`common/` splits along one line, and the line matters: **an experiment need not
solve with mcrdv at all.**

| module | knows about the solver? |
|---|---|
| `config` `record` `driver` `generate` `plotting` | no — a configuration is an opaque dict |
| `planners` `analysis` | yes — this is where mcrdv lives |
| `export` `style` `tidy` | no — drawing and reductions |

So a new experiment driving something else writes its own `execute` and gets
sweeping, chunking, recording, selection and tikz export for free.

## Writing a new experiment

Copy `smoke/`. The configuration describes one run:

```yaml
name: myexperiment
seed: 0

# `instance` mirrors Problem's own arguments, so it fully describes what is
# being solved -- including the horizon.
instance: {name: iridium33, num_chasers: 15, max_load: 8, max_time: 365d}

transfer: {type: qlaw, thrust: 3.0e-3}
schedule: {type: dp, grid_size: 32}
sequence: {type: hgs, preset: 4, config: {max_time: 2m, log_save: true}}
```

and `run.py` says what one run *is*:

```python
from common import driver, planners

def execute(config: dict) -> dict:
    return planners.solve(config)      # or anything at all

if __name__ == "__main__":
    driver.main(__file__, execute, warmup=planners.warm_up)
```

Whatever `execute` returns becomes `result` in the record. There is no schema:
keep the mission plan, a trace, timings, another solver's logs — whatever a
figure or a later question might want.

For mcrdv the YAML keys **are** the constructor parameters — `grid_size` is
`DpSchedule(grid_size=...)`, `crossover` is `HgsConfig(crossover=...)` — so any
knob a layer accepts is usable from a configuration with no code change. Times
accept suffixes: `2m`, `3h`, `365d`.

A sweep script in `sweeps/<name>.py` then says how to explore it:

```python
from common import generate

def campaign(base):
    return generate.grid(base, instance=INSTANCES, seed=range(5))
    # or generate.one_at_a_time(base, seed=range(5), **{"schedule.grid_size": [...]})

if __name__ == "__main__":
    generate.main(__file__, "myexperiment", campaign)
```

## Records

One pickle per run, holding three fields and nothing else:

```python
RunRecord(run_id, config, result)
```

Anything more would be a guess about what an experiment cares about. A run that
raises is still recorded, with `error` in `result`, so "crashed" stays
distinguishable from "never ran".

`record.row()` flattens a record to one row — config and result merged, nested
keys as dotted paths — which is what `plot.yaml` filters on and what
`collect.py` writes to `index.csv`. Values a table cannot hold (a `MissionPlan`,
a trace) are left out of the row but stay on the record.

## Figures and tikz

matplotlib is for looking at results while they come in; the paper draws them
with pgfplots. `plot.yaml` says *which* results a figure is about:

```yaml
source: results
figures:
  - name: cost
    select: {instance.name: iridium33}   # equality, or a list for membership
    group_by: num_targets
    value: cost
    reduce: median+iqr                   # or median, mean, raw
    x_scale: log
    xlabel: targets
```

and `plot.py` says how it is drawn. Selection is declarative because it genuinely
is; rendering is not, so it stays Python. `reduce: raw` hands a figure the
selected records untouched — that is how convergence resamples staircases and
showcase reads a plan leg by leg.

Without `--save` the figures are shown and nothing is written, which is what
you want while deciding whether a figure says anything. With `--save`, each one
writes three files under a shared stem, named exactly like a record —
`YYYYMMDD_HHMMSS_xxxxxxxx`:

| file | for |
|---|---|
| `<stem>.pdf` | a rendering to keep |
| `<stem>.dat` | `\addplot table[x=..., y=...]` |
| `<stem>.tex` | a paste-ready skeleton that already compiles |

So a figure sorts beside the runs it came from, and re-saving adds a version
rather than overwriting one already cited in a draft. Which figure is which is
printed when saving (`cost -> 20260918_092944_egurwslo`) and recorded in the
`.tex` header as `% Figure: cost`.

On a machine with no display, plotting without `--save` says so and stops rather
than silently drawing into the void.

The `.dat` is **not** the index: a figure shows medians, interquartile bands and
resampled curves computed in `plot.py`. Drawing through the recording wrappers
exports exactly those points, so the table and the figure cannot disagree:

```python
export.line(ax, x, median, name="hgs")
export.band(ax, x, lo, hi, name="hgs_iqr")
export.grouped_bars(ax, labels, {"hgs": values})
```

Series sharing an x-axis become columns of one table; series on different grids
get one file each, rather than a ragged table padded with holes.

## Things worth knowing

- **A convergence curve is dashed while it is not yet feasible.** `LogEntry`
  records whether the *incumbent* satisfies both limits, not just how many
  feasible individuals the population holds, because a cost logged before any
  feasible plan exists is not comparable with one logged after. The line goes
  solid only once every replication has found one.
- **Seeds are not decoration.** The search is stochastic, and a deeper preset can
  lose to a shallower one on a single seed. Every figure reduces over
  replications; none reports one run.
- **Budgets, not iteration counts.** Configurations stop the search on wall-clock
  time (`config: {max_time: 2m}`), which is what makes a Slurm walltime
  predictable and keeps instances of very different sizes comparable.
- **Setup is reported separately from search.** Building the table of per-leg
  bounds is quadratic in the catalogue size — about 22 s on `fengyun1C` — and
  summing it into the search time would hide the term that actually scales.
- **The search log is kept by `log_save`, printed by `verbose`.** The two are
  independent, and `log_save: true` under `sequence.config` records the
  convergence of a run that prints nothing. It is set in the configuration and
  nowhere else — `planners.solve` takes no such argument — so what an experiment
  keeps is readable from the file that describes it. Only `convergence/` asks
  for it.
- **Re-running a config adds a record, it does not replace one.** Names are
  timestamped, so repeats accumulate. Group on config *values*, never on a
  campaign's file names — `tmp/` is regenerated every time.
- **Swarm sizes for the large catalogues are a guess.** `cosmos2251` and
  `fengyun1C` have no established chaser count or capacity; the values in
  `sweeps/scaling.py` satisfy `chasers * load >= targets` and nothing more.
  Revisit them before quoting numbers.
- **`--cpus-per-task` and the bound table.** The only threaded section,
  `mcrdv.sequence.hgs.pair_lb`, sizes its pool from `os.cpu_count()` — the whole
  node, not the allocation — so a small `--cpus-per-task` oversubscribes and
  makes `t_setup` noisy. See the note at the end of `slurm/array.sbatch`.
