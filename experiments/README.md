# Experiments

Numerical experiments for the `mcrdv` paper.

**One configuration file describes exactly one run.** Sweeping across many runs
is a separate concern, and lives in `sweeps.yaml`. Each run writes one
self-contained record holding the configuration it used *and* its results, so a
result never depends on a file that might have been edited since.

```
experiments/
  _common/         shared core (see "How it fits together" below)
  sweeps.yaml      every campaign, and what running it costs on the cluster
  script.py        send, install, make, run, status, receive, clean
  tmp/             generated campaigns; disposable

  convergence/     incumbent cost against elapsed time
  sequence/        the search's presets: what each level costs and buys
  sensitivity/     one instance parameter at a time, on generated clouds
  showcase/        the structure of a produced mission plan
  schedule/        the schedule layer alone: gap to a fine grid, time per call
  optimality/      the search against the exact optimum of small instances
  transfer/        how the transfer layer's definition reshapes a plan
```

Every experiment folder has the same four parts:

| file | what it is |
|---|---|
| `run.yaml` | **one** run, by hand: an instance, a seed, a solver |
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
./showcase/run.py          # one run, from run.yaml
./showcase/plot.py         # draws its figures on screen
./showcase/plot.py --save  # writes them to figures/ instead
```

That is the whole one-shot path: no campaign, no cluster, nothing generated.
`run.py` writes one `YYYYMMDD_HHMMSS_xxxxxxxx.pkl` into `results/` and touches
nothing else.

## Running a campaign

A campaign is described entirely by `sweeps.yaml`: each sweep holds the `base`
configuration its axes vary from, and **no `run.yaml` is read**. The two are
deliberately independent — a `run.yaml` edited to try something by hand cannot
quietly change what a campaign means, and a campaign is reproducible from
`sweeps.yaml` and `script.py` alone. What they have in common is only their
shape: a sweep's `base` is written exactly as a `run.yaml` is.

Locally, build the campaign and run it in one process:

```bash
./script.py make -s sensitivity    # -> tmp/sensitivity/{configs,job.sh}
./sensitivity/run.py tmp/sensitivity/configs
```

On the cluster, `script.py` is the whole path. `send` and `receive` run on the
laptop, the rest on the cluster:

```bash
./script.py send                  # laptop: rsync the repository to scratch
                                  # then, on the cluster:
./script.py install               # once: build the environment
./script.py make                  # write every campaign and its job file
./script.py run -s sensitivity    # submit it as an array job
./script.py status -s sensitivity # did every configuration produce a record?
./script.py clean                 # forget the campaigns
                                  # back on the laptop:
./script.py receive               # bring the records home to plot them
```

`receive` is **one** transfer, however many campaigns are waiting: it rsyncs
the whole `experiments/` tree with filter rules narrowing it to the results of
the experiments asked for. A cluster behind two-factor authentication asks for
a password and a code on every connection, and an experiment at a time would
ask once per experiment. If you want `send` and `receive` to share a single
authentication too, enable connection multiplexing in `~/.ssh/config`:

```
Host rorqual.alliancecan.ca
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

`make` writes the job file with the paths of the repository it runs in, so it
belongs on the cluster, after `send` — which excludes `tmp/` for exactly that
reason. Run `status` *before* `clean`: comparing the records against the
campaign's manifest is the one check that catches a task that died, and a job
array otherwise leaves a perfectly healthy-looking results directory with holes
in it. Nothing else is lost by clearing a campaign — every configuration is
copied verbatim into the record of the run it produced.

Useful flags:

```bash
./sensitivity/run.py tmp/sensitivity/configs --dry-run      # this task's share
./sensitivity/run.py tmp/sensitivity/configs/run-0007.yaml  # reproduce one
./sensitivity/plot.py --results $SCRATCH/mcrdv/sensitivity --only cost_debris
./sensitivity/plot.py --save --out ~/paper/figures          # write elsewhere
```

### Sizing a campaign

A task runs its share **sequentially in one process**, which is deliberate:
numba compiles the solver in eight to twelve seconds and then never again in
that process, so one task per run would pay that toll on every run. `tasks` in
`sweeps.yaml` is the array size, and the walltime it needs is

```
configs / tasks x (search budget + setup) + ~15 s of compilation
```

Configs are spread across tasks by stride rather than in contiguous blocks,
since runs differ in cost by orders of magnitude and a stride keeps tasks
balanced.

`cpus` deserves one warning: the only threaded section,
`mcrdv.sequence.hgs.pair_lb`, sizes its pool from `os.cpu_count()` — the whole
node, not the allocation — so a small `cpus` oversubscribes and makes the
reported `t_setup` noisy. Either ask for a large share of the node, or read the
setup time on the biggest instances as an upper bound.

## How it fits together

`common/` splits along one line, and the line matters: **an experiment need not
solve with mcrdv at all.**

| module | knows about the solver? |
|---|---|
| `config` `record` `driver` `generate` `plotting` | no — a configuration is an opaque dict |
| `planners` `analysis` | yes — this is where mcrdv lives |
| `synthetic` `exact` `benchmark` `transfers` | yes — instances, layers and references the experiments add |
| `export` `style` `tidy` | no — drawing and reductions |

Those in the third row serve particular experiments but are not tied to them:

- `synthetic` generates a debris cloud from a single break-up (see *Instances*).
- `exact` finds the optimum of an instance of up to twenty targets by
  enumeration, exact for the transfer and schedule layers it is given.
- `benchmark` scores a schedule layer on a fixed bank of sequences against a
  reference layer, with no search involved.
- `transfers` holds transfer layers written here rather than shipped: the same
  Q-law physics under a different notion of what a transfer costs. They
  register like any other layer, so a configuration reaches them by name.

## Instances

`instance` mirrors `Problem`'s arguments, plus what says which targets to take:

```yaml
instance: {name: iridium33, num_chasers: 15, max_load: 8, max_time: 365d}   # a whole catalogue
instance: {name: cosmos2251, num_targets: 16, draw: 3, ...}                 # 16 of its targets
instance: {name: synthetic, num_targets: 400, draw: 0, cloud: {kick: 50}, ...}
```

- **A catalogue** is a file in `data/`, taken whole unless `num_targets` asks
  for a random subset. `draw` seeds which targets are taken, and a smaller subset
  of one draw is contained in every larger one.
- **A synthetic cloud** (`common/synthetic.py`) is one break-up: a parent on a
  circular orbit fragments, each fragment gets a random kick of `kick` m/s per
  component, and the cloud is observed `age` later. By then the J2 drift has
  spread the right ascensions the way it does in real clouds. The defaults give
  Fengyun-1C's shape. For one `draw`, a cloud of `n` targets is the first `n`
  fragments of every larger cloud, so growing `num_targets` adds debris to the
  same cloud.

`draw` and `seed` are separate axes on purpose. `draw` picks the instance and
`seed` drives the search, so replications of a search on one instance and
different instances of one size can be told apart.

So a new experiment driving something else writes its own `execute` and gets
sweeping, chunking, recording, selection and tikz export for free.

## What each experiment answers

| experiment | question | varies |
|---|---|---|
| `schedule` | How far is a schedule from optimal for a given grid, what do hints add, and what does a call cost? | grid size, hints, waiting |
| `optimality` | How far is the search from the proven optimum, and how fast does it get there? | size, search depth, instance draw |
| `convergence` | How does the incumbent improve over the budget? | seed |
| `sequence` | What does a preset of the search cost in time, and what does it buy in quality? | preset, catalogue |
| `sensitivity` | How does each property of the instance change what the solver costs to run and what the plan looks like? | debris, spread, chasers, horizon, load — one at a time |
| `transfer` | How does what a transfer is said to cost change the structure of the plan that comes back? | cost oracle, time weight |
| `showcase` | What does a produced plan look like? | seed |

`schedule` and `optimality` isolate one layer each: the first runs no search
and scores the schedule layer against a finer grid, while the second scores the
search against an optimum that is exact for the layers below it. Between them
they separate the error of the time grid from the error of the search.

`sensitivity` is the one that varies the *instance* rather than the solver.
Every other experiment holds the problem still and moves something inside the
planner; this one does the opposite, one property at a time — how much debris,
how far it is spread in right ascension, how many chasers, how long the mission
may run, how many targets one chaser may take. The instances are generated for
that reason: a shipped catalogue comes as a package whose size, spread and
altitude all change together, while a cloud of `n` fragments is the first `n`
of every larger cloud of the same draw, so a size curve adds debris rather than
comparing different clouds. Each parameter is read twice — what it costs to
solve, and what the plan that comes back looks like.

`convergence` and `sequence` divide the search between them: the first follows
one run of it over its budget, and the second compares the seven presets end to
end, each left to stop when it decides it is done. One says how a level gets to
its answer, the other which level to ask. `sequence` answers it as a Pareto
front of mission cost against what the search spent -- once in seconds, once in
genetic iterations, since a deep preset carries a far larger population and so
buys less per iteration and more per second.

`transfer` goes the other way: the solver is left exactly as it is and the
transfer layer under it is replaced -- delta-v alone, delta-v and when a target
is reached, delta-v and when the *hazardous* ones are, departures forbidden
inside recurring windows, and nothing departing at all before a given date.
Its figures are about the **structure** of the plans that come back rather than
their score: when the chasers fly and when they sit still, how long one pause
lasts, what a leg costs, how wide a chaser's slice of the sky is, how many
chasers are used, and where in a tour a hazardous object ends up. Every plan is
read on the one scale the oracles share -- each leg re-priced in delta-v at the
departure the plan chose -- so the differences are the planner's decisions and
not five different units. The layers live in
`_common/transfers.py`; nothing in the package changed to accommodate them.

## Writing a new experiment

Copy `showcase/`, the smallest of them. The configuration describes one run:

```yaml
name: myexperiment
seed: 0

# `instance` mirrors Problem's own arguments, so it fully describes what is
# being solved -- including the horizon.
instance: {name: iridium33, num_chasers: 15, max_load: 8, max_time: 365d}

# What solves it, and what that is built from.
solver:
  type: mcrdv
  args:
    transfer: {type: qlaw, args: {thrust: 3.0e-3}}
    schedule: {type: dp, args: {grid_size: 32}}
    sequence: {type: hgs, args: {preset: 4, max_time: 2m, log_save: true}}
```

and `run.py` says what one run *is*:

```python
from _common import driver, planners

def execute(config: dict) -> dict:
    return planners.solve(config)      # or anything at all

if __name__ == "__main__":
    driver.main(__file__, execute, warmup=planners.warm_up)
```

Whatever `execute` returns becomes `result` in the record. There is no schema:
keep the mission plan, a trace, timings, another solver's logs — whatever a
figure or a later question might want.

## Solvers, and what they are built from

Everything a configuration names is written the same way: a `type`, and the
`args` it is built from. `solver.type` picks the solver, `solver.args` is
whatever that solver takes, and mcrdv takes its three layers — each of them a
`type` and its own `args`, so the shape nests all the way down.

For mcrdv the argument names **are** the constructor parameters — `grid_size`
is `DpSchedule(grid_size=...)`, `crossover` is `HgsConfig(crossover=...)`, and
the sequence layer's `args` are its preset and the search configuration
flattened together — so any knob a layer accepts is usable from a configuration
with no code change. Times accept suffixes: `2m`, `3h`, `365d`. A key written
beside `type` rather than under `args` is an error rather than a silent
default.

Each layer is looked up by name in a small table in `planners`, and the tables
are not limited to what mcrdv ships. Besides `qlaw`, the transfer layer accepts
`qlaw_arrival`, `qlaw_urgency`, `qlaw_blackout` and `qlaw_embargo` -- the same
Q-law physics under a different cost, written in `_common/transfers.py` and
configured like any other layer:

```yaml
transfer: {type: qlaw_blackout, args: {period: 10d, duration: 3d}}
```

Another solver is a function and one line of registration:

```python
@planners.register("mysolver")
def mysolver(config: dict) -> dict:
    kind, args = planners.spec(config["solver"])   # args: {max_time: ..., ...}
    ...
    return {"cost": ..., "t_solve": ...}
```

`planners.solve` then dispatches on `solver.type`, and nothing else in the
harness needs to know what any solver is. `planners.warm_up` skips the
configurations that do not name mcrdv, having nothing to compile for them.

An entry in `sweeps.yaml` then says how to explore it, and what exploring it
costs on the cluster:

```yaml
# Written once, referred to by every sweep. Nothing reads this block.
common:
  solver: &mcrdv
    type: mcrdv
    args:
      transfer: {type: qlaw}
      schedule: {type: dp}
      sequence: {type: hgs, args: {preset: 5}}

sweeps:
  myexperiment:
    walltime: "01:00:00"
    memory: 8G
    cpus: 8
    tasks: 10
    # What the axes vary from, written as a run.yaml is. The experiment's own
    # run.yaml is never read: a campaign stands on this file alone.
    base:
      name: myexperiment
      seed: 0
      instance: {name: iridium33, num_chasers: 15, max_load: 8, max_time: 365d}
      solver: *mcrdv
    # Applied to the base once, before any axis: an anchor can be referred to
    # but not edited, and this is what adjusts it for this sweep.
    set:
      solver.args.sequence.args.max_time: 2m
    campaign:
      - axes:
          # Values that must move together travel in one bundle.
          set:
            - {instance.num_targets: 100, instance.num_chasers: 15}
            - {instance.num_targets: 200, instance.num_chasers: 30}
          solver.args.schedule.args.grid_size: [8, 16, 32]
          seed: {range: 5}
```

A sweep's own `set` is a bundle of dotted overrides applied to its base, once.
It is what makes a shared `base` or `solver` worth anchoring: the anchor gives
every sweep the same stack, and `set` adjusts the handful of knobs that sweep
needs. Overrides reach into the anchored value without touching it — each
configuration is built on a deep copy, so two sweeps sharing one anchor never
see each other's changes.

Every combination of the axes is generated, the first varying slowest; an axis
of a single value is a constant, and `mode: one_at_a_time` varies one axis at a
time from the base instead. Several arms under `campaign` are concatenated,
which is how a sweep compares things that do not share an axis — a search at
three depths, say, beside a construction that has none.

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
`script.py status` writes to `index.csv`. Values a table cannot hold (a
`MissionPlan`, a trace) are left out of the row but stay on the record.

## Figures and tikz

matplotlib is for looking at results while they come in; the paper draws them
with pgfplots. `plot.yaml` says *which* results a figure is about:

```yaml
# Where the records are, and which of them these figures are about.
source:
  folder: results
  select: {instance.name: iridium33}     # equality, or a list for membership

figures:
  - name: cost_small                     # the figure's own label
    draw: cost                           # which drawing function of plot.py
    source: {select: {num_targets: 110}} # added to the file's selection
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

**`source` is where a figure's records come from and which of them it is
about** — one question, written in one place. A bare `source: results` is just
the folder, which is the common case. The file's block is the default and a
figure's own is read against it, so the folder is written once at the top, each
figure adds the clauses that are its own, and a figure wins where both name a
key. A figure may name its own `folder` too, which is how one file can draw
from two result sets. Nothing else may sit in a `source`, and a `select` left
at the top of a figure is an error saying where it belongs.

`plot.py` registers drawing functions by what they draw (`gap`, `pareto`,
`sweep`...), never by the data they are shown. **Which data a figure is about
lives in the YAML alone**: to draw a figure for another catalogue, or for a new
swept axis, add an entry with its own `name` and `source` and the same `draw`,
with no change to any script. An entry without `draw` uses the function
registered under its `name`. An entry naming an unknown function is skipped,
with the list of the ones `plot.py` offers.

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
printed when saving (`cost  12/40 records -> 20260918_092944_egurwslo`) and
recorded in the `.tex` header as `% Figure: cost`.

Every figure prints how many of the loaded records its `select` kept, saved or
not. A results directory holds every run ever made in it -- re-running adds a
record rather than replacing one -- so the two counts differ as soon as a
folder holds more than one instance, and `12/40` is the figure saying which
part of the folder it describes.

On a machine with no display, plotting without `--save` says so and stops rather
than silently drawing into the void.

The `.dat` is **not** the index: a figure shows medians, interquartile bands and
resampled curves computed in `plot.py`. Drawing through the recording wrappers
exports exactly those points, so the table and the figure cannot disagree:

```python
export.line(ax, x, median, name="hgs")       # also steps, via drawstyle
export.band(ax, x, lo, hi, name="hgs_iqr")   # -> fill between
export.points(ax, x, y, name="runs")         # -> only marks
export.grouped_bars(ax, labels, {"hgs": values})
export.span(ax, 0.0, 3.0)                    # a shaded window, as context
export.series(ax, "chaser0", t, kind="segments", ...)   # record without drawing
```

Series sharing an x-axis become columns of one table; series on different grids
get one file each, rather than a ragged table padded with holes.

The `.tex` stub is written to draw what matplotlib drew, not merely to plot
every column it can find: markers stay markers, bars stay grouped and start at
zero, a curve drawn as steps stays steps, a band comes out as `fill between`,
and a column carried only to be read is written and never plotted. It compiles
as it stands -- `\input` it beside its `.dat` and restyle from there. A figure
whose matplotlib version differs from its tikz one is a bug in `export`, not
something to fix in the `.tex`.
