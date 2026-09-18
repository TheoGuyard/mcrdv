# mcrdv — Multi-chaser rendezvous planning software

`mcrdv` is a Python package to plan multi-chaser rendezvous missions in orbit. It allows planning a mission for a swarm of chasers that must visit a list of targets. Various operational constraints and objectives can be integrated during the planning process. `mcrdv` is designed to be **fast**, **ergonomic**, and **easy to extend** to facilitates its integration in real-world mission design workflows.

## Installation

`mcrdv` can be installed as follows:

```bash
git clone https://github.com/TheoGuyard/mcrdv.git
cd mcrdv
pip install -e .
```

Python `3.11+` is required. Use `pip install -e ".[dev]"` to install development dependencies for, and `pip install -e ".[experiments]"` to install experiments dependencies.

## Quickstart

The basic workflow of `mcrdv` is as follows:

```python
from mcrdv import DAY, Planner, Problem, centroid, load_states

# 1. Load a catalogue of targets (Keplerian elements, one row per object).
targets = load_states("data/iridium33.csv")

# 2. Add the depot, where the chasers start. By convention it is state 0.
states = [centroid(targets)] + targets

# 3. Describe the mission and its operational limits.
problem = Problem(
    states,
    num_chasers=15,          # chasers available at the depot
    max_time=365.0 * DAY,    # mission duration limit per chaser [s]
    max_load=10,             # targets a chaser can collect
)

# 4. Stack the three layers in the mission `Planner`.
planner = Planner()

# 5. Solve, inspect, save.
mission = planner.solve(problem)
print(mission)
mission.save("mission.csv")
```

```text
Initializing layers...
Layers initialized in 8.80 seconds          <- Numba compilation, once per process
Planning mission...
...
====================== Mission plan ======================
Feasible     : True
Total cost   : 114492.8
Chasers used : 13/15

Chaser 0 (feasible: True, cost: 64.4)
  src    dst             arrive       depart      leg cost
  0      19                  --    3003428.6          0.00
  19     94           3013785.5   22525714.3         31.07
  94     --          22536822.3           --         33.32
...
```

Each `ChaserPlan` in `mission.plans` holds the visited `sequence` (starting at the depot), the `depart` and `arrive` times in seconds, the leg `costs`, and its totals `cost` and `time`. By default, a leg's cost is its estimated `Δv` in `m/s`.

## Package design

### Modules

```
src/mcrdv/
├── problem.py        Problem instance, ChaserPlan, and MissionPlan
├── orbit.py          Orbital state representation and utilities
├── io.py             I/O to load and save states
├── planner.py        Planner and the transfer/schedule/sequence layer classes
├── kernel.py         Compilation utilities
├── transfer/         Concrete implementations of transfer layers
├── schedule/         Concrete implementations of schedule layers
└── sequence/         Concrete implementations of sequence layers
```

### The layer decomposition

`mcrdv` splits the mission planning task into three **layers** as follows:
```
Planner.solve(problem)
└── SequenceLayer           Which targets does each chaser visit, and in what order?
    └── ScheduleLayer       When should the chaser leave each target in a fixed sequence to visit?
        └── TransferLayer   What is the time and cost of a transfer between two targets at a given time?
```

| Layer | Default implementations |
|---|---|---|
| `TransferLayer` | `QlawTransfer` |
| `ScheduleLayer` | `DpSchedule` |
| `SequenceLayer` | `HgsSequence` |

A few things make the stack fast and easy to recombine:

- **Plug and play.** Each layer can be replaced with custom implementation and easily integrated to the `Planner`.
- **Bounds and hints.** Layers can expose information to help the other layers in their optimization task.
- **Caching.** The layers can cache results to avoid useless re-evaluations.
- **Compiled kernels.** Layers can be compiled with `numba` to deliver high performance.

## License

`mcrdv` is released under [MIT License](LICENSE).
