"""Code shared by the mcrdv experiments.

    runner     configurations, sweeps.yaml campaigns, the `run.py` loop and
                its records (knows nothing of mcrdv)
    planners    `instance:` -> Problem, `solver:` -> mcrdv stack, solving, and
                the metrics of a plan
    plotter    `plot.yaml` -> figures, and their export to pgfplots (knows
                nothing of mcrdv)

Code used by a single experiment lives in its own `run.py` or `plot.py`: its
layers register themselves in `planners.TRANSFERS`, `SCHEDULES`, `SEQUENCES`
(and generated instances in `planners.GENERATORS`), so configurations reach
them by name.
"""
