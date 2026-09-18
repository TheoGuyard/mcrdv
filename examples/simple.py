"""Run with `python examples/simple.py`."""

from mcrdv import DAY, Problem, Planner, centroid, load_states

# Load a target catalogue.
path = "data/iridium33.csv"
states = load_states(path)
print(f"Loaded {len(states)} states from {path}")

# Add depot at the centroid of the states before them.
states = [centroid(states)] + states

# Define the instance and its operational limits.
problem = Problem(
    states,
    15,  # chasers in the swarm
    365.0 * DAY,  # per-chaser mission duration limit
    10,  # per-chaser target capacity
)
print(problem)

# Plan the mission.
planner = Planner()
mission = planner.solve(problem)

print(mission)
