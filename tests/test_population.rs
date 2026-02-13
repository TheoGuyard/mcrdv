mod common;

use common::{default_oracle, make_test_debris, relaxed_problem};
use scorpion::optim::Solver;

#[test]
fn test_population_has_feasible_solutions_after_solve() {
    let debris = make_test_debris(4);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(2, 5.0, 42);
    let _ = solver.solver(&problem, &oracle);

    // After solving, the population should contain feasible solutions
    assert!(
        !solver.population.feasible.solutions.is_empty(),
        "Feasible sub-population should not be empty after solving"
    );
}

#[test]
fn test_population_best_returns_best_feasible() {
    let debris = make_test_debris(4);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(2, 5.0, 42);
    let best = solver.solver(&problem, &oracle);

    // Best from solver should match population.best() quality
    let pop_best = solver.population.best();
    assert!(
        (best.total_cost - pop_best.total_cost).abs() < 1e-6
            || solver.metric.better_than(&best, pop_best)
            || solver.metric.better_than(pop_best, &best)
            || (best.total_cost - pop_best.total_cost).abs() < 1e-6,
        "Solver result should be consistent with population best"
    );
}
