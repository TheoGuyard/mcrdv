mod common;

use common::{default_oracle, make_test_debris, relaxed_problem};
use scorpion::optim::Solver;
use scorpion::Problem;

#[test]
fn test_feasible_is_always_better_than_infeasible() {
    let debris = make_test_debris(5);

    // Relaxed problem for feasible solution
    let problem_ok = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver_ok = Solver::preset(0, 5.0, 42);
    let sol_ok = solver_ok.solver(&problem_ok, &oracle);
    assert!(sol_ok.is_feasible());

    // Tight problem for infeasible solution
    let problem_tight = Problem::new(
        &debris,
        "barycenter".to_string(),
        5,
        5,    // load is fine
        1.0,  // 1 second → almost certainly infeasible
    );
    let mut solver_tight = Solver::preset(0, 5.0, 99);
    let sol_tight = solver_tight.solver(&problem_tight, &oracle);

    // A feasible should beat an infeasible regardless of cost
    assert!(
        solver_ok.metric.better_than(&sol_ok, &sol_tight) || sol_tight.is_feasible(),
        "Feasible solution should be preferred over infeasible"
    );
}
