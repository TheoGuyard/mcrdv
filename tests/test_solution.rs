mod common;

use common::{default_oracle, make_test_debris, relaxed_problem};
use scorpion::optim::Solver;

#[test]
fn test_solution_total_load_covers_all_debris() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    // All 5 debris must be visited
    assert_eq!(sol.total_load(), 5, "All debris should be covered");
}

#[test]
fn test_solution_feasibility_with_relaxed_constraints() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    assert!(
        sol.is_feasible(),
        "Solution should be feasible with generous constraints"
    );
    assert_eq!(sol.load_excess, 0);
    assert!(sol.time_excess <= 0.0);
}

#[test]
fn test_solution_display_does_not_panic() {
    let debris = make_test_debris(4);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);
    // Display should not panic
    let display = format!("{}", sol);
    assert!(display.contains("Mission plan"));
    assert!(display.contains("feasible"));
}

#[test]
fn test_solution_total_time_is_max_of_sequences() {
    let debris = make_test_debris(4);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    let mission_time = sol
        .sequences
        .iter()
        .map(|s| s.time)
        .fold(0.0_f64, |a, b| a.max(b));
    assert!((sol.total_time() - mission_time).abs() < 1e-12);
}

#[test]
fn test_solution_pred_succ_consistency() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    // For every non-depot node, succ and pred should be consistent with
    // the sequence ordering.
    for seq in &sol.sequences {
        for i in 1..seq.indices.len() - 1 {
            let node = seq.indices[i];
            if node == 0 {
                continue;
            }
            assert_eq!(sol.pred[node], seq.indices[i - 1]);
            assert_eq!(sol.succ[node], seq.indices[i + 1]);
        }
    }
}

#[test]
fn test_solution_cost_is_sum_of_sequence_costs() {
    let debris = make_test_debris(4);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    let sum: f64 = sol.sequences.iter().map(|s| s.cost).sum();
    assert!(
        (sol.total_cost - sum).abs() < 1e-6,
        "total_cost should equal sum of sequence costs"
    );
}
