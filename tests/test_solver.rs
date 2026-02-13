mod common;

use common::{default_oracle, make_test_debris, relaxed_problem};
use scorpion::io::Loader;
use scorpion::optim::Solver;
use scorpion::orbit::State;
use scorpion::Problem;

#[test]
fn test_solver_level_0_produces_solution() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0, 10.0, 42);
    let sol = solver.solver(&problem, &oracle);
    assert_eq!(sol.total_load(), 5);
    assert!(sol.is_feasible());
}

#[test]
fn test_solver_level_2_produces_solution() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(2, 10.0, 42);
    let sol = solver.solver(&problem, &oracle);
    assert_eq!(sol.total_load(), 5);
    assert!(sol.is_feasible());
}

#[test]
fn test_solver_deterministic_with_same_seed() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();

    let mut solver1 = Solver::preset(0, 5.0, 123);
    let sol1 = solver1.solver(&problem, &oracle);

    let mut solver2 = Solver::preset(0, 5.0, 123);
    let sol2 = solver2.solver(&problem, &oracle);

    assert!(
        (sol1.total_cost - sol2.total_cost).abs() < 1e-6,
        "Same seed should produce identical cost"
    );
}

#[test]
fn test_solver_different_seeds_may_differ() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();

    let mut solver1 = Solver::preset(2, 5.0, 1);
    let sol1 = solver1.solver(&problem, &oracle);

    let mut solver2 = Solver::preset(2, 5.0, 9999);
    let sol2 = solver2.solver(&problem, &oracle);

    // They may or may not differ, but should both be valid
    assert_eq!(sol1.total_load(), 5);
    assert_eq!(sol2.total_load(), 5);
}

#[test]
fn test_solver_max_chasers_respected() {
    let debris = make_test_debris(8);
    let problem = Problem::new(
        &debris,
        "barycenter".to_string(),
        3,   // only 3 chasers
        8,   // generous load
        1e9, // generous time
    );
    let oracle = default_oracle();
    let mut solver = Solver::preset(0, 5.0, 42);
    let sol = solver.solver(&problem, &oracle);

    assert_eq!(sol.sequences.len(), 3, "Should have exactly max_chasers sequences");
    assert_eq!(sol.total_load(), 8, "All debris should be assigned");
}

#[test]
fn test_solver_single_debris() {
    let debris = make_test_debris(1);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0, 5.0, 42);
    let sol = solver.solver(&problem, &oracle);

    assert_eq!(sol.total_load(), 1);
    assert!(sol.is_feasible());
    assert!(sol.nb_sequences >= 1);
}

#[test]
fn test_solver_time_limit_respected() {
    let debris = make_test_debris(5);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();

    let time_limit = 2.0;
    let mut solver = Solver::preset(6, time_limit, 42);
    let start = std::time::Instant::now();
    let _ = solver.solver(&problem, &oracle);
    let elapsed = start.elapsed().as_secs_f64();

    // Allow some overhead but should not run much longer than limit
    assert!(
        elapsed < time_limit + 5.0,
        "Solver should respect time limit (took {:.2}s for limit {:.2}s)",
        elapsed,
        time_limit
    );
}

#[test]
fn test_solver_on_real_data_iridium_small() {
    // Use a subset of the real Iridium constellation data
    let all_debris = Loader::load("data/iridium.csv");
    let debris: Vec<State> = all_debris.into_iter().take(10).collect();
    let problem = Problem::new(
        &debris,
        "barycenter".to_string(),
        3,
        5,
        3.15e7, // ~1 year
    );
    let oracle = default_oracle();
    let mut solver = Solver::preset(2, 5.0, 42);
    let sol = solver.solver(&problem, &oracle);

    assert_eq!(sol.total_load(), 10, "All 10 debris should be collected");
}
