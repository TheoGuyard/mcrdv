mod common;

use common::{default_oracle, make_test_debris, relaxed_problem};
use scorpion::optim::Solver;

#[test]
fn test_empty_sequence() {
    let debris = make_test_debris(3);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();

    // Use solver level 0 (single local search) to produce a solution
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    // With 5 chasers and only 3 debris, some sequences should be empty
    let empty_count = sol.sequences.iter().filter(|s| s.load == 0).count();
    assert!(
        empty_count > 0,
        "With 5 chasers and 3 debris some sequences should be empty"
    );

    // Empty sequences should have load = 0 and cost = 0
    for seq in &sol.sequences {
        if seq.load == 0 {
            assert_eq!(seq.cost, 0.0);
            assert_eq!(seq.time, 0.0);
        }
    }
}
