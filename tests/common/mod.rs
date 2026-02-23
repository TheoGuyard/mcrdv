use scorpion::Problem;
use scorpion::orbit::{Oracle, State};

/// Build a small set of synthetic debris states for testing.
/// Returns up to 10 states with varying orbital elements to exercise the oracle
/// and routing logic without depending on external data files.
pub fn make_test_debris(n: usize) -> Vec<State> {
    assert!(n <= 10);
    let base_a = 7_000_000.0; // ~7 000 km semi-major axis
    let base_i = 1.5; // ~86° inclination (near-polar)
    (0..n)
        .map(|k| {
            let f = k as f64;
            State {
                a: base_a + f * 50_000.0,
                e: 0.001 + f * 0.001,
                i: base_i + f * 0.005,
                O: 2.0 + f * 0.3,
                o: 1.0 + f * 0.4,
                t: 0.5 + f * 0.6,
            }
        })
        .collect()
}

/// Create a default oracle using direct transfer strategy.
pub fn default_oracle() -> Oracle {
    Oracle::new("direct".to_string(), 0.003)
}

/// Create a relaxed problem with generous constraints so that feasible
/// solutions are easy to obtain.
pub fn relaxed_problem(debris: &Vec<State>) -> Problem {
    Problem::new(
        debris,
        "barycenter".to_string(),
        5,             // max chasers
        debris.len(),  // max load = all debris can fit in one chaser
        100_000_000.0, // max time (very generous)
    )
}
