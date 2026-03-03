use scorpion::Problem;
use scorpion::optim::Solver;
use scorpion::orbit::{Oracle, State};

fn make_test_debris(n: usize) -> Vec<State> {
    assert!(n <= 10);
    let base_a = 7_000_000.0;
    let base_i = 1.5;
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

fn default_oracle() -> Oracle {
    Oracle::new("direct".to_string(), 0.003, 6378e3)
}

#[test]
fn test_two_debris_single_chaser() {
    let debris = make_test_debris(2);
    let problem = Problem::new(
        &debris,
        "barycenter".to_string(),
        1, // single chaser
        2,
        1e9,
    );
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);
    assert_eq!(sol.total_load(), 2);
    assert_eq!(sol.sequences.len(), 1);
}

#[test]
fn test_max_load_equals_one() {
    // Each chaser can carry at most 1 debris → need at least as many chasers as debris
    let debris = make_test_debris(3);
    let problem = Problem::new(
        &debris,
        "barycenter".to_string(),
        3,
        1, // max load = 1
        1e9,
    );
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);
    assert_eq!(sol.total_load(), 3);
    // Each non-empty sequence should have at most 1 debris
    for seq in &sol.sequences {
        if sol.is_feasible() {
            assert!(seq.load <= 1, "Each chaser should visit at most 1 debris");
        }
    }
}

#[test]
fn test_oracle_different_thrust_levels() {
    let s1 = State {
        a: 7e6,
        e: 0.002,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let s2 = State {
        a: 7.1e6,
        e: 0.003,
        i: 1.51,
        O: 2.1,
        o: 1.1,
        t: 0.6,
    };

    let oracle_low = Oracle::new("direct".to_string(), 0.001, 6378e3);
    let oracle_high = Oracle::new("direct".to_string(), 0.01, 6378e3);

    let (cost_low, time_low) = oracle_low.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    let (cost_high, time_high) = oracle_high.evaluate(&s1, &s2, 0.0, f64::INFINITY);

    // Higher thrust → shorter transfer time
    assert!(
        time_high < time_low,
        "Higher thrust should yield shorter transfer time"
    );
    // Higher thrust → cost = thrust * time, not necessarily lower cost
    assert!(cost_low > 0.0 && cost_high > 0.0);
}
