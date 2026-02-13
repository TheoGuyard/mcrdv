use scorpion::orbit::State;
use scorpion::Problem;

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

#[test]
fn test_problem_new_barycenter() {
    let debris = make_test_debris(4);
    let problem = Problem::new(
        &debris,
        "barycenter".to_string(),
        4,
        4,
        1e8,
    );
    // states[0] is the barycenter, states[1..] are the debris
    assert_eq!(problem.states.len(), 5);

    // Barycenter should be the average of all debris elements
    let avg_a: f64 = debris.iter().map(|s| s.a).sum::<f64>() / 4.0;
    let avg_e: f64 = debris.iter().map(|s| s.e).sum::<f64>() / 4.0;
    assert!((problem.states[0].a - avg_a).abs() < 1e-6);
    assert!((problem.states[0].e - avg_e).abs() < 1e-10);
}

#[test]
fn test_problem_stores_parameters() {
    let debris = make_test_debris(3);
    let problem = Problem::new(
        &debris,
        "barycenter".to_string(),
        5,
        10,
        3.15e7,
    );
    assert_eq!(problem.max_chasers, 5);
    assert_eq!(problem.max_load, 10);
    assert_eq!(problem.mission_time, 3.15e7);
    assert_eq!(problem.chaser_start, "barycenter");
}

#[test]
#[should_panic(expected = "Number of debris exceeds total chaser capacity")]
fn test_problem_panics_on_insufficient_capacity() {
    let debris = make_test_debris(5);
    // 2 chasers * 2 load = 4 < 5 debris  → should panic
    let _ = Problem::new(
        &debris,
        "barycenter".to_string(),
        2,
        2,
        1e8,
    );
}

#[test]
#[should_panic(expected = "Invalid chaser start option")]
fn test_problem_panics_on_invalid_chaser_start() {
    let debris = make_test_debris(3);
    let _ = Problem::new(
        &debris,
        "invalid_method".to_string(),
        5,
        5,
        1e8,
    );
}
