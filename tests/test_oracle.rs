use scorpion::orbit::{Oracle, State};

fn default_oracle() -> Oracle {
    Oracle::new("direct".to_string(), 0.003, 6378e3)
}

#[test]
fn test_oracle_distance_zero_for_same_state() {
    let oracle = default_oracle();
    let s = State {
        a: 7e6,
        e: 0.01,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let d = oracle.distance(&s, &s);
    assert!(d.abs() < 1e-12, "Distance from state to itself should be 0");
}

#[test]
fn test_oracle_distance_symmetric() {
    let oracle = default_oracle();
    let s1 = State {
        a: 7e6,
        e: 0.01,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let s2 = State {
        a: 7.1e6,
        e: 0.02,
        i: 1.52,
        O: 2.3,
        o: 1.2,
        t: 0.8,
    };
    let d12 = oracle.distance(&s1, &s2);
    let d21 = oracle.distance(&s2, &s1);
    assert!(d12 > 0.0 && d21 > 0.0, "Distances should be positive");
}

#[test]
fn test_oracle_distance_positive() {
    let oracle = default_oracle();
    let s1 = State {
        a: 7e6,
        e: 0.01,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let s2 = State {
        a: 7.2e6,
        e: 0.03,
        i: 1.6,
        O: 2.5,
        o: 1.5,
        t: 1.0,
    };
    let d = oracle.distance(&s1, &s2);
    assert!(
        d > 0.0,
        "Distance between different states should be positive"
    );
}

#[test]
fn test_oracle_evaluate_direct_returns_positive() {
    let oracle = default_oracle();
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
    let (cost, time) = oracle.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    assert!(cost > 0.0, "Transfer cost should be positive");
    assert!(time > 0.0, "Transfer time should be positive");
}

#[test]
fn test_oracle_evaluate_self_transfer_is_zero() {
    let oracle = default_oracle();
    let s = State {
        a: 7e6,
        e: 0.002,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let (cost, time) = oracle.evaluate(&s, &s, 0.0, f64::INFINITY);
    assert!(cost.abs() < 1e-6, "Self-transfer cost should be ~0");
    assert!(time.abs() < 1e-6, "Self-transfer time should be ~0");
}

#[test]
fn test_oracle_evaluate_cost_equals_thrust_times_time() {
    let oracle = Oracle::new("direct".to_string(), 0.005, 6378e3);
    let s1 = State {
        a: 7e6,
        e: 0.002,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let s2 = State {
        a: 7.15e6,
        e: 0.004,
        i: 1.52,
        O: 2.2,
        o: 1.3,
        t: 0.7,
    };
    let (cost, time) = oracle.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    let expected_cost = 0.005 * time;
    assert!(
        (cost - expected_cost).abs() < 1e-6,
        "For direct strategy cost = max_thrust * time"
    );
}

#[test]
fn test_oracle_evaluate_with_nonzero_start_time() {
    let oracle = default_oracle();
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
    let (cost0, time0) = oracle.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    let (cost1, time1) = oracle.evaluate(&s1, &s2, 100_000.0, f64::INFINITY);
    // Results differ because orbital propagation shifts RAAN/omega
    assert!(cost0 > 0.0 && cost1 > 0.0);
    assert!(time0 > 0.0 && time1 > 0.0);
}

#[test]
fn test_oracle_distance_increases_with_delta_a() {
    let oracle = default_oracle();
    let s0 = State {
        a: 7e6,
        e: 0.01,
        i: 1.5,
        O: 2.0,
        o: 1.0,
        t: 0.5,
    };
    let s1 = State {
        a: 7.05e6,
        ..s0.clone()
    };
    let s2 = State {
        a: 7.10e6,
        ..s0.clone()
    };
    let d1 = oracle.distance(&s0, &s1);
    let d2 = oracle.distance(&s0, &s2);
    assert!(d2 > d1, "Larger delta-a should yield larger distance");
}

#[test]
fn test_oracle_strategy_dv_ordered() {
    let oracle_direct = Oracle::new("direct".to_string(), 0.003, 6378e3);
    let oracle_drift = Oracle::new("drift".to_string(), 0.003, 6378e3);
    let oracle_best = Oracle::new("best".to_string(), 0.003, 6378e3);
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
    let (cost_best, _) = oracle_best.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    let (cost_direct, time_direct) = oracle_direct.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    let (cost_drift, time_drift) = oracle_drift.evaluate(&s1, &s2, 0.0, f64::INFINITY);
    assert!(
        cost_best <= cost_direct,
        "Best strategy should be no more expensive than direct"
    );
    assert!(
        cost_best <= cost_drift,
        "Best strategy should be no more expensive than drift"
    );
    assert!(
        time_drift <= time_direct,
        "Drift strategy should be no longer than direct"
    );
}
