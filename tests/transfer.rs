//! Component tests for the transfer layer.

mod common;

use common::*;
use scorpion::orbit::{Catalog, DAY};
use scorpion::problem::{make_problem, Problem};
use scorpion::solver::TransferLayer;
use scorpion::transfer::qlaw::linspace;
use scorpion::transfer::{QlawConfig, QlawTransfer};

const TAU: f64 = std::f64::consts::TAU;

/// A `(i, j)` debris leg that has at least one RAAN-alignment hint.
fn first_leg_with_hint(
    transfer: &QlawTransfer,
    problem: &Problem,
) -> Option<(usize, usize, Vec<f64>)> {
    for i in 1..=problem.num_debris {
        for j in 1..=problem.num_debris {
            if i == j {
                continue;
            }
            let hints = transfer.hints(i, j);
            if !hints.is_empty() {
                return Some((i, j, hints.to_vec()));
            }
        }
    }
    None
}

#[test]
fn self_transfer_is_free() {
    let problem = problem();
    let transfer = transfer(&problem);
    for i in [1usize, 4, 7] {
        for t in [0.0, 1e6, 5e7] {
            let (dt, dv) = transfer.evaluate(i, i, t);
            assert_eq!(dt, 0.0, "self-transfer time not zero at t={t}");
            assert_eq!(dv, 0.0, "self-transfer fuel not zero at t={t}");
        }
    }
}

#[test]
fn time_is_fuel_over_thrust() {
    let problem = problem();
    let transfer = transfer(&problem);
    for t in linspace(0.0, problem.time_horizon, 200) {
        let (dt, dv) = transfer.evaluate(1, 7, t);
        assert!(dv >= 0.0, "negative fuel returned at t={t}");
        assert_approx(dt, dv / THRUST, 1e-12, "time != fuel / thrust");
    }
}

#[test]
fn thrust_and_factor_scaling() {
    // dv is thrust-independent; dt scales as 1/thrust. dv scales with factor.
    let problem = problem();
    let mk = |thrust: f64, factor: f64| {
        QlawTransfer::new(
            &problem,
            QlawConfig { thrust, factor, ..QlawConfig::default() },
        )
    };
    let a = mk(3e-4, 1.5);
    let b = mk(6e-4, 1.5);
    let c = mk(3e-4, 3.0);
    for t in linspace(0.0, problem.time_horizon, 50) {
        let (dta, dva) = a.evaluate(1, 7, t);
        let (dtb, dvb) = b.evaluate(1, 7, t);
        let (_dtc, dvc) = c.evaluate(1, 7, t);
        assert_approx(dva, dvb, 1e-12, "fuel should not depend on thrust");
        assert_approx(dtb, dta * (3e-4 / 6e-4), 1e-12, "time should scale as 1/thrust");
        assert_approx(dvc, dva * (3.0 / 1.5), 1e-12, "fuel should scale with factor");
    }
}

#[test]
fn cost_increases_with_raan_separation() {
    // Orbits identical but for their RAAN; cost must grow with the separation.
    let seps = [0.0, 0.3, 0.8, 1.5, 2.5];
    let n = seps.len();
    let catalog = Catalog::new(
        vec![7.0e6; n],
        vec![0.001; n],
        vec![1.0; n],
        seps.to_vec(),
        vec![0.3; n],
        vec![0.2; n],
    );
    let problem = make_problem(&catalog, 1, Some(3000.0 * DAY), None, None, "sum")
        .expect("problem must be constructible");
    let transfer = QlawTransfer::default(&problem);

    // State 1 has RAAN 0.0; states 2..5 have growing RAAN separation from it.
    let costs: Vec<f64> = (2..=5).map(|j| transfer.evaluate(1, j, 0.0).1).collect();
    assert!(
        costs.windows(2).all(|w| w[0] <= w[1]),
        "cost not monotonic in RAAN separation: {costs:?}"
    );
    assert!(costs[0] > 0.0, "a nonzero plane change should cost something");
}

#[test]
fn departure_hints_are_aligned_and_cheap() {
    let problem = problem();
    let transfer = transfer(&problem);
    let (i, j, hints) =
        first_leg_with_hint(&transfer, &problem).expect("no RAAN-alignment hints on any leg");
    let horizon = problem.time_horizon;
    assert!(
        hints.iter().all(|&h| (0.0..=horizon).contains(&h)),
        "hint outside [0, horizon]: {hints:?}"
    );

    // At a hint, the two nodes' (drifting) RAANs are aligned modulo 2*pi.
    let t = hints[0];
    let si = problem.catalog.state(i).propagate(t);
    let sj = problem.catalog.state(j).propagate(t);
    let d = (si.r - sj.r).abs().rem_euclid(TAU);
    let misalign = d.min(TAU - d);
    assert!(misalign < 1e-3, "hint is not RAAN-aligned (misalignment {misalign})");

    // And the hint is cheaper than the maximally misaligned epoch half a
    // drift period later.
    let rate = (problem.catalog.state(i).j2_secular_rates().0
        - problem.catalog.state(j).j2_secular_rates().0)
        .abs();
    if rate > 0.0 {
        let anti = t + 0.5 * TAU / rate;
        if anti <= horizon {
            let cost_hint = transfer.evaluate(i, j, t).1;
            let cost_anti = transfer.evaluate(i, j, anti).1;
            assert!(
                cost_hint <= cost_anti + 1e-9,
                "aligned hint not cheaper than anti-aligned epoch: {cost_hint} vs {cost_anti}"
            );
        }
    }
}

#[test]
fn evaluate_lb_is_a_lower_bound() {
    let problem = problem();
    let transfer = transfer(&problem);
    let (lb_t, lb_v) = transfer.evaluate_lb(1, 7);
    let mut min_t = f64::INFINITY;
    let mut min_v = f64::INFINITY;
    for t in linspace(0.0, problem.time_horizon, 60) {
        let (dt, dv) = transfer.evaluate(1, 7, t);
        min_t = min_t.min(dt);
        min_v = min_v.min(dv);
    }
    assert!(lb_v <= min_v + 1e-6, "fuel lower bound exceeds a sampled cost");
    assert!(lb_t <= min_t + 1e-6, "time lower bound exceeds a sampled time");
}

#[test]
fn caching_does_not_change_results() {
    let problem = problem();
    let cached = QlawTransfer::new(
        &problem,
        QlawConfig { thrust: THRUST, use_cache: true, ..QlawConfig::default() },
    );
    let direct = QlawTransfer::new(
        &problem,
        QlawConfig { thrust: THRUST, use_cache: false, ..QlawConfig::default() },
    );
    for t in linspace(0.0, problem.time_horizon, 32) {
        let (dta, dva) = cached.evaluate(1, 7, t);
        let (dtb, dvb) = direct.evaluate(1, 7, t);
        assert_eq!((dta, dva), (dtb, dvb), "caching changed evaluate at t={t}");
    }
    assert_eq!(
        cached.evaluate_lb(1, 7),
        direct.evaluate_lb(1, 7),
        "caching changed evaluate_lb"
    );
    assert_eq!(*cached.hints(1, 7), *direct.hints(1, 7), "caching changed hints");
}
