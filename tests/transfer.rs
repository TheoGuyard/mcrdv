//! The Q-law transfer layer.

mod common;

use mcrdv::solver::TransferLayer;
use mcrdv::orbit::MeeState;
use mcrdv::transfer::QlawTransfer;
use mcrdv::KepState;

fn layer() -> (std::rc::Rc<mcrdv::Problem>, QlawTransfer) {
    let problem = common::problem();
    let mut transfer = QlawTransfer::new();
    transfer.initialize(&problem);
    (problem, transfer)
}

#[test]
fn equinoctial_conversion_is_singularity_free_at_zero_eccentricity() {
    let kep = KepState::new(7.0e6, 0.0, 0.0, 0.0, 0.0, 0.3);
    let mee = MeeState::from_kep(&kep);
    assert_eq!(mee.a, kep.a);
    assert!(mee.f.abs() < 1e-15 && mee.g.abs() < 1e-15);
    assert!(mee.h.abs() < 1e-15 && mee.k.abs() < 1e-15);
}

#[test]
fn max_rates_are_finite_and_positive_on_a_real_orbit() {
    let mee = MeeState::from_kep(&KepState::new(7.15e6, 1e-3, 1.507, 0.65, 6.18, 0.1));
    let rates = mee.max_rates();
    for r in [rates.a, rates.f, rates.g] {
        assert!(r.is_finite() && r > 0.0, "got {r}");
    }
}

#[test]
fn time_and_cost_are_proportional_through_the_thrust() {
    let (problem, transfer) = layer();
    for t in [0.0, problem.max_time / 3.0, problem.max_time] {
        let (dt, dc) = transfer.evaluate(1, 5, t);
        assert!((dt - dc / transfer.thrust).abs() <= 1e-9 * dt.max(1.0));
    }
}

#[test]
fn a_transfer_to_the_same_orbit_is_free() {
    let (_, transfer) = layer();
    let (dt, dc) = transfer.evaluate(4, 4, 1234.0);
    assert_eq!(dc, 0.0);
    assert_eq!(dt, 0.0);
}

#[test]
fn the_cost_depends_on_the_departure_time() {
    // If it did not, the whole scheduling layer would be pointless.
    let (problem, transfer) = layer();
    let early = transfer.evaluate(1, 7, 0.0).1;
    let late = transfer.evaluate(1, 7, problem.max_time).1;
    assert!((early - late).abs() > 1e-6, "{early} vs {late}");
}

#[test]
fn the_bound_never_exceeds_a_sampled_cost() {
    let (problem, transfer) = layer();
    let horizon = problem.max_time;
    for (i, j) in [(0usize, 3usize), (2, 9), (5, 1), (11, 4)] {
        let (_, lb) = transfer.evaluate_lb(i, j, horizon);
        for k in 0..transfer.num_samples {
            let t = horizon * (k as f64) / ((transfer.num_samples - 1) as f64);
            assert!(
                lb <= transfer.evaluate(i, j, t).1 + 1e-9,
                "bound {lb} exceeded the cost at sample {k}"
            );
        }
    }
}

#[test]
fn the_bound_is_stable_across_calls() {
    // The second call is served from the dense table; it must agree exactly.
    let (problem, transfer) = layer();
    let first = transfer.evaluate_lb(2, 8, problem.max_time);
    let second = transfer.evaluate_lb(2, 8, problem.max_time);
    assert_eq!(first, second);
}

#[test]
fn a_shorter_window_can_only_raise_the_bound() {
    let (problem, transfer) = layer();
    let wide = transfer.evaluate_lb(3, 6, problem.max_time).1;
    let narrow = transfer.evaluate_lb(3, 6, problem.max_time / 4.0).1;
    assert!(narrow >= wide - 1e-9, "{narrow} < {wide}");
}

#[test]
fn hints_are_sorted_inside_the_window_and_capped() {
    let (problem, transfer) = layer();
    let horizon = problem.max_time;
    for (i, j) in [(0usize, 1usize), (4, 9), (7, 2)] {
        let hints = transfer.hints(i, j, horizon);
        assert!(!hints.is_empty(), "a leg should advertise some window");
        assert!(hints.len() <= transfer.num_hints, "{} hints", hints.len());
        assert!(hints.windows(2).all(|w| w[0] < w[1]), "hints must increase");
        assert!(hints.iter().all(|&t| (0.0..=horizon).contains(&t)));
    }
}

#[test]
fn a_lone_sample_with_no_hint_budget_yields_no_hint() {
    // A single sample has no neighbour to be compared against, so the local
    // minimum test has nothing to read; asking for no hint must still answer.
    let problem = common::problem();
    let mut transfer = QlawTransfer::with_options(3e-3, 1.5, 1, 0, 0.0, 1.0);
    transfer.initialize(&problem);
    assert!(transfer.hints(1, 2, problem.max_time).is_empty());
}

#[test]
fn hints_land_on_locally_cheap_departure_times() {
    // A hint should not be more expensive than both of its sampled neighbours.
    let (problem, transfer) = layer();
    let horizon = problem.max_time;
    let step = horizon / ((transfer.num_samples - 1) as f64);
    for &t in transfer.hints(1, 6, horizon).iter() {
        if t <= 0.0 || t >= horizon {
            continue;
        }
        let here = transfer.evaluate(1, 6, t).1;
        let before = transfer.evaluate(1, 6, t - step).1;
        let after = transfer.evaluate(1, 6, t + step).1;
        assert!(here <= before && here <= after, "{before} {here} {after}");
    }
}

#[test]
fn fewer_samples_than_hints_returns_every_sample() {
    let problem = common::problem();
    let mut transfer = QlawTransfer::with_options(3e-3, 1.5, 4, 10, 0.0, 1.0);
    transfer.initialize(&problem);
    assert_eq!(transfer.hints(1, 2, problem.max_time).len(), 4);
}
