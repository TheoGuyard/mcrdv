//! Orbital states, drift and geometric helpers.

use mcrdv::orbit::{centroid, circular_mean, KepState, DAY, TAU};

fn sample() -> KepState {
    KepState::new(7.15e6, 1e-3, 1.507, 0.658, 6.18, 0.1)
}

#[test]
fn drift_rates_are_derived_on_construction() {
    let s = sample();
    // Near-polar inclination makes the RAAN drift the slower of the two here.
    assert!(s.dr_ < 0.0, "a prograde orbit regresses its node");
    assert!(s.dr_.is_finite() && s.do_.is_finite());
}

#[test]
fn propagation_moves_only_the_two_drifting_angles() {
    let s = sample();
    let later = s.propagate(365.0 * DAY);

    assert_eq!(later.a, s.a);
    assert_eq!(later.e, s.e);
    assert_eq!(later.i, s.i);
    assert_eq!(later.t, s.t);
    assert_ne!(later.r, s.r);
}

#[test]
fn propagation_carries_the_rates_rather_than_deriving_them_again() {
    // The rates depend on a, e and i alone, none of which propagation changes,
    // so recomputing them would be redundant work -- and must give the same
    // answer as carrying them.
    let s = sample();
    let later = s.propagate(1234.5);
    let rebuilt = KepState::new(later.a, later.e, later.i, later.r, later.o, later.t);

    assert_eq!(later.dr_, s.dr_);
    assert_eq!(later.do_, s.do_);
    assert_eq!(later.dr_, rebuilt.dr_);
    assert_eq!(later.do_, rebuilt.do_);
}

#[test]
fn propagation_is_linear_in_time() {
    let s = sample();
    let a = s.propagate(1000.0);
    let b = s.propagate(2000.0);
    assert!((b.r - a.r - (a.r - s.r)).abs() < 1e-12);
}

#[test]
fn propagating_by_zero_changes_nothing() {
    let s = sample();
    assert_eq!(s.propagate(0.0), s);
}

#[test]
fn circular_mean_wraps_around_the_branch_cut() {
    // The arithmetic mean of these two is pi; the circular mean is near zero.
    let mean = circular_mean(&[0.1, TAU - 0.1]);
    assert!(!(1e-9..=TAU - 1e-9).contains(&mean), "got {mean}");
}

#[test]
fn circular_mean_is_in_range_and_agrees_when_angles_are_close() {
    let mean = circular_mean(&[1.0, 1.2, 1.4]);
    assert!((0.0..TAU).contains(&mean));
    assert!((mean - 1.2).abs() < 1e-9, "got {mean}");
}

#[test]
fn centroid_averages_sizes_and_wraps_angles() {
    let a = KepState::new(7.0e6, 1e-3, 1.5, 0.1, 0.0, 0.0);
    let b = KepState::new(7.2e6, 3e-3, 1.5, TAU - 0.1, 0.0, 0.0);
    let c = centroid(&[a, b]);

    assert!((c.a - 7.1e6).abs() < 1.0);
    assert!((c.e - 2e-3).abs() < 1e-12);
    assert!(!(1e-9..=TAU - 1e-9).contains(&c.r), "RAAN should wrap, got {}", c.r);
}

#[test]
fn centroid_of_one_state_is_that_state() {
    let s = sample();
    let c = centroid(&[s]);
    assert!((c.a - s.a).abs() < 1e-6);
    assert!((c.i - s.i).abs() < 1e-12);
}
