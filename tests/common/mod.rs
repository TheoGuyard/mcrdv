//! Shared fixtures for the scorpion component test suite.

#![allow(dead_code)]

use scorpion::io::read_debris;
use scorpion::orbit::{Catalog, DAY};
use scorpion::problem::{make_problem, ChaserPlan, Problem};
use scorpion::schedule::{CdConfig, CdScheduler, DpConfig, DpScheduler, NowaitConfig, NowaitScheduler};
use scorpion::solver::TransferLayer;
use scorpion::transfer::{QlawConfig, QlawTransfer};

/// The reference debris catalogue used by the whole suite.
pub const DATA: &str = "data/odrc.csv";

/// Thrust of the fixture oracle [m/s^2]. Low enough that transfer times are a
/// sizeable fraction of the horizon, so waiting genuinely matters.
pub const THRUST: f64 = 3e-4;

/// A multi-leg route where waiting pays off.
pub const SEQ: &[usize] = &[0, 2, 8, 5, 11, 3];

/// The reference debris catalogue (depot *not* included).
pub fn debris() -> Catalog {
    read_debris(DATA).expect("the reference debris catalogue must be readable")
}

/// A modest 3-chaser problem with generous limits.
pub fn problem() -> Problem {
    problem_with(3000.0 * DAY, 6)
}

/// A 3-chaser problem with an explicit horizon and load limit.
pub fn problem_with(max_time: f64, max_load: usize) -> Problem {
    make_problem(&debris(), 3, Some(max_time), Some(max_load), None, "sum")
        .expect("the fixture problem must be constructible")
}

/// A low-thrust Q-law oracle bound to `problem`.
pub fn transfer(problem: &Problem) -> QlawTransfer<'_> {
    transfer_with(problem, THRUST)
}

/// A Q-law oracle with an explicit thrust, bound to `problem`.
pub fn transfer_with(problem: &Problem, thrust: f64) -> QlawTransfer<'_> {
    QlawTransfer::new(problem, QlawConfig { thrust, ..QlawConfig::default() })
}

/// A no-wait scheduler bound to `problem` / `transfer`.
pub fn nowait<'p>(
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
) -> NowaitScheduler<'p> {
    NowaitScheduler::new(problem, transfer, NowaitConfig::default())
}

/// A coordinate-descent scheduler with the given knobs.
pub fn cd<'p>(
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
    config: CdConfig,
) -> CdScheduler<'p> {
    CdScheduler::new(problem, transfer, config)
}

/// A dynamic-programming scheduler with the given grid size.
pub fn dp<'p>(
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
    grid_size: usize,
) -> DpScheduler<'p> {
    DpScheduler::new(problem, transfer, DpConfig { grid_size, ..DpConfig::default() })
}

// --------------------------------------------------------------------------
// Assertions
// --------------------------------------------------------------------------

/// `a` and `b` agree to within `rel` relative error (or `abs` in absolute).
pub fn approx(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    if a == b {
        return true;
    }
    if !(a.is_finite() && b.is_finite()) {
        return false;
    }
    (a - b).abs() <= abs + rel * a.abs().max(b.abs())
}

#[track_caller]
pub fn assert_approx(a: f64, b: f64, rel: f64, what: &str) {
    assert!(
        approx(a, b, rel, 0.0),
        "{what}: {a:.12e} != {b:.12e} (relative error {:.3e} > {rel:.1e})",
        (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
    );
}

/// A waiting schedule is *coherent*: the sequence is preserved, no wait is
/// negative, every arrival is the chosen departure plus the oracle's leg time,
/// and the total cost reconstructs from the per-leg costs.
///
/// Port of `_assert_coherent` in `references/pyadr/tests/test_schedule.py`.
#[track_caller]
pub fn assert_coherent(plan: &ChaserPlan, transfer: &dyn TransferLayer, seq: &[usize]) {
    assert_eq!(plan.nodes_indices, seq, "sequence not preserved");
    assert!(
        plan.waiting_times.iter().all(|&w| w >= -1e-9),
        "negative waiting time in {:?}",
        plan.waiting_times
    );

    let depart = plan.departure_times();
    let mut recon = 0.0;
    for i in 1..seq.len() {
        let (dt, dv) = transfer.evaluate(seq[i - 1], seq[i], depart[i - 1]);
        assert!(
            (plan.meeting_times[i] - (depart[i - 1] + dt)).abs() <= 1.0,
            "arrival at leg {i} inconsistent with the oracle: {} vs {}",
            plan.meeting_times[i],
            depart[i - 1] + dt
        );
        recon += dv;
    }
    assert_approx(recon, plan.cost, 1e-6, "cost != sum of leg costs");
}
