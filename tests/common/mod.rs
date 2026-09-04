//! Shared fixtures for the integration tests.
//!
//! Each test binary links the whole module but uses only part of it.
#![allow(dead_code)]

use std::rc::Rc;

use mcrdv::{centroid, read_debris, DpSchedule, Problem, QlawTransfer, DAY};
use mcrdv::solver::ScheduleLayer;

/// The bundled 13-debris catalogue, prefixed with its centroid.
pub fn states() -> Vec<mcrdv::KepState> {
    let debris = read_debris("data/odrc.csv");
    let mut states = vec![centroid(&debris)];
    states.extend(debris);
    states
}

/// A small instance that every layer can solve quickly.
pub fn problem() -> Rc<Problem> {
    Rc::new(Problem::new(states(), 3, 2500.0 * DAY, 5))
}

/// An initialized scheduling layer over [`problem`].
pub fn schedule(problem: &Rc<Problem>) -> DpSchedule<QlawTransfer> {
    let mut layer = DpSchedule::new(QlawTransfer::new());
    layer.initialize(problem);
    layer
}

/// Every debris index collected by a mission, sorted.
pub fn collected(mission: &mcrdv::MissionPlan) -> Vec<usize> {
    let mut out: Vec<usize> = mission
        .plans
        .iter()
        .flat_map(|p| p.sequence[1..].iter().copied())
        .collect();
    out.sort_unstable();
    out
}
