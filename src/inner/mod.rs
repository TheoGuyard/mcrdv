pub mod qlaw;

use crate::inner::qlaw::{Qlaw, QlawParams};
use crate::problem::Problem;

/// Inner loop parameters
#[derive(Debug, Clone)]
pub enum InnerParams {
    Qlaw(QlawParams),
}

impl InnerParams {
    pub fn build(&self) -> Box<dyn InnerLoop> {
        match self {
            InnerParams::Qlaw(p) => Box::new(Qlaw::new(p)),
        }
    }
}

impl Default for InnerParams {
    fn default() -> Self {
        InnerParams::Qlaw(QlawParams::default())
    }
}


/// Inner loop output
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InnerOutput {
    /// Transfer time [s]
    pub time: f64,
    /// Transfer fuel [m/s]
    pub fuel: f64,
}


/// Inner loop interface
pub trait InnerLoop {
    /// Evaluate transfer between states at indices `src` and `dst` in
    /// `Problem`, stating at time `dt`. The chaser index is represented as
    /// `idx_ = problem.num_debris()`.
    fn solve(
        &self, 
        src: usize, 
        dst: usize, 
        dt: f64, 
        problem: &Problem
    ) -> InnerOutput;

    /// Time-independent lower bound on inner loop solve call
    fn solve_lb(
        &self, 
        _src: usize, 
        _dst: usize, 
        _problem: &Problem
    ) -> InnerOutput {
        InnerOutput { time: 0.0, fuel: 0.0 }
    }

    /// Give departure hints in `[0, horizon]` where transfer is expected to be
    /// cheapest between states `src` and `dst` in `Problem`. The chaser index
    /// is represented as `idx_ = problem.num_debris()`.
    fn departure_hints(
        &self,
        _src: usize,
        _dst: usize,
        _horizon: f64,
        _problem: &Problem,
    ) -> Vec<f64> {
        Vec::new()
    }
}
