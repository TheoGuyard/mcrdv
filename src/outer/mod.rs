pub mod hgs;

use crate::inner::InnerLoop;
use crate::middle::{MiddleLoop, MiddleOutput};
use crate::outer::hgs::{Hgs, HgsParams};
use crate::problem::Problem;

/// Outer loop parameters
#[derive(Debug, Clone)]
pub enum OuterParams {
    Hgs(HgsParams),
}

impl OuterParams {
    pub fn build(&self) -> Box<dyn OuterLoop> {
        match self {
            OuterParams::Hgs(p) => Box::new(Hgs::new(p)),
        }
    }
}

impl Default for OuterParams {
    fn default() -> Self {
        OuterParams::Hgs(HgsParams::default())
    }
}


/// Outer loop output
#[derive(Debug, Clone)]
pub struct OuterOutput {
    /// Route for each chaser
    pub routes: Vec<MiddleOutput>,
    /// Mission cost
    pub cost: f64,
    /// Mission feasibility
    pub feasible: bool,
}

impl OuterOutput {
    /// Count the number of active chasers in the mission
    pub fn active_chasers(&self) -> usize {
        self.routes.iter().filter(|r| !r.is_empty()).count()
    }
}


/// Outer loop interface
pub trait OuterLoop {
    /// Solve the problem using the provided inner and middle loops
    fn solve(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        problem: &Problem,
    ) -> OuterOutput;
}
