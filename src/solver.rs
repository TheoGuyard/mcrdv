//! MCRDV mission planner with nested layer decomposition.

use std::rc::Rc;
use std::time::Instant;

use crate::problem::{ChaserPlan, MissionPlan, Problem};

// ========================================================================= //
// Layers
// ========================================================================= //

/// Estimates the time and cost of a single transfer between states.
pub trait TransferLayer {
    /// Binds the layer to a problem instance, and clean outdated data.
    fn initialize(&mut self, problem: &Rc<Problem>);

    /// Time and cost of the transfer from state `i` to `j` at time `t`. Return
    /// an infinite cost for an infeasible transfer.
    fn evaluate(&self, i: usize, j: usize, t: f64) -> (f64, f64);

    /// Lower bound on [`Self::evaluate`] over the time range `[0, t]`.
    fn evaluate_lb(&self, i: usize, j: usize, t: f64) -> (f64, f64);

    /// Departure hints in the time range `[0, t]` for cheap transfer.
    fn hints(&self, i: usize, j: usize, t: f64) -> Vec<f64>;
}

/// Schedules a complete chaser plan along a sequence of debris to visit.
pub trait ScheduleLayer {
    /// Binds the layer to a problem instance, and clean outdated data.
    fn initialize(&mut self, problem: &Rc<Problem>);

    /// Schedules a complete chaser plan along a sequence of debris to visit.
    fn evaluate(&self, sequence: &[usize]) -> Rc<ChaserPlan>;

    /// A lower bound on the cost of any plan along some sequence.
    fn evaluate_lb(&self, sequence: &[usize]) -> f64;
}

/// Sequences a mission plan into individual chaser sequences to visit.
pub trait SequenceLayer {
    /// Binds the layer to a problem instance, and clean outdated data.
    fn initialize(&mut self, problem: &Rc<Problem>);

    /// Sequences a mission plan into individual chaser sequences to visit.
    fn evaluate(&mut self) -> MissionPlan;
}

// ========================================================================= //
// Solver
// ========================================================================= //

/// Solver consisted in nested layers.
#[derive(Debug, Clone)]
pub struct Solver<Q: SequenceLayer> {
    /// Top sequence layer referencing the ones below in a nested structure.
    pub sequence: Q,
    /// Verbosity toggle.
    pub verbose: bool,
}

impl<Q: SequenceLayer> Solver<Q> {
    /// Build a solver with nested layers.
    pub fn new(sequence: Q) -> Self {
        Self {
            sequence,
            verbose: true,
        }
    }

    /// Set the solver verbosity.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Plan a mission for a problem instance.
    pub fn solve(&mut self, problem: &Rc<Problem>) -> MissionPlan {
        let start = Instant::now();

        if self.verbose { 
            println!("Initializing layers..."); 
        }

        self.sequence.initialize(problem);
        
        if self.verbose {
            let elapsed = start.elapsed().as_secs_f64();
            println!("Layers initialized in {:.2} seconds", elapsed);
            println!("Starting solver...");
            println!();
        }

        let mission = self.sequence.evaluate();

        if self.verbose {
            let elapsed = start.elapsed().as_secs_f64();
            println!();
            println!("Solver terminated in {:.2} seconds", elapsed);
        }
        
        mission
    }
}
