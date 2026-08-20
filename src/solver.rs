//! Solver for the active debris removal mission planning problem.

use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use crate::problem::{ChaserPlan, MissionPlan};

// ========================================================================= //
// Layers
// ========================================================================= //


/// Abstract layer to evaluate the time and cost of a transfer.
pub trait TransferLayer {
    /// Prepare the layer before start of a solve.
    fn initialize(&self) -> ();

    /// Return `(time, cost)` of transfer `i -> j` departing at time `t`.
    fn evaluate(&self, i: usize, j: usize, t: f64) -> (f64, f64);

    /// Lower bound on `(time, cost)` of transfer `i -> j` over time horizon.
    fn evaluate_lb(&self, i: usize, j: usize) -> (f64, f64);

    /// Departure times of transfer `i -> j` where the cost is locally minimal.
    fn hints(&self, i: usize, j: usize) -> Rc<Vec<f64>>;

    /// Get the layer statistics.
    fn statistics(&self) -> HashMap<String, Box<dyn Any>>;
}

/// Abstract layer to evaluate the chaser plan of a fixed node sequence.
pub trait ScheduleLayer {
    /// Prepare the layer before start of a solve.
    fn initialize(&self) {}

    /// Evaluate the chaser plan for a fixed sequence of node indices.
    fn evaluate(&self, seq: &[usize]) -> Rc<ChaserPlan>;

    /// Evaluate lower bound on a plan cost for a fixed sequence of node indices.
    fn evaluate_lb(&self, seq: &[usize]) -> f64;

    /// Get the layer statistics.
    fn statistics(&self) -> HashMap<String, Box<dyn Any>>;
}

/// Abstract layer to evaluate a full mission plan of a problem.
pub trait SequenceLayer {
    /// Prepare the layer before start of a solve.
    fn initialize(&self) {}

    /// Evaluate a mission plan, reporting progress through `logger`.
    fn evaluate(&self) -> MissionPlan;

    /// Get the layer statistics.
    fn statistics(&self) -> HashMap<String, Box<dyn Any>>;
}

// ========================================================================= //
// Solver
// ========================================================================= //

pub struct Solver<'a> {
    pub transfer: &'a dyn TransferLayer,
    pub schedule: &'a dyn ScheduleLayer,
    pub sequence: &'a dyn SequenceLayer,
    pub verbose: bool,
}

impl<'a> Solver<'a> {
    pub fn new(
        transfer: &'a dyn TransferLayer,
        schedule: &'a dyn ScheduleLayer,
        sequence: &'a dyn SequenceLayer,
        verbose: bool,
    ) -> Self {
        Self {
            transfer,
            schedule,
            sequence,
            verbose,
        }
    }

    /// Solve the problem and return the best mission plan found.
    pub fn solve(&self) -> MissionPlan {
        let start = Instant::now();

        if self.verbose { println!("Initializing layers..."); }
        self.transfer.initialize();
        self.schedule.initialize();
        self.sequence.initialize();
        if self.verbose { println!("Layers initialized in {:.1} sec", start.elapsed().as_secs_f64()); }
        
        if self.verbose { println!("Stating solver..."); }
        let solution = self.sequence.evaluate();
        if self.verbose { println!("Solver finished in {:.1} sec", start.elapsed().as_secs_f64()); }
        
        solution
    }
}
