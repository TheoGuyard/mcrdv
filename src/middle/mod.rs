pub mod dynprog;

use crate::inner::InnerLoop;
use crate::middle::dynprog::{DynProg, DynProgParams};
use crate::problem::Problem;

/// Middle loop parameters
#[derive(Debug, Clone)]
pub enum MiddleParams {
    DynProg(DynProgParams),
}

impl MiddleParams {
    pub fn build(&self) -> Box<dyn MiddleLoop> {
        match self {
            MiddleParams::DynProg(p) => Box::new(DynProg::new(p)),
        }
    }
}

impl Default for MiddleParams {
    fn default() -> Self {
        MiddleParams::DynProg(DynProgParams::default())
    }
}


/// Middle loop output
#[derive(Debug, Clone)]
pub struct MiddleOutput {
    /// Chaser sequence `[chaser, debris_1, debris_2, ..., debris_n]`
    pub sequence: Vec<usize>,
    /// Per-leg wait time [s]
    pub wait: Vec<f64>,
    /// Per-leg transfer time [s]
    pub time: Vec<f64>,
    /// Per-leg meeting time [s] (`meet[i] = meet[i-1] + wait[i] + time[i]`)
    pub meet: Vec<f64>,
    /// Per-leg fuel consumption [m/s]
    pub fuel: Vec<f64>,
    /// Total wait time [s]
    pub total_wait: f64,
    /// Total transfer time [s]
    pub total_time: f64,
    /// Total fuel consumption [m/s]
    pub total_fuel: f64,
    /// Total load [debris]
    pub total_load: usize,
    /// Chaser plan cost
    pub cost: f64,
    /// Chaser plan feasibility
    pub feasible: bool,
}

impl MiddleOutput {
    pub fn empty(chaser: usize) -> Self {
        Self {
            sequence: vec![chaser; 1],
            wait: vec![0.0; 1],
            time: vec![0.0; 1],
            meet: vec![0.0; 1],
            fuel: vec![0.0; 1],
            total_wait: 0.0,
            total_time: 0.0,
            total_fuel: 0.0,
            total_load: 0,
            cost: 0.0,
            feasible: true,
        }
    }

    /// Whether the chaser visits any debris
    pub fn is_empty(&self) -> bool {
        self.sequence.len() <= 1
    }

    /// Overall mission time (meeting time of the final debris)
    pub fn mission_time(&self) -> f64 {
        *self.meet.last().unwrap_or(&0.0)
    }
}


/// Cheap lower bound on a chaser plan cost
#[derive(Debug, Clone, Copy)]
pub struct MiddleBound {
    pub total_time: f64,
    pub total_fuel: f64,
    pub total_load: usize,
    pub cost: f64,
}


/// Middle loop interface
pub trait MiddleLoop {
    
    /// Evaluate a sequence of debris visits for a chaser
    fn solve(
        &self,
        sequence: &[usize],
        inner: &dyn InnerLoop,
        problem: &Problem,
    ) -> MiddleOutput;

    /// Lower bound on the plan cost of a fixed sequence
    fn solve_lb(
        &self,
        sequence: &[usize],
        inner: &dyn InnerLoop,
        problem: &Problem,
    ) -> MiddleBound {
        compute_bound(sequence, inner, problem)
    }
}


/// Problem state index at position `pos` in a chaser sequence
pub(crate) fn index_at(sequence: &[usize], pos: usize, problem: &Problem) -> usize {
    if pos == 0 {
        problem.num_debris()
    } else {
        sequence[pos]
    }
}

/// Sum the per-leg inner-loop lower bounds into a plan-level lower bound
pub(crate) fn compute_bound(
    sequence: &[usize],
    inner: &dyn InnerLoop,
    problem: &Problem,
) -> MiddleBound {
    let mut total_time = 0.0;
    let mut total_fuel = 0.0;
    for i in 1..sequence.len() {
        let lb = inner.solve_lb(
            index_at(sequence, i - 1, problem),
            index_at(sequence, i, problem),
            problem,
        );
        total_time += lb.time;
        total_fuel += lb.fuel;
    }
    let total_load = sequence.len().saturating_sub(1);
    let cost = problem.cost(total_time, total_fuel);
    MiddleBound { total_time, total_fuel, total_load, cost }
}

/// Forward inner loop with no waiting time allowed
pub(crate) fn solve_no_wait(
    sequence: &[usize],
    inner: &dyn InnerLoop,
    problem: &Problem,
) -> MiddleOutput {
    let n = sequence.len();
    let mut wait = vec![0.0; n];
    let mut time = vec![0.0; n];
    let mut meet = vec![0.0; n];
    let mut fuel = vec![0.0; n];

    let mut curr = 0.0;
    for i in 1..n {
        let out = inner.solve(
            index_at(sequence, i - 1, problem),
            index_at(sequence, i, problem),
            curr,
            problem,
        );
        wait[i] = 0.0;
        time[i] = out.time;
        fuel[i] = out.fuel;
        curr += out.time;
        meet[i] = curr;
    }

    let total_wait = wait.iter().sum();
    let total_fuel = fuel.iter().sum();
    let total_time: f64 = time.iter().sum();
    let total_load = sequence.len() - 1;

    let tend = *meet.last().unwrap_or(&0.0);
    let cost = problem.cost(tend, total_fuel);
    let feasible = tend <= problem.max_time
        && total_fuel <= problem.max_fuel
        && total_load <= problem.max_load;

    MiddleOutput {
        sequence: sequence.to_vec(),
        wait,
        time,
        meet,
        fuel,
        total_wait,
        total_time,
        total_fuel,
        total_load,
        cost,
        feasible,
    }
}

/// Assemble a schedule from departure times
pub(crate) fn assemble(
    sequence: &[usize],
    departs: &[f64],
    inner: &dyn InnerLoop,
    problem: &Problem,
) -> MiddleOutput {
    let n = sequence.len();
    let mut wait = vec![0.0; n];
    let mut time = vec![0.0; n];
    let mut meet = vec![0.0; n];
    let mut fuel = vec![0.0; n];

    for i in 1..n {
        let out = inner.solve(
            index_at(sequence, i - 1, problem),
            index_at(sequence, i, problem),
            departs[i - 1],
            problem,
        );
        wait[i] = departs[i - 1] - meet[i - 1];
        time[i] = out.time;
        meet[i] = departs[i - 1] + out.time;
        fuel[i] = out.fuel;
    }

    let total_wait = wait.iter().sum();
    let total_time: f64 = time.iter().sum();
    let total_fuel = fuel.iter().sum();
    let total_load = sequence.len() - 1;

    let tend = *meet.last().unwrap_or(&0.0);
    let cost = problem.cost(tend, total_fuel);
    let feasible = tend <= problem.max_time
        && total_fuel <= problem.max_fuel
        && total_load <= problem.max_load;

    MiddleOutput {
        sequence: sequence.to_vec(),
        wait,
        time,
        meet,
        fuel,
        total_wait,
        total_time,
        total_fuel,
        total_load,
        cost,
        feasible,
    }
}
