use std::cell::RefCell;
use std::collections::HashMap;

use crate::inner::InnerLoop;
use crate::middle::{
    assemble, compute_bound, index_at, solve_no_wait, MiddleBound, MiddleLoop, MiddleOutput,
};
use crate::problem::Problem;

/// Dynamic programming time discretization mode
#[derive(Debug, Clone)]
pub enum TimeGrid {
    /// Uniform grid with a fixed step [s]
    Fixed(f64),
    /// Automatic grid combining a coarse uniform grid and departure hints
    Auto { coarse_steps: usize },
}


/// Dynamic programming parameters
#[derive(Debug, Clone)]
pub struct DynProgParams {
    /// Whether waiting is allowed
    pub allow_waiting: bool,
    /// Time grid steps
    pub grid_steps: usize,
    /// Whether to add departure hints
    pub departure_hints: bool,
}

impl Default for DynProgParams {
    fn default() -> Self {
        Self {
            allow_waiting: true,
            grid_steps: 12,
            departure_hints: true,
        }
    }
}


/// Dynamic programming flattened buffers
#[derive(Default)]
struct Scratch {
    /// `memo[pos * m + k]`: min cumulative fuel to be ready to depart from
    /// debris position `pos` at epoch index `k`
    memo: Vec<f64>,
    /// `back[pos * m + k]`: departure index achieving `memo[pos * m + k]`
    back: Vec<usize>,
}


/// Dynamic programming for the middle loop
pub struct DynProg {
    /// Parameters
    params: DynProgParams,
    /// Memoization table for `solve` calls
    memo: RefCell<HashMap<Vec<usize>, MiddleOutput>>,
    /// Memoization table for `solve_lb` calls
    memo_lb: RefCell<HashMap<Vec<usize>, MiddleBound>>,
    /// Reused buffers across `solve` calls
    scratch: RefCell<Scratch>,
}

impl DynProg {
    pub fn new(params: &DynProgParams) -> Self {
        Self {
            params: params.clone(),
            memo: RefCell::new(HashMap::new()),
            memo_lb: RefCell::new(HashMap::new()),
            scratch: RefCell::new(Scratch::default()),
        }
    }

    /// Departure time grid construction
    fn epochs(&self, sequence: &[usize], inner: &dyn InnerLoop, problem: &Problem) -> Vec<f64> {
        let steps = (self.params.grid_steps).max(1);
        let mut epochs: Vec<f64> = Vec::with_capacity(steps + 1);

        // Uniform grid
        for j in 0..=steps {
            epochs.push(j as f64 / steps as f64 * problem.max_time);
        }

        // Departure hints
        if self.params.departure_hints {
            let n = sequence.len();
            for i in 0..n - 1 {
                let src = index_at(sequence, i, problem);
                let dst = index_at(sequence, i + 1, problem);
                epochs.extend(inner.departure_hints(src, dst, problem.max_time, problem));
            }
            let eps = problem.max_time * 1e-6;
            epochs.sort_unstable_by(f64::total_cmp);
            epochs.dedup_by(|a, b| (*a - *b).abs() <= eps);
        }
        
        epochs
    }

    /// Run the waiting dynamic program over the precomputed departure grid
    fn schedule_with_waiting(
        &self,
        sequence: &[usize],
        grid: &[f64],
        inner: &dyn InnerLoop,
        problem: &Problem,
        fuel_cap: f64,
    ) -> Option<Vec<f64>> {
        let n = sequence.len();
        let m = grid.len();
        let size = n * m;

        let mut scratch = self.scratch.borrow_mut();
        let Scratch { memo, back } = &mut *scratch;

        // Grow buffers on demand
        if memo.len() < size { memo.resize(size, f64::INFINITY); }
        if back.len() < size { back.resize(size, usize::MAX); }
        memo[..size].fill(f64::INFINITY);
        back[..size].fill(usize::MAX);

        // Initial condition
        memo[..m].fill(0.0);

        // Recursive computation
        for i in 0..n - 2 {

            // Resolve state indices and buffer rows
            let src = index_at(sequence, i, problem);
            let dst = index_at(sequence, i + 1, problem);
            let row = i * m;
            let row_next = (i + 1) * m;

            // Explore each departure epoch
            for k in 0..m {
                let fuel_prev = memo[row + k];
                if fuel_prev.is_infinite() { continue; }

                let at_time = grid[k];
                let out = inner.solve(src, dst, at_time, problem);

                // Snap the arrival up to the next epoch after arrival
                let arrival = at_time + out.time;
                let kend = grid.partition_point(|&e| e < arrival);

                // Discard schedules whose arrival exceeds the time limit
                if kend >= m { continue; }

                // Discard schedules whose cumulative propellant exceeds the cap
                let fuel_next = fuel_prev + out.fuel;
                if fuel_next > fuel_cap { continue; }

                // Update best known arrival time
                if fuel_next < memo[row_next + kend] {
                    memo[row_next + kend] = fuel_next;
                    back[row_next + kend] = k;
                }
            }

            // Free waiting (later epoch inherits the cheaper arrival)
            for l in 1..m {
                if memo[row_next + l - 1] < memo[row_next + l] {
                    memo[row_next + l] = memo[row_next + l - 1];
                    back[row_next + l] = back[row_next + l - 1];
                }
            }
        }

        // Last leg with exact arrival time for numerical accuracy
        let last = n - 2;
        let src = index_at(sequence, last, problem);
        let dst = index_at(sequence, n - 1, problem);
        let row = last * m;

        let mut best_pred = usize::MAX; // cheapest fuel-feasible arrival
        let mut best_c = f64::INFINITY;
        let mut alt_pred = usize::MAX;  // minimum-fuel arrival (fallback)
        let mut alt_f = f64::INFINITY;
        for k in 0..m {
            let fuel_prev = memo[row + k];
            if fuel_prev.is_infinite() { continue; }

            let at_time = grid[k];
            let out = inner.solve(src, dst, at_time, problem);
            let arrival = at_time + out.time;
            let fuel_total = fuel_prev + out.fuel;

            // Exact fuel and time feasibility check
            if arrival > problem.max_time { continue; }
            if fuel_total < alt_f {
                alt_f = fuel_total;
                alt_pred = k;
            }

            if fuel_total <= problem.max_fuel {
                let c = problem.cost(arrival, fuel_total);
                if c < best_c {
                    best_c = c;
                    best_pred = k;
                }
            }
        }

        let chosen = if best_pred != usize::MAX { best_pred } else { alt_pred };
        if chosen == usize::MAX {
            // No admissible schedule found
            None
        } else {
            // Backtrack departure and waiting times
            let mut nodes = vec![0usize; n - 1];
            nodes[last] = chosen;
            for p in (1..n - 1).rev() {
                nodes[p - 1] = back[p * m + nodes[p]];
            }
            Some(nodes.iter().map(|&k| grid[k]).collect())
        }
    }
}

impl MiddleLoop for DynProg {
    fn solve(&self, sequence: &[usize], inner: &dyn InnerLoop, problem: &Problem) -> MiddleOutput {
        // Check if result already exists in memoization table
        {
            let memo = self.memo.borrow();
            if let Some(out) = memo.get(sequence) {
                return out.clone();
            }
        }
        
        let n = sequence.len();

        // Handle empty chaser mission with no debris collected (just the home).
        if n <= 1 {
            return MiddleOutput::empty(sequence[0]);
        }

        // Handle no-waiting case with a simple forward solution.
        if !self.params.allow_waiting {
            return solve_no_wait(sequence, inner, problem);
        }

        // Construct the departure time grid once and reuse it across passes.
        let epochs = self.epochs(sequence, inner, problem);

        // First pass: schedule with propellant pruning at the fuel budget.
        let departs = self.schedule_with_waiting(sequence, &epochs, inner, problem, problem.max_fuel);
        let mut out = match departs {
            None => solve_no_wait(sequence, inner, problem),
            Some(departs) => assemble(sequence, &departs, inner, problem),
        };

        // Propellant-based pruning
        if !out.feasible && problem.max_fuel.is_finite() {
            let departs = self.schedule_with_waiting(sequence, &epochs, inner, problem, f64::INFINITY);
            out = match departs {
                None => solve_no_wait(sequence, inner, problem),
                Some(departs) => assemble(sequence, &departs, inner, problem),
            };
        }

        // Memoize the result
        self.memo.borrow_mut().insert(sequence.to_vec(), out.clone());
        out
    }

    fn solve_lb(&self, sequence: &[usize], inner: &dyn InnerLoop, problem: &Problem) -> MiddleBound {

        // Serve from cache when available
        if let Some(bound) = self.memo_lb.borrow().get(sequence) {
            return *bound;
        }

        // Sum the per-leg inner-loop lower bounds and memoize
        let bound = compute_bound(sequence, inner, problem);
        self.memo_lb.borrow_mut().insert(sequence.to_vec(), bound);
        bound
    }
}
