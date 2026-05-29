use std::cell::RefCell;
use std::collections::HashMap;

use crate::inner::InnerLoop;
use crate::middle::{assemble, index_at, solve_no_wait, MiddleLoop, MiddleOutput};
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
    /// Time discretization mode
    pub time_grid: TimeGrid,
}

impl Default for DynProgParams {
    fn default() -> Self {
        Self {
            allow_waiting: true,
            time_grid: TimeGrid::Auto { coarse_steps: 12 },
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
    /// Reused buffers across `solve` calls
    scratch: RefCell<Scratch>,
}

impl DynProg {
    pub fn new(params: &DynProgParams) -> Self {
        Self {
            params: params.clone(),
            memo: RefCell::new(HashMap::new()),
            scratch: RefCell::new(Scratch::default()),
        }
    }

    /// Departure time grid construction
    fn epochs(&self, sequence: &[usize], inner: &dyn InnerLoop, problem: &Problem) -> Vec<f64> {
        match &self.params.time_grid {
            
            // Uniform grid
            TimeGrid::Fixed(step) => {
                let kmax = (problem.max_time / step).floor() as usize + 1;
                (0..kmax).map(|k| k as f64 * step).collect()
            }
            
            // Coarse uniform grid plus departure hints
            TimeGrid::Auto { coarse_steps } => {
                let steps = (*coarse_steps).max(1);
                let mut epochs: Vec<f64> = Vec::with_capacity(steps + 1);

                // Coarse uniform grid
                for j in 0..=steps {
                    epochs.push(j as f64 / steps as f64 * problem.max_time);
                }

                // Departure hints per leg 
                let n = sequence.len();
                for i in 0..n - 1 {
                    let src = index_at(sequence, i, problem);
                    let dst = index_at(sequence, i + 1, problem);
                    epochs.extend(inner.departure_hints(src, dst, problem.max_time, problem));
                }

                // Sort and drop near-duplicates
                let eps = problem.max_time * 1e-6;
                epochs.sort_unstable_by(f64::total_cmp);
                epochs.dedup_by(|a, b| (*a - *b).abs() <= eps);
                epochs
            }
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

        // Construct departure time grid
        let epochs = self.epochs(sequence, inner, problem);
        let m = epochs.len();
        let size = n * m;

        // Run over the flattened scratch buffers
        let departs: Option<Vec<f64>> = {
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

                    let at_time = epochs[k];
                    let out = inner.solve(src, dst, at_time, problem);

                    // Snap the arrival up to the next epoch after arrival
                    let arrival = at_time + out.time;
                    let kend = epochs.partition_point(|&e| e < arrival);

                    // Check if arrival exceeds max_time
                    if kend >= m { continue; }

                    // Update best known arrival time
                    let fuel_next = fuel_prev + out.fuel;
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

                let at_time = epochs[k];
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
                // No feasible schedule found
                None
            } else {
                // Backtrack departure and waiting times
                let mut nodes = vec![0usize; n - 1];
                nodes[last] = chosen;
                for p in (1..n - 1).rev() {
                    nodes[p - 1] = back[p * m + nodes[p]];
                }
                Some(nodes.iter().map(|&k| epochs[k]).collect())
            }
        };

        // Fall back to asap-depart if no waiting schedule is feasible
        let out = match departs {
            None => solve_no_wait(sequence, inner, problem),
            Some(departs) => assemble(sequence, &departs, inner, problem),
        };
        
        // Memoize the result
        self.memo.borrow_mut().insert(sequence.to_vec(), out.clone());
        out
    }
}
