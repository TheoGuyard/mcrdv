//! Dynamic-programming scheduler.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::problem::{ChaserPlan, Problem};
use crate::solver::{ScheduleLayer, TransferLayer};

/// Problem-independent parameters of the DP scheduler.
#[derive(Debug, Clone, Copy)]
pub struct DpConfig {
    /// Number of uniform grid steps.
    pub grid_size: usize,
    /// Whether to augment the grid with transfer departure hints.
    pub use_hints: bool,
    /// Whether to use caching.
    pub use_cache: bool,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            grid_size: 16,
            use_hints: true,
            use_cache: true,
        }
    }
}

/// Chaser-plan scheduler based on dynamic programming over a time grid.
pub struct DpScheduler<'p> {
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
    config: DpConfig,

    /// Cache: sequence -> chaser plan.
    cache_evaluate: RefCell<HashMap<Vec<usize>, Rc<ChaserPlan>>>,
    /// Cache: sequence -> chaser plan cost lower bound.
    cache_evaluate_lb: RefCell<HashMap<Vec<usize>, f64>>,
}

impl<'p> DpScheduler<'p> {
    pub fn new(problem: &'p Problem, transfer: &'p dyn TransferLayer, config: DpConfig) -> Self {
        Self {
            config,
            problem,
            transfer,
            cache_evaluate: RefCell::new(HashMap::new()),
            cache_evaluate_lb: RefCell::new(HashMap::new()),
        }
    }

    pub fn default(problem: &'p Problem, transfer: &'p dyn TransferLayer) -> Self {
        Self::new(problem, transfer, DpConfig::default())
    }

    /// Run the waiting dynamic program over a departure grid.
    fn _dp(&self, seq: &[usize], grid: &[f64], ub: f64) -> Option<ChaserPlan> {
        let n = seq.len() - 1;
        let m = grid.len();

        let mut memo = vec![f64::INFINITY; (n + 1) * m];
        let mut back = vec![usize::MAX; (n + 1) * m];
        
        memo[..m].fill(0.0);

        for i in 0..n {
            let (src, dst) = (seq[i], seq[i + 1]);
            let row_curr = i * m;
            let row_next = (i + 1) * m;

            for k in 0..m {
                let prev = memo[row_curr + k];
                
                // Skip infeasible partial schedules
                if !prev.is_finite() { continue; }
                
                // Evaluate the leg at this departure time
                let (dt, dc) = self.transfer.evaluate(src, dst, grid[k]);
                let arrival = grid[k] + dt;

                // Discard partial schedules that arrive after the horizon
                if arrival > self.problem.time_horizon { continue; }
                
                // Discard partial schedules exceeding the cost upper bound
                let cand = prev + dc;
                if cand > ub { continue; }
                
                // Snap the arrival up to the first grid epoch after arrival
                let kend = grid.partition_point(|&e| e < arrival);
                if kend >= m { continue; }
                if cand < memo[row_next + kend] {
                    memo[row_next + kend] = cand;
                    back[row_next + kend] = k;
                }
            }

            // Free waiting: a later epoch inherits a cheaper earlier arrival
            for l in 1..m {
                if memo[row_next + l - 1] < memo[row_next + l] {
                    memo[row_next + l] = memo[row_next + l - 1];
                    back[row_next + l] = back[row_next + l - 1];
                }
            }
        }

        // Check if any schedule if feasible
        if !memo[n * m + (m - 1)].is_finite() {
            return None;
        }

        // Backtrack the departure epoch index of each leg
        let mut d = vec![0usize; n];
        let mut c = m - 1;
        for p in (1..=n).rev() {
            let k = back[p * m + c];
            if k == usize::MAX {
                return None;
            }
            d[p - 1] = k;
            c = k;
        }

        // Assemble the plan from the departure times
        let departs: Vec<f64> = d.iter().map(|&k| grid[k]).collect();
        let n = seq.len();
        let mut meet = vec![0.0; n];
        let mut wait = vec![0.0; n];
        let mut cost = vec![0.0; n];
        for i in 1..n {
            let depart = departs[i - 1];
            let (dt, dc) = self.transfer.evaluate(seq[i - 1], seq[i], depart);
            wait[i - 1] = depart - meet[i - 1];
            meet[i] = depart + dt;
            cost[i] = dc;
        }
        Some(self._plan(seq, meet, wait, cost))
    }

    /// Wrap reconstructed schedule arrays into a feasibility-checked plan.
    fn _plan(&self, seq: &[usize], meet: Vec<f64>, wait: Vec<f64>, cost: Vec<f64>) -> ChaserPlan {
        let total: f64 = cost.iter().sum();
        let mut plan = ChaserPlan::new(seq.to_vec(), meet, wait, cost, total, true);
        plan.feasible = self.problem.is_feasible(&plan);
        plan
    }

    /// Forward pass when no waiting allowed.
    fn _forward(&self, seq: &[usize]) -> ChaserPlan {
        let n = seq.len();
        let mut meet = vec![0.0; n];
        let wait = vec![0.0; n];
        let mut cost = vec![0.0; n];
        let mut time = 0.0;
        for i in 1..n {
            let (dt, dc) = self.transfer.evaluate(seq[i - 1], seq[i], time);
            time += dt;
            cost[i] = dc;
            meet[i] = time;
        }
        self._plan(seq, meet, wait, cost)
    }

    /// Compute a plan for a sequence.
    fn _evaluate(&self, seq: &[usize]) -> ChaserPlan {
        // Handle empty sequences
        if seq.len() <= 1 { return ChaserPlan::empty(); }

        // Forward schedule with no waiting giving an upper bound
        let forward = self._forward(seq);
        let mut ub = self.problem.cost_cap();
        if forward.feasible { ub = ub.min(forward.cost); }
        
        // Construct coarse uniform time grid
        let steps = self.config.grid_size.max(1);
        let mut grid: Vec<f64> = (0..=steps)
            .map(|j| j as f64 / steps as f64 * self.problem.time_horizon)
            .collect();        

        // Add departure hints to the grid if requested
        if self.config.use_hints {
            for w in seq.windows(2) { grid.extend(self.transfer.hints(w[0], w[1]).to_vec()); }
            grid.sort_unstable_by(f64::total_cmp);
            let eps = self.problem.time_horizon * 1e-6;
            grid.dedup_by(|a, b| (*a - *b).abs() <= eps);
        }

        // Run the DP and return the plan found (forward plan if none feasible)
        match self._dp(seq, &grid, ub) {
            Some(plan) => plan,
            None => forward,
        }
    }

    /// Evaluate the lower bound for a sequence from per-leg lower bounds.
    fn _evaluate_lb(&self, seq: &[usize]) -> f64 {
        seq.windows(2).map(|w| self.transfer.evaluate_lb(w[0], w[1]).1).sum()
    }
}

impl<'p> ScheduleLayer for DpScheduler<'p> {
    fn initialize(&self) {
        self.cache_evaluate.borrow_mut().clear();
        self.cache_evaluate_lb.borrow_mut().clear();
    }

    fn evaluate(&self, seq: &[usize]) -> Rc<ChaserPlan> {
        if !self.config.use_cache {
            return Rc::new(self._evaluate(seq));
        }
        if let Some(plan) = self.cache_evaluate.borrow().get(seq) {
            return Rc::clone(plan);
        }
        let plan = Rc::new(self._evaluate(seq));
        self.cache_evaluate.borrow_mut().insert(seq.to_vec(), Rc::clone(&plan));
        plan
    }

    fn evaluate_lb(&self, seq: &[usize]) -> f64 {
        if !self.config.use_cache { 
            return self._evaluate_lb(seq); 
        }
        if let Some(lb) = self.cache_evaluate_lb.borrow().get(seq) { 
            return *lb; 
        }
        let lb = self._evaluate_lb(seq);
        self.cache_evaluate_lb.borrow_mut().insert(seq.to_vec(), lb);
        lb
    }

    fn statistics(&self) -> HashMap<String, Box<dyn Any>> {
        HashMap::new()
    }
}
