//! Coordinate-descent scheduler.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::problem::{ChaserPlan, Problem};
use crate::solver::{ScheduleLayer, TransferLayer};
use crate::transfer::qlaw::linspace;

/// Problem-independent parameters of the CD scheduler.
#[derive(Debug, Clone, Copy)]
pub struct CdConfig {
    /// Number of candidate departures per leg in the CD passes.
    pub samples: usize,
    /// Number of coordinate-descent passes refining the forward pass.
    pub max_iters: usize,
    /// Whether the forward pass may depart at the transfer departure hints.
    pub use_hints: bool,
    /// Whether to cache the results of the schedule evaluations.
    pub use_cache: bool,
}

impl Default for CdConfig {
    fn default() -> Self {
        Self {
            samples: 16,
            max_iters: 4,
            use_hints: true,
            use_cache: true,
        }
    }
}

/// Coordinate-descent scheduler.
///
/// Chooses a departure epoch per leg from a small set of candidate times, keeps
/// the cheapest one whose arrival still leaves room to finish within the
/// horizon, then optionally refines the departures by coordinate-descent passes.
pub struct CdScheduler<'p> {
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
    config: CdConfig,

    /// Cache: sequence -> chaser plan.
    cache_evaluate: RefCell<HashMap<Vec<usize>, Rc<ChaserPlan>>>,
    /// Cache: sequence -> chaser plan cost lower bound.
    cache_evaluate_lb: RefCell<HashMap<Vec<usize>, f64>>,
}

impl<'p> CdScheduler<'p> {
    pub fn new(problem: &'p Problem, transfer: &'p dyn TransferLayer, config: CdConfig) -> Self {
        Self {
            config,
            problem,
            transfer,
            cache_evaluate: RefCell::new(HashMap::new()),
            cache_evaluate_lb: RefCell::new(HashMap::new()),
        }
    }

    pub fn default(problem: &'p Problem, transfer: &'p dyn TransferLayer) -> Self {
        Self::new(problem, transfer, CdConfig::default())
    }

    /// Minimum transfer time of each leg of `seq`, i.e. `legmin[t - 1]` is a
    /// lower estimate of the duration of leg `t`.
    fn _legmin(&self, seq: &[usize]) -> Vec<f64> {
        (1..seq.len())
            .map(|t| self.transfer.evaluate_lb(seq[t - 1], seq[t]).0)
            .collect()
    }

    /// Evaluate the candidate departures `cand` for leg `i -> j` and return
    /// `(depart, arrival, cost)` for the cheapest candidate whose arrival is
    /// `<= bound`; if none qualifies, take the earliest arrival (best shot at
    /// staying feasible).
    fn _pick(&self, i: usize, j: usize, cand: &[f64], bound: f64) -> (f64, f64, f64) {
        debug_assert!(!cand.is_empty(), "no candidate departure for a leg");

        let mut best: Option<(f64, f64, f64)> = None; // (depart, arrival, cost)
        let mut best_key = f64::INFINITY; // cost when feasible, arrival otherwise
        let mut feasible_found = false;

        for &dep in cand {
            let (dt, dc) = self.transfer.evaluate(i, j, dep);
            let arr = dep + dt;
            let feasible = arr <= bound;

            // A feasible candidate always beats an infeasible incumbent; among
            // equals, the first one wins (matching `numpy.argmin` tie-breaks).
            let key = if feasible { dc } else { arr };
            let better = match (feasible, feasible_found) {
                (true, false) => true,
                (false, true) => false,
                _ => key < best_key,
            };
            if better {
                best_key = key;
                feasible_found |= feasible;
                best = Some((dep, arr, dc));
            }
        }

        best.expect("`_pick` called with an empty candidate set")
    }

    /// Forward pass: each leg departs at the cheapest candidate that keeps the
    /// remaining legs within the horizon.
    ///
    /// Returns the `(meeting, waiting, per-leg cost)` arrays, all of length
    /// `seq.len()`, with the per-leg cost stored at the index of the node the
    /// leg *arrives* at (so index `0` is always zero).
    fn _forward(&self, seq: &[usize], legmin: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = seq.len();
        let mut meet = vec![0.0; n];
        let mut wait = vec![0.0; n];
        let mut cost = vec![0.0; n];

        // tail[t] = minimum remaining leg time for legs t+1..n-1
        let mut tail = vec![0.0; n];
        for t in (1..n.saturating_sub(1)).rev() {
            tail[t] = tail[t + 1] + legmin[t];
        }

        let horizon = self.problem.time_horizon;
        let mut time = 0.0;
        let mut cand: Vec<f64> = Vec::new();
        for t in 1..n {
            // The no-wait departure comes first so that ties resolve to it.
            cand.clear();
            cand.push(time);
            if self.config.use_hints {
                cand.extend(
                    self.transfer
                        .hints(seq[t - 1], seq[t])
                        .iter()
                        .copied()
                        .filter(|&h| h >= time && h <= horizon),
                );
            }

            let bound = horizon - tail[t];
            let (dep, arr, dc) = self._pick(seq[t - 1], seq[t], &cand, bound);

            wait[t - 1] = dep - meet[t - 1];
            meet[t] = arr;
            cost[t] = dc;
            time = arr;
        }

        (meet, wait, cost)
    }

    /// Forward pass, then `max_iters` coordinate-descent refinements.
    fn _schedule(&self, seq: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = seq.len();

        // legmin[t-1] = minimum transfer time of leg t
        let legmin = self._legmin(seq);

        let (mut meet, mut wait, mut cost) = self._forward(seq, &legmin);
        if self.config.max_iters == 0 || n <= 2 {
            return (meet, wait, cost);
        }

        // depart[t-1] = best departure of leg t
        let mut depart: Vec<f64> = (1..n).map(|t| meet[t - 1] + wait[t - 1]).collect();

        let mut grid: Vec<f64> = Vec::with_capacity(self.config.samples + 1);
        for _ in 0..self.config.max_iters {
            let mut updated = false;
            for t in 1..n {
                // Departing later than the next leg already does would push that
                // leg back, so the arrival is capped by its current departure.
                let bound = if t == n - 1 {
                    self.problem.time_horizon
                } else {
                    depart[t]
                };

                let t0 = meet[t - 1];
                let t1 = t0.max(bound - legmin[t - 1]);

                grid.clear();
                grid.extend(linspace(t0, t1, self.config.samples));
                // `linspace` accumulates `t0 + step * k`, which can land an ulp
                // off `t1`; pin the endpoint so the grid spans exactly [t0, t1].
                *grid.last_mut().expect("samples >= 2") = t1;
                // Keep the incumbent departure as a (last, so lowest priority
                // on ties) candidate: a pass can then never worsen the plan.
                grid.push(depart[t - 1]);

                let (dep, arr, dc) = self._pick(seq[t - 1], seq[t], &grid, bound);

                if dep != depart[t - 1] {
                    updated = true;
                }
                depart[t - 1] = dep;
                wait[t - 1] = dep - meet[t - 1];
                meet[t] = arr;
                cost[t] = dc;
            }

            // A pass that changed no departure reproduces itself: stop early.
            if !updated {
                break;
            }
        }

        (meet, wait, cost)
    }

    /// Earliest-arrival schedule used as the fallback.
    fn _no_wait(&self, seq: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = seq.len();
        let mut meet = vec![0.0; n];
        let wait = vec![0.0; n];
        let mut cost = vec![0.0; n];
        let mut time = 0.0;
        for t in 1..n {
            let (dt, dc) = self.transfer.evaluate(seq[t - 1], seq[t], time);
            time += dt;
            cost[t] = dc;
            meet[t] = time;
        }
        (meet, wait, cost)
    }

    /// Wrap the schedule arrays into a cost-summed, feasibility-checked plan.
    fn _plan(
        &self,
        seq: &[usize],
        meet: Vec<f64>,
        wait: Vec<f64>,
        cost: Vec<f64>,
    ) -> ChaserPlan {
        let total: f64 = cost.iter().sum();
        let mut plan = ChaserPlan::new(seq.to_vec(), meet, wait, cost, total, true);
        plan.feasible = self.problem.is_feasible(&plan);
        plan
    }

    /// Compute a plan for a sequence.
    fn _evaluate(&self, seq: &[usize]) -> ChaserPlan {
        // Trivial sequence: nothing to schedule.
        if seq.len() <= 1 {
            return ChaserPlan::empty();
        }

        let (meet, wait, cost) = self._schedule(seq);

        // If the schedule overshoots the horizon, fall back to the no-wait plan,
        // which `_plan` then reports infeasible rather than returning a schedule
        // that spends more than the horizon on purpose.
        let (meet, wait, cost) = match meet.last() {
            Some(&end) if end > self.problem.time_horizon => self._no_wait(seq),
            _ => (meet, wait, cost),
        };

        self._plan(seq, meet, wait, cost)
    }

    /// Evaluate the lower bound for a sequence from per-leg lower bounds.
    fn _evaluate_lb(&self, seq: &[usize]) -> f64 {
        seq.windows(2).map(|w| self.transfer.evaluate_lb(w[0], w[1]).1).sum()
    }
}

impl<'p> ScheduleLayer for CdScheduler<'p> {
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
