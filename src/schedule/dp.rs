//! Schedule layer based on dynamic programming over discretized time.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::problem::{ChaserPlan, Problem};
use crate::solver::{ScheduleLayer, TransferLayer};

/// Sentinel for "no predecessor" in the backtracking table.
const NONE: usize = usize::MAX;

/// Schedule layer based on dynamic programming over discretized time.
pub struct DpSchedule<T: TransferLayer> {
    /// The layer supplying the leg oracles.
    transfer: T,
    /// Whether waiting on a debris orbit is allowed before departing.
    pub allow_waiting: bool,
    /// Number of steps of the uniform time grid.
    pub grid_size: usize,
    /// Whether to enrich the uniform grid with the transfer layer's hints.
    pub use_hints: bool,
    /// Whether to memoize plans and candidate grids.
    pub use_cache: bool,

    problem: Option<Rc<Problem>>,
    uniform: Rc<[f64]>,
    plans: RefCell<HashMap<Vec<usize>, Rc<ChaserPlan>>>,
    grids: RefCell<Vec<Option<Rc<[f64]>>>>,
}

impl<T: TransferLayer> DpSchedule<T> {
    /// Build a scheduler over `transfer` with the default settings.
    pub fn new(transfer: T) -> Self {
        Self::with_options(transfer, true, 32, true, true)
    }

    /// Build a scheduler, choosing every option.
    pub fn with_options(
        transfer: T,
        allow_waiting: bool,
        grid_size: usize,
        use_hints: bool,
        use_cache: bool,
    ) -> Self {
        assert!(grid_size >= 1, "parameter 'grid_size' must be positive");
        Self {
            transfer,
            allow_waiting,
            grid_size,
            use_hints,
            use_cache,
            problem: None,
            uniform: Rc::from(vec![].into_boxed_slice()),
            plans: RefCell::new(HashMap::new()),
            grids: RefCell::new(Vec::new()),
        }
    }

    /// The transfer layer underneath.
    pub fn transfer(&self) -> &T {
        &self.transfer
    }

    /// The transfer layer underneath, mutably.
    pub fn transfer_mut(&mut self) -> &mut T {
        &mut self.transfer
    }

    fn problem(&self) -> &Problem {
        self.problem
            .as_ref()
            .expect("schedule layer used before initialize()")
    }

    /// Candidate departure times for the leg `i -> j`, in increasing order.
    fn grid(&self, i: usize, j: usize) -> Rc<[f64]> {
        if !self.use_hints {
            return Rc::clone(&self.uniform);
        }
        let n = self.problem().states.len();
        let slot = i * n + j;

        if self.use_cache {
            if let Some(hit) = self.grids.borrow()[slot].as_ref() {
                return Rc::clone(hit);
            }
        }

        let horizon = self.problem().max_time;
        let hints = self.transfer.hints(i, j, horizon);
        let grid: Rc<[f64]> = if hints.is_empty() {
            Rc::clone(&self.uniform)
        } else {
            let mut merged: Vec<f64> = self.uniform.iter().copied().chain(hints).collect();
            merged.sort_by(f64::total_cmp);

            // Deduplicate the merged grid
            let eps = 1e-9 * horizon;
            let mut kept = Vec::with_capacity(merged.len());
            for (k, &t) in merged.iter().enumerate() {
                if k == 0 || t - merged[k - 1] > eps {
                    kept.push(t);
                }
            }
            Rc::from(kept.into_boxed_slice())
        };

        if self.use_cache {
            self.grids.borrow_mut()[slot] = Some(Rc::clone(&grid));
        }
        grid
    }

    /// Suffix sums of the per-leg lower bounds.
    fn tail_bounds(&self, sequence: &[usize]) -> Vec<f64> {
        let n = sequence.len();
        let horizon = self.problem().max_time;
        let mut tail = vec![0.0; n.max(1)];
        for i in (0..n.saturating_sub(1)).rev() {
            let (_, lb) = self.transfer.evaluate_lb(
                sequence[i], 
                sequence[i + 1], 
                horizon
            );
            tail[i] = tail[i + 1] + lb;
        }
        tail
    }

    /// The schedule with no waiting time allowed.
    fn nowait(&self, sequence: &[usize]) -> ChaserPlan {
        let n = sequence.len();
        let mut depart = vec![0.0; n];
        let mut arrive = vec![0.0; n];
        let mut costs = vec![0.0; n];

        let mut time = 0.0;
        for i in 0..n - 1 {
            let (dt, dc) = self.transfer.evaluate(
                sequence[i],
                sequence[i + 1],
                time
            );
            depart[i] = time;
            time += dt;
            arrive[i + 1] = time;
            costs[i + 1] = dc;
        }
        depart[n - 1] = arrive[n - 1];
        ChaserPlan::new(sequence.to_vec(), depart, arrive, costs)
    }

    /// Rebuild arrival times and leg costs from a choice of departure times.
    fn plan_from(&self, sequence: &[usize], departs: &[f64]) -> ChaserPlan {
        let n = sequence.len();
        let mut depart = vec![0.0; n];
        let mut arrive = vec![0.0; n];
        let mut costs = vec![0.0; n];

        for i in 0..n - 1 {
            let (dt, dc) = self.transfer.evaluate(
                sequence[i],
                sequence[i + 1],
                departs[i]
            );
            depart[i] = departs[i];
            arrive[i + 1] = departs[i] + dt;
            costs[i + 1] = dc;
        }
        depart[n - 1] = arrive[n - 1];
        ChaserPlan::new(sequence.to_vec(), depart, arrive, costs)
    }

    /// DP pass to construct a schedule (or None if no feasible one exists).
    fn dp(&self, sequence: &[usize], upper: f64) -> Option<ChaserPlan> {
        let n = sequence.len() - 1;
        let horizon = self.problem().max_time;

        // One candidate set per leg, plus the horizon alone for the last
        // object, where the plan cost is read off.
        let mut grids: Vec<Rc<[f64]>> = (0..n)
            .map(|i| self.grid(sequence[i], sequence[i + 1]))
            .collect();
        grids.push(Rc::from(vec![horizon].into_boxed_slice()));
        let tail = self.tail_bounds(sequence);

        // `cost[i][k]` is the least cost of reaching the `i`-th object early
        // enough to leave it at the `k`-th time of its grid.
        let mut cost: Vec<Vec<f64>> = grids.iter().map(|g| vec![f64::INFINITY; g.len()]).collect();
        let mut back: Vec<Vec<usize>> = grids.iter().map(|g| vec![NONE; g.len()]).collect();
        cost[0].fill(0.0);

        for i in 0..n {
            let (src, dst) = (sequence[i], sequence[i + 1]);
            let grid = Rc::clone(&grids[i]);
            let next = Rc::clone(&grids[i + 1]);

            for l in 0..grid.len() {
                if !cost[i][l].is_finite() {
                    continue;
                }
                let (dt, dc) = self.transfer.evaluate(src, dst, grid[l]);
                let arrival = grid[l] + dt;
                if arrival > horizon {
                    continue;
                }
                // Earliest candidate departure from the next object that the
                // chaser can still make.
                let k = next.partition_point(|&t| t < arrival);
                if k >= next.len() {
                    continue;
                }
                let candidate = cost[i][l] + dc;
                if candidate + tail[i + 1] > upper {
                    continue;
                }
                if candidate < cost[i + 1][k] {
                    cost[i + 1][k] = candidate;
                    back[i + 1][k] = l;
                }
            }

            // Waiting is free, so being ready earlier is never worse: a later
            // departure inherits any cheaper way of arriving before it. This
            // makes each row non-increasing.
            let row = &mut cost[i + 1];
            let trace = &mut back[i + 1];
            for k in 1..row.len() {
                if row[k - 1] < row[k] {
                    row[k] = row[k - 1];
                    trace[k] = trace[k - 1];
                }
            }
        }

        if !cost[n].last()?.is_finite() {
            return None;
        }

        // Walk the choices back from the last object.
        let mut chosen = vec![0usize; n];
        let mut k = cost[n].len() - 1;
        for i in (1..=n).rev() {
            let l = back[i][k];
            if l == NONE {
                return None;
            }
            chosen[i - 1] = l;
            k = l;
        }

        let departs: Vec<f64> = (0..n).map(|i| grids[i][chosen[i]]).collect();
        Some(self.plan_from(sequence, &departs))
    }
}

impl<T: TransferLayer> ScheduleLayer for DpSchedule<T> {
    fn initialize(&mut self, problem: &Rc<Problem>) {
        self.transfer.initialize(problem);

        let steps = self.grid_size;
        let uniform: Vec<f64> = (0..=steps)
            .map(|k| problem.max_time * (k as f64) / (steps as f64))
            .collect();
        self.uniform = Rc::from(uniform.into_boxed_slice());

        let n = problem.states.len();
        self.problem = Some(Rc::clone(problem));
        self.plans.borrow_mut().clear();
        self.grids.replace(vec![None; n * n]);
    }

    fn evaluate(&self, sequence: &[usize]) -> Rc<ChaserPlan> {
        assert!(
            !sequence.is_empty(),
            "empty sequence received in the schedule layer"
        );
        if sequence.len() == 1 {
            // No leg to price. The plan is the object it starts on, which is
            // the departure orbit for every caller in the crate but need not be.
            let zero = vec![0.0];
            return Rc::new(ChaserPlan::new(
                sequence.to_vec(),
                zero.clone(),
                zero.clone(),
                zero,
            ));
        }
        if self.use_cache {
            if let Some(hit) = self.plans.borrow().get(sequence) {
                return Rc::clone(hit);
            }
        }

        // The no-wait schedule answers three questions at once: it is the plan
        // when waiting is off, it decides schedulability, and its cost is the
        // incumbent the dynamic program prunes against.
        let nowait = self.nowait(sequence);
        let plan = if !self.allow_waiting || nowait.time() > self.problem().max_time {
            nowait
        } else {
            let cost = nowait.cost();
            self.dp(sequence, cost).unwrap_or(nowait)
        };

        let plan = Rc::new(plan);
        if self.use_cache {
            self.plans
                .borrow_mut()
                .insert(sequence.to_vec(), Rc::clone(&plan));
        }
        plan
    }

    fn evaluate_lb(&self, sequence: &[usize]) -> f64 {
        self.tail_bounds(sequence)[0]
    }
}
