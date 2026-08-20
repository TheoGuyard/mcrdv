//! Forward scheduler with no waiting allowed between legs.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::problem::{ChaserPlan, Problem};
use crate::solver::{ScheduleLayer, TransferLayer};

/// Problem-independent parameters of the no-wait scheduler.
#[derive(Debug, Clone, Copy)]
pub struct NowaitConfig {
    /// Whether to use caching.
    pub use_cache: bool,
}

impl Default for NowaitConfig {
    fn default() -> Self {
        Self {
            use_cache: true,
        }
    }
}

/// Forward scheduler with no waiting allowed between legs.
pub struct NowaitScheduler<'p> {
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
    config: NowaitConfig,

    /// cache_evaluate: sequence -> chaser plan.
    cache_evaluate: RefCell<HashMap<Vec<usize>, Rc<ChaserPlan>>>,
    /// cache_evaluate: sequence -> chaser plan cost lower bound.
    cache_evaluate_lb: RefCell<HashMap<Vec<usize>, f64>>,
}

impl<'p> NowaitScheduler<'p> {
    pub fn new(problem: &'p Problem, transfer: &'p dyn TransferLayer, config: NowaitConfig) -> Self {
        Self {
            problem,
            transfer,
            config,
            cache_evaluate: RefCell::new(HashMap::new()),
            cache_evaluate_lb: RefCell::new(HashMap::new()),
        }
    }

    pub fn default(problem: &'p Problem, transfer: &'p dyn TransferLayer) -> Self {
        Self::new(problem, transfer, NowaitConfig::default())
    }

    /// Forward pass with no waiting allowed.
    fn _evaluate(&self, seq: &[usize]) -> ChaserPlan {
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
        let total: f64 = cost.iter().sum();
        let mut plan = ChaserPlan::new(
            seq.to_vec(),
            meet,
            wait,
            cost,
            total,
            true
        );
        plan.feasible = self.problem.is_feasible(&plan);
        plan
    }

    /// Evaluate the lower bound for a sequence from per-leg lower bounds.
    fn _evaluate_lb(&self, seq: &[usize]) -> f64 {
        seq.windows(2).map(|w| self.transfer.evaluate_lb(w[0], w[1]).1).sum()
    }
}

impl<'p> ScheduleLayer for NowaitScheduler<'p> {
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
