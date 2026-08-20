//! Q-law transfer estimator.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::f64::consts::TAU;

use crate::orbit::MeeState;
use crate::problem::Problem;
use crate::solver::TransferLayer;

pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![start]; }
    let step = (stop - start) / (n - 1) as f64;
    (0..n).map(|k| start + step * k as f64).collect()
}

/// Differential RAAN-rate threshold below which alignment never repeats.
const RATE_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub struct QlawConfig {
    /// Chaser thrust acceleration [m/s^2].
    pub thrust: f64,
    /// Q-law scaling factor on the element-difference norm.
    pub factor: f64,
    /// Number of grid samples used by `evaluate_lb`.
    pub lb_samples: usize,
    /// Whether to use caching.
    pub use_cache: bool,
}

impl Default for QlawConfig {
    fn default() -> Self {
        Self {
            thrust: 3e-3,
            factor: 1.5,
            lb_samples: 64,
            use_cache: true,
        }
    }
}

pub struct QlawTransfer<'p> {
    problem: &'p Problem,
    config: QlawConfig,

    /// Cache: leg `(i, j, t)` -> `(time, cost)`.
    cache_evaluate: RefCell<HashMap<(usize, usize, u64), (f64, f64)>>,
    /// Cache: leg `(i, j)` -> lower bound on `(time, cost)`.
    cache_evaluate_lb: RefCell<HashMap<(usize, usize), (f64, f64)>>,
    /// Cache: leg `(i, j)` -> departure hints.
    cache_hints: RefCell<HashMap<(usize, usize), Rc<Vec<f64>>>>,
}

impl<'p> QlawTransfer<'p> {
    pub fn new(problem: &'p Problem, config: QlawConfig) -> Self {
        Self {
            problem,
            config,
            cache_evaluate: RefCell::new(HashMap::new()),
            cache_evaluate_lb: RefCell::new(HashMap::new()),
            cache_hints: RefCell::new(HashMap::new()),
        }
    }

    pub fn default(problem: &'p Problem) -> Self {
        Self::new(problem, QlawConfig::default())
    }

    // Estimate of transfer cost and time.
    fn _evaluate(&self, i: usize, j: usize, t: f64) -> (f64, f64) {        
        let cat = &self.problem.catalog;
        let src = MeeState::from_kep(&cat.state(i).propagate(t));
        let dst = MeeState::from_kep(&cat.state(j).propagate(t));

        let mrates = src.max_rates();
        let sqnorm = ((src.a - dst.a) / mrates.a).powi(2)
            + ((src.f - dst.f) / mrates.f).powi(2)
            + ((src.g - dst.g) / mrates.g).powi(2)
            + ((src.h - dst.h) / mrates.h).powi(2)
            + ((src.k - dst.k) / mrates.k).powi(2);
        
        let dv = self.config.factor * sqnorm.sqrt();
        let dt = dv / self.config.thrust;
        (dt, dv)
    }

    /// Lower bound estimate of transfer cost and time over a time grid.
    fn _evaluate_lb(&self, i: usize, j: usize) -> (f64, f64) {
        let mut min_dt = f64::INFINITY;
        let mut min_dv = f64::INFINITY;
        let grid = linspace(
            0.0,
            self.problem.time_horizon, 
            self.config.lb_samples.max(1)
        );
        for t in grid {
            let (dt, dc) = self.evaluate(i, j, t);
            min_dt = min_dt.min(dt);
            min_dv = min_dv.min(dc);
        }
        (min_dt, min_dv)
    }

    /// Analytic departure hints from RAAN-alignment pattern.
    fn _hints(&self, i: usize, j: usize) -> Vec<f64> {
        let si = &self.problem.catalog.state(i);
        let sj = &self.problem.catalog.state(j);
        
        let d_rate = si.j2_secular_rates().0 - sj.j2_secular_rates().0;
        if d_rate.abs() < RATE_EPS { return Vec::new(); }
        
        let d_raan = si.r - sj.r;
        let period = TAU / d_rate.abs();
        let mut tbase = (-d_raan / d_rate).rem_euclid(period);
        let mut hints = Vec::new();
        while tbase <= self.problem.time_horizon {
            hints.push(tbase);
            tbase += period;
        }
        
        hints
    }

}

impl<'p> TransferLayer for QlawTransfer<'p> {
    fn initialize(&self) -> () {
        self.cache_evaluate.borrow_mut().clear();
        self.cache_evaluate_lb.borrow_mut().clear();
        self.cache_hints.borrow_mut().clear();
    }

    fn evaluate(&self, i: usize, j: usize, t: f64) -> (f64, f64) {
        if !self.config.use_cache {
            return self._evaluate(i, j, t);
        }

        let key = (i, j, t.to_bits());

        if let Some(&result) = self.cache_evaluate.borrow().get(&key) {
            return result;
        }

        let result = self._evaluate(i, j, t);
        self.cache_evaluate.borrow_mut().insert(key, result);

        result
    }

    fn evaluate_lb(&self, i: usize, j: usize) -> (f64, f64) {
        if !self.config.use_cache {
            return self._evaluate_lb(i, j);
        }

        let key = (i, j);

        if let Some(&result) = self.cache_evaluate_lb.borrow().get(&key) {
            return result;
        }

        let result = self._evaluate_lb(i, j);
        self.cache_evaluate_lb.borrow_mut().insert(key, result);

        result
    }

    fn hints(&self, i: usize, j: usize) -> Rc<Vec<f64>> {
        if !self.config.use_cache {
            return Rc::new(self._hints(i, j));
        }

        let key = (i, j);

        if let Some(result) = self.cache_hints.borrow().get(&key) {
            return Rc::clone(result);
        }

        let result = Rc::new(self._hints(i, j));
        self.cache_hints.borrow_mut().insert(key, Rc::clone(&result));

        result
    }

    fn statistics(&self) -> HashMap<String, Box<dyn Any>> {
        HashMap::new()
    }
}
