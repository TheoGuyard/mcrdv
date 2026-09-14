//! Transfer layer based on an analytic Q-law estimate.

use std::cell::RefCell;
use std::rc::Rc;

use crate::orbit::MeeState;
use crate::problem::Problem;
use crate::solver::TransferLayer;

/// Transfer layer built on the Q-law estimate.
#[derive(Debug, Clone)]
pub struct QlawTransfer {
    /// Thrust acceleration of a chaser \[m/s^2\].
    pub thrust: f64,
    /// Dimensionless factor scaling the estimate.
    pub factor: f64,
    /// Departure times sampled per leg when bounding or hinting.
    pub num_samples: usize,
    /// Most hints returned for one leg.
    pub num_hints: usize,
    /// Cost factor for the transfer time.
    pub cost_factor_time: f64,
    /// Cost factor for the transfer fuel.
    pub cost_factor_fuel: f64,

    problem: Option<Rc<Problem>>,
    lb_cache: RefCell<Vec<Option<(f64, f64)>>>,
}

impl QlawTransfer {
    /// Build a transfer layer with default options.
    pub fn new() -> Self {
        Self::with_options(3e-3, 1.5, 512, 16, 0.0, 1.0)
    }

    /// Build a transfer layer.
    pub fn with_options(
        thrust: f64, 
        factor: f64, 
        num_samples: usize, 
        num_hints: usize,
        cost_factor_time: f64,
        cost_factor_fuel: f64,
    ) -> Self {
        assert!(thrust > 0.0, "thrust must be positive");
        assert!(factor > 0.0, "factor must be positive");
        assert!(num_samples >= 1, "num_samples must be at least 1");
        assert!(cost_factor_time >= 0.0, "cost_factor_time must be non-negative");
        assert!(cost_factor_fuel >= 0.0, "cost_factor_fuel must be non-negative");
        Self {
            thrust,
            factor,
            num_samples,
            num_hints,
            cost_factor_time,
            cost_factor_fuel,
            problem: None,
            lb_cache: RefCell::new(Vec::new()),
        }
    }

    /// The problem instance.
    fn problem(&self) -> &Problem {
        self.problem
            .as_ref()
            .expect("transfer layer used before initialize()")
    }

    /// The `k`-th of `num_samples` departure times spanning `[0, t]`.
    #[inline]
    fn sample_time(&self, k: usize, t: f64) -> f64 {
        if self.num_samples <= 1 {
            0.0
        } else {
            t * (k as f64) / ((self.num_samples - 1) as f64)
        }
    }

    /// The cost profile of a leg, sampled over `[0, t]`.
    fn profile(&self, i: usize, j: usize, t: f64) -> Vec<f64> {
        (0..self.num_samples)
            .map(|k| self.evaluate(i, j, self.sample_time(k, t)).1)
            .collect()
    }

    /// The cheapest sampled cost of a leg over `[0, t]`.
    fn sampled_min(&self, i: usize, j: usize, t: f64) -> f64 {
        (0..self.num_samples)
            .map(|k| self.evaluate(i, j, self.sample_time(k, t)).1)
            .fold(f64::INFINITY, f64::min)
    }
}

impl Default for QlawTransfer {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferLayer for QlawTransfer {
    fn initialize(&mut self, problem: &Rc<Problem>) {
        let n = problem.states.len();
        self.problem = Some(Rc::clone(problem));
        self.lb_cache.replace(vec![None; n * n]);
    }

    fn evaluate(&self, i: usize, j: usize, t: f64) -> (f64, f64) {
        let states = &self.problem().states;
        let src = MeeState::from_kep(&states[i].propagate(t));
        let dst = MeeState::from_kep(&states[j].propagate(t));
        let rts = src.max_rates();

        let da = (src.a - dst.a) / rts.a;
        let df = (src.f - dst.f) / rts.f;
        let dg = (src.g - dst.g) / rts.g;
        let dh = (src.h - dst.h) / rts.h;
        let dk = (src.k - dst.k) / rts.k;

        let dv = self.factor * (
            da * da + 
            df * df + 
            dg * dg + 
            dh * dh + 
            dk * dk
        ).sqrt();
        let dt = dv / self.thrust;
        let dc = self.cost_factor_time * dt + self.cost_factor_fuel * dv;

        (dt, dc)
    }

    fn evaluate_lb(&self, i: usize, j: usize, t: f64) -> (f64, f64) {
        let problem = self.problem();
        if t != problem.max_time {
            let best = self.sampled_min(i, j, t);
            return (best / self.thrust, best);
        }

        let n = problem.states.len();
        let slot = i * n + j;
        if let Some(hit) = self.lb_cache.borrow()[slot] {
            return hit;
        }
        let best = self.sampled_min(i, j, t);
        let value = (best / self.thrust, best);
        self.lb_cache.borrow_mut()[slot] = Some(value);
        value
    }

    fn hints(&self, i: usize, j: usize, t: f64) -> Vec<f64> {
        let costs = self.profile(i, j, t);
        let m = costs.len();

        // Fewer samples than hints asked for: every sample is a hint. A lone
        // sample stops here too, having no neighbour to be compared against.
        if m <= self.num_hints || m < 2 {
            return (0..m.min(self.num_hints)).map(|k| self.sample_time(k, t)).collect();
        }

        // Strict interior local minima of the sampled profile.
        let mut keep: Vec<usize> = (1..m - 1)
            .filter(|&k| costs[k] < costs[k - 1] && costs[k] <= costs[k + 1])
            .collect();

        // A profile that only decreases towards an endpoint is minimal there.
        if costs[0] <= costs[1] {
            keep.push(0);
        }
        if costs[m - 1] < costs[m - 2] {
            keep.push(m - 1);
        }
        keep.sort_unstable();

        // Too many windows: keep the cheapest, then restore time order.
        if keep.len() > self.num_hints {
            keep.sort_by(|&a, &b| costs[a].total_cmp(&costs[b]));
            keep.truncate(self.num_hints);
            keep.sort_unstable();
        }
        keep.into_iter().map(|k| self.sample_time(k, t)).collect()
    }
}
