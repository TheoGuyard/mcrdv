use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::inner::{InnerLoop, InnerOutput};
use crate::orbit::{MU, State};
use crate::problem::Problem;

/// Differential threshold below which relative RAAN is considered frozen.
const RATE_EPS: f64 = 1e-12;


/// Modified equinoctial state `(a, f, g, h, k, l)`.
#[derive(Debug, Clone, Copy)]
struct MeeState {
    a: f64,
    f: f64,
    g: f64,
    h: f64,
    k: f64,
    /// True longitude. Carried for completeness; not used by the estimator.
    #[allow(dead_code)]
    l: f64,
}

impl MeeState {
    /// Build from a Keplerian state
    fn from_kep(s: &State) -> MeeState {
        let tan_half_i = (0.5 * s.i).tan();
        MeeState {
            a: s.a,
            f: s.e * (s.o + s.r).cos(),
            g: s.e * (s.o + s.r).sin(),
            h: tan_half_i * s.r.cos(),
            k: tan_half_i * s.r.sin(),
            l: s.r + s.o + s.t,
        }
    }

    /// Maximum drift rate of each modified equinoctial element per thrust unit
    #[inline]
    fn max_rates(&self) -> MeeState {
        let e2 = self.f * self.f + self.g * self.g;
        let e = e2.sqrt();
        let p = self.a * (1.0 - e2);
        let q = (p / MU).sqrt();
        let r = 1.0 + self.h * self.h + self.k * self.k;

        MeeState {
            a: 2.0 * ((self.a.powi(3) / MU) * (1.0 + e) / (1.0 - e)).sqrt(),
            f: 2.0 * q,
            g: 2.0 * q,
            h: 0.5 * q * r / ((1.0 - self.g * self.g).sqrt() + self.f),
            k: 0.5 * q * r / ((1.0 - self.f * self.f).sqrt() + self.g),
            l: q * (self.h * self.l.sin() - self.k * self.l.cos()) / (1.0 - e2).sqrt() + (MU / self.a.powi(3)).sqrt()
        }
    }
}


/// Qlaw memoization key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct QlawKey {
    src: usize,
    dst: usize,
    dt: u64,
}


/// Qlaw parameters
#[derive(Debug, Clone)]
pub struct QlawParams {
    /// Maximum chaser thrust [m/s^2]
    pub max_thrust: f64,
    /// Number of steps for lower bound scan
    pub lb_steps: usize,
}

impl Default for QlawParams {
    fn default() -> Self {
        Self {
            max_thrust: 0.003,
            lb_steps: 8,
        }
    }
}


/// Qlaw implementation for the inner loop
pub struct Qlaw {
    /// Parameters
    params: QlawParams,
    /// Memoization table for `solve` calls
    memo: RefCell<HashMap<QlawKey, InnerOutput>>,
    /// Memoization table for `solve_lb` calls
    memo_lb: RefCell<HashMap<(usize, usize), InnerOutput>>,
}

impl Qlaw {
    pub fn new(params: &QlawParams) -> Self {
        Self {
            params: params.clone(),
            memo: RefCell::new(HashMap::new()),
            memo_lb: RefCell::new(HashMap::new()),
        }
    }

    /// Q-law propellant estimate between two modified equinoctial states
    #[inline]
    fn fuel_qlaw(&self, src: &MeeState, dst: &MeeState) -> f64 {
        let rates = src.max_rates();

        let sqnorm = ((src.a - dst.a) / rates.a).powi(2)
            + ((src.f - dst.f) / rates.f).powi(2)
            + ((src.g - dst.g) / rates.g).powi(2)
            + ((src.h - dst.h) / rates.h).powi(2)
            + ((src.k - dst.k) / rates.k).powi(2);

        1.5 * sqnorm.sqrt()
    }
}

impl InnerLoop for Qlaw {
    fn solve(&self, src: usize, dst: usize, dt: f64, problem: &Problem) -> InnerOutput {

        // Check if result already exists in memoization table
        let key = QlawKey { src, dst, dt: dt.to_bits() };
        {
            let memo = self.memo.borrow();
            if let Some(out) = memo.get(&key) { return *out; }
        }

        // Resolve indices to states
        let home = problem.num_debris();
        let src_state = if src == home { &problem.chasers } else { &problem.debris[src] };
        let dst_state = if dst == home { &problem.chasers } else { &problem.debris[dst] };

        // Propagate states to the departure time
        let src_dt = src_state.propagate(dt);
        let dst_dt = dst_state.propagate(dt);

        // Convert states to modified equinoctial elements
        let src_mee = MeeState::from_kep(&src_dt);
        let dst_mee = MeeState::from_kep(&dst_dt);

        // Estimate fuel and time based on Q-law
        let fuel = self.fuel_qlaw(&src_mee, &dst_mee);
        let time = fuel / self.params.max_thrust;

        // Build output
        let out = InnerOutput { time, fuel };

        // Memoize the result
        self.memo.borrow_mut().insert(key, out);

        out
    }

    fn solve_lb(&self, src: usize, dst: usize, problem: &Problem) -> InnerOutput {
        
        // Check if result already exists in memoization table
        if let Some(out) = self.memo_lb.borrow().get(&(src, dst)) {
            return *out;
        }

        let horizon = problem.max_time;

        // Candidate departure times: RAAN-alignment hints plus a coarse scan
        let mut epochs = self.departure_hints(src, dst, horizon, problem);
        epochs.push(horizon);
        for j in 1..self.params.lb_steps {
            epochs.push(j as f64 / self.params.lb_steps as f64 * horizon);
        }

        // Minimize time and fuel over the scanned departures
        let init = self.solve(src, dst, 0.0, problem);
        let mut min_time = init.time;
        let mut min_fuel = init.fuel;
        for t in epochs {
            let out = self.solve(src, dst, t, problem);
            if out.time < min_time {
                min_time = out.time;
            }
            if out.fuel < min_fuel {
                min_fuel = out.fuel;
            }
        }
        let lb = InnerOutput { time: min_time, fuel: min_fuel };

        // Memoize the result
        self.memo_lb.borrow_mut().insert((src, dst), lb);

        lb
    }

    /// Analytic departure hints based on the relative-RAAN alignment epochs
    fn departure_hints(&self, src: usize, dst: usize, horizon: f64, problem: &Problem) -> Vec<f64> {
        
        // Resolve indices to states
        let home = problem.num_debris();
        let src_state = if src == home { &problem.chasers } else { &problem.debris[src] };
        let dst_state = if dst == home { &problem.chasers } else { &problem.debris[dst] };

        // Differential nodal precession rate (time-invariant under J2 secular)
        let (r_dot_src, _) = src_state.j2_secular_rates();
        let (r_dot_dst, _) = dst_state.j2_secular_rates();
        let d_rate = r_dot_src - r_dot_dst;

        // If relative RAAN is frozen, waiting cannot realign the states
        if d_rate.abs() < RATE_EPS { return Vec::new(); }

        // RAAN alignment lattice
        let d_raan = src_state.r - dst_state.r;
        let period = TAU / d_rate.abs();
        let base = -d_raan / d_rate;

        // Hints evaluation
        let mut hints = Vec::new();
        let mut t = base.rem_euclid(period);
        while t <= horizon {
            hints.push(t);
            t += period;
        }

        hints
    }
}
