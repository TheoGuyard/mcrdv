use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::inner::{InnerLoop, InnerOutput};
use crate::orbit::{RE, MU, wrap_pi, State};
use crate::problem::Problem;

/// Below this differential nodal-precession rate [rad/s], the relative RAAN is
/// treated as frozen, so there are no alignment epochs to wait for.
const RATE_EPS: f64 = 1e-12;


/// Non-dimensionalized Keplerian state
#[derive(Debug, Clone, Copy)]
struct NdimState {
    a: f64,
    e: f64,
    i: f64,
    r: f64,
    o: f64,
    #[allow(dead_code)]
    t: f64,
}


/// Non-dimensionalization scaling system
struct Scalings {
    /// Length unit [m]
    lu: f64,
    /// Time unit [s]
    tu: f64,
    /// Velocity unit [m/s]
    vu: f64,
}

impl Scalings {
    fn new() -> Self {
        let lu = RE;
        let tu = (lu.powi(3) / MU).sqrt();
        let vu = lu / tu;
        Self { lu, tu, vu }
    }

    #[inline]
    fn to_ndim_state(&self, s: &State) -> NdimState {
        NdimState {
            a: s.a / self.lu,
            e: s.e,
            i: s.i,
            r: s.r,
            o: s.o,
            t: s.t,
        }
    }

    #[inline]
    fn to_ndim_thrust(&self, thrust: f64) -> f64 {
        thrust * self.tu * self.tu / self.lu
    }

    #[inline]
    fn to_dim_time(&self, time_ndim: f64) -> f64 {
        time_ndim * self.tu
    }

    #[inline]
    fn to_dim_fuel(&self, fuel_ndim: f64) -> f64 {
        fuel_ndim * self.vu
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
            max_thrust: 3.0e-3,
            lb_steps: 8,
        }
    }
}


/// Qlaw implementation for the inner loop
pub struct Qlaw {
    /// Parameters
    params: QlawParams,
    /// Non-dimensionalization scaling system
    scalings: Scalings,
    /// Memoization table for `solve` calls
    memo: RefCell<HashMap<QlawKey, InnerOutput>>,
    /// Memoization table for `solve_lb` calls
    memo_lb: RefCell<HashMap<(usize, usize), InnerOutput>>,
}

impl Qlaw {
    pub fn new(params: &QlawParams) -> Self {
        Self {
            params: params.clone(),
            scalings: Scalings::new(),
            memo: RefCell::new(HashMap::new()),
            memo_lb: RefCell::new(HashMap::new()),
        }
    }

    /// Maximum drift rates of the non-dimensionalized Keplerian state
    #[inline]
    fn oexx_by_accel(&self, s: &NdimState) -> NdimState {
        let p = s.a * (1.0 - s.e.powi(2));
        let h = p.sqrt();

        let axx = 2.0 * (s.a.powi(3) * (1.0 + s.e) / (1.0 - s.e)).sqrt();
        let exx = 2.0 * p / h;
        let ixx = p / (h * ((1.0 - s.e.powi(2) * s.o.sin().powi(2)).sqrt() - s.e * s.o.cos().abs()));
        let rxx = p / (h * s.i.sin() * ((1.0 - s.e.powi(2) * s.o.cos().powi(2)).sqrt() - s.e * s.o.sin().abs()));

        NdimState { a: axx, e: exx, i: ixx, r: rxx, o: f64::NAN, t: f64::NAN }
    }

    /// Q-law fuel estimate
    #[inline]
    fn fuel_qlaw(&self, src: &NdimState, dst: &NdimState) -> f64 {
        let d_a = src.a - dst.a;
        let d_e = src.e - dst.e;
        let d_i = wrap_pi(src.i - dst.i);
        let d_r = wrap_pi(src.r - dst.r);

        let oexx = self.oexx_by_accel(src);

        let sqnorm = (d_a / oexx.a).powi(2)
            + (d_e / oexx.e).powi(2)
            + (d_i / oexx.i).powi(2)
            + (d_r / oexx.r).powi(2);

        1.5 * sqnorm.sqrt()
    }
}

impl InnerLoop for Qlaw {
    fn solve(&self, src: usize, dst: usize, dt: f64, problem: &Problem) -> InnerOutput {

        // Check if result already exists in memoization table
        let key = QlawKey {
            src,
            dst,
            dt: dt.to_bits(),
        };
        {
            let memo = self.memo.borrow();
            if let Some(out) = memo.get(&key) {
                return *out;
            }
        }

        // Resolve indices to states
        let home = problem.num_debris();
        let src_state = if src == home { &problem.chasers } else { &problem.debris[src] };
        let dst_state = if dst == home { &problem.chasers } else { &problem.debris[dst] };

        // Propagate states to the departure time
        let src_dt = src_state.propagate(dt);
        let dst_dt = dst_state.propagate(dt);

        // Non-dimensionalize states
        let src_ndim = self.scalings.to_ndim_state(&src_dt);
        let dst_ndim = self.scalings.to_ndim_state(&dst_dt);

        // Estimate fuel and time based on Q-law
        let fuel_ndim = self.fuel_qlaw(&src_ndim, &dst_ndim);
        let fmax_ndim = self.scalings.to_ndim_thrust(self.params.max_thrust);
        let time_ndim = fuel_ndim / fmax_ndim;

        // Build output
        let out = InnerOutput {
            time: self.scalings.to_dim_time(time_ndim),
            fuel: self.scalings.to_dim_fuel(fuel_ndim),
        };

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

        // Departure hints based on small coarse scan
        let mut epochs = self.departure_hints(src, dst, horizon, problem);
        epochs.push(horizon);
        for j in 1..self.params.lb_steps {
            epochs.push(j as f64 / self.params.lb_steps as f64 * horizon);
        }

        // Minimum-fuel transfer departure hints
        let mut best = self.solve(src, dst, 0.0, problem);
        for t in epochs {
            let out = self.solve(src, dst, t, problem);
            if out.fuel < best.fuel {
                best = out;
            }
        }

        // Memoize the result
        self.memo_lb.borrow_mut().insert((src, dst), best);

        best
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
