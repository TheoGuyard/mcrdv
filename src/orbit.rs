use std::error::Error;
use std::f64::consts::{PI, TAU};

use serde::{Deserialize, Serialize};

/// Earth gravitational parameter [m^3/s^2]
pub const MU: f64 = 3.986_004e14;

/// Earth second zonal harmonic [dimensionless]
pub const J2: f64 = 1.082_626e-3;

/// Earth equatorial radius [m]
pub const RE: f64 = 6.378_137e6;


/// Orbital state in Keplerian elements
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct State {
    /// Semi-major axis [m]
    pub a: f64,
    /// Eccentricity [proportion]
    pub e: f64,
    /// Inclination [rad]
    pub i: f64,
    /// Right ascension of the ascending node [rad]
    pub r: f64,
    /// Argument of perigee [rad]
    pub o: f64,
    /// True anomaly [rad]
    pub t: f64,
}

impl State {
    /// J2 secular drift rates in RAAN and argument of perigee [rad/s]
    pub fn j2_secular_rates(&self) -> (f64, f64) {
        let n = (MU / self.a.powi(3)).sqrt();
        let p = self.a * (1.0 - self.e * self.e);
        let f = 1.5 * J2 * (RE / p).powi(2) * n;
        let r_dt = -f * self.i.cos();
        let o_dt = f * (2.0 - 2.5 * self.i.sin().powi(2));
        (r_dt, o_dt)
    }

    /// Propagate state forward in time by `dt` seconds
    pub fn propagate(&self, dt: f64) -> State {
        let (r_dt, o_dt) = self.j2_secular_rates();
        State {
            r: self.r + r_dt * dt,
            o: self.o + o_dt * dt,
            ..*self
        }
    }
}


/// Wrap angle `x` to `[-π, π)`
pub fn wrap_pi(x: f64) -> f64 {
    let t = x.rem_euclid(TAU);
    if t > PI {
        t - TAU
    } else {
        t
    }
}

/// Circular mean of angles wrapped to `[0, 2π)`
fn circular_mean(angles: impl Iterator<Item = f64>) -> f64 {
    let (s, c) = angles.fold((0.0, 0.0), |(s, c), a| (s + a.sin(), c + a.cos()));
    let mean = s.atan2(c);
    if mean < 0.0 {
        mean + TAU
    } else {
        mean
    }
}

/// Centroid of orbital states
pub fn centroid(states: &[State]) -> Result<State, Box<dyn Error>> {
    if states.is_empty() {
        return Err("cannot compute centroid of empty state list".into());
    }
    let n = states.len() as f64;
    let a = states.iter().map(|s| s.a).sum::<f64>() / n;
    let e = states.iter().map(|s| s.e).sum::<f64>() / n;
    let i = states.iter().map(|s| s.i).sum::<f64>() / n;
    let r = circular_mean(states.iter().map(|s| s.r));
    let o = circular_mean(states.iter().map(|s| s.o));
    let t = circular_mean(states.iter().map(|s| s.t));
    Ok(State { a, e, i, r, o, t })
}