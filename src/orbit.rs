//! Orbital states representation and utilities.

use std::f64::consts::{PI, TAU};

use serde::{Deserialize, Serialize};

// ========================================================================= //
// Constants
// ========================================================================= //

/// One day [s].
pub const DAY: f64 = 86_400.0;
/// Earth gravitational parameter [m^3/s^2].
pub const MU: f64 = 3.986_004e14;
/// Earth equatorial radius [m].
pub const RE: f64 = 6.378_137e6;
/// Earth second zonal harmonic [-].
pub const J2: f64 = 1.082_626e-3;

// ========================================================================= //
// Geometric utilities
// ========================================================================= //

/// Wrap an angle to `[-pi, pi)`.
pub fn wrap_pi(x: f64) -> f64 {
    let t = x.rem_euclid(TAU);
    if t > PI {
        t - TAU
    } else {
        t
    }
}

/// Mean of an array of angles wrapped to `[0, 2*pi)`.
pub fn circular_mean(x: &[f64]) -> f64 {
    let s: f64 = x.iter().map(|a| a.sin()).sum();
    let c: f64 = x.iter().map(|a| a.cos()).sum();
    let mean = s.atan2(c);
    if mean < 0.0 {
        mean + TAU
    } else {
        mean
    }
}

// ========================================================================= //
// Orbital state representation
// ========================================================================= //

/// Orbital state in Keplerian elements.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KepState {
    /// Semi-major axis [m].
    pub a: f64,
    /// Eccentricity [-].
    pub e: f64,
    /// Inclination [rad].
    pub i: f64,
    /// Right ascension of the ascending node [rad].
    pub r: f64,
    /// Argument of periapsis [rad].
    pub o: f64,
    /// True anomaly [rad].
    pub t: f64,
}

impl KepState {
    /// Construct a Keplerian state from its six elements.
    pub fn new(a: f64, e: f64, i: f64, r: f64, o: f64, t: f64) -> Self {
        Self { a, e, i, r, o, t }
    }

    /// Secular J2 drift rates `(r_dot, o_dot)` of the right ascension and
    /// argument of periapsis [rad/s]. Time-invariant under the secular model.
    pub fn j2_secular_rates(&self) -> (f64, f64) {
        let m = (MU / self.a.powi(3)).sqrt();
        let p = self.a * (1.0 - self.e * self.e);
        let f = 1.5 * J2 * (RE / p).powi(2) * m;
        let r_dot = -f * self.i.cos();
        let o_dot = f * (2.0 - 2.5 * self.i.sin().powi(2));
        (r_dot, o_dot)
    }

    /// Propagate the state forward in time by `dt` seconds under secular J2
    /// precession.
    pub fn propagate(&self, dt: f64) -> KepState {
        let (r_dot, o_dot) = self.j2_secular_rates();
        KepState {
            r: self.r + r_dot * dt,
            o: self.o + o_dot * dt,
            ..*self
        }
    }
}

/// Orbital state in modified equinoctial elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeeState {
    /// Semi-major axis [m].
    pub a: f64,
    /// `e * cos(o + r)` [-].
    pub f: f64,
    /// `e * sin(o + r)` [-].
    pub g: f64,
    /// `tan(i/2) * cos(r)` [-].
    pub h: f64,
    /// `tan(i/2) * sin(r)` [-].
    pub k: f64,
    /// True longitude `r + o + t` [rad].
    pub l: f64,
}

impl MeeState {
    /// Convert a Keplerian state to modified equinoctial elements.
    pub fn from_kep(s: &KepState) -> MeeState {
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

    /// Maximal drift rate of each element per unit of thrust acceleration.
    pub fn max_rates(&self) -> MeeState {
        let e2 = self.f * self.f + self.g * self.g;
        let e = e2.sqrt();
        let p = self.a * (1.0 - e2);
        let q = (p / MU).sqrt();
        let s2 = 1.0 + self.h * self.h + self.k * self.k;
        MeeState {
            a: 2.0 * ((self.a.powi(3) / MU) * (1.0 + e) / (1.0 - e)).sqrt(),
            f: 2.0 * q,
            g: 2.0 * q,
            h: 0.5 * q * s2 / ((1.0 - self.g * self.g).sqrt() + self.f),
            k: 0.5 * q * s2 / ((1.0 - self.f * self.f).sqrt() + self.g),
            l: q * (self.h * self.l.sin() - self.k * self.l.cos()) / (1.0 - e2).sqrt()
                + (MU / self.a.powi(3)).sqrt(),
        }
    }
}

// ========================================================================= //
// Catalog of orbital states
// ========================================================================= //

/// Catalog of orbital states at a common epoch.
///
/// The six Keplerian elements are stored column-wise so that the catalog maps
/// naturally onto the rows of a debris csv file. When `has_depot` is set, the
/// state at index `0` is the swarm depot and indices `1..n` are the debris.
#[derive(Debug, Clone, PartialEq)]
pub struct Catalog {
    pub a: Vec<f64>,
    pub e: Vec<f64>,
    pub i: Vec<f64>,
    pub r: Vec<f64>,
    pub o: Vec<f64>,
    pub t: Vec<f64>,
    /// Whether index `0` is a depot rather than a debris.
    pub has_depot: bool,
    /// Total number of states (depot included if any).
    pub n: usize,
}

impl Catalog {
    /// Build a catalog from the per-element columns (no depot by default).
    pub fn new(a: Vec<f64>, e: Vec<f64>, i: Vec<f64>, r: Vec<f64>, o: Vec<f64>, t: Vec<f64>) -> Self {
        let n = a.len();
        assert!(
            e.len() == n && 
            i.len() == n && 
            r.len() == n && 
            o.len() == n && 
            t.len() == n,
            "catalog columns must have equal length"
        );
        Self {
            a,
            e,
            i,
            r,
            o,
            t,
            has_depot: false,
            n,
        }
    }

    /// Number of states in the catalog.
    pub fn len(&self) -> usize { self.n }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool { self.n == 0 }

    /// Number of depots (`0` or `1`).
    pub fn num_depots(&self) -> usize { usize::from(self.has_depot) }

    /// Number of debris (all states excluding the depot).
    pub fn num_debris(&self) -> usize { self.n - self.num_depots() }

    /// State at catalog index `idx` as a `KepState` struct.
    pub fn state(&self, idx: usize) -> KepState {
        KepState {
            a: self.a[idx],
            e: self.e[idx],
            i: self.i[idx],
            r: self.r[idx],
            o: self.o[idx],
            t: self.t[idx],
        }
    }

    /// States as a vector of `KepState` structs.
    pub fn states(&self) -> Vec<KepState> {
        (0..self.n).map(|idx| self.state(idx)).collect()
    }

    /// Indices of the depots.
    pub fn depots_indices(&self) -> Vec<usize> {
        if self.has_depot {
            vec![0]
        } else {
            vec![]
        }
    }

    /// Indices of the debris.
    pub fn debris_indices(&self) -> Vec<usize> {
        (self.num_depots()..self.n).collect()
    }

    /// Centroid of the debris states.
    pub fn centroid_debris(&self) -> KepState {
        let idx = self.debris_indices();
        let mean = |sel: &dyn Fn(&KepState) -> f64| -> f64 {
            idx.iter().map(|&k| sel(&self.state(k))).sum::<f64>() / idx.len() as f64
        };
        let circ = |sel: &dyn Fn(&KepState) -> f64| -> f64 {
            let vals: Vec<f64> = idx.iter().map(|&k| sel(&self.state(k))).collect();
            circular_mean(&vals)
        };
        KepState {
            a: mean(&|s| s.a),
            e: mean(&|s| s.e),
            i: mean(&|s| s.i),
            r: circ(&|s| s.r),
            o: circ(&|s| s.o),
            t: circ(&|s| s.t),
        }
    }

    /// Prepend the debris centroid as a depot at index `0`.
    pub fn add_depot(&mut self) {
        assert!(!self.has_depot, "catalog already has a depot");
        let depot = self.centroid_debris();
        self.a.insert(0, depot.a);
        self.e.insert(0, depot.e);
        self.i.insert(0, depot.i);
        self.r.insert(0, depot.r);
        self.o.insert(0, depot.o);
        self.t.insert(0, depot.t);
        self.has_depot = true;
        self.n += 1;
    }
}
