//! Orbital state representation and geometric utilities.

use std::f64::consts::PI;

// ========================================================================= //
// Constants
// ========================================================================= //

/// Full geometric turn \[rad\].
pub const TAU: f64 = 2.0 * PI;
/// One day \[s\].
pub const DAY: f64 = 86_400.0;
/// Earth gravitational parameter \[m^3/s^2\].
pub const MU: f64 = 3.986_004e14;
/// Earth equatorial radius \[m\].
pub const RE: f64 = 6.378_137e6;
/// Earth second zonal harmonic \[-\].
pub const J2: f64 = 1.082_626e-3;

// ========================================================================= //
// Orbital state
// ========================================================================= //

/// Orbital state in Keplerian elements at a reference time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KepState {
    /// Semi-major axis \[m\].
    pub a: f64,
    /// Eccentricity \[-\].
    pub e: f64,
    /// Inclination \[rad\].
    pub i: f64,
    /// Right ascension of the ascending node \[rad\].
    pub r: f64,
    /// Argument of periapsis \[rad\].
    pub o: f64,
    /// True anomaly \[rad\].
    pub t: f64,
    /// Secular drift rate of the right ascension of the ascending node \[rad/s\].
    pub dr_: f64,
    /// Secular drift rate of the argument of periapsis \[rad/s\].
    pub do_: f64,
}

impl KepState {
    /// Build a state from Keplerian elements.
    pub fn new(a: f64, e: f64, i: f64, r: f64, o: f64, t: f64) -> Self {
        let n = (MU / (a * a * a)).sqrt();
        let p = a * (1.0 - e * e);
        let f = 1.5 * J2 * (RE / p).powi(2) * n;
        let dr_ = -f * i.cos();
        let do_ = f * (2.0 - 2.5 * i.sin().powi(2));
        Self {a, e, i, r, o, t, dr_, do_}
    }

    /// State elements as an array `[a, e, i, r, o, t]`.
    pub fn as_array(&self) -> [f64; 6] {
        [self.a, self.e, self.i, self.r, self.o, self.t]
    }

    /// Advance the state by `dt` seconds according to J2 secular drift.
    #[inline]
    pub fn propagate(&self, dt: f64) -> Self {
        Self {
            r: self.r + self.dr_ * dt,
            o: self.o + self.do_ * dt,
            ..*self
        }
    }
}

/// Orbital state in modified equinoctial elements at reference time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeeState {
    /// Semi-major axis \[m\].
    pub a: f64,
    /// `e * cos(o + r)` \[-\].
    pub f: f64,
    /// `e * sin(o + r)` \[-\].
    pub g: f64,
    /// `tan(i / 2) * cos(r)` \[-\].
    pub h: f64,
    /// `tan(i / 2) * sin(r)` \[-\].
    pub k: f64,
    /// True longitude `r + o + t` \[rad\].
    pub l: f64,
}

impl MeeState {
    /// Convert from Keplerian elements.
    #[inline]
    pub fn from_kep(kep: &KepState) -> Self {
        let w = kep.o + kep.r;
        let t = (0.5 * kep.i).tan();
        Self {
            a: kep.a,
            f: kep.e * w.cos(),
            g: kep.e * w.sin(),
            h: t * kep.r.cos(),
            k: t * kep.r.sin(),
            l: w + kep.t,
        }
    }

    /// The fastest rate of change of each element, per thrust unit.
    ///
    /// The result is packed into a `MeeState` for convenience, but its fields
    /// are *rates*, not elements: multiplying them by a thrust acceleration
    /// \[m/s^2\] gives the maximum achievable rate of the matching element.
    #[inline]
    pub fn max_rates(&self) -> Self {
        let v = self.f * self.f + self.g * self.g;
        let e = v.sqrt();
        let p = self.a * (1.0 - v);
        let q = (p / MU).sqrt();
        let s = 1.0 + self.h * self.h + self.k * self.k;

        let da = 2.0 * ((self.a.powi(3) / MU) * (1.0 + e) / (1.0 - e)).sqrt();
        let df = 2.0 * q;
        let dg = 2.0 * q;
        let dh = 0.5 * q * s / (self.f + (1.0 - self.g * self.g).sqrt());
        let dk = 0.5 * q * s / (self.g + (1.0 - self.f * self.f).sqrt());
        let dl = q * (self.h * self.l.sin() - self.k * self.l.cos())
            / (1.0 - v).sqrt()
            + (MU / self.a.powi(3)).sqrt();

        Self {
            a: da,
            f: df,
            g: dg,
            h: dh,
            k: dk,
            l: dl,
        }
    }
}

// ========================================================================= //
// Geometric utilities
// ========================================================================= //

/// Circular mean of angles, wrapped to `[0, 2*pi)`.
pub fn circular_mean(angles: &[f64]) -> f64 {
    let (s, c) = angles
        .iter()
        .fold((0.0, 0.0), |(s, c), a| (s + a.sin(), c + a.cos()));
    let mean = s.atan2(c);
    if mean < 0.0 { mean + TAU } else { mean }
}

/// State at the centroid of a cluster.
pub fn centroid(states: &[KepState]) -> KepState {
    let n = states.len() as f64;
    let mean_values = |f: fn(&KepState) -> f64| {
        states.iter().map(f).sum::<f64>() / n
    };
    let mean_angles = |f: fn(&KepState) -> f64| {
        circular_mean(&states.iter().map(f).collect::<Vec<_>>())
    };
    KepState::new(
        mean_values(|s| s.a),
        mean_values(|s| s.e),
        mean_values(|s| s.i),
        mean_angles(|s| s.r),
        mean_angles(|s| s.o),
        mean_angles(|s| s.t),
    )
}
