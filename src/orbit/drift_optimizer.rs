pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

// This crate provides functions for optimizing low-thrust orbital transfers.
// It is a Rust implementation of the Python code found in
// `scripts/compact_cost_function/inner.py`.

use nalgebra::{SVector, Vector6};
use roots::{find_root_newton_raphson, SimpleConvergency};
use std::f64::consts::PI;

use crate::orbit::constants::{J2, MU, RE};
use crate::orbit::State;

/// A scaling system for non-dimensionalization.
pub struct Scalings {
    /// Length unit [m].
    pub lu: f64,
    /// Time unit [s].
    pub tu: f64,
    /// Velocity unit [m/s].
    pub vu: f64,
}

impl Scalings {
    /// Create a new scaling system.
    pub fn new() -> Self {
        let lu = RE;
        let tu = (lu.powi(3) / MU).sqrt();
        let vu = lu / tu;
        Self { lu, tu, vu }
    }

    /// Convert a dimensional state to a non-dimensional Keplerian state.
    pub fn to_nondim_state(&self, state: &State) -> NonDimKep {
        NonDimKep::new(
            state.a / self.lu,
            state.e,
            state.i,
            state.O,
            state.o,
            state.t,
        )
    }
}

/// Keplerian orbital elements (a, e, i, raan, aop, ta).
/// (semimajor axis, eccentricity, inclination, right ascension of the ascending node,
/// argument of periapsis, true anomaly)
pub type NonDimKep = SVector<f64, 6>;

/// Wraps an angle to be in the range [-pi, pi].
///
/// # Arguments
///
/// * `x` - Angle in radians.
///
/// # Returns
///
/// Wrapped angle in radians.
pub fn wrap_angle(x: f64) -> f64 {
    (x + PI).rem_euclid(2.0 * PI) - PI
}

/// Shortest phase difference between two angles, i.e. how much `a` is ahead of `b`.
/// Returns in range [-pi, pi].
///
/// Source: https://stackoverflow.com/a/2007279
///
/// # Arguments
///
/// * `a` - First angle.
/// * `b` - Second angle.
///
/// # Returns
///
/// Shortest phase difference between the two angles.
pub fn delta_angle(a: f64, b: f64) -> f64 {
    wrap_angle(a - b)
}

/// Given an angle in radians, return the equivalent angle that goes the "other way" around the circle.
/// For example, if the input angle is PI/4, the output will be -7PI/4.
///
/// # Arguments
///
/// * `angle` - Angle in radians.
///
/// # Returns
///
/// Equivalent angle going the other way around the circle.
pub fn go_other_way(angle: f64) -> f64 {
    angle - angle.signum() * 2.0 * PI
}

/// Propagate a state forwards in time, using time-averaged J2 perturbations.
///
/// Accounts for secular variations in RAAN and AOP, but ignores short-period effects
/// including changes in true anomaly.
///
/// Quantities must be nondimensionalized such that mu = 1 and R_EARTH = 1.
///
/// # Arguments
///
/// * `states` - Initial orbital elements in Keplerian elements.
/// * `ts` - Time to propagate forwards in non-dimensional time units.
///
/// # Returns
///
/// Propagated Keplerian state vector.
pub fn raan_j2_prop_vec(state: &NonDimKep, ts: f64) -> NonDimKep {
    let a = state[0];
    let e = state[1];
    let i = state[2];

    let raan_rate = J2 * -3.0 / 2.0 * i.cos() * (1.0 - e.powi(2)).powi(-2) * a.powf(-3.5);
    let aop_rate =
        J2 * 3.0 / 4.0 * (5.0 * i.cos().powi(2) - 1.0) * (1.0 - e.powi(2)).powi(-2) * a.powf(-3.5);

    let mut secular_rates = Vector6::zeros();
    secular_rates[3] = raan_rate;
    secular_rates[4] = aop_rate;

    state + ts * secular_rates
}

/// Compute the maximum rates of change of the orbital elements
/// per unit acceleration using the Q-law formulation.
///
/// # Arguments
///
/// * `xc` - Orbital state in Keplerian elements.
///
/// # Returns
///
/// Orbital element scales.
pub fn oexx_by_f_kep(xc: &NonDimKep) -> Vector6<f64> {
    let (a, e, i, _, argp, _) = (xc[0], xc[1], xc[2], xc[3], xc[4], xc[5]);
    let p = a * (1.0 - e.powi(2));
    let h = p.sqrt();

    let axx = 2.0 * (a.powi(2) * (1.0 + e) / (1.0 - e)).sqrt();
    let exx = 2.0 * p / h;
    let ixx = p / (h * ((1.0 - e.powi(2) * argp.sin().powi(2)).sqrt() - e * argp.cos().abs()));
    let raanx =
        p / (h * i.sin() * ((1.0 - e.powi(2) * argp.cos().powi(2)).sqrt() - e * argp.sin().abs()));

    Vector6::new(axx, exx, ixx, raanx, f64::NAN, f64::NAN)
}

/// Estimate delta-v cost to go between two orbits (defined in Keplerian elements)
/// using the Q-law. Inputs must be non-dimensionalized such that mu = 1.
/// Output is in non-dimensional velocity units.
///
/// # Arguments
///
/// * `xc` - Chaser orbital state in Keplerian elements.
/// * `xt` - Target orbital state in Keplerian elements.
/// * `phi` - Optional parameter for RAAN matching [0, 1]. If None, computes the time to go
///           based on actual elements. Otherwise, computes time to go assuming
///           that only (1-phi) of the RAAN difference needs to be corrected.
///
/// # Returns
///
/// Estimated delta-v cost (non-dimensional velocity units).
pub fn dv_go_qlaw(xc: &NonDimKep, xt: &NonDimKep, phi: Option<f64>) -> f64 {
    let mut dx = xc - xt;
    dx[3] = delta_angle(xc[3], xt[3]);
    dx[2] = delta_angle(xc[2], xt[2]);

    if let Some(phi_val) = phi {
        dx[3] *= 1.0 - phi_val;
    }

    let oexx = oexx_by_f_kep(xc);
    let mut norm_vec = Vector6::zeros();
    for i in 0..4 {
        norm_vec[i] = dx[i] / oexx[i];
    }

    3.0 / 2.0 * norm_vec.fixed_rows::<4>(0).norm()
}

/// Estimate time to go between two orbits (defined in Keplerian elements)
/// using the Q-law. Inputs must be non-dimensionalized such that mu = 1.
/// Output is in non-dimensional time units.
///
/// # Arguments
///
/// * `xc` - Chaser orbital state in Keplerian elements.
/// * `xt` - Target orbital state in Keplerian elements.
/// * `f` - Maximum acceleration of the spacecraft (non-dimensional).
/// * `phi` - Optional parameter for RAAN matching [0, 1].
///
/// # Returns
///
/// Estimated time to go (non-dimensional time units).
pub fn time_to_go_qlaw(xc: &NonDimKep, xt: &NonDimKep, f: f64, phi: Option<f64>) -> f64 {
    let dv = dv_go_qlaw(xc, xt, phi);
    dv / f
}

/// RAAN drift amount due to a circle-to-circle raising or lowering maneuver.
///
/// # Arguments
///
/// * `x` - Orbital state in Keplerian elements.
/// * `f` - Maximum acceleration of the spacecraft (m/s^2).
/// * `a_i` - Initial semi-major axis (m).
/// * `a_f` - Final semi-major axis (m).
///
/// # Returns
///
/// The change in RAAN due to the orbit raise maneuver.
pub fn c2c_raan_change(x: &NonDimKep, f: f64, a_i: f64, a_f: f64) -> f64 {
    let (_, e, i, ..) = (x[0], x[1], x[2]);
    let coeff = -3.0 / 16.0 * J2;
    let difference = (1.0 / a_i.powi(4) - 1.0 / a_f.powi(4)).abs();
    coeff * i.cos() / (f * (1.0 - e.powi(2)).powi(2)) * difference
}

/// Estimate time needed to make drift-based transfer.
///
/// # Arguments
///
/// * `ch_kep` - Chaser orbital state.
/// * `tg_kep` - Target orbital state.
/// * `a_d` - Drift orbit semi-major axis.
/// * `phi` - RAAN matching parameter [0, 1].
/// * `f` - Maximum spacecraft acceleration.
/// * `long_way` - Whether to take the long way around for RAAN matching.
///
/// # Returns
///
/// The estimated time needed for the drift transfer.
pub fn drift_dt(
    ch_kep: &NonDimKep,
    tg_kep: &NonDimKep,
    a_d: f64,
    phi: f64,
    f: f64,
    long_way: bool,
) -> f64 {
    let (a_c, e_c, i_c) = (ch_kep[0], ch_kep[1], ch_kep[2]);
    let (a_t, e_t, i_t) = (tg_kep[0], tg_kep[1], tg_kep[2]);
    let mut initial_draan = delta_angle(ch_kep[3], tg_kep[3]);

    let dr1 = c2c_raan_change(ch_kep, f, a_c, a_d);
    let dr3 = c2c_raan_change(ch_kep, f, a_d, a_t);
    let dt1 = ((1.0 / a_c).sqrt() - (1.0 / a_d).sqrt()).abs() / f;
    let dt3 = ((1.0 / a_t).sqrt() - (1.0 / a_d).sqrt()).abs() / f;
    let mut ch_kep_mod = ch_kep.clone_owned();
    ch_kep_mod[0] = a_t;
    let dt4 = time_to_go_qlaw(&ch_kep_mod, tg_kep, f, Some(phi));

    let ch_rate = -3.0 / 2.0 * J2 * (i_c.cos() / (1.0 - e_c.powi(2)).powi(2) * a_d.powf(-3.5));
    let tg_rate = -3.0 / 2.0 * J2 * (i_t.cos() / (1.0 - e_t.powi(2)).powi(2) * a_t.powf(-3.5));
    let rel_rate = ch_rate - tg_rate;

    let extra_draan = dr1 + dr3 - tg_rate * (dt1 + dt3);

    if long_way {
        initial_draan = go_other_way(initial_draan);
    }

    let draan_to_go = initial_draan * phi + extra_draan;
    let mut dt2 = draan_to_go / -rel_rate;

    if dt2 < 0.0 {
        dt2 = f64::INFINITY;
    }

    dt1 + dt2 + dt3 + dt4
}

/// Estimate total delta-v needed for a drift transfer.
///
/// # Arguments
///
/// * `ch_kep` - Chaser orbital state.
/// * `tg_kep` - Target orbital state.
/// * `a_d` - Drift orbit semi-major axis.
/// * `phi` - RAAN matching parameter [0, 1].
///
/// # Returns
///
/// Estimated total delta-v for the drift transfer.
pub fn drift_dv(ch_kep: &NonDimKep, tg_kep: &NonDimKep, a_d: f64, phi: f64) -> f64 {
    let a_c = ch_kep[0];
    let a_t = tg_kep[0];
    let dv1 = ((1.0 / a_c).sqrt() - (1.0 / a_d).sqrt()).abs();
    let dv3 = ((1.0 / a_t).sqrt() - (1.0 / a_d).sqrt()).abs();
    let mut ch_kep_mod = ch_kep.clone_owned();
    ch_kep_mod[0] = a_t;
    let dv4 = dv_go_qlaw(&ch_kep_mod, tg_kep, Some(phi));
    dv1 + dv3 + dv4
}

/// Compute the drift fraction parameter phi given parameters a_d and T.
///
/// # Arguments
///
/// * `xc` - Chaser orbital state.
/// * `xt` - Target orbital state.
/// * `f` - Maximum spacecraft acceleration.
/// * `a_d` - Estimated drift orbit semi-major axis.
/// * `t_final` - Total time for the transfer.
/// * `long_way` - Whether to take the long way around for RAAN matching.
///
/// # Returns
///
/// RAAN matching parameter phi [0, 1].
pub fn phi_from_ad(
    xc: &NonDimKep,
    xt: &NonDimKep,
    f: f64,
    a_d: f64,
    t_final: f64,
    long_way: bool,
) -> f64 {
    let (a_c, e_c, i_c) = (xc[0], xc[1], xc[2]);
    let (a_t, e_t, i_t) = (xt[0], xt[1], xt[2]);

    let dr1 = c2c_raan_change(xc, f, a_c, a_d);
    let dr3 = c2c_raan_change(xc, f, a_d, a_t);
    let dt1 = ((1.0 / a_c).sqrt() - (1.0 / a_d).sqrt()).abs() / f;
    let dt3 = ((1.0 / a_t).sqrt() - (1.0 / a_d).sqrt()).abs() / f;

    let ch_rate = -3.0 / 2.0 * J2 * (i_c.cos() / (1.0 - e_c.powi(2)).powi(2) * a_d.powf(-3.5));
    let tg_rate = -3.0 / 2.0 * J2 * (i_t.cos() / (1.0 - e_t.powi(2)).powi(2) * a_t.powf(-3.5));
    let rel_rate = ch_rate - tg_rate;

    let extra_draan = dr1 + dr3 - tg_rate * (dt1 + dt3);
    let t_extra = extra_draan / -rel_rate;

    let mut initial_draan = delta_angle(xc[3], xt[3]);
    if long_way {
        initial_draan = go_other_way(initial_draan);
    }

    let k = t_final - dt1 - dt3 - t_extra;
    let a = initial_draan / -rel_rate;
    let mut xc_mod = xc.clone_owned();
    xc_mod[0] = a_t;
    let oexx = oexx_by_f_kep(&xc_mod);
    let l = 3.0 / 2.0 / f;
    let b = ((e_c - e_t) / oexx[1]).powi(2) + ((i_c - i_t) / oexx[2]).powi(2);
    let c = (initial_draan / oexx[3]).powi(2);

    let a_quad = (a / l).powi(2) - c;
    let b_quad = 2.0 * c - 2.0 * a * k / l.powi(2);
    let c_quad = (k / l).powi(2) - (b + c);

    (-b_quad - (b_quad.powi(2) - 4.0 * a_quad * c_quad).sqrt()) / (2.0 * a_quad)
}

/// Computes an approximate drift orbit semi-major axis given phi and t_final.
fn approx_ad_from_phi(
    xc: &NonDimKep,
    xt: &NonDimKep,
    f: f64,
    phi: f64,
    t_final: f64,
    long_way: bool,
) -> f64 {
    let mut draan = delta_angle(xc[3], xt[3]);
    if long_way {
        draan = go_other_way(draan);
    }
    draan *= phi;

    let (_, e_c, i_c, ..) = (xc[0], xc[1], xc[2]);
    let (a_t, e_t, i_t, ..) = (xt[0], xt[1], xt[2]);

    let mut xc_mod = xc.clone_owned();
    xc_mod[0] = a_t;
    let dt4 = time_to_go_qlaw(&xc_mod, xt, f, Some(phi));
    let dt2 = t_final - dt4;

    let coeff = (1.0 - e_c.powi(2)).powi(2) / (3.0 / 2.0 * J2 * dt2 * i_c.cos());
    let term_1 = draan * coeff;
    let term_2 = (1.0 - e_c.powi(2)).powi(2) / (1.0 - e_t.powi(2)).powi(2) * i_t.cos() / i_c.cos()
        * a_t.powf(-3.5);
    let a_d = (term_1 + term_2).powf(-2.0 / 7.0);

    let rel_rate = -3.0 / 2.0
        * J2
        * (i_c.cos() / (1.0 - e_c.powi(2)).powi(2) * a_d.powf(-3.5)
            - i_t.cos() / (1.0 - e_t.powi(2)).powi(2) * a_t.powf(-3.5));

    if rel_rate.signum() == -draan.signum() {
        a_d
    } else {
        f64::NAN
    }
}

/// Iteratively computes drift orbit semi-major axis given phi and T.
///
/// # Arguments
///
/// * `xc` - Chaser orbital state.
/// * `xt` - Target orbital state.
/// * `f` - Maximum spacecraft acceleration.
/// * `phi` - RAAN matching parameter [0, 1].
/// * `t_final` - Total time for the transfer.
/// * `long_way` - Whether to take the long way around for RAAN matching.
///
/// # Returns
///
/// Estimated drift orbit semi-major axis.
pub fn ad_from_phi(
    xc: &NonDimKep,
    xt: &NonDimKep,
    f: f64,
    phi: f64,
    t_final: f64,
    long_way: bool,
) -> f64 {
    let a_d_init = approx_ad_from_phi(xc, xt, f, phi, t_final, long_way);
    if a_d_init.is_nan() {
        return f64::NAN;
    }

    let residual = |a_d: f64| -> f64 {
        let phi_val = phi_from_ad(xc, xt, f, a_d, t_final, long_way);
        phi - phi_val
    };
    let d_residual = |a_d: f64| -> f64 {
        let h = 1e-6;
        (residual(a_d + h) - residual(a_d - h)) / (2.0 * h)
    };

    let mut convergency = SimpleConvergency {
        eps: 1e-6,
        max_iter: 20,
    };
    let root = find_root_newton_raphson(a_d_init, &residual, &d_residual, &mut convergency);

    match root {
        Ok(val) => val,
        Err(_) => f64::NAN,
    }
}

/// Parameters for `opt_dv_given_time`.
#[derive(Debug, Clone, Copy)]
pub struct OptParams {
    pub a_d_min: f64,
}

impl Default for OptParams {
    fn default() -> Self {
        Self { a_d_min: 1.0 }
    }
}

/// Optimize a drift transfer for minimum delta-v under a given time constraint.
///
/// Expects nondimensionalized inputs.
///
/// # Arguments
///
/// * `ch_kep` - Chaser Keplerian elements.
/// * `tg_kep` - Target Keplerian elements.
/// * `f` - Spacecraft acceleration magnitude.
/// * `t_final` - Final time for the transfer.
/// * `params` - Additional parameters for optimization.
///
/// # Returns
///
/// A tuple containing:
/// - `(a_d, phi, long_way)`: The optimal parameters.
/// - `dv`: The corresponding minimum delta-v.
pub fn opt_dv_given_time(
    ch_kep: &NonDimKep,
    tg_kep: &NonDimKep,
    f: f64,
    t_final: f64,
    params: &OptParams,
) -> ((f64, f64, bool), f64) {
    let a_d_min = params.a_d_min;

    let opt1 = |long_way: bool| -> (f64, f64) {
        let a_d = ad_from_phi(ch_kep, tg_kep, f, 1.0, t_final, long_way);
        if a_d.is_nan() {
            return (a_d, f64::INFINITY);
        }
        let dv = drift_dv(ch_kep, tg_kep, a_d, 1.0);
        (a_d, dv)
    };

    let (a_d_1_sw, dv_1_sw) = opt1(false);
    let (a_d_1_lw, dv_1_lw) = opt1(true);

    let dv_1_sw_masked = if a_d_1_sw < a_d_min || a_d_1_sw.is_nan() {
        f64::INFINITY
    } else {
        dv_1_sw
    };
    let dv_1_lw_masked = if a_d_1_lw < a_d_min || a_d_1_lw.is_nan() {
        f64::INFINITY
    } else {
        dv_1_lw
    };

    let opt_capped = |long_way: bool| -> (f64, f64) {
        let phi = phi_from_ad(ch_kep, tg_kep, f, a_d_min, t_final, long_way);
        let dv = drift_dv(ch_kep, tg_kep, a_d_min, phi);

        if !(0.0..=1.0).contains(&phi) || phi.is_nan() {
            (phi, f64::INFINITY)
        } else {
            (phi, dv)
        }
    };

    let (phi_sw, dv_sw) = opt_capped(false);
    let (phi_lw, dv_lw) = opt_capped(true);

    let sols = [dv_1_sw_masked, dv_1_lw_masked, dv_sw, dv_lw];
    let ad_vals = [a_d_1_sw, a_d_1_lw, a_d_min, a_d_min];
    let phi_vals = [1.0, 1.0, phi_sw, phi_lw];
    let long_way_vals = [false, true, false, true];

    let mut best_idx = 0;
    let mut min_dv = f64::INFINITY;
    for (i, &dv) in sols.iter().enumerate() {
        if dv < min_dv {
            min_dv = dv;
            best_idx = i;
        }
    }

    (
        (
            ad_vals[best_idx],
            phi_vals[best_idx],
            long_way_vals[best_idx],
        ),
        sols[best_idx],
    )
}

/// Optimize the transfer from a chaser to a target over a given time of flight.
///
/// States must be nondimensionalized Keplerian elements.
///
/// # Arguments
///
/// * `ch_kep` - Initial state of the chaser.
/// * `tg_kep` - Initial state of the target.
/// * `tof` - Time of flight.
/// * `spacecraft_accel` - Spacecraft acceleration.
/// * `min_sma` - Minimum semi-major axis.
/// * `strategy` - "best", "drift", or "direct".
/// * `tof_tol` - Time of flight tolerance.
///
/// # Returns
///
/// A tuple containing:
/// - `dv`: The delta-v for the transfer.
/// - `phi`: The drift phase angle (phi), or 0.0 for a direct transfer.
/// - `is_drift`: Boolean indicating if a drift transfer was used.
/// - `long_way`: Boolean indicating if the long-way transfer was used.
pub fn optimize_segment(
    ch_kep: &NonDimKep,
    tg_kep: &NonDimKep,
    tof: f64,
    spacecraft_accel: f64,
    min_sma: f64,
    strategy: &str,
    tof_tol: f64,
) -> (f64, f64, bool, bool) {
    let direct_transfer = || -> f64 {
        let dv = dv_go_qlaw(ch_kep, tg_kep, None);
        let dt = time_to_go_qlaw(ch_kep, tg_kep, spacecraft_accel, None);
        if dt < tof {
            dv
        } else {
            f64::INFINITY
        }
    };

    let drift_transfer = || -> (f64, f64, bool) {
        let params = OptParams { a_d_min: min_sma };
        let ((a_d, phi, long_way), dv) =
            opt_dv_given_time(ch_kep, tg_kep, spacecraft_accel, tof, &params);

        let dt = drift_dt(ch_kep, tg_kep, a_d, phi, spacecraft_accel, long_way);

        if (dt / tof - 1.0).abs() > tof_tol {
            (f64::INFINITY, phi, long_way)
        } else {
            (dv, phi, long_way)
        }
    };

    match strategy {
        "best" => {
            let (dv_drift, phi, long_way) = drift_transfer();
            let dv_direct = direct_transfer();

            if dv_drift < dv_direct {
                (dv_drift, phi, true, long_way)
            } else {
                (dv_direct, 0.0, false, false)
            }
        }
        "drift" => {
            let (dv, phi, long_way) = drift_transfer();
            (dv, phi, true, long_way)
        }
        "direct" => (direct_transfer(), 0.0, false, false),
        _ => panic!("Unknown strategy: {}", strategy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_angle() {
        assert!((wrap_angle(4.0 * PI) - 0.0).abs() < 1e-9);
        assert!((wrap_angle(3.0 * PI) - -PI).abs() < 1e-9);
        assert!((wrap_angle(-3.0 * PI) - PI).abs() < 1e-9);
    }

    #[test]
    fn test_delta_angle() {
        assert!((delta_angle(PI / 4.0, -PI / 4.0) - PI / 2.0).abs() < 1e-9);
        assert!((delta_angle(-PI / 4.0, PI / 4.0) - -PI / 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_go_other_way() {
        assert!((go_other_way(PI / 4.0) - (-7.0 * PI / 4.0)).abs() < 1e-9);
        assert!((go_other_way(-PI / 4.0) - (7.0 * PI / 4.0)).abs() < 1e-9);
    }
}
