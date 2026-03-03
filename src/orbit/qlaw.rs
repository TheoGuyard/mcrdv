use crate::orbit::constants::{J2, PI};
use crate::orbit::state::NdState;

/// Wraps an angle to be in the range [-pi, pi]
#[inline]
pub fn wrap_angle(a: f64) -> f64 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// Delta between two angles wrapped in range [-pi, pi]
#[inline]
pub fn delta_angle(a: f64, b: f64) -> f64 {
    wrap_angle(a - b)
}

/// Reverse angle the other way around the circle (e.g. PI/4 -> -7PI/4)
#[inline]
pub fn reverse_angle(a: f64) -> f64 {
    a - a.signum() * 2.0 * PI
}

/// Compute the maximum rates of change of state per acceleration unit
#[inline]
pub fn oexx_by_accel(state: &NdState) -> NdState {
    let p = state.a * (1.0 - state.e.powi(2));
    let h = p.sqrt();

    let axx = 2.0 * (state.a.powi(2) * (1.0 + state.e) / (1.0 - state.e)).sqrt();
    let exx = 2.0 * p / h;
    let ixx = p
        / (h * ((1.0 - state.e.powi(2) * state.o.sin().powi(2)).sqrt()
            - state.e * state.o.cos().abs()));
    let oxx = p
        / (h * state.i.sin()
            * ((1.0 - state.e.powi(2) * state.o.cos().powi(2)).sqrt()
                - state.e * state.o.sin().abs()));

    NdState {
        a: axx,
        e: exx,
        i: ixx,
        o: oxx,
        O: f64::NAN,
        t: f64::NAN,
    }
}

/// Estimate delta-v between states using the Q-law.
///
/// Note: Inputs and outputs are non-dimensionalized so that mu = 1.
///
/// Note: `phi` is an optional parameter for RAAN matching proportion [0, 1].
/// If None, computes delta-v based on actual elements. Otherwise, computes
/// delta-v assuming that only (1-phi) of the RAAN difference needs to be
/// corrected.
#[inline]
pub fn dv_qlaw(src: &NdState, dst: &NdState, phi: Option<f64>) -> f64 {
    let mut diff = NdState {
        a: src.a - dst.a,
        e: src.e - dst.e,
        i: delta_angle(src.i, dst.i),
        o: delta_angle(src.o, dst.o),
        O: f64::NAN,
        t: f64::NAN,
    };

    if let Some(phi_val) = phi {
        diff.o *= 1.0 - phi_val;
    }

    // Maximum rates of change of source state per acceleration unit
    let oexx = oexx_by_accel(src);

    // Squared norm of state difference normalized by oexx over (a, e, i, o)
    let mut sqnorm = 0.0;
    sqnorm += (diff.a / oexx.a).powi(2);
    sqnorm += (diff.e / oexx.e).powi(2);
    sqnorm += (diff.i / oexx.i).powi(2);
    sqnorm += (diff.o / oexx.o).powi(2);

    3.0 / 2.0 * sqnorm.sqrt()
}

/// Estimate delta-t between states using the Q-law, given max thrust `fmax`.
///
/// Note: Inputs and outputs are non-dimensionalized so that mu = 1.
///
/// Note: `phi` is an optional parameter for RAAN matching proportion [0, 1].
/// If None, computes delta-t based on actual elements. Otherwise, computes
/// delta-t assuming that only (1-phi) of the RAAN difference needs to be
/// corrected.
#[inline]
pub fn dt_qlaw(src: &NdState, dst: &NdState, fmax: f64, phi: Option<f64>) -> f64 {
    let dv = dv_qlaw(src, dst, phi);
    dv / fmax
}

/// RAAN drift amount due to a circle-to-circle raising or lowering maneuver
/// of a state to another semi-major axis `a`, given max thrust `fmax`.
///
/// Note: Inputs and outputs are non-dimensionalized so that mu = 1.
#[inline]
pub fn c2c_raan_change(state: &NdState, a: f64, fmax: f64) -> f64 {
    let coeff = -3.0 / 16.0 * J2;
    let cdiff = (1.0 / state.a.powi(4) - 1.0 / a.powi(4)).abs();
    coeff * state.i.cos() / (fmax * (1.0 - state.e.powi(2)).powi(2)) * cdiff
}

/// Compute the drift fraction parameter phi between two states, given drift
/// semi-major axis `ad`, max thrust `fmax`, total time `tmax`, and long-way
/// choice `lw`.
#[inline]
pub fn phi_from_ad(src: &NdState, dst: &NdState, ad: f64, fmax: f64, tmax: f64, lw: bool) -> f64 {
    // Intermediate state with drift orbit semi-major axis
    let int = &mut src.clone();
    int.a = ad;

    let dr1 = c2c_raan_change(src, int.a, fmax);
    let dr3 = c2c_raan_change(int, dst.a, fmax);
    let dt1 = ((1.0 / src.a).sqrt() - (1.0 / ad).sqrt()).abs() / fmax;
    let dt3 = ((1.0 / dst.a).sqrt() - (1.0 / ad).sqrt()).abs() / fmax;

    let src_rate = -3.0 / 2.0 * J2 * (src.i.cos() / (1.0 - src.e.powi(2)).powi(2) * ad.powf(-3.5));
    let dst_rate =
        -3.0 / 2.0 * J2 * (dst.i.cos() / (1.0 - dst.e.powi(2)).powi(2) * dst.a.powf(-3.5));
    let rel_rate = src_rate - dst_rate;
    let extra_dt = (dr1 + dr3 - dst_rate * (dt1 + dt3)) / -rel_rate;

    let rel_raan = if lw {
        reverse_angle(delta_angle(src.O, dst.O))
    } else {
        delta_angle(src.O, dst.O)
    };

    let k = tmax - dt1 - dt3 - extra_dt;
    let a = rel_raan / -rel_rate;

    // Matching state with destination orbit semi-major axis
    let fnl = &mut src.clone();
    fnl.a = dst.a;

    let oexx = oexx_by_accel(fnl);
    let l = 3.0 / 2.0 / fmax;
    let b = ((src.e - dst.e) / oexx.e).powi(2) + ((src.i - dst.i) / oexx.i).powi(2);
    let c = (rel_raan / oexx.O).powi(2);

    let a_quad = (a / l).powi(2) - c;
    let b_quad = 2.0 * c - 2.0 * a * k / l.powi(2);
    let c_quad = (k / l).powi(2) - (b + c);

    (-b_quad - (b_quad.powi(2) - 4.0 * a_quad * c_quad).sqrt()) / (2.0 * a_quad)
}

/// Approximate drift orbit semi-major axis between two states, given drift
/// fraction parameter `phi`, max thrust `fmax`, total time `tmax`, and
/// long-way choice `lw`.
#[inline]
fn approx_ad_from_phi(
    src: &NdState,
    dst: &NdState,
    phi: f64,
    fmax: f64,
    tmax: f64,
    lw: bool,
) -> f64 {
    let rel_raan = if lw {
        reverse_angle(delta_angle(src.O, dst.O))
    } else {
        delta_angle(src.O, dst.O)
    } * phi;

    let mut src_match = src.clone();
    src_match.a = dst.a;

    let dt4 = dt_qlaw(&src_match, dst, fmax, Some(phi));
    let dt2 = tmax - dt4;

    let c1 = (1.0 - src.e.powi(2)).powi(2) / (3.0 / 2.0 * J2 * dt2 * src.i.cos());
    let c2 = rel_raan * c1;
    let c3 = (1.0 - src.e.powi(2)).powi(2) / (1.0 - dst.e.powi(2)).powi(2) * dst.i.cos()
        / src.i.cos()
        * dst.a.powf(-3.5);
    let ad = (c2 + c3).powf(-2.0 / 7.0);

    let rel_rate = -3.0 / 2.0
        * J2
        * (src.i.cos() / (1.0 - src.e.powi(2)).powi(2) * ad.powf(-3.5)
            - dst.i.cos() / (1.0 - dst.e.powi(2)).powi(2) * dst.a.powf(-3.5));

    if rel_rate.signum() == -rel_raan.signum() {
        ad
    } else {
        f64::NAN
    }
}

/// Iteratively compute drift orbit semi-major axis between two states, given
/// drift fraction parameter `phi`, max thrust `fmax`, total time `tmax`, and
/// long-way choice `lw`.
pub fn ad_from_phi(src: &NdState, dst: &NdState, phi: f64, fmax: f64, tmax: f64, lw: bool) -> f64 {
    let func = |ad: f64| -> f64 {
        let phi_val = phi_from_ad(src, dst, ad, fmax, tmax, lw);
        phi - phi_val
    };

    let grad = |ad: f64| -> f64 {
        let h = 1e-6;
        (func(ad + h) - func(ad - h)) / (2.0 * h)
    };

    // Starting point
    let mut ad = approx_ad_from_phi(src, dst, phi, fmax, tmax, lw);
    if ad.is_nan() {
        return f64::NAN;
    }

    // Approximation using Newton-Raphson method
    let tol_abs = 1e-6;
    let max_its = 20;
    let mut it = 0;
    loop {
        let f_ad = func(ad);
        let g_ad = grad(ad);

        if f_ad.abs() < tol_abs {
            break;
        }

        // Derivative is 0; try to correct the bad starting point
        if g_ad.abs() < tol_abs {
            if it == 0 {
                ad += 1.0;
                it += 1;
                continue;
            } else {
                it = max_its;
                break;
            }
        }

        let ad_next = ad - f_ad / g_ad;

        if (ad_next - ad).abs() < tol_abs {
            break;
        }

        ad = ad_next;
        it += 1;

        if it >= max_its {
            break;
        }
    }

    match it >= max_its {
        true => ad,
        false => f64::NAN,
    }
}
