use crate::orbit::constants::{J2, MU, RE};
use crate::orbit::qlaw::{
    ad_from_phi, c2c_raan_change, delta_angle, dt_qlaw, dv_qlaw, phi_from_ad, reverse_angle,
};
use crate::orbit::state::{NdState, Scalings, State};

/// Physical oracle to perform operations on orbital states
pub struct Oracle {
    /// Type of transfer strategy to use for maneuvers. Current choices are:
    /// - "direct": direct transfer using maximum thrust
    /// - "drift": drift transfer using intermediate orbit
    /// - "best": select best strategy among all available
    pub strategy: String,
    /// Maximum thrust for maneuvers [m/s^2]
    pub max_thrust: f64,
    /// Minimum semimajor axis allowed [m]
    pub min_sma: f64,
    /// Non-dimensionalization scaling system
    pub scalings: Scalings,
}

/// Signature of oracle evaluation function, regardless of the strategy used
type EvaluationType = fn(&Oracle, &NdState, &NdState, f64, f64, f64) -> (f64, f64);

impl Oracle {
    pub fn new(strategy: String, max_thrust: f64, min_sma: f64) -> Self {
        Self {
            strategy,
            max_thrust,
            min_sma,
            scalings: Scalings::default(),
        }
    }

    /// Evaluate cost and time of transfer between two orbital states
    ///
    /// Note: if the transfer cannot be completed within the stop time,
    /// infinite delta-v and minimum delta-t required to make it feasible are
    /// returned.
    pub fn evaluate(&self, src: &State, dst: &State, src_time: f64, dst_time: f64) -> (f64, f64) {
        // Propagate states to the transfer start time
        let src_pp = self.propagate(src, src_time);
        let dst_pp = self.propagate(dst, src_time);

        // Pass time constraint into evaluation function (needed for drift)
        let tmax_nd = self.scalings.to_nondim_time(dst_time - src_time);

        // Non-dimensionalize propagated states and physical parameters
        let src_nd = self.scalings.to_nondim_state(&src_pp);
        let dst_nd = self.scalings.to_nondim_state(&dst_pp);
        let fmax_nd = self.scalings.to_nondim_thrust(self.max_thrust);
        let amin_nd = self.scalings.to_nondim_sma(self.min_sma);

        // Evaluate transfer delta-v and delta-t based on selected strategy
        let (dv_nd, dt_nd) = match self.strategy.as_str() {
            "direct" => self.evaluate_direct(&src_nd, &dst_nd, fmax_nd, amin_nd, tmax_nd),
            "drift" => self.evaluate_drift(&src_nd, &dst_nd, fmax_nd, amin_nd, tmax_nd),
            "best" => self.evaluate_best(&src_nd, &dst_nd, fmax_nd, amin_nd, tmax_nd),
            _ => panic!("Unsupported transfer strategy: {}", self.strategy),
        };

        // Re-dimensionalize back delta-v and delta-t
        let dv = self.scalings.to_dim_dv(dv_nd);
        let dt = self.scalings.to_dim_dt(dt_nd);

        // Check if transfer can be completed before stop time
        if src_time + dt > dst_time {
            (f64::INFINITY, dt)
        } else {
            (dv, dt)
        }
    }

    /// Time-independent distance estimate between two orbital states
    pub fn distance(&self, src: &State, dst: &State) -> f64 {
        let (cost, _) = self.evaluate(src, dst, 0.0, f64::INFINITY);
        cost
    }

    /// Propagate orbital elements forward by `dt` seconds
    #[allow(non_snake_case)]
    fn propagate(&self, state: &State, dt: f64) -> State {
        let prefactor =
            (J2 * RE.powi(2)) * (MU / state.a.powi(7)).sqrt() / (1.0 - state.e * state.e).powi(2);

        let dot_O = -1.5 * prefactor * state.i.cos();
        let dot_o = 0.75 * prefactor * (5.0 * state.i.cos().powi(2) - 1.0);

        State {
            a: state.a,
            e: state.e,
            i: state.i,
            O: state.O + dot_O * dt,
            o: state.o + dot_o * dt,
            t: state.t,
        }
    }

    // ==================== Direct transfer ==================== //

    /// Evaluate cost and time of direct transfer between two orbital states
    ///
    /// Note: Inputs and outputs are non-dimensionalized via `Scalings`.
    fn evaluate_direct(
        &self,
        src: &NdState,
        dst: &NdState,
        fmax: f64,
        _amin: f64,
        _tmax: f64,
    ) -> (f64, f64) {
        let dv = dv_qlaw(src, dst, None);
        let dt = dt_qlaw(src, dst, fmax, None);
        (dv, dt)
    }

    // ==================== Drift transfer ==================== //

    /// Evaluate cost and time of drift transfer between two orbital states
    ///
    /// Note: Inputs and outputs are non-dimensionalized via `Scalings`.
    fn evaluate_drift(
        &self,
        src: &NdState,
        dst: &NdState,
        fmax: f64,
        amin: f64,
        tmax: f64,
    ) -> (f64, f64) {
        let (dv, ad, phi, lw) = self.drift_dv_given_dt(src, dst, fmax, amin, tmax);
        let dt = self.drift_dt(src, dst, fmax, ad, phi, lw);

        (dv, dt)
    }

    /// Delta-t for drift transfer between states, given maximum thrust
    /// `fmax_nd` drift orbit semi-major axis `a`, RAAN proportion `phi`, and
    /// `lw` choice.
    ///  
    /// Note: Inputs and outputs are non-dimensionalized via `Scalings`.
    fn drift_dt(
        &self,
        src: &NdState,
        dst: &NdState,
        fmax: f64,
        ad: f64,
        phi: f64,
        lw: bool,
    ) -> f64 {
        // Initial state with intermediate drift orbit semi-major axis
        let int = &mut src.clone();
        int.a = ad;

        // Initial state with final drift orbit semi-major axis
        let fnl = &mut src.clone();
        fnl.a = dst.a;

        let dr1 = c2c_raan_change(src, ad, fmax);
        let dr3 = c2c_raan_change(int, dst.a, fmax);

        let dt1 = ((1.0 / src.a).sqrt() - (1.0 / ad).sqrt()).abs() / fmax;
        let dt3 = ((1.0 / dst.a).sqrt() - (1.0 / ad).sqrt()).abs() / fmax;
        let dt4 = dt_qlaw(fnl, dst, fmax, Some(phi));

        let src_rate =
            -3.0 / 2.0 * J2 * (src.i.cos() / (1.0 - src.e.powi(2)).powi(2) * ad.powf(-3.5));
        let dst_rate =
            -3.0 / 2.0 * J2 * (dst.i.cos() / (1.0 - dst.e.powi(2)).powi(2) * dst.a.powf(-3.5));
        let rel_rate = src_rate - dst_rate;

        let src_dr2 = if lw {
            reverse_angle(delta_angle(src.O, dst.O))
        } else {
            delta_angle(src.O, dst.O)
        };
        let dst_dr2 = dr1 + dr3 - dst_rate * (dt1 + dt3);

        let dr2 = src_dr2 * phi + dst_dr2;
        let dt2 = dr2 / -rel_rate;

        if dt2 < 0.0 {
            f64::INFINITY
        } else {
            dt1 + dt2 + dt3 + dt4
        }
    }

    /// Delta-v for drift transfer between states, given drift orbit semi-major
    /// axis `a` and RAAN proportion `phi`.
    ///  
    /// Note: Inputs and outputs are non-dimensionalized via `Scalings`.
    fn drift_dv(&self, src: &NdState, dst: &NdState, ad: f64, phi: f64) -> f64 {
        let mut src_match = src.clone();
        src_match.a = dst.a;
        let dv1 = ((1.0 / src.a).sqrt() - (1.0 / ad).sqrt()).abs();
        let dv2 = 0.0;
        let dv3 = ((1.0 / dst.a).sqrt() - (1.0 / ad).sqrt()).abs();
        let dv4 = dv_qlaw(&src_match, dst, Some(phi));
        dv1 + dv2 + dv3 + dv4
    }

    /// Optimize a drift transfer for minimum delta-v under a given time constraint.
    ///
    /// Expects non-dimensionalized inputs.
    ///
    /// # Arguments
    ///
    /// * `src` - Chaser Keplerian elements.
    /// * `dst` - Target Keplerian elements.
    /// * `fmax_nd` - Spacecraft acceleration magnitude.
    /// * `amin_nd` - Minimum allowed drift orbit semi-major axis.
    /// * `tmax_dst` - Final time for the transfer.
    /// * `params` - Additional parameters for optimization.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `(a_d, phi, long_way)`: The optimal parameters.
    /// - `dv`: The corresponding minimum delta-v.
    fn drift_dv_given_dt(
        &self,
        src: &NdState,
        dst: &NdState,
        fmax: f64,
        amin: f64,
        tmax: f64,
    ) -> (f64, f64, f64, bool) {
        let mut best_dv = f64::INFINITY;
        let mut best_ad = f64::NAN;
        let mut best_phi = f64::NAN;
        let mut best_lw = false;

        // Try phi = 1 (max RAAN matching) and get ad from phi
        let phi = 1.0;
        for &lw in [false, true].iter() {
            let ad = ad_from_phi(src, dst, phi, fmax, tmax, lw);
            if ad < amin || ad.is_nan() {
                continue;
            }
            let dv = self.drift_dv(src, dst, ad, phi);
            if dv < best_dv {
                best_dv = dv;
                best_ad = ad;
                best_phi = phi;
                best_lw = lw;
            }
        }

        // Try ad = amin (min semi-major axis) and get phi from ad
        let ad = amin;
        for &lw in [false, true].iter() {
            let phi = phi_from_ad(src, dst, amin, fmax, tmax, lw);
            if !(0.0..=1.0).contains(&phi) || phi.is_nan() {
                continue;
            }
            let dv = self.drift_dv(src, dst, amin, phi);
            if dv < best_dv {
                best_dv = dv;
                best_ad = ad;
                best_phi = phi;
                best_lw = lw;
            }
        }

        (best_dv, best_ad, best_phi, best_lw)
    }

    // ==================== Best transfer ==================== //

    /// Evaluate cost and time of best transfer between two orbital states
    ///
    /// Note: Inputs and outputs are non-dimensionalized via `Scalings`.
    fn evaluate_best(
        &self,
        src: &NdState,
        dst: &NdState,
        fmax: f64,
        amin: f64,
        tmax: f64,
    ) -> (f64, f64) {
        let strategies: [EvaluationType; 2] = [Self::evaluate_direct, Self::evaluate_drift];

        strategies
            .iter()
            .map(|f| f(self, src, dst, fmax, amin, tmax))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
    }
}
