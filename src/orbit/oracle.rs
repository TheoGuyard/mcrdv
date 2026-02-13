use crate::orbit::constants::{J2, MU, RE};
use crate::orbit::drift_optimizer;
use crate::orbit::state::State;

/// Physical oracle to perform operations on orbital states
pub struct Oracle {
    /// Type of transfer strategy to use for maneuvers. Current choices are:
    /// - "direct": direct transfer using maximum thrust
    /// - "drift": drift transfer using intermediate orbit
    /// - "best": select best strategy among all available
    strategy: String,
    /// Maximum thrust for maneuvers [m/s^2]
    max_thrust: f64,
    /// Minimum semimajor axis allowed during drift [m]
    min_sma: f64,
}

impl Oracle {
    pub fn new(strategy: String, max_thrust: f64, min_sma: f64) -> Self {
        Self {
            strategy: strategy,
            max_thrust,
            min_sma,
        }
    }

    /// Evaluate cost and time of transfer between two orbital states
    pub fn evaluate(&self, src: &State, dst: &State, start_time: f64) -> (f64, f64) {
        // Propagate states to the transfer start time
        let src_prop = self.propagate(src, start_time);
        let dst_prop = self.propagate(dst, start_time);

        // Evaluate transfer cost and time based on selected strategy
        match self.strategy.as_str() {
            "direct" => self.evaluate_direct(&src_prop, &dst_prop),
            "drift" => self.evaluate_drift(&src_prop, &dst_prop),
            "best" => self.evaluate_best(&src_prop, &dst_prop),
            _ => panic!("Unsupported transfer strategy: {}", self.strategy),
        }
    }

    /// Time-independent distance estimate between two orbital states
    pub fn distance(&self, src: &State, dst: &State) -> f64 {
        // Nondimensionalize states
        let sc = drift_optimizer::Scalings::new();
        let ch_kep = sc.to_nondim_state(src);
        let tg_kep = sc.to_nondim_state(dst);

        let dv_nd = drift_optimizer::dv_go_qlaw(&ch_kep, &tg_kep, None);
        let dv = dv_nd * sc.vu;
        // The distance is measured as the propellant cost of a direct transfer.
        return dv;
        // TODO: we can change this to e.g. prioritize RAAN differences
    }

    /// Propagate orbital elements forward by `dt` seconds
    #[allow(non_snake_case)]
    fn propagate(&self, state: &State, dt: f64) -> State {
        let prefactor =
            (J2 * RE.powi(2)) * (MU / state.a.powi(7)).sqrt() / (1.0 - state.e * state.e).powi(2);

        let dot_O = -1.5 * prefactor * state.i.cos();
        let dot_o = 0.75 * prefactor * (5.0 * state.i.cos().powi(2) - 1.0);

        let state_at_dt = State {
            a: state.a,
            e: state.e,
            i: state.i,
            O: state.O + dot_O * dt,
            o: state.o + dot_o * dt,
            t: state.t,
        };

        return state_at_dt;
    }

    /// Evaluate cost and time of direct transfer between two orbital states
    fn evaluate_direct(&self, src: &State, dst: &State) -> (f64, f64) {
        let sc = drift_optimizer::Scalings::new();
        let ch_kep = sc.to_nondim_state(src);
        let tg_kep = sc.to_nondim_state(dst);
        let f = self.max_thrust * sc.tu * sc.tu / sc.lu;

        let dv_nd = drift_optimizer::dv_go_qlaw(&ch_kep, &tg_kep, None);
        let dt_nd = drift_optimizer::time_to_go_qlaw(&ch_kep, &tg_kep, f, None);

        let dv = dv_nd * sc.vu;
        let dt = dt_nd * sc.tu;

        (dv, dt)
    }

    /// Evaluate cost and time of drift transfer between two orbital states
    fn evaluate_drift(&self, src: &State, dst: &State) -> (f64, f64) {
        // TEMPORARY WORKAROUND: will be replaced by more sophisticated strategy in future.
        // Currently, we try to pick a transfer N times as long as the direct transfer
        // Such that we get we get a "sufficient decrease" in cost compared to the direct transfer, while not drifting for too long.

        let (dv_direct, dt_direct) = self.evaluate_direct(src, dst);

        let factors = [1.0, 1.25, 1.5, 2.0, 3.0];
        // e.g. a 2x increase in ToF should net a >2x decrease in cost to be worthwhile
        // pick the SHORTEST drift transfer that meets this criterion (otherwise, use a direct transfer)
        for &factor in factors.iter() {
            let (dv_drift, dt_drift) = self.evaluate_drift_nx(src, dst, factor);
            let performance_ratio = dv_drift / dv_direct * (dt_drift / dt_direct);
            if performance_ratio < 0.99 {
                return (dv_drift, dt_drift);
            }
        }
        (dv_direct, dt_direct)
    }

    /// Helper function for the above
    fn evaluate_drift_nx(&self, src: &State, dst: &State, factor: f64) -> (f64, f64) {
        let sc = drift_optimizer::Scalings::new();
        let ch_kep = sc.to_nondim_state(src);
        let tg_kep = sc.to_nondim_state(dst);
        let f = self.max_thrust * sc.tu * sc.tu / sc.lu;
        let min_sma = self.min_sma / sc.lu;

        // Heuristic ToF for drift transfer is `factor` times the direct transfer time
        let tof_direct_nd = drift_optimizer::time_to_go_qlaw(&ch_kep, &tg_kep, f, None);
        let tof_nd = factor * tof_direct_nd;

        let params = drift_optimizer::OptParams { a_d_min: min_sma };
        let ((a_d, phi, long_way), dv) =
            drift_optimizer::opt_dv_given_time(&ch_kep, &tg_kep, f, tof_nd, &params);

        let dt = drift_optimizer::drift_dt(&ch_kep, &tg_kep, a_d, phi, f, long_way);

        let dv = dv * sc.vu;
        let dt = dt * sc.tu;
        (dv, dt)
    }

    /// Evaluate cost and time of best transfer between two orbital states
    fn evaluate_best(&self, src: &State, dst: &State) -> (f64, f64) {
        let (cost_direct, time_direct) = self.evaluate_direct(src, dst);
        let (cost_drift, time_drift) = self.evaluate_drift(src, dst);

        if cost_direct < cost_drift {
            (cost_direct, time_direct)
        } else {
            (cost_drift, time_drift)
        }
    }
}
