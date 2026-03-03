use crate::orbit::State;
use crate::orbit::constants::{J2, MU, RE};

/// Physical oracle to perform operations on orbital states
pub struct Oracle {
    /// Type of transfer strategy to use for maneuvers. Current choices are:
    /// - "direct": direct transfer using maximum thrust
    /// - "drift": drift transfer using intermediate orbit
    /// - "best": select best strategy among all available
    pub strategy: String,
    /// Maximum thrust for maneuvers [m/s^2]
    pub max_thrust: f64,
}

/// Signature of oracle evaluation function, regardless of the strategy used
type EvaluationType = fn(&Oracle, &State, &State) -> (f64, f64);

impl Oracle {
    pub fn new(strategy: String, max_thrust: f64) -> Self {
        Self {
            strategy,
            max_thrust,
        }
    }

    /// Evaluate cost and time of transfer between two orbital states
    /// Note: if the transfer cannot be completed within the stop time, cost is
    /// set to infinity, and the time returned is the smallest possible time of
    /// flight to make the transfer feasible
    pub fn evaluate(
        &self,
        src: &State,
        dst: &State,
        start_time: f64,
        stop_time: f64,
    ) -> (f64, f64) {
        // Propagate states to the transfer start time
        let src_prop = self.propagate(src, start_time);
        let dst_prop = self.propagate(dst, start_time);

        // Evaluate cost and minimum transfer time based on selected strategy
        let (cost, time) = match self.strategy.as_str() {
            "direct" => self.evaluate_direct(&src_prop, &dst_prop),
            "drift" => self.evaluate_drift(&src_prop, &dst_prop),
            "best" => self.evaluate_best(&src_prop, &dst_prop),
            _ => panic!("Unsupported transfer strategy: {}", self.strategy),
        };

        // Check if transfer can be completed within stop time
        if start_time + time > stop_time {
            (f64::INFINITY, time)
        } else {
            (cost, time)
        }
    }

    /// Time-independent distance estimate between two orbital states
    pub fn distance(&self, src: &State, dst: &State) -> f64 {
        ((src.a - dst.a).powi(2)
            + (src.e - dst.e).powi(2)
            + (src.i - dst.i).powi(2)
            + (src.O - dst.O).powi(2)
            + (src.o - dst.o).powi(2)
            + (src.t - dst.t).powi(2))
        .sqrt()
    }

    /// Propagate orbital elements forward by `dt` seconds
    #[allow(non_snake_case)]
    fn propagate(&self, state: &State, dt: f64) -> State {
        let scale =
            (J2 * RE.powi(2)) * (MU / state.a.powi(7)).sqrt() / (1.0 - state.e * state.e).powi(2);

        let dot_O = -1.5 * scale * state.i.cos();
        let dot_o = 0.75 * scale * (5.0 * state.i.cos().powi(2) - 1.0);

        State {
            a: state.a,
            e: state.e,
            i: state.i,
            O: state.O + dot_O * dt,
            o: state.o + dot_o * dt,
            t: state.t,
        }
    }

    /// Time-of-flight between two orbital states using direct transfer
    #[allow(non_snake_case)]
    fn time_direct(&self, src: &State, dst: &State) -> f64 {
        // Precomputed quantities
        let epow2 = src.e.powi(2);
        let scale = self.max_thrust * (src.a * (1.0 - epow2) / MU).sqrt();
        let cos_o = src.o.cos();
        let sin_o = src.o.sin();

        // Differences between chaser and target elements
        let delta_a = src.a - dst.a;
        let delta_e = src.e - dst.e;
        let delta_i = src.i - dst.i;
        let delta_O = src.O - dst.O;

        // Maximum rates of change
        let dotxx_a = 2.0 * scale * src.a / (1.0 - src.e);
        let dotxx_e = 2.0 * scale;
        let dotxx_i = scale / ((1.0 - epow2 * sin_o.powi(2)).sqrt() - src.e * cos_o.abs());
        let dotxx_O =
            scale / (src.i.sin() * ((1.0 - epow2 * cos_o.powi(2)).sqrt() - src.e * sin_o.abs()));

        // Time of flight estimate
        ((delta_a / dotxx_a).powi(2)
            + (delta_e / dotxx_e).powi(2)
            + (delta_i / dotxx_i).powi(2)
            + (delta_O / dotxx_O).powi(2))
        .sqrt()
    }

    /// Evaluate cost and time of direct transfer between two orbital states
    fn evaluate_direct(&self, src: &State, dst: &State) -> (f64, f64) {
        let time = self.time_direct(src, dst);
        let cost = self.max_thrust * time;

        (cost, time)
    }

    /// Evaluate cost and time of drift transfer between two orbital states
    #[allow(dead_code)]
    fn evaluate_drift(&self, _src: &State, _dst: &State) -> (f64, f64) {
        unimplemented!("Drift transfer strategy not implemented yet");
    }

    /// Evaluate cost and time of best transfer between two orbital states
    fn evaluate_best(&self, src: &State, dst: &State) -> (f64, f64) {
        let strategies: [EvaluationType; 2] = [Self::evaluate_direct, Self::evaluate_drift];

        strategies
            .iter()
            .map(|f| f(self, src, dst))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
    }
}
