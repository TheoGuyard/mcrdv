use crate::orbit::constants::{MU, RE};

/// Orbital state represented in Keplerian elements.
///
/// Fields:
/// - a: Semi-major axis        [meters]
/// - e: Eccentricity           [proportion]
/// - i: Inclination            [radians]
/// - O: RAAN                   [radians]
/// - o: Argument of perigee    [radians]
/// - t: True anomaly           [radians]
#[derive(Clone)]
#[allow(non_snake_case)]
pub struct State {
    pub a: f64,
    pub e: f64,
    pub i: f64,
    pub O: f64,
    pub o: f64,
    pub t: f64,
}

/// Non-dimensionalized orbital state represented in Keplerian elements.
///
/// Fields:
/// - a: Non-dim. semi-major axis   [dimensionless]
/// - e: Eccentricity               [proportion]
/// - i: Inclination                [radians]
/// - O: RAAN                       [radians]
/// - o: Argument of perigee        [radians]
/// - t: True anomaly               [radians]
#[derive(Clone)]
#[allow(non_snake_case)]
pub struct NdState {
    pub a: f64,
    pub e: f64,
    pub i: f64,
    pub O: f64,
    pub o: f64,
    pub t: f64,
}

/// Non-dimensionalization scaling system
pub struct Scalings {
    /// Length unit [meters]
    pub lu: f64,
    /// Time unit [seconds]
    pub tu: f64,
    /// Velocity unit [meters/seconds]
    pub vu: f64,
}

impl Scalings {
    pub fn default() -> Self {
        let lu = RE;
        let tu = (lu.powi(3) / MU).sqrt();
        let vu = lu / tu;
        Self { lu, tu, vu }
    }

    /// Convert Keplarian state to non-dimensionalized Keplerian state
    #[inline]
    pub fn to_nondim_state(&self, state: &State) -> NdState {
        NdState {
            a: state.a / self.lu,
            e: state.e,
            i: state.i,
            O: state.O,
            o: state.o,
            t: state.t,
        }
    }

    /// Convert thrust in [m/s^2] to non-dimensionalized thrust
    #[inline]
    pub fn to_nondim_thrust(&self, thrust: f64) -> f64 {
        thrust * self.tu * self.tu / self.lu
    }

    /// Convert semi-major axis in [m] to non-dimensionalized semi-major axis
    #[inline]
    pub fn to_nondim_sma(&self, sma: f64) -> f64 {
        sma / self.lu
    }

    /// Convert time in [seconds] to non-dimensionalized time
    #[inline]
    pub fn to_nondim_time(&self, time: f64) -> f64 {
        time / self.tu
    }

    /// Revert non-dimensionalized delta-t to dimensional delta-t in seconds
    #[inline]
    pub fn to_dim_dt(&self, dt_nd: f64) -> f64 {
        dt_nd * self.tu
    }

    /// Revert non-dimensionalized delta-v to dimensional delta-v in m/s
    #[inline]
    pub fn to_dim_dv(&self, dv_nd: f64) -> f64 {
        dv_nd * self.vu
    }
}
