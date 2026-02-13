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