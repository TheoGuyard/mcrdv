use serde::Deserialize;

use crate::orbit::State;

// Row of csv file storing orbital states given in Keplerian elements
#[derive(Debug, Deserialize)]
struct Row {
    #[serde(rename = "a[m]")]
    a: f64,
    #[serde(rename = "e[prop]")]
    e: f64,
    #[serde(rename = "i[rad]")]
    i: f64,
    #[serde(rename = "r[rad]")]
    r: f64,
    #[serde(rename = "o[rad]")]
    o: f64,
    #[serde(rename = "t[rad]")]
    t: f64,
}

/// Load orbital states from a csv file
pub fn load_states(path: &str) -> Result<Vec<State>, csv::Error> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut states = Vec::new();
    for record in reader.deserialize() {
        let rec: Row = record?;
        states.push(State { a: rec.a, e: rec.e, i: rec.i, r: rec.r, o: rec.o, t: rec.t });
    }
    Ok(states)
}
