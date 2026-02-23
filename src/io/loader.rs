use crate::orbit::State;
use std::fs;

const HEADER: &str = "a[m],e[prop],i[rad],Omega[rad],omega[rad],theta[rad]";

/// Load debris states from input .csv file
///
/// The .csv file header must match the `HEADER` constant, and each line should
/// contain 6 comma-separated numerical values as follows:
///
/// ```text
/// a[m],e[prop],i[rad],Omega[rad],omega[rad],theta[rad]
/// 7164040.5518,0.0019,1.5079,2.8765,0.8909,5.3923
/// 6989199.3166,0.0030,1.5081,2.6417,2.1655,1.8687
/// 7148540.2308,0.0025,1.5081,2.8457,1.0069,5.3600
/// 7159283.7674,0.0017,1.5075,2.7557,0.8765,2.4102
/// ```
pub struct Loader;

impl Loader {
    pub fn load(path: &str) -> Vec<State> {
        // Check if file exists
        if !std::path::Path::new(path).exists() {
            panic!("File not found: {}", path);
        }

        let file = fs::read_to_string(path).unwrap();
        let mut lines = file.lines();

        // Check header format
        let line = lines.next();
        if line.is_none() {
            panic!("Empty file: {}", path);
        }
        if line.unwrap() != HEADER {
            panic!("Invalid header: expected '{}'", HEADER);
        }

        // Load debris states
        let mut states: Vec<State> = Vec::new();
        for line in lines {
            let elems: Vec<&str> = line.split(',').collect();
            let values: Vec<f64> = elems.iter().map(|x| x.parse::<f64>().unwrap()).collect();
            if values.len() != 6 {
                panic!("Invalid state line: {}", line);
            }
            let state = State {
                a: values[0],
                e: values[1],
                i: values[2],
                O: values[3],
                o: values[4],
                t: values[5],
            };
            states.push(state);
        }

        states
    }
}
