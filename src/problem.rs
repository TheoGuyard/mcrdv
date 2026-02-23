use crate::orbit::State;

/// Space debris remediation problem instance
pub struct Problem {
    /// List of orbital states
    /// - states[0]   : initial and return position of all chasers
    /// - states[1..] : position of debris to be removed
    pub states: Vec<State>,

    /// Chaser starting position
    /// - "barycenter": all chasers start at the barycenter of debris
    pub chaser_start: String,

    /// Maximum number of chasers available
    pub max_chasers: usize,

    /// Maximum debris load per chaser
    pub max_load: usize,

    /// Maximum mission duration per chaser [seconds]
    pub mission_time: f64,
}

impl Problem {
    pub fn new(
        debris: &Vec<State>,
        chaser_start: String,
        max_chasers: usize,
        max_load: usize,
        mission_time: f64,
    ) -> Self {
        // Sanity checks
        if debris.len() > max_chasers * max_load {
            panic!(
                "Error: Number of debris exceeds total chaser capacity. \
                Consider increasing parameters max_chasers or max_load."
            );
        }

        // Owned debris states to be modified
        let mut states = debris.to_owned();

        // Set chasers start position (insert as first element of the states vector)
        match chaser_start.as_str() {
            "barycenter" => {
                let chaser = State {
                    a: states.iter().map(|s| s.a).sum::<f64>() / (states.len() as f64),
                    e: states.iter().map(|s| s.e).sum::<f64>() / (states.len() as f64),
                    i: states.iter().map(|s| s.i).sum::<f64>() / (states.len() as f64),
                    O: states.iter().map(|s| s.O).sum::<f64>() / (states.len() as f64),
                    o: states.iter().map(|s| s.o).sum::<f64>() / (states.len() as f64),
                    t: states.iter().map(|s| s.t).sum::<f64>() / (states.len() as f64),
                };
                states.insert(0, chaser);
            }
            _ => panic!("Invalid chaser start option: {}", chaser_start),
        }

        Self {
            states,
            chaser_start,
            max_chasers,
            max_load,
            mission_time,
        }
    }
}
