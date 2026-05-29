use std::error::Error;

use crate::orbit::State;

/// Multi-chaser active debris removal problem instance
#[derive(Debug, Clone)]
pub struct Problem {
    /// Debris states at the mission start
    pub debris: Vec<State>,
    /// Chaser swarm state at the mission start
    pub chasers: State,
    /// Number of available chasers
    pub num_chasers: usize,
    /// Maximum mission time per chaser [s]
    pub max_time: f64,
    /// Maximum mission fuel per chaser [m/s]
    pub max_fuel: f64,
    /// Maximum mission load per chaser [debris]
    pub max_load: usize,
    /// Objective weight on mission time [1/s]
    pub factor_time: f64,
    /// Objective weight on mission fuel [s/m]
    pub factor_fuel: f64,
}

impl Problem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        debris: Vec<State>,
        chasers: State,
        num_chasers: usize,
        max_time: f64,
        max_fuel: f64,
        max_load: usize,
        factor_time: f64,
        factor_fuel: f64,
    ) -> Result<Self, Box<dyn Error>> {

        // Sanity checks
        if debris.is_empty() { return Err("debris list is empty".into()); }
        if num_chasers == 0 { return Err("num_chasers must be positive".into()); }
        if debris.len() > num_chasers * max_load { return Err("too many debris for the given number of chasers and max load".into()); }
        if max_time <= 0.0 { return Err("max_time must be positive".into()); }
        if max_fuel <= 0.0 { return Err("max_fuel must be positive".into()); }
        if max_load == 0 { return Err("max_load must be positive".into()); }
        if factor_time < 0.0 { return Err("factor_time must be non-negative".into()); }
        if factor_fuel < 0.0 { return Err("factor_fuel must be non-negative".into()); }

        Ok(Self {
            debris,
            chasers,
            num_chasers,
            max_time,
            max_fuel,
            max_load,
            factor_time,
            factor_fuel,
        })
    }

    pub fn num_debris(&self) -> usize {
        self.debris.len()
    }

    pub fn cost(&self, time: f64, fuel: f64) -> f64 {
        self.factor_time * time + self.factor_fuel * fuel
    }

    pub fn has_objective(&self) -> bool {
        self.factor_time > 0.0 || self.factor_fuel > 0.0
    }
}
