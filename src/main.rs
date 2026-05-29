use std::error::Error;

use scorpion::io::load_states;
use scorpion::orbit::centroid;
use scorpion::problem::Problem;
use scorpion::solver::{Params, Solver};

fn main() -> Result<(), Box<dyn Error>> {

    // Initialize logger
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .without_time()
        .init();

    // Define problem data
    let debris_path = "data/iridium33.csv";
    let max_time    = 365.0 * 86_400.0;     // [s]
    let max_fuel    = f64::INFINITY;        // [m/s]
    let max_load    = 5;                    // [debris]
    let factor_time = 0.0;                  // [1/s]
    let factor_fuel = 1.0;                  // [s/m]

    // Load debris from file
    let debris = load_states(debris_path)?;

    // Define the swarm position
    let chasers = centroid(&debris)?;

    // Set number of chasers based on the debris and the maximum load
    let fac_chasers = 1.5;
    let min_chasers = ((debris.len() as f64) / (max_load as f64)).ceil();
    let num_chasers = (fac_chasers * min_chasers) as usize;

    // Instantiate problem
    let problem = Problem::new(
        debris,
        chasers,
        num_chasers,
        max_time,
        max_fuel,
        max_load,
        factor_time,
        factor_fuel,
    )?;

    // Instantiate solver
    let params = Params::default();
    let solver = Solver::new(&params);

    // Solve problem
    let solution = solver.solve(&problem);

    println!("{solution}");

    Ok(())
}
