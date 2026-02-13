use scorpion::io::{Loader, Writer};
use scorpion::optim::Solver;
use scorpion::orbit::Oracle;
use scorpion::Problem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load debris from input file
    let debris = Loader::load("data/odrc.csv");

    // Set problem data
    let problem = Problem::new(
        &debris,                  // debris states
        "barycenter".to_string(), // where do chasers start and end ("barycenter", "first")
        3,                        // maximum number of chasers
        8,                        // maximum debris load per chaser
        60. * 60. * 24. * 3000.,  // maximum mission time in seconds
    );

    // Set oracle (used to optimize maneuvers between states)
    let oracle = Oracle::new(
        "best".to_string(), // transfer strategy ("direct", "drift", "best")
        1e-3,               // maximum chaser thrust in m/s^2
        6878e3,             // minimum semimajor axis allowed during drift in m
    );

    // Set solver (used to optimize chaser routes)
    let mut solver = Solver::preset(
        4,    // aggressiveness level between 0 and 6 (higher -> slower but better solution)
        60.0, // time limit for the solver in seconds
        0,    // random seed used in genetic operations
    );

    // Run solver and recover best solution
    let solution = solver.solver(&problem, &oracle);

    // Print best solution
    println!("\n{}", solution);

    // Write solution to file if output path provided
    Writer::write("examples/drift.txt", &solution);

    Ok(())
}
