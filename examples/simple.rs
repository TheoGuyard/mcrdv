use scorpion::Problem;
use scorpion::io::{Loader, Writer};
use scorpion::optim::Solver;
use scorpion::orbit::Oracle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load debris from input file
    let debris = Loader::load("data/iridium33.csv");

    // Set problem data
    let problem = Problem::new(
        &debris,                  // debris states
        "barycenter".to_string(), // where do chasers start and end ("barycenter", "first")
        20,                       // maximum number of chasers
        20,                       // maximum debris load per chaser
        60. * 60. * 24. * 365.,   // maximum mission time in seconds
    );

    // Set oracle (used to optimize maneuvers between states)
    let oracle = Oracle::new(
        "direct".to_string(), // transfer strategy ("direct", "drift", "best")
        0.003,                // maximum chaser thrust in m/s^2
    );

    // Set solver (used to optimize chaser routes)
    let mut solver = Solver::preset(
        3, // aggressiveness level between 0 and 6 (higher -> slower but better solution)
    );

    // Run solver to recover best solution and trace
    let (solution, _trace) = solver.solve(&problem, &oracle);

    // Print best solution
    println!("\n{}", solution);

    // Write solution to file if output path provided
    Writer::write("examples/simple.txt", &solution);

    Ok(())
}
