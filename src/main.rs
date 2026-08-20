//! Entry point: end-to-end mission solve with the full layer stack.

use std::error::Error;

use scorpion::orbit::DAY;
use scorpion::io::{print_mission, read_debris};
use scorpion::problem::make_problem;
use scorpion::schedule::CdScheduler;
use scorpion::sequence::HgsSequencer;
use scorpion::transfer::QlawTransfer;
use scorpion::solver::Solver;

fn main() -> Result<(), Box<dyn Error>> {

    // Load debris catalog
    let path = "data/iridium33.csv";
    let debris = read_debris(path)?;
    println!("Loaded {} debris from {}", debris.len(), path);

    // Build problem
    let problem = make_problem(
        &debris,
        15,                  // number of chasers
        Some(365.0 * DAY),   // per-chaser mission-time limit [s]
        Some(10),            // per-chaser debris-load limit
        Some(25_000.0),      // per-chaser cost (delta-v) limit [m/s]
        "sum",               // mission objective aggregation
    )?;

    // Build layers
    let transfer = QlawTransfer::default(&problem);
    let schedule = CdScheduler::default(&problem, &transfer);
    let sequence = HgsSequencer::default(&problem, &transfer, &schedule);

    // Solve the problem
    let solver = Solver::new(&transfer, &schedule, &sequence, true);
    let result = solver.solve();
    print_mission(&result, &problem);
    
    Ok(())
}
