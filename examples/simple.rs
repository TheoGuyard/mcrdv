//! Minimal end-to-end example: plan a mission from a debris catalog.
//!
//! Port of `references/pyadr/examples/simple.py`.
//!
//! Run from the crate root (`delivery/scorpion`):
//!
//! ```text
//! cargo run --release --example simple
//! ```

use std::error::Error;

use scorpion::io::{print_mission, read_debris};
use scorpion::orbit::DAY;
use scorpion::problem::make_problem;
use scorpion::schedule::{NowaitConfig, NowaitScheduler};
use scorpion::sequence::{HgsConfig, HgsSequencer};
use scorpion::solver::Solver;
use scorpion::transfer::{QlawConfig, QlawTransfer};

const DATA: &str = "data/odrc.csv";

fn main() -> Result<(), Box<dyn Error>> {
    // Load a cluster of debris.
    let debris = read_debris(DATA)?;
    println!("loaded {} debris from {DATA}", debris.len());

    // Define the problem. The depot is placed at the debris centroid.
    let problem = make_problem(
        &debris,
        3,                  // number of chasers in the swarm
        Some(365.0 * DAY),  // per-chaser mission-time limit [s]
        Some(10),           // per-chaser debris-load limit [debris]
        Some(25_000.0),     // per-chaser fuel (delta-v) limit [m/s]
        "sum",              // mission cost = sum of the chaser costs
    )?;

    // Build the layer stack. Unlike `pyadr`, the Rust `Solver` has no implicit
    // default stack: every layer is constructed explicitly and borrows the
    // problem, so the wiring is visible here. This is the stack `pyadr.Solver()`
    // fills in by default -- swap `NowaitScheduler` for
    // `CdScheduler::default(&problem, &transfer)` to let the chasers wait for a
    // cheap RAAN alignment before each transfer (see `examples/medium.rs`).
    let transfer = QlawTransfer::new(&problem, QlawConfig::default());
    let schedule = NowaitScheduler::new(&problem, &transfer, NowaitConfig::default());
    let sequence = HgsSequencer::new(&problem, &transfer, &schedule, HgsConfig::default());

    // Solve the mission planning problem.
    let solver = Solver::new(&transfer, &schedule, &sequence, true);
    let mission = solver.solve();

    // Display the resulting mission plan.
    print_mission(&mission, &problem);

    Ok(())
}
