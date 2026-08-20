//! A larger mission over the Iridium-33 debris cloud.
//!
//! Port of `references/pyadr/examples/medium.py`.
//!
//! Run from the crate root (`delivery/scorpion`):
//!
//! ```text
//! cargo run --release --example medium
//! ```

use std::error::Error;

use scorpion::io::{print_mission, read_debris};
use scorpion::orbit::DAY;
use scorpion::problem::make_problem;
use scorpion::schedule::{CdConfig, CdScheduler};
use scorpion::sequence::{HgsConfig, HgsSequencer};
use scorpion::solver::Solver;
use scorpion::transfer::{QlawConfig, QlawTransfer};

const DATA: &str = "data/iridium33.csv";

fn main() -> Result<(), Box<dyn Error>> {
    let debris = read_debris(DATA)?;
    println!("loaded {} debris from {DATA}", debris.len());

    let problem = make_problem(
        &debris,
        15,                 // number of chasers in the swarm
        Some(365.0 * DAY),  // per-chaser mission-time limit [s]
        Some(10),           // per-chaser debris-load limit [debris]
        Some(25_000.0),     // per-chaser fuel (delta-v) limit [m/s]
        "sum",
    )?;

    // Solve the mission planning problem with a coordinate-descent scheduler.
    let transfer = QlawTransfer::new(&problem, QlawConfig::default());
    let schedule = CdScheduler::new(&problem, &transfer, CdConfig::default());
    let sequence = HgsSequencer::new(
        &problem,
        &transfer,
        &schedule,
        HgsConfig {
            log_iter: 1,
            nb_neighbors: 1,
            ..HgsConfig::default()
        },
    );

    let solver = Solver::new(&transfer, &schedule, &sequence, true);
    let mission = solver.solve();

    print_mission(&mission, &problem);

    Ok(())
}
