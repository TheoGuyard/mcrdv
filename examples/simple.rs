//! Run with `cargo run --release --example simple`.

use mcrdv::{centroid, format_mission, read_debris, DAY};
use mcrdv::{DpSchedule, HgsSequence, Problem, QlawTransfer, Solver};

fn main() {
    // Load a debris catalogue.
    let path = "data/iridium33.csv";
    let debris = read_debris(path);
    println!("Loaded {} debris from {path}", debris.len());

    // Index 0 is the orbit the chasers depart from, by convention.
    let mut states = vec![centroid(&debris)];
    states.extend(debris);

    // Define the instance and its operational limits.
    let problem = Problem::new(
        states,
        15,           // chasers in the swarm
        365.0 * DAY,  // per-chaser mission duration limit
        10,           // per-chaser debris capacity
    );
    println!("{problem}");

    // Build the solver layer stack from the bottom up.
    let transfer = QlawTransfer::new();
    let schedule = DpSchedule::new(transfer);
    let sequence = HgsSequence::new(schedule);
    let mut solver = Solver::new(sequence);
    
    // Solver the problem
    let mission = solver.solve(&problem);

    println!("{}", format_mission(&problem, &mission));
}
