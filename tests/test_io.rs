mod common;

use std::fs;

use common::{default_oracle, make_test_debris, relaxed_problem};
use scorpion::io::{Loader, Writer};
use scorpion::optim::Solver;

#[test]
fn test_loader() {
    let states = Loader::load("data/ordc.csv");
    assert!(
        !states.is_empty(),
        "Should load at least one debris from ordc.csv"
    );
    // Check that values are physically reasonable
    for s in &states {
        assert!(s.a > 6e6 && s.a < 1e8, "Semi-major axis out of range");
        assert!(s.e >= 0.0 && s.e < 1.0, "Eccentricity out of range");
        assert!(s.i >= 0.0, "Inclination should be non-negative");
    }
}

#[test]
#[should_panic]
fn test_loader_nonexistent_file_panics() {
    let _ = Loader::load("data/nonexistent_file_12345.csv");
}

#[test]
fn test_writer_roundtrip() {
    let debris = make_test_debris(3);
    let problem = relaxed_problem(&debris);
    let oracle = default_oracle();
    let mut solver = Solver::preset(0);
    let (sol, _) = solver.solve(&problem, &oracle);

    let path = "target/tmp/test_output.txt";
    fs::create_dir_all("target/tmp").unwrap();
    Writer::write(path, &sol);

    let content = fs::read_to_string(path).unwrap();
    assert!(content.contains("Solution"));
    assert!(content.contains("feasible"));
    assert!(content.contains("total cost"));
    assert!(content.contains("depot"));

    // Clean up
    let _ = fs::remove_file(path);
}
