//! Reading catalogues and reporting missions.

mod common;

use std::fs;

use mcrdv::io::{format_mission, read_debris, write_mission, CSV_HEADER};
use mcrdv::{ChaserPlan, MissionPlan};

#[test]
fn every_bundled_catalogue_reads() {
    for (name, count) in [
        ("odrc", 13usize),
        ("cosmos1408", 4),
        ("iridium33", 110),
        ("cosmos2251", 583),
        ("fengyun1C", 1847),
    ] {
        let debris = read_debris(format!("data/{name}.csv"));
        assert_eq!(debris.len(), count, "{name}");
    }
}

#[test]
fn the_elements_of_the_first_row_are_read_in_order() {
    let debris = read_debris("data/odrc.csv");
    let first = debris[0];
    assert!((first.a - 7_164_040.551_8).abs() < 1e-6);
    assert!((first.e - 0.0019).abs() < 1e-12);
    assert!((first.i - 1.5079).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "invalid csv header")]
fn a_wrong_header_is_rejected() {
    let path = std::env::temp_dir().join("mcrdv_bad_header.csv");
    fs::write(&path, "a,b,c,d,e,f\n1,2,3,4,5,6\n").unwrap();
    let _ = read_debris(&path);
}

#[test]
#[should_panic(expected = "line 3: cannot parse 'oops' as a number")]
fn a_malformed_row_names_its_line() {
    let path = std::env::temp_dir().join("mcrdv_bad_row.csv");
    fs::write(&path, format!("{CSV_HEADER}\n1,2,3,4,5,6\n1,2,oops,4,5,6\n")).unwrap();
    let _ = read_debris(&path);
}

#[test]
#[should_panic(expected = "cannot open 'data/does_not_exist.csv'")]
fn a_missing_file_reports_the_underlying_error() {
    let _ = read_debris("data/does_not_exist.csv");
}

#[test]
fn blank_lines_are_skipped() {
    let path = std::env::temp_dir().join("mcrdv_blank.csv");
    fs::write(&path, format!("{CSV_HEADER}\n7e6,0.001,1.5,0.1,0.2,0.3\n\n")).unwrap();
    assert_eq!(read_debris(&path).len(), 1);
    let _ = fs::remove_file(&path);
}

#[test]
fn a_written_mission_can_be_read_back() {
    let mission = MissionPlan::new(vec![
        ChaserPlan::new(vec![0, 3], vec![1.5, 2.5], vec![0.0, 9.5], vec![0.0, 42.25]),
        ChaserPlan::empty(),
    ]);
    let path = std::env::temp_dir().join("mcrdv_mission.csv");
    write_mission(&mission, &path, 6);

    let text = fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "chaser,node,depart,arrive,cost");
    assert_eq!(lines.len(), 1 + 2 + 1, "two rows for chaser 0, one for chaser 1");
    assert!(lines[2].starts_with("0,3,2.500000,9.500000,42.250000"), "{}", lines[2]);
    let _ = fs::remove_file(&path);
}

#[test]
#[should_panic(expected = "cannot write")]
fn an_unwritable_destination_is_reported() {
    let mission = MissionPlan::new(vec![ChaserPlan::empty()]);
    // A path whose parent directory does not exist cannot be created.
    let path = std::env::temp_dir().join("mcrdv_absent_dir").join("mission.csv");
    write_mission(&mission, &path, 6);
}

#[test]
fn the_report_shows_the_headline_figures() {
    let problem = common::problem();
    let mission = MissionPlan::new(vec![ChaserPlan::empty(); problem.num_chasers]);
    let report = format_mission(&problem, &mission);

    assert!(report.contains("Mission plan"));
    assert!(report.contains("Feasible     : true"));
    assert!(report.contains("Chasers used : 0/3"));
    // An idle chaser reports its arrival rather than a dash.
    assert!(report.contains("Chaser 2 (feasible: true, cost: 0.0)"));
}
