//! Instance validation and feasibility.

mod common;

use mcrdv::{ChaserPlan, MissionPlan, Problem, DAY};

#[test]
fn a_sound_instance_is_accepted() {
    let problem = Problem::new(common::states(), 3, 2500.0 * DAY, 5);
    assert_eq!(problem.num_chasers, 3);
    assert_eq!(problem.num_debris(), 13);
}

#[test]
#[should_panic(expected = "'num_chasers' must be positive")]
fn the_swarm_size_must_be_positive() {
    Problem::new(common::states(), 0, 2500.0 * DAY, 5);
}

#[test]
#[should_panic(expected = "'max_time' must be positive")]
fn the_time_limit_must_be_positive() {
    Problem::new(common::states(), 3, 0.0, 5);
}

#[test]
#[should_panic(expected = "'max_time' must be finite")]
fn the_time_limit_must_be_finite() {
    Problem::new(common::states(), 3, f64::INFINITY, 5);
}

#[test]
#[should_panic(expected = "'max_load' must be positive")]
fn the_load_limit_must_be_positive() {
    Problem::new(common::states(), 3, 2500.0 * DAY, 0);
}

#[test]
#[should_panic(expected = "'states' must be of length >= 2")]
fn an_instance_without_debris_is_refused() {
    Problem::new(common::states()[..1].to_vec(), 3, 2500.0 * DAY, 5);
}

#[test]
#[should_panic(expected = "insufficient capacity over the swarm")]
fn a_swarm_too_small_to_carry_the_catalogue_is_refused() {
    // 13 debris, but 2 chasers holding 5 each can only take 10.
    Problem::new(common::states(), 2, 2500.0 * DAY, 5);
}

#[test]
fn exactly_sufficient_capacity_is_accepted() {
    let problem = Problem::new(common::states(), 13, 2500.0 * DAY, 1);
    assert_eq!(problem.num_debris(), 13);
}

#[test]
fn an_empty_plan_is_feasible_and_costs_nothing() {
    let problem = common::problem();
    let plan = ChaserPlan::empty();
    assert_eq!(plan.load(), 0);
    assert_eq!(plan.cost(), 0.0);
    assert_eq!(plan.time(), 0.0);
    assert!(problem.is_plan_feasible(&plan));
}

#[test]
fn an_empty_plan_reports_itself_as_empty() {
    // A plan always holds the object it starts from, so emptiness is about the
    // load and not about the length of the sequence.
    assert!(ChaserPlan::empty().is_empty());
    let plan = ChaserPlan::new(vec![0, 1], vec![0.0, 5.0], vec![0.0, 5.0], vec![0.0, 2.0]);
    assert!(!plan.is_empty());
}

#[test]
fn a_plan_over_the_load_limit_is_infeasible() {
    let problem = common::problem();
    let n = problem.max_load + 2;
    let plan = ChaserPlan::new(
        (0..=n).collect(),
        vec![0.0; n + 1],
        vec![0.0; n + 1],
        vec![0.0; n + 1],
    );
    assert!(!problem.is_plan_feasible(&plan));
}

#[test]
fn a_plan_over_the_time_limit_is_infeasible() {
    let problem = common::problem();
    let over = problem.max_time + 10.0 * Problem::TIME_TOLERANCE;
    let plan = ChaserPlan::new(vec![0, 1], vec![0.0, over], vec![0.0, over], vec![0.0, 1.0]);
    assert!(!problem.is_plan_feasible(&plan));
}

#[test]
fn the_time_limit_carries_a_small_tolerance() {
    // Accumulated arrival times drift by fractions of a second; a plan must not
    // be called infeasible for that.
    let problem = common::problem();
    let just_over = problem.max_time + 0.5 * Problem::TIME_TOLERANCE;
    let plan = ChaserPlan::new(
        vec![0, 1],
        vec![0.0, just_over],
        vec![0.0, just_over],
        vec![0.0, 1.0],
    );
    assert!(problem.is_plan_feasible(&plan));
}

#[test]
fn an_infinite_cost_makes_a_plan_infeasible() {
    let problem = common::problem();
    let plan = ChaserPlan::new(vec![0, 1], vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, f64::INFINITY]);
    assert!(!problem.is_plan_feasible(&plan));
}

#[test]
fn a_mission_is_feasible_only_when_every_plan_is() {
    let problem = common::problem();
    let bad = ChaserPlan::new(vec![0, 1], vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, f64::INFINITY]);
    let mission = MissionPlan::new(vec![ChaserPlan::empty(), bad]);
    assert!(!problem.is_feasible(&mission));
}

#[test]
fn mission_cost_is_the_sum_over_the_swarm() {
    let a = ChaserPlan::new(vec![0, 1], vec![0.0; 2], vec![0.0; 2], vec![0.0, 3.0]);
    let b = ChaserPlan::new(vec![0, 2], vec![0.0; 2], vec![0.0; 2], vec![0.0, 4.0]);
    assert_eq!(MissionPlan::new(vec![a, b]).cost(), 7.0);
}

#[test]
fn the_debris_range_excludes_the_departure_orbit() {
    let problem = common::problem();
    assert_eq!(problem.num_debris(), problem.states.len() - 1);
    assert_eq!(problem.debris().next(), Some(1));
}
