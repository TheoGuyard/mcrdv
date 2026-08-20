//! Component tests for the `Solver` wiring and objective aggregation.

mod common;

use common::*;
use scorpion::orbit::DAY;
use scorpion::problem::{make_problem, MissionPlan, Objective, ObjectiveMax, ObjectiveSum, Problem};
use scorpion::schedule::CdConfig;
use scorpion::sequence::{HgsConfig, HgsSequencer};
use scorpion::solver::Solver;

fn quiet(seed: u64) -> HgsConfig {
    HgsConfig {
        seed,
        log_iter: 0,
        pop_init: 10,
        max_iter: 150,
        max_nimp: 100,
        ..HgsConfig::default()
    }
}

fn solve(objective: &str) -> (MissionPlan, Problem) {
    let problem = make_problem(
        &debris(),
        3,
        Some(2500.0 * DAY),
        Some(6),
        None,
        objective,
    )
    .expect("problem must be constructible");
    let mission = {
        let transfer = transfer_with(&problem, 3e-3);
        let schedule = cd(&problem, &transfer, CdConfig::default());
        let sequence = HgsSequencer::new(&problem, &transfer, &schedule, quiet(0));
        Solver::new(&transfer, &schedule, &sequence, false).solve()
    };
    (mission, problem)
}

#[test]
fn mission_collected_excludes_depot() {
    let (mission, problem) = solve("sum");
    assert!(
        !mission.collected().contains(&0),
        "depot wrongly counted as collected debris"
    );
    assert_eq!(
        mission.collected(),
        problem.debris(),
        "not every debris collected exactly once"
    );
}

#[test]
fn objective_sum_and_max() {
    let (total, _) = solve("sum");
    let (worst, _) = solve("max");

    let sum: f64 = total.chaser_plans.iter().map(|p| p.cost).sum();
    assert_approx(total.cost, sum, 1e-9, "sum objective is not the sum of chaser costs");

    let mx = worst
        .chaser_plans
        .iter()
        .map(|p| p.cost)
        .fold(f64::NEG_INFINITY, f64::max);
    assert_approx(worst.cost, mx, 1e-9, "max objective is not the max chaser cost");
}

#[test]
fn objective_aggregations_are_correct() {
    let costs = [1.0, 2.0, 3.0];
    assert_eq!(ObjectiveSum.aggregate(&costs), 6.0);
    assert_eq!(ObjectiveMax.aggregate(&costs), 3.0);
    assert_eq!(ObjectiveSum.aggregate(&[]), 0.0);
    assert_eq!(ObjectiveMax.aggregate(&[]), 0.0, "an empty max must not be -inf");
}

#[test]
fn solver_runs_the_full_stack() {
    let (mission, problem) = solve("sum");
    assert!(mission.cost > 0.0, "a real mission cannot be free");
    assert!(
        mission.chaser_plans.len() <= problem.num_chasers,
        "solver used more routes than chasers"
    );
}
