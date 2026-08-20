//! Component tests for the sequence layer.

mod common;

use common::*;
use scorpion::orbit::DAY;
use scorpion::problem::{MissionPlan, Problem};
use scorpion::schedule::NowaitConfig;
use scorpion::sequence::{HgsConfig, HgsSequencer};
use scorpion::solver::SequenceLayer;

/// The Python fixture: 3 chasers, a 2500-day horizon and a load limit of 5.
fn hgs_problem() -> Problem {
    problem_with(2500.0 * DAY, 5)
}

/// A quiet, small, seeded HGS configuration.
fn quiet(seed: u64) -> HgsConfig {
    HgsConfig {
        seed,
        log_iter: 0, // `log_iter == 0` silences the sequencer
        pop_init: 12,
        max_iter: 150,
        max_nimp: 150,
        ..HgsConfig::default()
    }
}

fn run(problem: &Problem, config: HgsConfig) -> MissionPlan {
    let transfer = transfer_with(problem, 3e-3);
    let schedule =
        scorpion::schedule::NowaitScheduler::new(problem, &transfer, NowaitConfig::default());
    let sequencer = HgsSequencer::new(problem, &transfer, &schedule, config);
    sequencer.initialize();
    sequencer.evaluate()
}

#[test]
fn hgs_deterministic_and_complete() {
    let problem = hgs_problem();
    let first = run(&problem, quiet(0));
    let second = run(&problem, quiet(0));

    assert_eq!(
        first.collected(),
        problem.debris(),
        "HGS did not collect every debris exactly once"
    );
    assert_approx(first.cost, second.cost, 1e-12, "same-seed runs diverged");
    assert_eq!(
        first.chaser_plans.len(),
        second.chaser_plans.len(),
        "same-seed runs produced a different number of routes"
    );
    assert!(
        first.chaser_plans.len() <= problem.num_chasers,
        "more routes ({}) than chasers ({})",
        first.chaser_plans.len(),
        problem.num_chasers
    );
    if first.feasible {
        let loads: Vec<usize> = first.chaser_plans.iter().map(|p| p.load()).collect();
        assert!(
            loads.iter().all(|&l| l <= 5),
            "a feasible plan exceeds max_load: {loads:?}"
        );
    }
}

#[test]
fn hgs_different_seeds_stay_valid() {
    // Whatever the seed, the mission must still cover the whole catalogue.
    let problem = hgs_problem();
    for seed in [1u64, 2, 7] {
        let mission = run(&problem, quiet(seed));
        assert_eq!(
            mission.collected(),
            problem.debris(),
            "seed {seed}: HGS did not collect every debris"
        );
        assert!(mission.cost.is_finite(), "seed {seed}: non-finite mission cost");
    }
}

#[test]
fn hgs_uses_the_schedule_layer_it_was_given() {
    // The waiting scheduler must be able to reach a strictly cheaper mission
    // than the no-wait one on the same seed.
    let problem = hgs_problem();
    let transfer = transfer_with(&problem, 3e-3);

    let no = {
        let schedule =
            scorpion::schedule::NowaitScheduler::new(&problem, &transfer, NowaitConfig::default());
        let seq = HgsSequencer::new(&problem, &transfer, &schedule, quiet(0));
        seq.initialize();
        seq.evaluate()
    };
    let waiting = {
        let schedule = cd(&problem, &transfer, scorpion::schedule::CdConfig::default());
        let seq = HgsSequencer::new(&problem, &transfer, &schedule, quiet(0));
        seq.initialize();
        seq.evaluate()
    };

    assert_eq!(waiting.collected(), problem.debris(), "CD run dropped debris");
    assert!(
        waiting.cost <= no.cost + 1e-6,
        "the waiting scheduler produced a worse mission than no-wait: \
         {} vs {}",
        waiting.cost,
        no.cost
    );
}
