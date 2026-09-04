//! The three layers wired together, end to end.

mod common;

use std::rc::Rc;

use mcrdv::{
    DpSchedule, HgsConfig, HgsSequence, Problem, QlawTransfer, Solver, DAY,
};

pub type DefaultSequence = HgsSequence<DpSchedule<QlawTransfer>>;
pub type DefaultSolver = Solver<DefaultSequence>;

/// Build a solver with the default layer stack and default settings.
pub fn default_solver() -> DefaultSolver {
    Solver::new(
        HgsSequence::new(
            DpSchedule::new(QlawTransfer::new())
        )
    )
}


/// The default wiring, on a search budget sized for a test suite.
fn quick() -> Solver<HgsSequence<DpSchedule<QlawTransfer>>> {
    let config = HgsConfig {
        seed: 0,
        verbose: false,
        max_iter: 20,
        pop_init: 4,
        ..HgsConfig::default()
    };
    Solver::new(HgsSequence::with_config(
        DpSchedule::new(QlawTransfer::new()),
        config,
    ))
    .verbose(false)
}

#[test]
fn a_solved_mission_collects_the_whole_catalogue() {
    let problem = common::problem();
    let mission = quick().solve(&problem);
    assert_eq!(
        common::collected(&mission),
        (1..problem.states.len()).collect::<Vec<_>>()
    );
    assert_eq!(mission.plans.len(), problem.num_chasers);
}

#[test]
fn a_comfortable_instance_is_solved_feasibly() {
    let problem = common::problem();
    let mission = quick().solve(&problem);
    assert!(problem.is_feasible(&mission), "cost {}", mission.cost());
}

#[test]
fn the_duration_limit_holds_on_every_feasible_plan() {
    let problem = common::problem();
    let mission = quick().solve(&problem);
    for plan in &mission.plans {
        if problem.is_plan_feasible(plan) {
            assert!(plan.time() <= problem.max_time + Problem::TIME_TOLERANCE);
        }
    }
}

#[test]
fn solving_twice_gives_the_same_mission() {
    let problem = common::problem();
    let mut solver = quick();
    assert_eq!(solver.solve(&problem), solver.solve(&problem));
}

#[test]
fn a_reused_solver_carries_no_state_between_instances() {
    let mut solver = quick();
    let first = common::problem();
    let mission = solver.solve(&first);
    assert_eq!(
        common::collected(&mission),
        (1..first.states.len()).collect::<Vec<_>>()
    );

    let other = mcrdv::read_debris("data/iridium33.csv");
    let mut states = vec![mcrdv::centroid(&other)];
    states.extend(other);
    let second = Rc::new(Problem::new(states, 12, 2500.0 * DAY, 12));

    let mission = solver.solve(&second);
    assert_eq!(
        common::collected(&mission),
        (1..second.states.len()).collect::<Vec<_>>()
    );
}

#[test]
fn the_verbosity_switch_is_carried_on_the_solver() {
    let solver = quick();
    assert!(!solver.verbose);
    assert!(Solver::new(HgsSequence::new(DpSchedule::new(QlawTransfer::new()))).verbose);
}

#[test]
fn the_layers_are_reachable_through_the_stack() {
    let solver = quick();
    // Ownership nesting means the three can never be wired to different
    // instances, so there is nothing to validate at run time.
    let transfer = solver.sequence.schedule().transfer();
    assert_eq!(transfer.num_samples, 512);
}

#[test]
fn the_default_stack_solves_the_bundled_instance() {
    let problem = common::problem();
    let mut solver = default_solver().verbose(false);
    solver.sequence.config = HgsConfig {
        verbose: false,
        max_iter: 20,
        pop_init: 4,
        ..HgsConfig::default()
    };
    let mission = solver.solve(&problem);
    assert_eq!(
        common::collected(&mission),
        (1..problem.states.len()).collect::<Vec<_>>()
    );
}
