//! The dynamic-programming schedule layer.

mod common;

use std::rc::Rc;

use mcrdv::solver::{ScheduleLayer, TransferLayer};
use mcrdv::{DpSchedule, QlawTransfer};

fn sequences() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1],
        vec![0, 5, 3],
        vec![0, 2, 7, 4],
        vec![0, 9, 1, 6, 11],
        vec![0, 12, 8, 3, 10, 5],
    ]
}

#[test]
fn a_lone_departure_orbit_yields_the_empty_plan() {
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    let plan = schedule.evaluate(&[0]);
    assert_eq!(plan.load(), 0);
    assert_eq!(plan.cost(), 0.0);
}

#[test]
fn a_lone_object_is_scheduled_on_that_object() {
    // There is no leg to price, but the plan must still be about the object it
    // was handed rather than about the departure orbit.
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    let plan = schedule.evaluate(&[5]);
    assert_eq!(plan.sequence, vec![5]);
    assert_eq!(plan.load(), 0);
    assert_eq!(plan.cost(), 0.0);
}

#[test]
fn a_plan_follows_the_sequence_it_was_given() {
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    for sequence in sequences() {
        let plan = schedule.evaluate(&sequence);
        assert_eq!(plan.sequence, sequence);
        assert_eq!(plan.load(), sequence.len() - 1);
        assert_eq!(plan.depart.len(), sequence.len());
    }
}

#[test]
fn arrivals_follow_from_the_departures_through_the_oracle() {
    // The reported timeline must be exactly what the transfer layer implies,
    // not an independent accumulation that could drift from it.
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    for sequence in sequences() {
        let plan = schedule.evaluate(&sequence);
        for i in 0..plan.len() - 1 {
            let (dt, dc) =
                schedule.transfer().evaluate(sequence[i], sequence[i + 1], plan.depart[i]);
            let expected = plan.depart[i] + dt;
            assert!(
                (plan.arrive[i + 1] - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "leg {i} of {sequence:?}"
            );
            assert!((plan.costs[i + 1] - dc).abs() <= 1e-9 * dc.abs().max(1.0));
        }
        assert_eq!(plan.arrive[0], 0.0);
        assert_eq!(plan.costs[0], 0.0);
    }
}

#[test]
fn a_chaser_never_departs_before_it_arrives() {
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    for sequence in sequences() {
        let plan = schedule.evaluate(&sequence);
        for i in 0..plan.len() - 1 {
            assert!(
                plan.depart[i] >= plan.arrive[i] - 1e-6,
                "departs {} before arriving {} at position {i}",
                plan.depart[i],
                plan.arrive[i]
            );
        }
    }
}

#[test]
fn waiting_is_never_worse_than_the_no_wait_schedule() {
    // Optimizing departure times searches a superset of the no-wait schedule,
    // so it cannot come back more expensive.
    let problem = common::problem();
    let mut waiting = DpSchedule::new(QlawTransfer::new());
    let mut nowait = DpSchedule::with_options(QlawTransfer::new(), false, 32, true, true);
    waiting.initialize(&problem);
    nowait.initialize(&problem);

    let mut strictly_better = 0;
    for sequence in sequences() {
        let w = waiting.evaluate(&sequence).cost();
        let n = nowait.evaluate(&sequence).cost();
        assert!(w <= n + 1e-6, "{sequence:?}: waiting {w} > no-wait {n}");
        if w < n - 1e-6 {
            strictly_better += 1;
        }
    }
    assert!(strictly_better > 0, "loitering should pay on at least one sequence");
}

#[test]
fn the_bound_never_exceeds_the_scheduled_cost() {
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    for sequence in sequences() {
        let bound = schedule.evaluate_lb(&sequence);
        let cost = schedule.evaluate(&sequence).cost();
        assert!(bound <= cost + 1e-6, "{sequence:?}: bound {bound} > cost {cost}");
    }
}

#[test]
fn the_bound_of_a_single_object_is_zero() {
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    assert_eq!(schedule.evaluate_lb(&[0]), 0.0);
}

#[test]
fn the_cache_hands_back_the_very_same_plan() {
    let problem = common::problem();
    let schedule = common::schedule(&problem);
    let first = schedule.evaluate(&[0, 4, 2]);
    let second = schedule.evaluate(&[0, 4, 2]);
    assert!(Rc::ptr_eq(&first, &second));
}

#[test]
fn results_do_not_depend_on_the_cache_being_on() {
    let problem = common::problem();
    let mut cached = DpSchedule::with_options(QlawTransfer::new(), true, 32, true, true);
    let mut uncached = DpSchedule::with_options(QlawTransfer::new(), true, 32, true, false);
    cached.initialize(&problem);
    uncached.initialize(&problem);
    for sequence in sequences() {
        assert_eq!(
            cached.evaluate(&sequence).cost(),
            uncached.evaluate(&sequence).cost(),
            "{sequence:?}"
        );
    }
}

#[test]
fn hints_do_not_make_the_schedule_worse() {
    // They only add candidate departure times, so the optimum can only improve.
    let problem = common::problem();
    let mut with = DpSchedule::with_options(QlawTransfer::new(), true, 32, true, true);
    let mut without = DpSchedule::with_options(QlawTransfer::new(), true, 32, false, true);
    with.initialize(&problem);
    without.initialize(&problem);
    for sequence in sequences() {
        let a = with.evaluate(&sequence).cost();
        let b = without.evaluate(&sequence).cost();
        assert!(a <= b + 1e-6, "{sequence:?}: with hints {a} > without {b}");
    }
}

#[test]
fn initialize_drops_the_memoized_plans() {
    let problem = common::problem();
    let mut schedule = DpSchedule::new(QlawTransfer::new());
    schedule.initialize(&problem);
    let first = schedule.evaluate(&[0, 1, 2]);

    schedule.initialize(&problem);
    let second = schedule.evaluate(&[0, 1, 2]);

    assert!(!Rc::ptr_eq(&first, &second), "the plan cache survived initialize()");
    assert_eq!(first.cost(), second.cost(), "yet the answer must not change");
}

#[test]
fn re_initializing_on_another_catalogue_reprices_from_scratch() {
    // A stale cache would answer the second instance with the first one's plan.
    let first = common::problem();
    let mut schedule = DpSchedule::new(QlawTransfer::new());
    schedule.initialize(&first);
    let before = schedule.evaluate(&[0, 1, 2]).cost();

    let other = mcrdv::read_debris("data/iridium33.csv");
    let mut states = vec![mcrdv::centroid(&other)];
    states.extend(other);
    let second = Rc::new(mcrdv::Problem::new(states, 3, 2500.0 * mcrdv::DAY, 40));

    schedule.initialize(&second);
    let after = schedule.evaluate(&[0, 1, 2]).cost();
    assert!((before - after).abs() > 1e-6, "state survived initialize(): {before} vs {after}");
}

#[test]
#[should_panic(expected = "grid_size")]
fn a_zero_grid_is_refused() {
    let _ = DpSchedule::with_options(QlawTransfer::new(), true, 0, true, true);
}
