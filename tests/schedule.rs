//! Component tests for the schedule layer.

mod common;

use common::*;
use scorpion::schedule::CdConfig;
use scorpion::solver::ScheduleLayer;

/// The Python-default CD knobs, as a starting point for the variations below.
fn cfg() -> CdConfig {
    CdConfig::default()
}

// ------------------------------- coherence -------------------------------- //

#[test]
fn nowait_coherence() {
    let problem = problem();
    let transfer = transfer(&problem);
    let plan = nowait(&problem, &transfer).evaluate(SEQ);
    assert!(
        plan.waiting_times.iter().all(|&w| w == 0.0),
        "no-wait schedule has waiting: {:?}",
        plan.waiting_times
    );
    assert_coherent(&plan, &transfer, SEQ);
}

#[test]
fn cd_coherence() {
    let problem = problem();
    let transfer = transfer(&problem);
    let config = CdConfig { samples: 16, max_iters: 2, ..cfg() };
    let plan = cd(&problem, &transfer, config).evaluate(SEQ);
    assert_coherent(&plan, &transfer, SEQ);
    assert!(
        plan.mission_time() <= problem.time_horizon * (1.0 + 1e-9),
        "CD schedule overshoots the horizon: {} > {}",
        plan.mission_time(),
        problem.time_horizon
    );
}

#[test]
fn dp_coherence() {
    let problem = problem();
    let transfer = transfer(&problem);
    let plan = dp(&problem, &transfer, 24).evaluate(SEQ);
    assert_coherent(&plan, &transfer, SEQ);
}

#[test]
fn lower_bounds_are_valid() {
    let problem = problem();
    let transfer = transfer(&problem);

    let schedulers: Vec<(&str, Box<dyn ScheduleLayer>)> = vec![
        ("NowaitScheduler", Box::new(nowait(&problem, &transfer))),
        ("CdScheduler", Box::new(cd(&problem, &transfer, CdConfig { max_iters: 2, ..cfg() }))),
        ("DpScheduler", Box::new(dp(&problem, &transfer, 16))),
    ];
    for (name, sched) in schedulers {
        let plan = sched.evaluate(SEQ);
        assert!(
            sched.evaluate_lb(SEQ) <= plan.cost + 1e-6,
            "{name}: lower bound {} exceeds the achieved cost {}",
            sched.evaluate_lb(SEQ),
            plan.cost
        );
    }
}

// -------------------------- waiting beats no-wait ------------------------- //

#[test]
fn waiting_schedulers_beat_nowait() {
    let problem = problem();
    let transfer = transfer(&problem);
    let no = nowait(&problem, &transfer).evaluate(SEQ).cost;
    let c = cd(&problem, &transfer, CdConfig { samples: 24, max_iters: 3, ..cfg() })
        .evaluate(SEQ)
        .cost;
    let d = dp(&problem, &transfer, 32).evaluate(SEQ).cost;
    assert!(c < no, "CdScheduler ({c}) did not beat NowaitScheduler ({no})");
    assert!(d < no, "DpScheduler ({d}) did not beat NowaitScheduler ({no})");
}

// ----------------------------- knob behaviour ----------------------------- //

#[test]
fn cd_max_iters_never_worsen() {
    let problem = problem();
    let transfer = transfer(&problem);
    let costs: Vec<f64> = [0usize, 1, 2, 4]
        .iter()
        .map(|&max_iters| {
            cd(&problem, &transfer, CdConfig { samples: 24, max_iters, ..cfg() })
                .evaluate(SEQ)
                .cost
        })
        .collect();
    assert!(
        costs.windows(2).all(|w| w[1] <= w[0] + 1e-6),
        "more coordinate-descent iters made the schedule worse: {costs:?}"
    );
}

#[test]
fn cd_forward_pass_beats_no_wait() {
    // The forward pass (max_iters = 0) already departs at the RAAN-alignment
    // hints, so waiting there must be at least as good as the no-wait schedule
    // -- and in practice strictly cheaper on a multi-leg route.
    let problem = problem();
    let transfer = transfer(&problem);
    let forward = cd(&problem, &transfer, CdConfig { max_iters: 0, ..cfg() })
        .evaluate(SEQ)
        .cost;
    let no = nowait(&problem, &transfer).evaluate(SEQ).cost;
    assert!(
        forward <= no + 1e-6,
        "CD forward pass ({forward:.1}) is worse than no-wait ({no:.1})"
    );
}

#[test]
fn cd_without_hints_matches_no_wait() {
    // With hints disabled and no refinement, CD reduces to the no-wait schedule.
    let problem = problem();
    let transfer = transfer(&problem);
    let config = CdConfig { max_iters: 0, use_hints: false, ..cfg() };
    let plan = cd(&problem, &transfer, config).evaluate(SEQ);
    let reference = nowait(&problem, &transfer).evaluate(SEQ);
    assert_approx(
        plan.cost,
        reference.cost,
        1e-9,
        "Cd(use_hints=false, max_iters=0) cost != no-wait",
    );
    assert_eq!(
        plan.meeting_times, reference.meeting_times,
        "Cd(use_hints=false, max_iters=0) meeting times differ from no-wait"
    );
}

#[test]
fn cd_more_samples_never_worsen() {
    // The incumbent departure is always among the candidates, and the grids are
    // over the same interval, so a finer grid can only help.
    let problem = problem();
    let transfer = transfer(&problem);
    let costs: Vec<f64> = [2usize, 8, 16, 32, 64]
        .iter()
        .map(|&samples| {
            cd(&problem, &transfer, CdConfig { samples, max_iters: 3, ..cfg() })
                .evaluate(SEQ)
                .cost
        })
        .collect();
    assert!(
        costs.iter().all(|&c| c <= costs[0] + 1e-6),
        "a finer CD grid made the schedule worse than the 2-point grid: {costs:?}"
    );
}

#[test]
fn dp_finer_grid_never_worsens() {
    // Nested grids (each a superset of the previous) -> DP optimum is monotone.
    let problem = problem();
    let transfer = transfer(&problem);
    let costs: Vec<f64> = [4usize, 8, 16, 32]
        .iter()
        .map(|&gs| dp(&problem, &transfer, gs).evaluate(SEQ).cost)
        .collect();
    assert!(
        costs.windows(2).all(|w| w[1] <= w[0] + 1e-6),
        "a finer DP grid made the schedule worse: {costs:?}"
    );
}

#[test]
fn dp_hints_help() {
    use scorpion::schedule::{DpConfig, DpScheduler};
    let problem = problem();
    let transfer = transfer(&problem);
    let with = DpScheduler::new(
        &problem,
        &transfer,
        DpConfig { grid_size: 8, use_hints: true, use_cache: true },
    )
    .evaluate(SEQ)
    .cost;
    let without = DpScheduler::new(
        &problem,
        &transfer,
        DpConfig { grid_size: 8, use_hints: false, use_cache: true },
    )
    .evaluate(SEQ)
    .cost;
    assert!(
        with <= without + 1e-6,
        "RAAN hints made the DP schedule worse: {with} vs {without}"
    );
}

// ------------------------------- robustness ------------------------------- //

#[test]
fn over_horizon_is_infeasible_not_a_crash() {
    // A one-second horizon cannot fit any transfer: report infeasible, no crash.
    let problem = problem_with(1.0, 6);
    let transfer = transfer_with(&problem, 3e-3);
    let seq: &[usize] = &[0, 2, 8, 5];

    for (name, plan) in [
        ("DpScheduler", dp(&problem, &transfer, 8).evaluate(seq)),
        ("CdScheduler", cd(&problem, &transfer, cfg()).evaluate(seq)),
        ("NowaitScheduler", nowait(&problem, &transfer).evaluate(seq)),
    ] {
        assert!(!plan.feasible, "{name}: over-horizon schedule wrongly reported feasible");
        assert!(
            plan.meeting_times.iter().all(|t| t.is_finite()),
            "{name}: non-finite meeting times"
        );
    }
}

#[test]
fn cd_over_horizon_falls_back_to_no_wait() {
    // When the refined schedule would end past the horizon, CD returns the
    // earliest-arrival (no-wait) plan instead -- the best shot at feasibility.
    // This route waits its way past a 3000-day horizon; `tests/parity.rs` pins
    // the resulting numbers against the Python reference.
    let problem = problem();
    let transfer = transfer(&problem);
    let seq: &[usize] = &[0, 10, 11, 4, 7, 13, 8, 6, 12];
    let plan = cd(&problem, &transfer, cfg()).evaluate(seq);
    let reference = nowait(&problem, &transfer).evaluate(seq);
    assert!(
        plan.mission_time() > problem.time_horizon,
        "the fixture no longer overshoots the horizon; pick a longer route"
    );
    assert!(!plan.feasible, "an over-horizon plan must report itself infeasible");
    assert_eq!(
        plan.meeting_times, reference.meeting_times,
        "CD did not fall back to the no-wait schedule past the horizon"
    );
    assert!(
        plan.waiting_times.iter().all(|&w| w == 0.0),
        "the fallback plan still waits: {:?}",
        plan.waiting_times
    );
}

#[test]
fn cd_refinement_improves_and_reports_it() {
    // A route where the coordinate descent genuinely moves off the forward pass.
    // The refined plan must be coherent (its cost reconstructs from the schedule
    // it actually flies), not merely cheaper on paper.
    let problem = problem();
    let transfer = transfer(&problem);
    for seq in [&[0usize, 11, 13, 9, 7, 5][..], &[0, 12, 1, 11, 9, 10][..]] {
        let refined = cd(&problem, &transfer, CdConfig { samples: 16, max_iters: 2, ..cfg() })
            .evaluate(seq);
        let forward = cd(&problem, &transfer, CdConfig { max_iters: 0, ..cfg() }).evaluate(seq);
        assert!(
            refined.cost < forward.cost - 1.0,
            "{seq:?}: refinement did not improve on the forward pass ({} vs {})",
            refined.cost,
            forward.cost
        );
        assert_coherent(&refined, &transfer, seq);
        assert!(
            refined.mission_time() <= problem.time_horizon * (1.0 + 1e-9),
            "{seq:?}: refined schedule overshoots the horizon"
        );
    }
}

#[test]
fn cache_matches_no_cache() {
    let problem = problem();
    let transfer = transfer(&problem);
    let cached = cd(
        &problem,
        &transfer,
        CdConfig { max_iters: 2, use_cache: true, ..cfg() },
    )
    .evaluate(SEQ)
    .cost;
    let direct = cd(
        &problem,
        &transfer,
        CdConfig { max_iters: 2, use_cache: false, ..cfg() },
    )
    .evaluate(SEQ)
    .cost;
    assert_approx(cached, direct, 1e-12, "CdScheduler: caching changed the result");

    let cached_dp = dp(&problem, &transfer, 16).evaluate(SEQ).cost;
    let direct_dp = {
        use scorpion::schedule::{DpConfig, DpScheduler};
        DpScheduler::new(
            &problem,
            &transfer,
            DpConfig { grid_size: 16, use_hints: true, use_cache: false },
        )
        .evaluate(SEQ)
        .cost
    };
    assert_approx(cached_dp, direct_dp, 1e-12, "DpScheduler: caching changed the result");
}

#[test]
fn repeated_evaluation_is_stable() {
    // The plan cache must hand back the same plan, not a re-scheduled one.
    let problem = problem();
    let transfer = transfer(&problem);
    let sched = cd(&problem, &transfer, cfg());
    let first = sched.evaluate(SEQ);
    let second = sched.evaluate(SEQ);
    assert_eq!(*first, *second, "a cached CD plan changed between evaluations");
}

#[test]
fn trivial_sequence_is_empty() {
    let problem = problem();
    let transfer = transfer(&problem);
    let schedulers: Vec<(&str, Box<dyn ScheduleLayer>)> = vec![
        ("NowaitScheduler", Box::new(nowait(&problem, &transfer))),
        ("CdScheduler", Box::new(cd(&problem, &transfer, cfg()))),
        ("DpScheduler", Box::new(dp(&problem, &transfer, 16))),
    ];
    for (name, sched) in schedulers {
        let plan = sched.evaluate(&[0]);
        assert_eq!(plan.load(), 0, "{name}: depot-only route has nonzero load");
        assert_eq!(plan.cost, 0.0, "{name}: empty route costs");
    }
}

#[test]
fn per_leg_costs_are_indexed_by_arrival_node() {
    // `per_leg_costs[i]` is the cost of the leg *arriving* at node `i`, so index
    // 0 is always zero and the entries sum to the plan cost. `io::format_mission`
    // relies on this layout.
    let problem = problem();
    let transfer = transfer(&problem);
    let schedulers: Vec<(&str, Box<dyn ScheduleLayer>)> = vec![
        ("NowaitScheduler", Box::new(nowait(&problem, &transfer))),
        ("CdScheduler", Box::new(cd(&problem, &transfer, cfg()))),
        ("DpScheduler", Box::new(dp(&problem, &transfer, 16))),
    ];
    for (name, sched) in schedulers {
        let plan = sched.evaluate(SEQ);
        assert_eq!(plan.per_leg_costs[0], 0.0, "{name}: node 0 carries a leg cost");
        assert!(
            plan.per_leg_costs[1..].iter().all(|&c| c > 0.0),
            "{name}: a leg has zero cost: {:?}",
            plan.per_leg_costs
        );
        assert_approx(
            plan.per_leg_costs.iter().sum::<f64>(),
            plan.cost,
            1e-12,
            &format!("{name}: per-leg costs do not sum to the plan cost"),
        );
    }
}
