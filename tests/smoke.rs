//! Basic smoke and pipeline tests for scorpion.

mod common;

use common::*;
use scorpion::io::{format_mission, read_debris, write_mission};
use scorpion::orbit::{KepState, DAY, J2, MU, RE};
use scorpion::problem::{
    make_problem, ChaserPlan, Constraint, MaxLoad, MaxTime, MissionPlan, Problem,
};
use scorpion::schedule::CdConfig;
use scorpion::sequence::{HgsConfig, HgsSequencer};
use scorpion::solver::{ScheduleLayer, Solver, TransferLayer};
use scorpion::transfer::qlaw::linspace;

// ---------------------------------- io ----------------------------------- //

#[test]
fn read_debris_catalogue() {
    let catalog = debris();
    assert!(catalog.len() >= 10, "the reference catalogue is unexpectedly small");
    for state in catalog.states() {
        for v in [state.a, state.e, state.i, state.r, state.o, state.t] {
            assert!(v.is_finite(), "a catalogue element did not parse to a finite float");
        }
    }
}

#[test]
fn read_debris_rejects_a_bad_header() {
    let path = std::env::temp_dir().join("mcadr_bad_header.csv");
    std::fs::write(&path, "a,b,c\n1,2,3\n").expect("temp file must be writable");
    assert!(read_debris(&path).is_err(), "a bad header must be rejected");
    let _ = std::fs::remove_file(&path);
}

// --------------------------------- orbit --------------------------------- //

#[test]
fn kepstate_propagate() {
    let s = KepState::new(7.0e6, 0.01, 1.0, 0.5, 0.3, 0.2);
    let dt = 100.0 * DAY;
    // Expected J2 secular precession rates of the nodes.
    let f = 1.5 * J2 * (RE / (s.a * (1.0 - s.e * s.e))).powi(2) * (MU / s.a.powi(3)).sqrt();
    let dr = -f * s.i.cos();
    let dou = f * (2.0 - 2.5 * s.i.sin().powi(2));

    let moved = s.propagate(dt);
    assert_approx(moved.r, s.r + dr * dt, 1e-12, "RAAN precession");
    assert_approx(moved.o, s.o + dou * dt, 1e-12, "argument-of-periapsis precession");
    assert_eq!(
        (moved.a, moved.e, moved.i, moved.t),
        (s.a, s.e, s.i, s.t),
        "only the precessing nodes may move"
    );
}

// -------------------------------- problem -------------------------------- //

#[test]
fn make_problem_prepends_the_depot() {
    let catalog = debris();
    let problem = problem_with(3000.0 * DAY, 5);
    assert_eq!(problem.catalog.len(), catalog.len() + 1, "the depot is prepended");
    assert_eq!(problem.num_debris, catalog.len());
    assert_eq!(problem.num_chasers, 3);
    assert_eq!(problem.debris(), (1..=catalog.len()).collect::<Vec<_>>());
    assert_eq!(problem.time_horizon, 3000.0 * DAY);

    // The depot is the cluster centroid of the semi-major axes.
    let mean = catalog.a.iter().sum::<f64>() / catalog.a.len() as f64;
    assert_approx(problem.catalog.state(0).a, mean, 1e-12, "depot semi-major axis");
}

#[test]
fn make_problem_rejects_a_trivially_infeasible_load() {
    let catalog = debris();
    let err = make_problem(&catalog, 1, Some(3000.0 * DAY), Some(1), None, "sum");
    assert!(err.is_err(), "one chaser of capacity 1 cannot collect the catalogue");
}

#[test]
fn constraints_and_objectives() {
    let plan = ChaserPlan::new(
        vec![0, 1, 2, 3],
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0; 4],
        vec![0.0; 4],
        10.0,
        true,
    );
    assert!(MaxLoad::new(5).violation(&plan) < 0.0, "load 3 <= 5 -> satisfied");
    assert!(MaxLoad::new(2).violation(&plan) > 0.0, "load 3 > 2 -> violated");
    assert_approx(MaxTime::new(2.5).violation(&plan), 0.5, 1e-12, "time 3 - 2.5");
}

// ------------------------------- transfer -------------------------------- //

#[test]
fn transfer_oracle() {
    let problem = problem();
    let transfer = transfer(&problem);
    let grid = linspace(0.0, problem.time_horizon, 500);
    let dv: Vec<f64> = grid
        .iter()
        .map(|&t| {
            let (dt, dv) = transfer.evaluate(1, 7, t);
            assert!(dv >= 0.0, "negative fuel");
            assert_approx(dt, dv / THRUST, 1e-12, "time is fuel / thrust");
            dv
        })
        .collect();
    let lo = dv.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = dv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo > 1.0, "cost does not genuinely vary with the departure epoch");
}

// -------------------------------- schedule ------------------------------- //

#[test]
fn nowait_scheduler() {
    let problem = problem();
    let transfer = transfer(&problem);
    let sched = nowait(&problem, &transfer);
    let seq: &[usize] = &[0, 2, 5, 8];
    let plan = sched.evaluate(seq);

    assert_eq!(plan.nodes_indices, seq);
    assert!(plan.waiting_times.iter().all(|&w| w == 0.0));
    // Coherence: each arrival is the previous arrival plus the leg time.
    for i in 1..seq.len() {
        let (dt, _) = transfer.evaluate(seq[i - 1], seq[i], plan.meeting_times[i - 1]);
        assert_approx(
            plan.meeting_times[i],
            plan.meeting_times[i - 1] + dt,
            1e-12,
            "arrival",
        );
    }
    assert_approx(
        plan.cost,
        plan.per_leg_costs.iter().sum::<f64>(),
        1e-12,
        "cost != sum of leg costs",
    );
    assert!(sched.evaluate_lb(seq) <= plan.cost + 1e-6);
}

#[test]
fn cd_scheduler() {
    let problem = problem();
    let transfer = transfer(&problem);
    let seq: &[usize] = &[0, 2, 8, 5, 11];
    let sched = cd(
        &problem,
        &transfer,
        CdConfig { samples: 16, max_iters: 2, ..CdConfig::default() },
    );
    let plan = sched.evaluate(seq);
    assert_coherent(&plan, &transfer, seq);
    assert!(plan.mission_time() <= problem.time_horizon * (1.0 + 1e-9));
    assert!(sched.evaluate_lb(seq) <= plan.cost + 1e-6);

    // Refinement passes never cost more than the greedy forward pass.
    let greedy = cd(&problem, &transfer, CdConfig { max_iters: 0, ..CdConfig::default() })
        .evaluate(seq)
        .cost;
    assert!(plan.cost <= greedy + 1e-6, "refinement made the plan worse");
}

#[test]
fn dp_scheduler() {
    let problem = problem();
    let transfer = transfer(&problem);
    let seq: &[usize] = &[0, 2, 8, 5, 11];
    let sched = dp(&problem, &transfer, 24);
    let plan = sched.evaluate(seq);
    assert_coherent(&plan, &transfer, seq);
    assert!(sched.evaluate_lb(seq) <= plan.cost + 1e-6);
}

// --------------------------------- solver -------------------------------- //

fn solve(num_chasers: usize, max_load: usize, seed: u64) -> (MissionPlan, Problem) {
    solve_within(num_chasers, max_load, 2500.0 * DAY, seed)
}

fn solve_within(
    num_chasers: usize,
    max_load: usize,
    max_time: f64,
    seed: u64,
) -> (MissionPlan, Problem) {
    let problem = make_problem(
        &debris(),
        num_chasers,
        Some(max_time),
        Some(max_load),
        None,
        "sum",
    )
    .expect("problem must be constructible");
    let mission = {
        let transfer = transfer_with(&problem, 3e-3);
        let schedule = nowait(&problem, &transfer);
        let sequence = HgsSequencer::new(
            &problem,
            &transfer,
            &schedule,
            HgsConfig { seed, log_iter: 0, max_iter: 300, max_nimp: 150, ..HgsConfig::default() },
        );
        Solver::new(&transfer, &schedule, &sequence, false).solve()
    };
    (mission, problem)
}

#[test]
fn end_to_end() {
    let (mission, problem) = solve(3, 5, 0);
    assert!(mission.feasible, "the reference mission should be feasible");
    assert!(mission.cost > 0.0);
    assert_eq!(mission.collected(), problem.debris(), "every debris collected once");

    // Same seed -> identical result.
    let (again, _) = solve(3, 5, 0);
    assert_approx(again.cost, mission.cost, 1e-12, "same-seed runs diverged");
}

#[test]
fn infeasible_is_reported() {
    // A single chaser given ten days to collect the whole catalogue cannot
    // finish in time: report infeasible, never as a bogus zero-cost empty plan.
    //
    // `references/pyadr`'s version of this test asks for one chaser of capacity
    // 3, which `make_problem` rejects outright (13 debris > 1 * 3) -- the Python
    // test errors rather than asserting anything (see
    // `make_problem_rejects_a_trivially_infeasible_load`), so the load limit is
    // set to the catalogue size here and the horizon carries the infeasibility.
    let (mission, problem) = solve_within(1, debris().len(), 10.0 * DAY, 0);
    assert!(!mission.feasible, "an impossible mission was reported feasible");
    assert!(mission.cost >= 0.0, "a mission cost must never be negative");
    assert_eq!(
        mission.collected(),
        problem.debris(),
        "an infeasible mission must still be a complete plan, not an empty one"
    );
}


// ----------------------------- mission report ---------------------------- //

#[test]
fn mission_report_is_well_formed() {
    // `format_mission` walks `per_leg_costs` from index 1, so a report whose
    // cumulative cost matches the plan cost also pins that layout.
    let (mission, problem) = solve(3, 5, 0);
    let report = format_mission(&mission, &problem);
    assert!(report.contains("Mission plan"), "missing banner");
    assert!(report.contains("Mission cost"), "missing cost line");
    for plan in mission.chaser_plans.iter().filter(|p| p.load() > 0) {
        let cum: f64 = plan.per_leg_costs[1..].iter().sum();
        assert_approx(cum, plan.cost, 1e-9, "reported cumulative cost != plan cost");
    }

    let path = std::env::temp_dir().join("mcadr_mission_smoke.csv");
    write_mission(&mission, &path).expect("mission must be writable");
    let written = std::fs::read_to_string(&path).expect("written mission must be readable");
    assert!(written.starts_with("chaser,node,meet,wait,cost"), "bad csv header");
    let _ = std::fs::remove_file(&path);
}
