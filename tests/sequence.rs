//! The clustering baseline and the hybrid genetic search.

mod common;

use mcrdv::solver::SequenceLayer;
use mcrdv::{
    ClusterSequence, Crossover, DpSchedule, Generation, HgsConfig, HgsSequence, Ordering,
    Problem, QlawTransfer, DAY, DEPOT,
};

fn quiet() -> HgsConfig {
    HgsConfig {
        seed: 0,
        verbose: false,
        max_iter: 25,
        pop_init: 4,
        ..HgsConfig::default()
    }
}

fn hgs(config: HgsConfig) -> HgsSequence<DpSchedule<QlawTransfer>> {
    HgsSequence::with_config(DpSchedule::new(QlawTransfer::new()), config)
}

// ------------------------------------------------------------- clustering //

#[test]
fn the_baseline_collects_every_debris_exactly_once() {
    let problem = common::problem();
    let mut layer = ClusterSequence::new(DpSchedule::new(QlawTransfer::new()));
    layer.initialize(&problem);
    let mission = layer.evaluate();

    assert_eq!(common::collected(&mission), (1..problem.states.len()).collect::<Vec<_>>());
    assert_eq!(mission.plans.len(), problem.num_chasers);
    assert!(mission.plans.iter().all(|p| p.sequence[0] == DEPOT));
}

#[test]
fn the_baseline_is_deterministic() {
    let problem = common::problem();
    let run = || {
        let mut layer = ClusterSequence::new(DpSchedule::new(QlawTransfer::new()));
        layer.initialize(&problem);
        layer.evaluate()
    };
    assert_eq!(run(), run());
}

// ------------------------------------------------------------------- HGS //

#[test]
fn the_search_collects_every_debris_exactly_once() {
    let problem = common::problem();
    let mut layer = hgs(quiet());
    layer.initialize(&problem);
    let mission = layer.evaluate();

    assert_eq!(common::collected(&mission), (1..problem.states.len()).collect::<Vec<_>>());
    assert_eq!(mission.plans.len(), problem.num_chasers);
    assert!(mission.plans.iter().all(|p| p.sequence[0] == DEPOT));
}

#[test]
fn the_mission_cost_is_the_sum_of_its_plans() {
    let problem = common::problem();
    let mut layer = hgs(quiet());
    layer.initialize(&problem);
    let mission = layer.evaluate();
    let sum: f64 = mission.plans.iter().map(|p| p.cost()).sum();
    assert!((mission.cost() - sum).abs() < 1e-9);
}

#[test]
fn the_same_seed_gives_the_same_mission() {
    let problem = common::problem();
    let run = || {
        let mut layer = hgs(quiet());
        layer.initialize(&problem);
        layer.evaluate()
    };
    assert_eq!(run(), run());
}

#[test]
fn different_seeds_explore_differently() {
    let problem = common::problem();
    let plans: Vec<Vec<Vec<usize>>> = (1..=4)
        .map(|seed| {
            let config = HgsConfig { seed, max_iter: 5, pop_init: 3, ..quiet() };
            let mut layer = hgs(config);
            layer.initialize(&problem);
            layer
                .evaluate()
                .plans
                .iter()
                .map(|p| p.sequence.clone())
                .collect()
        })
        .collect();
    assert!(plans.windows(2).any(|w| w[0] != w[1]), "every seed gave the same plan");
}

#[test]
fn the_search_beats_the_clustering_baseline() {
    // The whole point of searching is to do better than the one-shot sweep.
    let problem = common::problem();
    let mut baseline = ClusterSequence::new(DpSchedule::new(QlawTransfer::new()));
    baseline.initialize(&problem);
    let reference = baseline.evaluate().cost();

    let mut layer = hgs(quiet());
    layer.initialize(&problem);
    let found = layer.evaluate().cost();

    assert!(found < reference, "search {found} did not beat baseline {reference}");
}

#[test]
fn idle_chasers_come_back_as_empty_plans() {
    let debris = mcrdv::read_debris("data/odrc.csv");
    let mut states = vec![mcrdv::centroid(&debris)];
    states.extend(debris);
    let problem = std::rc::Rc::new(Problem::new(states, 8, 2500.0 * DAY, 20));

    let mut layer = hgs(quiet());
    layer.initialize(&problem);
    let mission = layer.evaluate();

    assert_eq!(mission.plans.len(), 8);
    assert!(mission.plans.iter().any(|p| p.load() == 0), "8 chasers for 13 debris");
    assert_eq!(common::collected(&mission), (1..problem.states.len()).collect::<Vec<_>>());
}

#[test]
fn the_iteration_budget_is_respected() {
    let problem = common::problem();
    let config = HgsConfig { max_iter: 1, pop_init: 2, ..quiet() };
    let mut layer = hgs(config);
    layer.initialize(&problem);
    assert_eq!(common::collected(&layer.evaluate()).len(), problem.num_debris());
}

#[test]
fn a_vanishing_time_budget_still_returns_the_initial_best() {
    let problem = common::problem();
    let config = HgsConfig { max_iter: usize::MAX, max_time: 0.0, pop_init: 2, ..quiet() };
    let mut layer = hgs(config);
    layer.initialize(&problem);
    assert_eq!(common::collected(&layer.evaluate()).len(), problem.num_debris());
}

#[test]
fn a_zero_cadence_prints_no_progress_rows() {
    // `log_iter == 0` documents "never report" and must not divide by zero.
    let problem = common::problem();
    let config = HgsConfig { verbose: true, log_iter: 0, max_iter: 4, pop_init: 2, ..quiet() };
    let mut layer = hgs(config);
    layer.initialize(&problem);
    let mission = layer.evaluate();
    assert_eq!(common::collected(&mission).len(), problem.num_debris());
}

#[test]
fn every_ordering_and_operator_pairing_produces_a_complete_mission() {
    let problem = common::problem();
    for ordering in [Ordering::Index, Ordering::Raan, Ordering::Nearest] {
        for crossover in [Crossover::Ox, Crossover::Erx, Crossover::Srex] {
            for seed in 0..3 {
                let config = HgsConfig {
                    seed, ordering, crossover, max_iter: 6, pop_init: 3, ..quiet()
                };
                let mut layer = hgs(config);
                layer.initialize(&problem);
                let mission = layer.evaluate();
                assert_eq!(
                    common::collected(&mission),
                    (1..problem.states.len()).collect::<Vec<_>>(),
                    "{ordering:?}/{crossover:?}/seed {seed}"
                );
            }
        }
    }
}

#[test]
fn random_generation_also_covers_the_catalogue() {
    let problem = common::problem();
    let config = HgsConfig { generation: Generation::Random, ..quiet() };
    let mut layer = hgs(config);
    layer.initialize(&problem);
    assert_eq!(
        common::collected(&layer.evaluate()),
        (1..problem.states.len()).collect::<Vec<_>>()
    );
}

#[test]
fn neighbour_lists_are_capped_and_exclude_the_debris_itself() {
    let problem = common::problem();
    let config = HgsConfig { nb_neighbors: 4, ..quiet() };
    let mut layer = hgs(config);
    layer.initialize(&problem);

    let context = layer.context().expect("initialize installs a context");
    for d in problem.debris() {
        let peers = &context.neighbors[d];
        assert!(peers.len() <= 4, "debris {d} has {} peers", peers.len());
        assert!(!peers.contains(&d));
        assert!(!peers.contains(&DEPOT));
    }
}

#[test]
fn presets_are_ordered_by_effort() {
    let presets: Vec<HgsConfig> = (0..7).map(HgsConfig::preset).collect();
    for pair in presets.windows(2) {
        assert!(pair[1].max_iter >= pair[0].max_iter);
        assert!(pair[1].pop_max >= pair[0].pop_max);
    }
}

#[test]
fn the_defaults_match_preset_four() {
    let d = HgsConfig::default();
    let p = HgsConfig::preset(4);
    assert_eq!(
        (d.log_iter, d.max_iter, d.max_nimp, d.pop_init, d.pop_min, d.pop_max, d.nb_close, d.nb_elite),
        (p.log_iter, p.max_iter, p.max_nimp, p.pop_init, p.pop_min, p.pop_max, p.nb_close, p.nb_elite)
    );
}

#[test]
#[should_panic(expected = "preset level")]
fn an_unknown_preset_is_refused() {
    let _ = HgsConfig::preset(99);
}
