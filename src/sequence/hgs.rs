//! Hybrid Genetic Search sequence layer.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::problem::{ChaserPlan, Constraint, MissionPlan, Objective, Problem};
use crate::solver::{ScheduleLayer, SequenceLayer, TransferLayer};

fn fcmp(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

// ========================================================================= //
// Configuration
// ========================================================================= //

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HgsGeneration {
    /// Randomly assign debris to chaser plans.
    Random,
    /// Nearest-neighbor greedy assignment.
    Greedy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HgsOrdering {
    /// Keep plans in their construction order.
    Index,
    /// Order plans by ascending mean RAAN.
    Raan,
    /// Chain plans greedily by transfer cost at their boundaries.
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HgsCrossover {
    /// Order crossover on the giant tour.
    Ox,
    /// Edge-recombination crossover on the giant tour.
    Erx,
    /// Selective plan exchange between the parent plan sets.
    Srex,
}

#[derive(Debug, Clone)]
pub struct HgsConfig {
    pub seed: u64,
    // termination
    pub max_iter: usize,
    pub max_nimp: usize,
    pub max_time: f64,
    pub log_iter: usize,
    // generation
    pub generation: HgsGeneration,
    pub pop_init: usize,
    // population
    pub pop_min: usize,
    pub pop_max: usize,
    pub nb_close: usize,
    pub nb_elite: usize,
    pub adapt_iter: usize,
    pub clone_eps: f64,
    // crossover
    pub nb_pick: usize,
    pub threshold_factor: f64,
    pub ordering: HgsOrdering,
    pub crossover: HgsCrossover,
    // local search
    pub nb_neighbors: usize,
    pub neighbor_samples: usize,
    // penalties
    pub penalty_increase: f64,
    pub penalty_decrease: f64,
    pub penalty_min: f64,
    pub penalty_max: f64,
    pub target_ratio: f64,
}

impl Default for HgsConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            // termination
            max_iter: 5_000,
            max_nimp: 500,
            max_time: f64::INFINITY,
            log_iter: 10,
            // generation
            generation: HgsGeneration::Greedy,
            pop_init: 20,
            // population
            pop_min: 10,
            pop_max: 20,
            nb_close: 2,
            nb_elite: 3,
            adapt_iter: 10,
            clone_eps: 1e-6,
            // crossover
            nb_pick: 2,
            threshold_factor: 1.5,
            ordering: HgsOrdering::Raan,
            crossover: HgsCrossover::Ox,
            // local search
            nb_neighbors: 10,
            neighbor_samples: 16,
            // penalties
            penalty_increase: 2.0,
            penalty_decrease: 0.5,
            penalty_min: 1e-9,
            penalty_max: 1e9,
            target_ratio: 0.9,
        }
    }
}

impl HgsConfig {
    pub fn preset(level: usize) -> Self {
        let mut config = Self::default();
        if level > 6 { panic!("HGS preset level must be between 0 and 6."); }
        match level.min(6) {
            0 => {
                // Local search on a single initial individual
                config.log_iter = 1;
                config.max_iter = 0;
                config.max_nimp = 0;
                config.pop_init = 1;
                config.pop_min = 2;
                config.pop_max = 2;
                config.nb_close = 1;
                config.nb_elite = 1;
            }
            1 => {
                // Local search on multiple initial individuals
                config.log_iter = 1;
                config.max_iter = 0;
                config.max_nimp = 0;
                config.pop_init = 5;
                config.pop_min = 2;
                config.pop_max = 3;
                config.nb_close = 1;
                config.nb_elite = 1;
            }
            2 => {
                // Very shallow genetic selection
                config.log_iter = 1;
                config.max_iter = 50;
                config.max_nimp = 10;
                config.pop_init = 6;
                config.pop_min = 3;
                config.pop_max = 6;
                config.nb_close = 1;
                config.nb_elite = 1;
            }
            3 => {
                // Shallow genetic selection
                config.log_iter = 1;
                config.max_iter = 500;
                config.max_nimp = 100;
                config.pop_init = 10;
                config.pop_min = 5;
                config.pop_max = 10;
                config.nb_close = 2;
                config.nb_elite = 2;
            }
            4 => {
                // Medium genetic selection
                config.log_iter = 10;
                config.max_iter = 5_000;
                config.max_nimp = 500;
                config.pop_init = 20;
                config.pop_min = 10;
                config.pop_max = 20;
                config.nb_close = 2;
                config.nb_elite = 3;
            }
            5 => {
                // Deep genetic selection
                config.log_iter = 10;
                config.max_iter = 50_000;
                config.max_nimp = 5_000;
                config.pop_init = 24;
                config.pop_min = 12;
                config.pop_max = 32;
                config.nb_close = 3;
                config.nb_elite = 4;
            }
            6 => {
                // Very deep genetic selection
                config.log_iter = 100;
                config.max_iter = 200_000;
                config.max_nimp = 10_000;
                config.pop_init = 50;
                config.pop_min = 25;
                config.pop_max = 65;
                config.nb_close = 3;
                config.nb_elite = 8;
            }
            // Unreachable: `level.min(6)` keeps the match exhaustive over 0..=6.
            _ => {}
        }
        config
    }
}

// ========================================================================= //
// Context
// ========================================================================= //

/// Context helper with shared data, search state and methods across one solve.
struct Ctx<'a> {
    problem: &'a Problem,
    transfer: &'a dyn TransferLayer,
    schedule: &'a dyn ScheduleLayer,
    config: &'a HgsConfig,
    depot: usize,
    debris: Vec<usize>,
    neighbors: Vec<Vec<usize>>,
    metric: Metric,
    population: Population,
}

impl<'a> Ctx<'a> {
    fn new(
        problem: &'a Problem,
        transfer: &'a dyn TransferLayer,
        schedule: &'a dyn ScheduleLayer,
        config: &'a HgsConfig,
    ) -> Self {
        assert!(problem.catalog.depots_indices() == [0], "HGS only supports a single depot");
        let neighbors = build_neighbors(problem, transfer, config);
        Ctx {
            problem,
            transfer,
            schedule,
            config,
            depot: 0,
            debris: problem.debris(),
            neighbors,
            metric: Metric::new(problem, transfer, 0, config),
            population: Population::new(problem),
        }
    }

    /// Insert an individual into the population (adapting penalties as needed).
    fn add(&mut self, ind: Individual) {
        self.population.add(self.config, &self.debris, &mut self.metric, ind);
    }

    /// Best individual currently in the population.
    fn best(&self) -> Option<Individual> {
        self.population.best(&self.metric)
    }

    /// Penalized value of the current best individual.
    fn best_value(&self) -> f64 {
        self.best().map_or(f64::INFINITY, |b| self.metric.value(&b))
    }

    fn constraints(&self) -> &[Box<dyn Constraint>] {
        &self.problem.constraints
    }

    fn objective(&self) -> &dyn Objective {
        self.problem.objective.as_ref()
    }

    /// Evaluate a partial debris sequence (depot prepended).
    fn eval_debris(&self, debris: &[usize]) -> Rc<ChaserPlan> {
        let mut seq = Vec::with_capacity(debris.len() + 1);
        seq.push(self.depot);
        seq.extend_from_slice(debris);
        self.schedule.evaluate(&seq)
    }

    /// Evaluate a chaser sequence (assuming depot in first position).
    fn eval(&self, seq: &[usize]) -> Rc<ChaserPlan> {
        self.schedule.evaluate(seq)
    }

    /// Cheap cost lower bound of a chaser sequence, used to screen moves before
    /// paying for the full schedule evaluation. Since the penalty term is always
    /// non-negative, `plan_value >= cost >= lb`, so a candidate whose lb already
    /// reaches the value to beat cannot improve and can be skipped.
    fn lb(&self, seq: &[usize]) -> f64 {
        self.schedule.evaluate_lb(seq)
    }

    /// Per-constraint violation of a chaser plan.
    fn plan_excess(&self, plan: &ChaserPlan) -> Vec<f64> {
        self.constraints()
            .iter()
            .map(|c| c.violation(plan).max(0.0))
            .collect()
    }
}

/// Mean orbital-distance between two nodes over the horizon.
fn orbital_distance(problem: &Problem, transfer: &dyn TransferLayer, config: &HgsConfig, i: usize, j: usize) -> f64 {
    let m = config.neighbor_samples.max(1);
    let horizon = problem.time_horizon;
    let mut s = 0.0;
    for q in 0..m {
        let t = horizon * (q as f64 + 0.5) / m as f64;
        s += transfer.evaluate(i, j, t).1;
    }
    s / m as f64
}

/// Build per-debris neighbor lists ordered by the orbital-distance proxy.
fn build_neighbors(problem: &Problem, transfer: &dyn TransferLayer, config: &HgsConfig) -> Vec<Vec<usize>> {
    let debris = problem.debris();
    let mut neighbors = vec![Vec::new(); problem.catalog.n];
    for &i in &debris {
        let mut others: Vec<usize> = debris.iter().copied().filter(|&j| j != i).collect();
        others.sort_by(|&a, &b| {
            fcmp(
                orbital_distance(problem, transfer, config, i, a),
                orbital_distance(problem, transfer, config, i, b),
            )
        });
        others.truncate(config.nb_neighbors);
        neighbors[i] = others;
    }
    neighbors
}

// ========================================================================= //
// Individual
// ========================================================================= //

/// A candidate mission: one chaser plan per chaser, with derived metrics.
#[derive(Clone)]
struct Individual {
    plans: Vec<Rc<ChaserPlan>>,
    cost: f64,
    feasible: bool,
    violation: Vec<f64>,
    active: usize,
    pred: Vec<usize>,
    succ: Vec<usize>,
}

impl Individual {
    fn new(ctx: &Ctx, plans: Vec<Rc<ChaserPlan>>) -> Self {
        let nc = ctx.constraints().len();
        let costs: Vec<f64> = plans.iter().map(|r| r.cost).collect();
        let cost = ctx.objective().aggregate(&costs);
        let feasible = plans.iter().all(|r| r.feasible);
        let mut violation = vec![0.0; nc];
        let mut active = 0;
        let mut pred = vec![0usize; ctx.problem.catalog.n];
        let mut succ = vec![0usize; ctx.problem.catalog.n];
        for plan in &plans {
            if nc > 0 {
                for (c, e) in ctx.plan_excess(plan).into_iter().enumerate() {
                    violation[c] += e;
                }
            }
            let seq = &plan.nodes_indices;
            if seq.len() > 1 {
                active += 1;
                let n = seq.len();
                for i in 1..n {
                    let d = seq[i];
                    pred[d] = if i == 1 { ctx.depot } else { seq[i - 1] };
                    succ[d] = if i == n - 1 { ctx.depot } else { seq[i + 1] };
                }
            }
        }
        Self {
            plans,
            cost,
            feasible,
            violation,
            active,
            pred,
            succ,
        }
    }

    /// Recompute derived metrics after the plans were mutated.
    fn refresh(&mut self, ctx: &Ctx) {
        let plans = std::mem::take(&mut self.plans);
        *self = Individual::new(ctx, plans);
    }
}

/// Normalized broken-pairs distance between two individuals.
fn broken_pair_distance(a: &Individual, b: &Individual, debris: &[usize]) -> f64 {
    let n = debris.len();
    if n == 0 {
        return 0.0;
    }
    let mut count = 0usize;
    for &d in debris {
        let (asucc, bsucc) = (a.succ[d], b.succ[d]);
        let (apred, bpred) = (a.pred[d], b.pred[d]);
        let broken = asucc != bsucc && asucc != bpred;
        let orphan = apred == 0 && bpred != 0 && bsucc != 0;
        count += broken as usize + orphan as usize;
    }
    count as f64 / n as f64
}

// ========================================================================= //
// Metric
// ========================================================================= //

/// Penalized cost with one adaptive penalty per problem constraint.
struct Metric {
    penalties: Vec<f64>,
}

impl Metric {
    fn new(
        problem: &Problem, 
        transfer: &dyn TransferLayer, 
        depot: usize, 
        config: &HgsConfig
    ) -> Self {
        // Scale each penalty in magnitude ~ total cost / unitary violation
        let cost: f64 = problem
            .debris()
            .iter()
            .map(|&d| transfer.evaluate_lb(depot, d).1)
            .sum::<f64>()
            .max(1.0);
        let empty = ChaserPlan::empty();
        let penalties = problem
            .constraints
            .iter()
            .map(
                |c| 
                (-cost / c.violation(&empty)).clamp(
                    config.penalty_min, 
                    config.penalty_max
                )
            )
            .collect();
        Self { penalties }
    }

    fn plan_value(&self, problem: &Problem, plan: &ChaserPlan) -> f64 {
        if self.penalties.is_empty() { return plan.cost; }
        plan.cost + problem.constraints
            .iter()
            .zip(&self.penalties)
            .map(|(c, p)| p * c.violation(plan).max(0.0))
            .sum::<f64>()
    }

    fn value(&self, ind: &Individual) -> f64 {
        if self.penalties.is_empty() { return ind.cost; }
        ind.cost + ind.violation
            .iter()
            .zip(&self.penalties)
            .map(|(e, p)| e * p)
            .sum::<f64>()
    }

    fn adapt(&mut self, config: &HgsConfig, ratios: &[f64]) {
        for (c, &ratio) in ratios.iter().enumerate() {
            self.penalties[c] = if ratio < config.target_ratio {
                (config.penalty_increase * self.penalties[c]).clamp(
                    config.penalty_min,
                    config.penalty_max
                )
            } else {
                (config.penalty_decrease * self.penalties[c]).clamp(
                    config.penalty_min,
                    config.penalty_max
                )
            };
        }
    }
}

// ========================================================================= //
// Population
// ========================================================================= //

struct Subpopulation {
    individuals: Vec<Individual>,
    score: Vec<f64>,
    dist: Vec<Vec<f64>>,
}

impl Subpopulation {
    fn new() -> Self {
        Self {
            individuals: Vec::new(),
            score: Vec::new(),
            dist: Vec::new(),
        }
    }

    fn add(&mut self, config: &HgsConfig, debris: &[usize], metric: &Metric, ind: Individual) {
        let m = self.individuals.len();
        let row: Vec<f64> = (0..m)
            .map(|j| broken_pair_distance(&ind, &self.individuals[j], debris))
            .collect();
        for (j, r) in self.dist.iter_mut().enumerate() {
            r.push(row[j]);
        }
        let mut new_row = row;
        new_row.push(0.0);
        self.dist.push(new_row);
        self.individuals.push(ind);

        self.update_score(config, metric);
        if self.individuals.len() > config.pop_max {
            self.survivors_selection(config, metric);
        }
    }

    fn remove(&mut self, idx: usize) {
        self.individuals.remove(idx);
        self.dist.remove(idx);
        for r in self.dist.iter_mut() {
            r.remove(idx);
        }
    }

    fn update_score(&mut self, config: &HgsConfig, metric: &Metric) {
        let n = self.individuals.len();
        self.score = vec![0.0; n];
        if n <= 1 {
            return;
        }
        let denom = (n - 1) as f64;

        // Cost rank.
        let values: Vec<f64> = self.individuals.iter().map(|i| metric.value(i)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| fcmp(values[a], values[b]));
        let mut cost_pos = vec![0usize; n];
        for (rank, &idx) in order.iter().enumerate() {
            cost_pos[idx] = rank;
        }

        // Diversity rank (descending average distance to the nb_close nearest),
        // read from the maintained distance matrix.
        let kk = config.nb_close.min(n - 1);
        let mut avg_dist = vec![0.0; n];
        if kk > 0 {
            for (i, a) in avg_dist.iter_mut().enumerate() {
                let mut ds: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| self.dist[i][j]).collect();
                ds.sort_by(|x, y| fcmp(*x, *y));
                *a = ds[..kk].iter().sum::<f64>() / kk as f64;
            }
        }
        let mut div_order: Vec<usize> = (0..n).collect();
        div_order.sort_by(|&a, &b| fcmp(avg_dist[b], avg_dist[a]));
        let mut div_pos = vec![0usize; n];
        for (rank, &idx) in div_order.iter().enumerate() {
            div_pos[idx] = rank;
        }

        let scale = 1.0 - config.nb_elite as f64 / n as f64;
        for i in 0..n {
            let ord_rank = cost_pos[i] as f64 / denom;
            self.score[i] = if cost_pos[i] < config.nb_elite {
                ord_rank
            } else {
                ord_rank + scale * div_pos[i] as f64 / denom
            };
        }
    }

    /// Minimum distance from individual `i` to any other (self excluded).
    fn nearest_min(&self, i: usize) -> f64 {
        let mut m = f64::INFINITY;
        for (j, &d) in self.dist[i].iter().enumerate() {
            if j != i {
                m = m.min(d);
            }
        }
        m
    }

    fn worst_index(&self, config: &HgsConfig) -> usize {
        let mut worst_idx = 0;
        let mut worst_clone = self.nearest_min(0) <= config.clone_eps;
        let mut worst_score = self.score[0];
        for i in 1..self.individuals.len() {
            let is_clone = self.nearest_min(i) <= config.clone_eps;
            let score = self.score[i];
            if (is_clone && !worst_clone) || (is_clone == worst_clone && score > worst_score) {
                worst_clone = is_clone;
                worst_score = score;
                worst_idx = i;
            }
        }
        worst_idx
    }

    fn survivors_selection(&mut self, config: &HgsConfig, metric: &Metric) {
        while self.individuals.len() > config.pop_min {
            let idx = self.worst_index(config);
            self.remove(idx);
            self.update_score(config, metric);
        }
    }

    fn best(&self, metric: &Metric) -> Option<&Individual> {
        self.individuals
            .iter()
            .min_by(|a, b| fcmp(metric.value(a), metric.value(b)))
    }
}

struct Population {
    feasible: Subpopulation,
    infeasible: Subpopulation,
    windows: Vec<VecDeque<bool>>,
    count: usize,
}

impl Population {
    fn new(problem: &Problem) -> Self {
        Self {
            feasible: Subpopulation::new(),
            infeasible: Subpopulation::new(),
            windows: vec![VecDeque::new(); problem.constraints.len()],
            count: 0,
        }
    }

    fn add(&mut self, config: &HgsConfig, debris: &[usize], metric: &mut Metric, ind: Individual) {
        self.count += 1;
        for (c, window) in self.windows.iter_mut().enumerate() {
            window.push_back(ind.violation[c] == 0.0);
        }
        if ind.feasible {
            self.feasible.add(config, debris, metric, ind);
        } else {
            self.infeasible.add(config, debris, metric, ind);
        }

        let adapt = config.adapt_iter;
        if adapt > 0 && !self.windows.is_empty() && self.count.is_multiple_of(adapt) {
            let ratios: Vec<f64> = self
                .windows
                .iter()
                .map(|w| {
                    let hits = w.iter().rev().take(adapt).filter(|&&b| b).count();
                    hits as f64 / adapt as f64
                })
                .collect();
            metric.adapt(config, &ratios);
            self.feasible.update_score(config, metric);
            self.infeasible.update_score(config, metric);
        }
    }

    fn pick(&self, config: &HgsConfig, rng: &mut SmallRng) -> Individual {
        let nf = self.feasible.individuals.len();
        let ni = self.infeasible.individuals.len();
        let total = nf + ni;
        let k = config.nb_pick.max(1).min(total);
        let mut chosen = 0usize;
        let mut best_score = f64::INFINITY;
        for d in 0..k {
            let g = rng.gen_range(0..total);
            let score = if g < nf {
                self.feasible.score[g]
            } else {
                self.infeasible.score[g - nf]
            };
            if d == 0 || score < best_score {
                best_score = score;
                chosen = g;
            }
        }
        if chosen < nf {
            self.feasible.individuals[chosen].clone()
        } else {
            self.infeasible.individuals[chosen - nf].clone()
        }
    }

    fn best(&self, metric: &Metric) -> Option<Individual> {
        self.feasible
            .best(metric)
            .or_else(|| self.infeasible.best(metric))
            .cloned()
    }
}

// ========================================================================= //
// Generation
// ========================================================================= //

fn nowait_feasible(ctx: &Ctx, plan: &[usize], meet: &[f64], cost: &[f64]) -> bool {
    let mut seq = Vec::with_capacity(plan.len() + 1);
    seq.push(ctx.depot);
    seq.extend_from_slice(plan);
    let plan = ChaserPlan::new(
        seq,
        meet.to_vec(),
        vec![0.0; meet.len()],
        cost.to_vec(),
        cost.iter().sum(),
        true,
    );
    ctx.problem.is_feasible(&plan)
}

fn assemble(ctx: &Ctx, buckets: &[Vec<usize>]) -> Individual {
    let empty: Vec<usize> = Vec::new();
    let plans: Vec<Rc<ChaserPlan>> = (0..ctx.problem.num_chasers)
        .map(|c| ctx.eval_debris(buckets.get(c).unwrap_or(&empty)))
        .collect();
    Individual::new(ctx, plans)
}

fn generate(ctx: &Ctx, rng: &mut SmallRng) -> Individual {
    match ctx.config.generation {
        HgsGeneration::Random => generate_random(ctx, rng),
        HgsGeneration::Greedy => generate_greedy(ctx, rng),
    }
}

fn generate_greedy(ctx: &Ctx, rng: &mut SmallRng) -> Individual {
    let mut available = vec![false; ctx.problem.catalog.n];
    for &d in &ctx.debris {
        available[d] = true;
    }
    let mut ordered = ctx.debris.clone();
    ordered.sort_by(|&a, &b| {
        fcmp(
            ctx.transfer.evaluate(ctx.depot, a, 0.0).1,
            ctx.transfer.evaluate(ctx.depot, b, 0.0).1,
        )
    });
    // Windowed shuffle so greedy seeds differ across calls.
    let len = ordered.len();
    for i in 0..len.saturating_sub(1) {
        let end = (i + 4).min(len - 1);
        let j = i + rng.gen_range(0..end - i + 1);
        ordered.swap(i, j);
    }

    let mut buckets: Vec<Vec<usize>> = Vec::new();
    for &start in &ordered {
        if !available[start] || buckets.len() >= ctx.problem.num_chasers {
            continue;
        }
        available[start] = false;
        let (dt0, dv0) = ctx.transfer.evaluate(ctx.depot, start, 0.0);
        let mut plan = vec![start];
        let mut meet = vec![0.0, dt0];
        let mut cost = vec![0.0, dv0];
        loop {
            let current = *plan.last().unwrap();
            let t = *meet.last().unwrap();
            let mut best: Option<usize> = None;
            let mut best_dt = 0.0;
            let mut best_cost = f64::INFINITY;
            for &d in &ordered {
                if !available[d] {
                    continue;
                }
                let (leg_t, leg_f) = ctx.transfer.evaluate(current, d, t);
                if leg_f >= best_cost {
                    continue;
                }
                let mut m2 = meet.clone();
                m2.push(t + leg_t);
                let mut c2 = cost.clone();
                c2.push(leg_f);
                let mut r2 = plan.clone();
                r2.push(d);
                if nowait_feasible(ctx, &r2, &m2, &c2) {
                    best = Some(d);
                    best_dt = leg_t;
                    best_cost = leg_f;
                }
            }
            match best {
                Some(d) => {
                    available[d] = false;
                    plan.push(d);
                    meet.push(t + best_dt);
                    cost.push(best_cost);
                }
                None => break,
            }
        }
        buckets.push(plan);
    }
    let leftover: Vec<usize> = ctx.debris.iter().copied().filter(|&d| available[d]).collect();
    if !leftover.is_empty() {
        if buckets.is_empty() {
            buckets.push(Vec::new());
        }
        buckets.last_mut().unwrap().extend(leftover);
    }
    assemble(ctx, &buckets)
}

fn generate_random(ctx: &Ctx, rng: &mut SmallRng) -> Individual {
    let mut remaining = ctx.debris.clone();
    remaining.shuffle(rng);
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    while !remaining.is_empty() && buckets.len() < ctx.problem.num_chasers {
        let mut plan: Vec<usize> = Vec::new();
        let mut meet = vec![0.0];
        let mut cost = vec![0.0];
        let mut prev = ctx.depot;
        let mut t = 0.0;
        let mut i = 0;
        while i < remaining.len() {
            let node = remaining[i];
            let (leg_t, leg_f) = ctx.transfer.evaluate(prev, node, t);
            let mut m2 = meet.clone();
            m2.push(t + leg_t);
            let mut c2 = cost.clone();
            c2.push(leg_f);
            let mut r2 = plan.clone();
            r2.push(node);
            if !nowait_feasible(ctx, &r2, &m2, &c2) {
                i += 1;
                continue;
            }
            plan.push(node);
            meet = m2;
            cost = c2;
            t += leg_t;
            prev = node;
            let last = remaining.len() - 1;
            remaining[i] = remaining[last];
            remaining.pop();
        }
        buckets.push(plan);
    }
    if !remaining.is_empty() {
        if buckets.is_empty() {
            buckets.push(Vec::new());
        }
        buckets.last_mut().unwrap().extend(remaining);
    }
    assemble(ctx, &buckets)
}

// ========================================================================= //
// Crossover
// ========================================================================= //

fn giant_tour(ctx: &Ctx, ind: &Individual) -> Vec<usize> {
    let active: Vec<usize> = (0..ind.plans.len())
        .filter(|&i| ind.plans[i].nodes_indices.len() > 1)
        .collect();
    let order = match ctx.config.ordering {
        HgsOrdering::Index => active,
        HgsOrdering::Raan => order_by_raan(ctx, ind, active),
        HgsOrdering::Nearest => order_by_nearest(ctx, ind, active),
    };
    let mut tour = Vec::new();
    for idx in order {
        tour.extend_from_slice(&ind.plans[idx].nodes_indices[1..]);
    }
    tour
}

fn order_by_raan(ctx: &Ctx, ind: &Individual, active: Vec<usize>) -> Vec<usize> {
    let mut keyed: Vec<(f64, usize)> = active
        .into_iter()
        .map(|i| {
            let seq = &ind.plans[i].nodes_indices;
            let mean = seq[1..].iter().map(|&d| ctx.problem.catalog.r[d]).sum::<f64>()
                / (seq.len() - 1) as f64;
            (mean, i)
        })
        .collect();
    keyed.sort_by(|a, b| fcmp(a.0, b.0));
    keyed.into_iter().map(|(_, i)| i).collect()
}

fn order_by_nearest(ctx: &Ctx, ind: &Individual, active: Vec<usize>) -> Vec<usize> {
    if active.len() <= 1 {
        return active;
    }
    let first = |i: usize| ind.plans[i].nodes_indices[1];
    let last = |i: usize| *ind.plans[i].nodes_indices.last().unwrap();
    let cost = |i: usize, j: usize| orbital_distance(ctx.problem, ctx.transfer, ctx.config, i, j);

    let mut remaining = active;
    let start = remaining
        .iter()
        .copied()
        .min_by(|&a, &b| fcmp(cost(ctx.depot, first(a)), cost(ctx.depot, first(b))))
        .unwrap();
    remaining.retain(|&i| i != start);
    let mut order = vec![start];
    while !remaining.is_empty() {
        let tail = last(*order.last().unwrap());
        let next = remaining
            .iter()
            .copied()
            .min_by(|&a, &b| fcmp(cost(tail, first(a)), cost(tail, first(b))))
            .unwrap();
        remaining.retain(|&i| i != next);
        order.push(next);
    }
    order
}

fn ox(p1: &[usize], p2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
    let n = p1.len();
    if n < 2 {
        return p1.to_vec();
    }
    let maxid = *p1.iter().max().unwrap();
    let mut child = vec![0usize; n];
    let mut added = vec![false; maxid + 1];
    let start = rng.gen_range(0..n);
    let mut end = rng.gen_range(0..n);
    while end == start {
        end = rng.gen_range(0..n);
    }
    let stop = (end + 1) % n;
    let mut j = start;
    loop {
        let pos = j % n;
        child[pos] = p1[pos];
        added[p1[pos]] = true;
        j += 1;
        if j % n == stop {
            break;
        }
    }
    let mut pos = stop;
    for t in 0..n {
        let idx = p2[(stop + t) % n];
        if idx < added.len() && !added[idx] {
            child[pos] = idx;
            added[idx] = true;
            pos = (pos + 1) % n;
        }
    }
    child
}

fn erx(p1: &[usize], p2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
    let n = p1.len();
    if n < 2 {
        return p1.to_vec();
    }
    let maxid = p1.iter().chain(p2.iter()).copied().max().unwrap();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); maxid + 1];
    for w in p1.windows(2).chain(p2.windows(2)) {
        let (a, b) = (w[0], w[1]);
        if !adj[a].contains(&b) {
            adj[a].push(b);
        }
        if !adj[b].contains(&a) {
            adj[b].push(a);
        }
    }

    let mut visited = vec![false; maxid + 1];
    let mut child = Vec::with_capacity(n);
    let mut current = p1[0];
    for _ in 0..n {
        child.push(current);
        visited[current] = true;
        // Drop the visited node from its neighbours' lists.
        let nbrs = std::mem::take(&mut adj[current]);
        for &y in &nbrs {
            adj[y].retain(|&z| z != current);
        }
        if child.len() == n {
            break;
        }
        // Prefer a live neighbour with the fewest remaining neighbours; if the
        // node is isolated, restart from a random unvisited node.
        let live: Vec<usize> = nbrs.into_iter().filter(|&y| !visited[y]).collect();
        current = if !live.is_empty() {
            let min_deg = live.iter().map(|&y| adj[y].len()).min().unwrap();
            let tied: Vec<usize> = live.into_iter().filter(|&y| adj[y].len() == min_deg).collect();
            tied[rng.gen_range(0..tied.len())]
        } else {
            let rem: Vec<usize> = p1.iter().copied().filter(|&y| !visited[y]).collect();
            rem[rng.gen_range(0..rem.len())]
        };
    }
    child
}

fn split(ctx: &Ctx, giant: &[usize], k_goal: usize) -> Vec<Vec<usize>> {
    let n = giant.len();
    let k = k_goal.max(1);
    if n == 0 {
        return Vec::new();
    }
    let inf = f64::INFINITY;
    let mut memo = vec![vec![inf; n + 1]; k + 1];
    let mut pred = vec![vec![0usize; n + 1]; k + 1];
    memo[0][0] = 0.0;
    let tf = ctx.config.threshold_factor;
    let relax = if tf > 0.0 { 1.0 - 1.0 / tf } else { 0.0 };

    for s in 1..=k {
        for i in (s - 1)..n {
            let base = memo[s - 1][i];
            if base >= inf {
                continue;
            }
            for j in i..n {
                let plan = ctx.eval_debris(&giant[i..=j]);
                let c = base + ctx.metric.plan_value(ctx.problem, &plan);
                if c < memo[s][j + 1] {
                    memo[s][j + 1] = c;
                    pred[s][j + 1] = i;
                }
                // Relaxed agnostic pruning: stop extending once a constraint is
                // violated past the relaxation factor.
                let prune = ctx
                    .constraints()
                    .iter()
                    .any(|con| con.violation(&plan) > relax * con.metric(&plan));
                if prune {
                    break;
                }
            }
        }
    }

    let mut best_k = k;
    let mut best_c = memo[k][n];
    #[allow(clippy::needless_range_loop)]
    for s in 1..k {
        if memo[s][n] < best_c {
            best_c = memo[s][n];
            best_k = s;
        }
    }
    if best_c >= inf {
        return vec![giant.to_vec()];
    }

    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let mut j = n;
    for s in (1..=best_k).rev() {
        let i = pred[s][j];
        bounds.push((i, j));
        j = i;
    }
    bounds.reverse();
    bounds.iter().map(|&(a, b)| giant[a..b].to_vec()).collect()
}

fn debris_plans(ind: &Individual) -> Vec<Vec<usize>> {
    ind.plans
        .iter()
        .filter(|r| r.nodes_indices.len() > 1)
        .map(|r| r.nodes_indices[1..].to_vec())
        .collect()
}

/// Greedily insert `d` at the cheapest feasible position over `plans`.
fn cheapest_insert(ctx: &Ctx, plans: &mut [Vec<usize>], d: usize) {
    let mut best: Option<(usize, usize, f64)> = None;
    for (ri, plan) in plans.iter().enumerate() {
        let base = ctx.metric.plan_value(ctx.problem, &ctx.eval_debris(plan));
        for pos in 0..=plan.len() {
            let mut cand = plan.clone();
            cand.insert(pos, d);
            let delta = ctx.metric.plan_value(ctx.problem, &ctx.eval_debris(&cand)) - base;
            if best.is_none_or(|(_, _, b)| delta < b) {
                best = Some((ri, pos, delta));
            }
        }
    }
    if let Some((ri, pos, _)) = best {
        plans[ri].insert(pos, d);
    } else if let Some(r) = plans.first_mut() {
        r.push(d);
    }
}

/// Selective plan exchange: keep most of `p1`, replace a contiguous block of
/// its plans with one from `p2`, then repair duplicates and reinsert orphans.
fn srex(ctx: &Ctx, p1: &Individual, p2: &Individual, rng: &mut SmallRng) -> Vec<Vec<usize>> {
    let a = debris_plans(p1);
    let b = debris_plans(p2);
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    let (na, nb) = (a.len(), b.len());
    let num = 1 + rng.gen_range(0..na.min(nb));
    let sa = rng.gen_range(0..na);
    let sb = rng.gen_range(0..nb);

    let mut removed = vec![false; na];
    let mut injected = vec![false; ctx.problem.catalog.n];
    let mut result: Vec<Vec<usize>> = Vec::new();
    for t in 0..num {
        removed[(sa + t) % na] = true;
        let plan = &b[(sb + t) % nb];
        for &d in plan {
            injected[d] = true;
        }
        result.push(plan.clone());
    }

    let mut served = injected.clone();
    for (i, plan) in a.iter().enumerate() {
        if removed[i] {
            continue;
        }
        let kept: Vec<usize> = plan.iter().copied().filter(|&d| !injected[d]).collect();
        for &d in &kept {
            served[d] = true;
        }
        result.push(kept);
    }

    let mut orphans: Vec<usize> = Vec::new();
    for (i, plan) in a.iter().enumerate() {
        if removed[i] {
            for &d in plan {
                if !served[d] {
                    served[d] = true;
                    orphans.push(d);
                }
            }
        }
    }

    while result.len() < ctx.problem.num_chasers {
        result.push(Vec::new());
    }
    for &d in &orphans {
        cheapest_insert(ctx, &mut result, d);
    }
    result
}

fn crossover(ctx: &Ctx, rng: &mut SmallRng) -> Individual {
    let p1 = ctx.population.pick(ctx.config, rng);
    let p2 = ctx.population.pick(ctx.config, rng);
    let buckets = match ctx.config.crossover {
        HgsCrossover::Srex => srex(ctx, &p1, &p2, rng),
        kind => {
            let t1 = giant_tour(ctx, &p1);
            let t2 = giant_tour(ctx, &p2);
            let extra = if rng.gen::<f64>() < 0.1 { 1 } else { 0 };
            let k_goal = (p1.active.max(1) + extra).min(ctx.problem.num_chasers);
            let tour = match kind {
                HgsCrossover::Erx => erx(&t1, &t2, rng),
                _ => ox(&t1, &t2, rng),
            };
            split(ctx, &tour, k_goal)
        }
    };
    assemble(ctx, &buckets)
}

// ========================================================================= //
// Local search
// ========================================================================= //

fn rebuild_state(ind: &Individual, state_seq: &mut [usize], state_pos: &mut [usize]) {
    for (s, plan) in ind.plans.iter().enumerate() {
        for (p, &idx) in plan.nodes_indices.iter().enumerate() {
            if p != 0 {
                state_seq[idx] = s;
                state_pos[idx] = p;
            }
        }
    }
}

/// Refresh the (plan, position) state of the nodes in plan `s` only. Any node
/// that left `s` is re-registered when the plan it moved into is also refreshed.
fn rebuild_plan(ind: &Individual, s: usize, state_seq: &mut [usize], state_pos: &mut [usize]) {
    for (p, &idx) in ind.plans[s].nodes_indices.iter().enumerate() {
        if p != 0 {
            state_seq[idx] = s;
            state_pos[idx] = p;
        }
    }
}

fn intra_relocate(ctx: &Ctx, ind: &mut Individual, s: usize, pos: usize) -> bool {
    let seq = ind.plans[s].nodes_indices.clone();
    let length = seq.len();
    if length < 3 || pos == 0 || pos >= length {
        return false;
    }
    let base = ctx.metric.plan_value(ctx.problem, &ind.plans[s]);
    let state = seq[pos];
    for insert_at in 1..length {
        if insert_at == pos {
            continue;
        }
        let mut cand = seq.clone();
        cand.remove(pos);
        cand.insert(insert_at, state);
        if ctx.lb(&cand) >= base {
            continue;
        }
        let mo = ctx.eval(&cand);
        if ctx.metric.plan_value(ctx.problem, &mo) < base {
            ind.plans[s] = mo;
            return true;
        }
    }
    false
}

fn intra_swap(ctx: &Ctx, ind: &mut Individual, s: usize, pos: usize) -> bool {
    let seq = ind.plans[s].nodes_indices.clone();
    let length = seq.len();
    if length < 3 || pos == 0 || pos >= length {
        return false;
    }
    let base = ctx.metric.plan_value(ctx.problem, &ind.plans[s]);
    for t in (pos + 1)..length {
        let mut cand = seq.clone();
        cand.swap(pos, t);
        if ctx.lb(&cand) >= base {
            continue;
        }
        let mo = ctx.eval(&cand);
        if ctx.metric.plan_value(ctx.problem, &mo) < base {
            ind.plans[s] = mo;
            return true;
        }
    }
    false
}

fn two_opt(ctx: &Ctx, ind: &mut Individual, s: usize, pos: usize) -> bool {
    let seq = ind.plans[s].nodes_indices.clone();
    let length = seq.len();
    if length < 3 || pos == 0 || pos >= length - 1 {
        return false;
    }
    let base = ctx.metric.plan_value(ctx.problem, &ind.plans[s]);
    for end in (pos + 1)..length {
        let mut cand = seq.clone();
        cand[pos..=end].reverse();
        if ctx.lb(&cand) >= base {
            continue;
        }
        let mo = ctx.eval(&cand);
        if ctx.metric.plan_value(ctx.problem, &mo) < base {
            ind.plans[s] = mo;
            return true;
        }
    }
    false
}

fn inter_relocate(
    ctx: &Ctx,
    ind: &mut Individual,
    s1: usize,
    pos1: usize,
    s2: usize,
    pos2: usize,
) -> bool {
    if s1 == s2 {
        return false;
    }
    let seq1 = ind.plans[s1].nodes_indices.clone();
    let seq2 = ind.plans[s2].nodes_indices.clone();
    let (len1, len2) = (seq1.len(), seq2.len());
    if pos1 == 0 || pos1 >= len1 || pos2 == 0 || pos2 > len2 {
        return false;
    }
    let old = ctx.metric.plan_value(ctx.problem, &ind.plans[s1])
        + ctx.metric.plan_value(ctx.problem, &ind.plans[s2]);
    let u = seq1[pos1];

    // Move u from s1 into s2.
    let mut new1 = seq1.clone();
    new1.remove(pos1);
    let mut new2 = seq2.clone();
    new2.insert(pos2.min(new2.len()), u);
    if ctx.lb(&new1) + ctx.lb(&new2) < old {
        let mo1 = ctx.eval(&new1);
        let mo2 = ctx.eval(&new2);
        if ctx.metric.plan_value(ctx.problem, &mo1) + ctx.metric.plan_value(ctx.problem, &mo2) < old {
            ind.plans[s1] = mo1;
            ind.plans[s2] = mo2;
            return true;
        }
    }

    // Swap u with the node at pos2.
    if pos2 < len2 {
        let w = seq2[pos2];
        let mut n1 = seq1.clone();
        n1[pos1] = w;
        let mut n2 = seq2.clone();
        n2[pos2] = u;
        if ctx.lb(&n1) + ctx.lb(&n2) < old {
            let mo1 = ctx.eval(&n1);
            let mo2 = ctx.eval(&n2);
            if ctx.metric.plan_value(ctx.problem, &mo1) + ctx.metric.plan_value(ctx.problem, &mo2) < old {
                ind.plans[s1] = mo1;
                ind.plans[s2] = mo2;
                return true;
            }
        }
    }

    false
}

fn two_opt_star(
    ctx: &Ctx,
    ind: &mut Individual,
    s1: usize,
    pos1: usize,
    s2: usize,
    pos2: usize,
) -> bool {
    if s1 == s2 {
        return false;
    }
    let seq1 = ind.plans[s1].nodes_indices.clone();
    let seq2 = ind.plans[s2].nodes_indices.clone();
    let (len1, len2) = (seq1.len(), seq2.len());
    if pos1 == 0 || pos1 > len1 || pos2 == 0 || pos2 > len2 {
        return false;
    }
    let old = ctx.metric.plan_value(ctx.problem, &ind.plans[s1])
        + ctx.metric.plan_value(ctx.problem, &ind.plans[s2]);
    let mut new1 = seq1[..pos1].to_vec();
    new1.extend_from_slice(&seq2[pos2..len2]);
    let mut new2 = seq2[..pos2].to_vec();
    new2.extend_from_slice(&seq1[pos1..len1]);
    if ctx.lb(&new1) + ctx.lb(&new2) >= old {
        return false;
    }
    let mo1 = ctx.eval(&new1);
    let mo2 = ctx.eval(&new2);
    if ctx.metric.plan_value(ctx.problem, &mo1) + ctx.metric.plan_value(ctx.problem, &mo2) < old {
        ind.plans[s1] = mo1;
        ind.plans[s2] = mo2;
        true
    } else {
        false
    }
}

fn improve(ind: &mut Individual, ctx: &Ctx, rng: &mut SmallRng) {
    let mut state_seq = vec![0usize; ctx.problem.catalog.n];
    let mut state_pos = vec![0usize; ctx.problem.catalog.n];
    let mut when_tested = vec![0u64; ctx.problem.catalog.n];
    let mut when_modified = vec![1u64; ind.plans.len()];
    let mut num_moves = 1u64;
    rebuild_state(ind, &mut state_seq, &mut state_pos);

    let mut improved = true;
    let mut loop_num = 0u64;
    while improved {
        improved = false;
        loop_num += 1;
        let mut clients = ctx.debris.clone();
        clients.shuffle(rng);
        for &u in &clients {
            let last_tested = when_tested[u];
            when_tested[u] = num_moves;
            let su = state_seq[u];
            let pu = state_pos[u];

            // Inter-plan moves over u's neighbours.
            let mut did = false;
            let len_n = ctx.neighbors[u].len();
            if len_n > 0 {
                let start = rng.gen_range(0..len_n);
                for off in 0..len_n {
                    let v = ctx.neighbors[u][(start + off) % len_n];
                    let sv = state_seq[v];
                    if su == sv {
                        continue;
                    }
                    // Skip if neither touched plan changed since last test.
                    if when_modified[su].max(when_modified[sv]) <= last_tested {
                        continue;
                    }
                    let pv = state_pos[v];
                    if inter_relocate(ctx, ind, su, pu, sv, pv + 1)
                        || (pu == 1 && inter_relocate(ctx, ind, sv, pv, su, pu))
                        || two_opt_star(ctx, ind, su, pu, sv, pv + 1)
                    {
                        num_moves += 1;
                        when_modified[su] = num_moves;
                        when_modified[sv] = num_moves;
                        rebuild_plan(ind, su, &mut state_seq, &mut state_pos);
                        rebuild_plan(ind, sv, &mut state_seq, &mut state_pos);
                        improved = true;
                        did = true;
                        break;
                    }
                }
            }
            if did {
                continue;
            }

            // Empty-plan moves (after the first sweep).
            let su = state_seq[u];
            let pu = state_pos[u];
            if loop_num > 1 {
                if let Some(empty_s) = ind.plans.iter().position(|r| r.nodes_indices.len() <= 1) {
                    if empty_s != su
                        && (two_opt_star(ctx, ind, su, pu, empty_s, 1)
                            || inter_relocate(ctx, ind, su, pu, empty_s, 1))
                    {
                        num_moves += 1;
                        when_modified[su] = num_moves;
                        when_modified[empty_s] = num_moves;
                        rebuild_plan(ind, su, &mut state_seq, &mut state_pos);
                        rebuild_plan(ind, empty_s, &mut state_seq, &mut state_pos);
                        improved = true;
                        continue;
                    }
                }
            }

            // Intra-plan moves (only if u's plan changed since last test).
            let su = state_seq[u];
            let pu = state_pos[u];
            if when_modified[su] > last_tested {
                if intra_relocate(ctx, ind, su, pu) {
                    num_moves += 1;
                    when_modified[su] = num_moves;
                    rebuild_plan(ind, su, &mut state_seq, &mut state_pos);
                    improved = true;
                }
                let su = state_seq[u];
                let pu = state_pos[u];
                if intra_swap(ctx, ind, su, pu) {
                    num_moves += 1;
                    when_modified[su] = num_moves;
                    rebuild_plan(ind, su, &mut state_seq, &mut state_pos);
                    improved = true;
                }
                let su = state_seq[u];
                let pu = state_pos[u];
                if two_opt(ctx, ind, su, pu) {
                    num_moves += 1;
                    when_modified[su] = num_moves;
                    rebuild_plan(ind, su, &mut state_seq, &mut state_pos);
                    improved = true;
                }
            }
        }
    }
    ind.refresh(ctx);
}

// ========================================================================= //
// HGS
// ========================================================================= //

fn finalize(ctx: &Ctx, ind: &Individual) -> MissionPlan {
    let mut plans: Vec<ChaserPlan> = ind.plans
        .iter()
        .map(|p| (**p).clone())
        .collect();
    while plans.len() < ctx.problem.num_chasers {
        plans.push(ChaserPlan::empty());
    }
    let costs: Vec<f64> = plans.iter().map(|p| p.cost).collect();
    let cost = ctx.objective().aggregate(&costs);
    let feasible = plans.iter().all(|p| ctx.problem.is_feasible(p));
    MissionPlan::new(plans, cost, feasible)
}

pub struct HgsSequencer<'p> {
    problem: &'p Problem,
    transfer: &'p dyn TransferLayer,
    schedule: &'p dyn ScheduleLayer,
    config: HgsConfig,
}

impl<'p> HgsSequencer<'p> {
    pub fn new(
        problem: &'p Problem,
        transfer: &'p dyn TransferLayer,
        schedule: &'p dyn ScheduleLayer,
        config: HgsConfig,
    ) -> Self {
        Self {
            problem,
            transfer,
            schedule,
            config,
        }
    }

    /// Build a sequencer with the default config.
    pub fn default(
        problem: &'p Problem,
        transfer: &'p dyn TransferLayer,
        schedule: &'p dyn ScheduleLayer,
    ) -> Self {
        Self::new(problem, transfer, schedule, HgsConfig::default())
    }

    fn log_init(&self) {
        println!("Constructing initial population...");
    }

    fn log_iter(&self, start: Instant, iter: usize, cost: f64, feas: bool) {        
        let timer = start.elapsed().as_secs_f64();        
        let cost_str = if cost.is_finite() {
            format!("{cost:.1}")
        } else {
            "--".to_string()
        };
        let feas_str = if feas { "true" } else { "false" };
        println!("  {timer:9.1}  {iter:7}  {cost_str:>13}  {feas_str:>8}");
    }

    fn log_head(&self) {
        println!("Starting genetic search...");
        println!(
            "  {:>9}  {:>7}  {:>13}  {:>8}",
            "time", "iter", "cost", "feasible"
        );
    }
}

impl<'p> SequenceLayer for HgsSequencer<'p> {
    fn initialize(&self) -> () {}

    fn evaluate(&self) -> MissionPlan {
        let start = Instant::now();

        let mut rng = SmallRng::seed_from_u64(self.config.seed);
        let mut ctx = Ctx::new(self.problem, self.transfer, self.schedule, &self.config);

        // Generate initial population
        if self.config.log_iter > 0 { self.log_init(); }
        for _ in 0..self.config.pop_init {
            let mut ind = generate(&ctx, &mut rng);
            improve(&mut ind, &ctx, &mut rng);
            ctx.add(ind);
        }

        // Run genetic search
        if self.config.log_iter > 0 { self.log_head(); }
        let mut best = ctx.best_value();
        let mut iter = 0usize;
        let mut nimp = 0usize;
        loop {
            // Stopping criteria
            if iter >= self.config.max_iter { break; }
            if nimp >= self.config.max_nimp { break; }
            if start.elapsed().as_secs_f64() >= self.config.max_time { break; }

            // Build new individual in the population
            let mut child = crossover(&ctx, &mut rng);
            improve(&mut child, &ctx, &mut rng);
            ctx.add(child);
            iter += 1;

            // Update best solution
            let curr = ctx.best_value();
            if curr < best { best = curr; nimp = 0; } else { nimp += 1; }

            // Log iteration
            if iter.is_multiple_of(self.config.log_iter) {
                match ctx.best() {
                    Some(b) => self.log_iter(start, iter, b.cost, b.feasible),
                    None => self.log_iter(start, iter, f64::INFINITY, false),
                }
            }
        }

        // Retrieve best solution and log final iteration
        let plan = ctx.best().expect("no best solution found");
        if self.config.log_iter > 0 {
            self.log_iter(start, iter, plan.cost, plan.feasible);
        }

        finalize(&ctx, &plan)
    }

    fn statistics(&self) -> HashMap<String, Box<dyn Any>> {
        HashMap::new()
    }
}

// ========================================================================= //
// Tests
// ========================================================================= //

/// Unit tests for the HGS internals.
///
/// Port of the `test_hgs_*` cases in
/// `references/pyadr/tests/test_sequence.py`. They live here rather than under
/// `tests/` because `Ctx`, `Individual`, `generate`, `improve`, `giant_tour`
/// and `split` are private to the crate.
#[cfg(test)]
mod tests {
    use super::*;

    use crate::io::read_debris;
    use crate::orbit::DAY;
    use crate::problem::make_problem;
    use crate::schedule::{NowaitConfig, NowaitScheduler};
    use crate::transfer::{QlawConfig, QlawTransfer};

    /// The Python fixture: 3 chasers, a 2500-day horizon and a load limit of 5.
    fn fixture_problem() -> Problem {
        let debris = read_debris("data/odrc.csv").expect("catalogue must be readable");
        make_problem(&debris, 3, Some(2500.0 * DAY), Some(5), None, "sum")
            .expect("problem must be constructible")
    }

    fn quiet(seed: u64) -> HgsConfig {
        HgsConfig { seed, log_iter: 0, ..HgsConfig::default() }
    }

    /// Run `body` with a fully wired context over the fixture problem.
    fn with_ctx(seed: u64, body: impl FnOnce(&Ctx, &mut SmallRng)) {
        let problem = fixture_problem();
        let transfer = QlawTransfer::new(
            &problem,
            QlawConfig { thrust: 3e-3, ..QlawConfig::default() },
        );
        let schedule = NowaitScheduler::new(&problem, &transfer, NowaitConfig::default());
        let config = quiet(seed);
        let ctx = Ctx::new(&problem, &transfer, &schedule, &config);
        let mut rng = SmallRng::seed_from_u64(seed);
        body(&ctx, &mut rng);
    }

    #[test]
    fn local_search_improves_the_greedy_start() {
        with_ctx(1, |ctx, rng| {
            let mut ind = generate(ctx, rng);
            let before = ctx.metric.value(&ind);
            improve(&mut ind, ctx, rng);
            let after = ctx.metric.value(&ind);
            assert!(
                after <= before + 1e-6,
                "local search made the solution worse: {before:.1} -> {after:.1}"
            );
            assert!(
                after < before,
                "local search did not improve the greedy start: {before:.1} -> {after:.1}"
            );
        });
    }

    #[test]
    fn split_preserves_all_debris() {
        with_ctx(2, |ctx, rng| {
            let ind = generate(ctx, rng);
            let giant = giant_tour(ctx, &ind);
            let buckets = split(ctx, &giant, ctx.problem.num_chasers);

            let mut flat: Vec<usize> = buckets.iter().flatten().copied().collect();
            let mut expected = giant.clone();
            flat.sort_unstable();
            expected.sort_unstable();
            assert_eq!(flat, expected, "split dropped or duplicated debris");
            assert!(
                buckets.len() <= ctx.problem.num_chasers,
                "split used more routes ({}) than chasers ({})",
                buckets.len(),
                ctx.problem.num_chasers
            );
        });
    }

    #[test]
    fn generated_individuals_cover_the_catalogue() {
        with_ctx(3, |ctx, rng| {
            for _ in 0..5 {
                let ind = generate(ctx, rng);
                let mut visited: Vec<usize> = ind
                    .plans
                    .iter()
                    .flat_map(|p| p.nodes_indices.iter().copied())
                    .filter(|&n| n != ctx.depot)
                    .collect();
                visited.sort_unstable();
                let before = visited.len();
                visited.dedup();
                assert_eq!(before, visited.len(), "a debris was visited twice");
                assert_eq!(visited, ctx.debris, "a generated individual dropped debris");
            }
        });
    }

    #[test]
    fn local_search_preserves_the_catalogue() {
        with_ctx(4, |ctx, rng| {
            let mut ind = generate(ctx, rng);
            improve(&mut ind, ctx, rng);
            let mut visited: Vec<usize> = ind
                .plans
                .iter()
                .flat_map(|p| p.nodes_indices.iter().copied())
                .filter(|&n| n != ctx.depot)
                .collect();
            visited.sort_unstable();
            visited.dedup();
            assert_eq!(visited, ctx.debris, "local search dropped or duplicated debris");
        });
    }
}
