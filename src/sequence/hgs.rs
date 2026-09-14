//! Sequence layer based on a hybrid genetic search.

use std::collections::{BTreeSet};
use std::rc::Rc;
use std::time::Instant;

use crate::orbit::TAU;
use crate::problem::{ChaserPlan, MissionPlan, Problem, DEPOT};
use crate::solver::{ScheduleLayer, SequenceLayer};

use rand::rngs::SmallRng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};

// ========================================================================= //
// Configuration
// ========================================================================= //

/// Giant-tour ordering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Keep the plans in the order they were built.
    Index,
    /// By circular mean RAAN, read at the times the debris are visited.
    Raan,
    /// Chain the plans greedily by the cost of the transfer joining them.
    Nearest,
}

/// Crossover operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossover {
    /// Order crossover on the giant tour.
    Ox,
    /// Edge-recombination crossover on the giant tour.
    Erx,
    /// Selective exchange of whole chaser plans.
    Srex,
}

/// Initial population generation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generation {
    /// Nearest-neighbor construction on the cheapest reachable debris.
    Greedy,
    /// Random assignment of debris to chasers.
    Random,
}

/// Hybrid genetic search configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HgsConfig {
    // -- global -----------------------------------------------------------
    /// Random seed.
    pub seed: u64,
    /// Toggle verbosity.
    pub verbose: bool,
    /// Verbosity iteration step.
    pub log_iter: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Maximum number of iterations without improvement.
    pub max_nimp: usize,
    /// Maximum time budget in seconds.
    pub max_time: f64,

    // -- generation -------------------------------------------------------
    /// Strategy used to build the initial population.
    pub generation: Generation,
    /// Number of individuals in the initial population.
    pub pop_init: usize,

    // -- population -------------------------------------------------------
    /// Minimum population size.
    pub pop_min: usize,
    /// Maximum population size.
    pub pop_max: usize,
    /// Number of nearest individuals averaged in the diversity measure.
    pub nb_close: usize,
    /// Number of elite individuals with no diversity measure.
    pub nb_elite: usize,
    /// Number of iterations between penalties adaptation.
    pub adapt_iter: usize,
    /// Distance below which two individuals count as clones.
    pub clone_eps: f64,

    // -- crossover --------------------------------------------------------
    /// Number of candidates drawn in each parent tournament.
    pub nb_pick: usize,
    /// Infeasibility factor during split operation of a giant tour.
    pub threshold_factor: f64,
    /// Giant-tour ordering.
    pub ordering: Ordering,
    /// Crossover operator.
    pub crossover: Crossover,

    // -- local search -----------------------------------------------------
    /// Size of the neighborhood for local search moves.
    pub nb_neighbors: usize,

    // -- penalties --------------------------------------------------------
    /// Multiplier applied to increase a penalty.
    pub penalty_increase: f64,
    /// Multiplier applied to decrease a penalty.
    pub penalty_decrease: f64,
    /// Lower bound on the penalties range.
    pub penalty_min: f64,
    /// Upper bound on the penalties range.
    pub penalty_max: f64,
    /// Target ratio of feasible individuals in the population.
    pub target_ratio: f64,
}

impl Default for HgsConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            verbose: true,
            log_iter: 10,

            max_iter: 5_000,
            max_nimp: 500,
            max_time: f64::INFINITY,

            generation: Generation::Greedy,
            pop_init: 20,

            pop_min: 10,
            pop_max: 20,
            nb_close: 2,
            nb_elite: 3,
            adapt_iter: 10,
            clone_eps: 1e-6,

            nb_pick: 4,
            threshold_factor: 1.5,
            ordering: Ordering::Nearest,
            crossover: Crossover::Ox,

            nb_neighbors: 10,

            penalty_increase: 2.0,
            penalty_decrease: 0.5,
            penalty_min: 1e-9,
            penalty_max: 1e9,
            target_ratio: 0.9,
        }
    }
}

impl HgsConfig {
    /// Preset configuration with level from 0 to 6 (higher = deeper search).
    pub fn preset(level: usize) -> Self {
        // log, iter, nimp, init, min, max, close, elite
        let table = [
            (1usize, 0usize, 0usize, 1usize, 2usize, 2usize, 1usize, 1usize),
            (1, 0, 0, 5, 2, 3, 1, 1),
            (1, 50, 10, 6, 3, 6, 1, 1),
            (1, 500, 100, 10, 5, 10, 2, 2),
            (10, 5_000, 500, 20, 10, 20, 2, 3),
            (10, 50_000, 5_000, 24, 12, 32, 3, 4),
            (100, 200_000, 10_000, 50, 25, 65, 3, 8),
        ];
        let &(log, iter, nimp, init, lo, hi, close, elite) = table
            .get(level)
            .expect("HGS preset level must be between 0 and 6");
        Self {
            log_iter: log,
            max_iter: iter,
            max_nimp: nimp,
            pop_init: init,
            pop_min: lo,
            pop_max: hi,
            nb_close: close,
            nb_elite: elite,
            ..Self::default()
        }
    }

    /// Set configuration seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set configuration verbosity.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set configuration maximum time budget in seconds.
    pub fn with_max_time(mut self, max_time: f64) -> Self {
        self.max_time = max_time;
        self
    }
}

// ========================================================================= //
// Individual
// ========================================================================= //

/// Individual in the population as a candidate mission plan.
#[derive(Debug, Clone)]
pub struct Individual {
    /// One plan per chaser, some possibly idle.
    pub plans: Vec<Rc<ChaserPlan>>,
    /// Mission cost, the sum of the chaser plan costs.
    pub cost: f64,
    /// Total excess over the load limit, and over the duration limit.
    pub violation: [f64; 2],
    /// Whether every plan satisfies both operational limits.
    pub feasible: bool,
    /// Number of chasers that actually collect something.
    pub active: usize,
    /// For each debris, the object visited just before it, [`DEPOT`] for none.
    pub pred: Vec<usize>,
    /// For each debris, the object visited just after it, [`DEPOT`] for none.
    pub succ: Vec<usize>,
}

impl Individual {
    /// Derive every ranking quantity from a set of chaser plans.
    pub fn new(problem: &Rc<Problem>, plans: Vec<Rc<ChaserPlan>>) -> Self {
        let n = problem.states.len();
        let mut individual = Self {
            plans,
            cost: 0.0,
            violation: [0.0, 0.0],
            feasible: true,
            active: 0,
            pred: vec![DEPOT; n],
            succ: vec![DEPOT; n],
        };
        individual.refresh(problem);
        individual
    }

    /// Recompute the derived quantities after the plans were replaced.
    pub fn refresh(&mut self, problem: &Rc<Problem>) {
        self.cost = self.plans.iter().map(|p| p.cost()).sum();
        self.pred.iter_mut().for_each(|x| *x = DEPOT);
        self.succ.iter_mut().for_each(|x| *x = DEPOT);

        let mut load_excess = 0.0;
        let mut time_excess = 0.0;
        self.active = 0;
        self.feasible = true;

        for plan in &self.plans {
            load_excess += plan.load().saturating_sub(problem.max_load) as f64;
            time_excess += (plan.time() - problem.max_time).max(0.0);
            self.feasible &= problem.is_plan_feasible(plan);

            let seq = &plan.sequence;
            if seq.len() > 1 {
                self.active += 1;
                let last = seq.len() - 1;
                for i in 1..seq.len() {
                    let d = seq[i];
                    self.pred[d] = if i == 1 { DEPOT } else { seq[i - 1] };
                    self.succ[d] = if i == last { DEPOT } else { seq[i + 1] };
                }
            }
        }
        self.violation = [load_excess, time_excess];
    }
}

/// Broken-pair distance between two individuals.
pub fn broken_pair_distance(a: &Individual, b: &Individual, debris: &[usize]) -> f64 {
    if debris.is_empty() {
        return 0.0;
    }
    let mut count = 0usize;
    for &d in debris {
        let (asucc, bsucc) = (a.succ[d], b.succ[d]);
        let (apred, bpred) = (a.pred[d], b.pred[d]);
        count += usize::from(asucc != bsucc && asucc != bpred);
        count += usize::from(apred == DEPOT && bpred != DEPOT && bsucc != DEPOT);
    }
    count as f64 / debris.len() as f64
}

/// Unified metric to evaluate both feasible and infeasible individuals.
#[derive(Debug, Clone)]
pub struct Metric {
    /// Constraint penalties: [load, time]
    pub penalties: [f64; 2],
    config: HgsConfig,
}

impl Metric {
    /// Initialize metric and scale the penalties to the instance.
    pub fn new<S: ScheduleLayer>(problem: &Rc<Problem>, schedule: &S, config: HgsConfig) -> Self {
        let scale = problem
            .debris()
            .map(|d| schedule.evaluate_lb(&[DEPOT, d]))
            .sum::<f64>()
            .max(1.0);
        let clamp = |x: f64| x.max(config.penalty_min).min(config.penalty_max);
        Self {
            penalties: [
                clamp(scale / problem.max_load as f64),
                clamp(scale / problem.max_time),
            ],
            config,
        }
    }

    /// Penalized cost of a single chaser plan.
    pub fn plan_value(&self, problem: &Rc<Problem>, plan: &ChaserPlan) -> f64 {
        let excesses = [
            plan.load().saturating_sub(problem.max_load) as f64,
            (plan.time() - problem.max_time).max(0.0)
        ];
        plan.cost() + self.penalties
            .iter()
            .zip(excesses)
            .map(|(p, e)| p * e).sum::<f64>()
    }

    /// Penalized cost of a whole mission plan.
    pub fn value(&self, individual: &Individual) -> f64 {
        individual.cost + self.penalties
            .iter()
            .zip(individual.violation)
            .map(|(p, v)| p * v).sum::<f64>()
    }

    /// Raise the penalties that are too weak, lower those that are too strong.
    pub fn adapt(&mut self, ratios: [f64; 2]) {
        for (penalty, ratio) in self.penalties.iter_mut().zip(ratios) {
            let factor = if ratio < self.config.target_ratio {
                self.config.penalty_increase
            } else {
                self.config.penalty_decrease
            };
            *penalty = (factor * *penalty)
                .max(self.config.penalty_min)
                .min(self.config.penalty_max);
        }
    }
}

// ========================================================================= //
// Population
// ========================================================================= //

/// Subpopulation of feasible or infeasible individuals.
#[derive(Debug, Clone)]
pub struct Subpopulation {
    /// The individuals of the subpopulation.
    pub individuals: Vec<Individual>,
    /// Biased fitness of each individual (lower is better).
    pub score: Vec<f64>,
    /// Pairwise broken-pair distances between the individuals.
    dist: Vec<Vec<f64>>,
    config: HgsConfig,
}

impl Subpopulation {
    /// An empty subpopulation.
    pub fn new(config: HgsConfig) -> Self {
        Self {
            individuals: Vec::new(),
            score: Vec::new(),
            dist: Vec::new(),
            config,
        }
    }

    /// Number of individuals held.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Whether no individual is held.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Insert an individual, trimming the subpopulation if it overflows.
    pub fn add(&mut self, debris: &[usize], metric: &Metric, individual: Individual) {
        let mut row: Vec<f64> = self
            .individuals
            .iter()
            .map(|other| broken_pair_distance(&individual, other, debris))
            .collect();
        for (j, previous) in self.dist.iter_mut().enumerate() {
            previous.push(row[j]);
        }
        row.push(0.0);
        self.dist.push(row);
        self.individuals.push(individual);

        self.update_score(metric);
        if self.individuals.len() > self.config.pop_max {
            self.survivors_selection(metric);
        }
    }

    /// Drop one individual and the distance row and column that describe it.
    fn remove(&mut self, index: usize) {
        self.individuals.remove(index);
        self.dist.remove(index);
        for row in &mut self.dist {
            row.remove(index);
        }
    }

    /// Recompute the biased fitness of every individual.
    pub fn update_score(&mut self, metric: &Metric) {
        let n = self.individuals.len();
        self.score = vec![0.0; n];
        if n <= 1 {
            return;
        }
        let denom = (n - 1) as f64;

        // Rank on penalized cost, cheapest first.
        let values: Vec<f64> = self.individuals.iter().map(|i| metric.value(i)).collect();
        let mut by_cost: Vec<usize> = (0..n).collect();
        by_cost.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let mut cost_pos = vec![0usize; n];
        for (rank, &i) in by_cost.iter().enumerate() {
            cost_pos[i] = rank;
        }

        // Rank on diversity, most isolated first, from the maintained matrix.
        let close = self.config.nb_close.min(n - 1);
        let mut spread = vec![0.0; n];
        if close > 0 {
            for (i, value) in spread.iter_mut().enumerate() {
                let mut others: Vec<f64> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| self.dist[i][j])
                    .collect();
                others.sort_by(f64::total_cmp);
                *value = others[..close].iter().sum::<f64>() / close as f64;
            }
        }
        let mut by_spread: Vec<usize> = (0..n).collect();
        by_spread.sort_by(|&a, &b| spread[b].total_cmp(&spread[a]));
        let mut div_pos = vec![0usize; n];
        for (rank, &i) in by_spread.iter().enumerate() {
            div_pos[i] = rank;
        }

        let scale = 1.0 - self.config.nb_elite as f64 / n as f64;
        for i in 0..n {
            let base = cost_pos[i] as f64 / denom;
            self.score[i] = if cost_pos[i] < self.config.nb_elite {
                base
            } else {
                base + scale * div_pos[i] as f64 / denom
            };
        }
    }

    /// Distance from individual `i` to its closest neighbour.
    fn nearest_min(&self, i: usize) -> f64 {
        (0..self.dist[i].len())
            .filter(|&j| j != i)
            .map(|j| self.dist[i][j])
            .fold(f64::INFINITY, f64::min)
    }

    /// The individual to drop: a clone if there is one, else the worst ranked.
    fn worst_index(&self) -> usize {
        let mut worst = 0;
        let mut worst_clone = self.nearest_min(0) <= self.config.clone_eps;
        let mut worst_score = self.score[0];
        for i in 1..self.individuals.len() {
            let clone = self.nearest_min(i) <= self.config.clone_eps;
            let score = self.score[i];
            if (clone && !worst_clone) || (clone == worst_clone && score > worst_score) {
                worst = i;
                worst_clone = clone;
                worst_score = score;
            }
        }
        worst
    }

    /// Trim the subpopulation back to its minimum size.
    fn survivors_selection(&mut self, metric: &Metric) {
        while self.individuals.len() > self.config.pop_min {
            let index = self.worst_index();
            self.remove(index);
            self.update_score(metric);
        }
    }

    /// The cheapest individual held, if any.
    pub fn best(&self, metric: &Metric) -> Option<&Individual> {
        self.individuals
            .iter()
            .min_by(|a, b| metric.value(a).total_cmp(&metric.value(b)))
    }
}

/// Feasible and infeasible individual populations.
#[derive(Debug, Clone)]
pub struct Population {
    /// Individuals respecting both operational limits.
    pub feasible: Subpopulation,
    /// Individuals violating at least one limit.
    pub infeasible: Subpopulation,
    /// Recent feasibility record, one window per limit.
    windows: [Vec<bool>; 2],
    /// Insertions so far, which paces the penalty adaptation.
    count: usize,
    config: HgsConfig,
}

impl Population {
    /// An empty population.
    pub fn new(config: HgsConfig) -> Self {
        Self {
            feasible: Subpopulation::new(config),
            infeasible: Subpopulation::new(config),
            windows: [Vec::new(), Vec::new()],
            count: 0,
            config,
        }
    }

    /// Total number of individuals, both statuses together.
    pub fn len(&self) -> usize {
        self.feasible.len() + self.infeasible.len()
    }

    /// Whether the population holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert an individual and periodically retune the penalties.
    pub fn add(&mut self, debris: &[usize], metric: &mut Metric, individual: Individual) {
        self.count += 1;
        let window = self.config.adapt_iter;
        for (c, history) in self.windows.iter_mut().enumerate() {
            history.push(individual.violation[c] == 0.0);
            if window > 0 && history.len() > window {
                let excess = history.len() - window;
                history.drain(..excess);
            }
        }

        let target = if individual.feasible {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };
        target.add(debris, metric, individual);

        if window > 0 && self.count % window == 0 {
            let ratios = [
                self.windows[0].iter().filter(|&&b| b).count() as f64 / window as f64,
                self.windows[1].iter().filter(|&&b| b).count() as f64 / window as f64,
            ];
            metric.adapt(ratios);
            self.feasible.update_score(metric);
            self.infeasible.update_score(metric);
        }
    }

    /// Binary tournament over the whole population; the best rank wins.
    pub fn pick(&self, rng: &mut SmallRng) -> Option<Individual> {
        let total = self.len();
        if total == 0 {
            return None;
        }
        let nf = self.feasible.len();
        let draws = self.config.nb_pick.max(1).min(total);

        let mut chosen = 0;
        let mut best_score = f64::INFINITY;
        for d in 0..draws {
            let g = rng.random_range(0..total);
            let score = if g < nf {
                self.feasible.score[g]
            } else {
                self.infeasible.score[g - nf]
            };
            if d == 0 || score < best_score {
                chosen = g;
                best_score = score;
            }
        }
        let individual = if chosen < nf {
            &self.feasible.individuals[chosen]
        } else {
            &self.infeasible.individuals[chosen - nf]
        };
        Some(individual.clone())
    }

    /// The cheapest feasible individual, or the cheapest one at all.
    pub fn best(&self, metric: &Metric) -> Option<&Individual> {
        self.feasible.best(metric).or_else(|| self.infeasible.best(metric))
    }
}

// ========================================================================= //
// Local search
// ========================================================================= //

/// Local search operator.
pub struct Search<'a, S: ScheduleLayer> {
    /// The state of the run, which also provides the random scan order.
    context: &'a mut Context<S>,
    /// The individual being improved, rewritten in place.
    individual: &'a mut Individual,
    /// The plan currently holding each debris.
    holder: Vec<usize>,
    /// The position of each debris in the plan holding it.
    rank: Vec<usize>,
    /// Reading of the clock when each debris was last examined.
    tested: Vec<u64>,
    /// Reading of the clock when each plan last changed.
    touched: Vec<u64>,
    /// Ticks once per accepted move, and orders the two records above.
    clock: u64,
}

impl<'a, S: ScheduleLayer> Search<'a, S> {
    /// Prepare the search on `individual`.
    pub fn new(context: &'a mut Context<S>, individual: &'a mut Individual) -> Self {
        let states = context.problem.states.len();
        let plans = individual.plans.len();
        let mut search = Self {
            context,
            individual,
            holder: vec![0; states],
            rank: vec![0; states],
            tested: vec![0; states],
            touched: vec![1; plans],
            clock: 1,
        };
        for s in 0..plans {
            search.register(s);
        }
        search
    }

    /// Record where each debris of one plan currently sits.
    fn register(&mut self, s: usize) {
        for (position, &node) in self.individual.plans[s].sequence.iter().enumerate() {
            if position != 0 {
                self.holder[node] = s;
                self.rank[node] = position;
            }
        }
    }

    /// Lower bound on the cost of a candidate chaser sequence.
    fn bound(&self, sequence: &[usize]) -> f64 {
        self.context.schedule.evaluate_lb(sequence)
    }

    /// Screen a candidate move on the bound, then price it for real.
    fn accept(&mut self, slots: &[usize], candidates: &[Vec<usize>], base: f64) -> bool {
        // The bound costs a table lookup; the schedule costs a dynamic program.
        if candidates.iter().map(|c| self.bound(c)).sum::<f64>() >= base {
            return false;
        }
        let plans: Vec<_> = candidates
            .iter()
            .map(|c| self.context.schedule.evaluate(c))
            .collect();
        if plans.iter().map(|p| self.context.value(p)).sum::<f64>() >= base {
            return false;
        }
        for (&slot, plan) in slots.iter().zip(plans) {
            self.individual.plans[slot] = plan;
        }
        true
    }

    // ------------------------- Intra-plan moves ------------------------- //

    /// Move one debris elsewhere in its own plan.
    fn intra_relocate(&mut self, s: usize, pos: usize) -> bool {
        let seq = self.individual.plans[s].sequence.clone();
        let n = seq.len();
        if n < 3 || pos == 0 || pos >= n {
            return false;
        }
        let base = self.context.value(&self.individual.plans[s]);
        let node = seq[pos];
        let mut rest = seq.clone();
        rest.remove(pos);

        for insert_at in 1..n {
            if insert_at == pos {
                continue;
            }
            let mut candidate = rest.clone();
            candidate.insert(insert_at, node);
            if self.accept(&[s], &[candidate], base) {
                return true;
            }
        }
        false
    }

    /// Exchange two debris within the same plan.
    fn intra_swap(&mut self, s: usize, pos: usize) -> bool {
        let seq = self.individual.plans[s].sequence.clone();
        let n = seq.len();
        if n < 3 || pos == 0 || pos >= n {
            return false;
        }
        let base = self.context.value(&self.individual.plans[s]);
        for other in (pos + 1)..n {
            let mut candidate = seq.clone();
            candidate.swap(pos, other);
            if self.accept(&[s], &[candidate], base) {
                return true;
            }
        }
        false
    }

    /// Reverse a segment of a plan.
    fn two_opt(&mut self, s: usize, pos: usize) -> bool {
        let seq = self.individual.plans[s].sequence.clone();
        let n = seq.len();
        if n < 3 || pos == 0 || pos >= n - 1 {
            return false;
        }
        let base = self.context.value(&self.individual.plans[s]);
        for end in (pos + 1)..n {
            let mut candidate = seq.clone();
            candidate[pos..=end].reverse();
            if self.accept(&[s], &[candidate], base) {
                return true;
            }
        }
        false
    }

    // ------------------------- Inter-plan moves ------------------------- //

    /// Hand one debris over to another chaser, or trade two of them.
    fn inter_relocate(&mut self, s1: usize, pos1: usize, s2: usize, pos2: usize) -> bool {
        if s1 == s2 {
            return false;
        }
        let seq1 = self.individual.plans[s1].sequence.clone();
        let seq2 = self.individual.plans[s2].sequence.clone();
        let (n1, n2) = (seq1.len(), seq2.len());
        if pos1 == 0 || pos1 >= n1 || pos2 == 0 || pos2 > n2 {
            return false;
        }
        let base = self.context.value(&self.individual.plans[s1])
            + self.context.value(&self.individual.plans[s2]);
        let node = seq1[pos1];

        let mut moved1 = seq1.clone();
        moved1.remove(pos1);
        let mut moved2 = seq2.clone();
        moved2.insert(pos2.min(n2), node);
        if self.accept(&[s1, s2], &[moved1, moved2], base) {
            return true;
        }

        if pos2 < n2 {
            let mut traded1 = seq1.clone();
            let mut traded2 = seq2.clone();
            traded1[pos1] = seq2[pos2];
            traded2[pos2] = node;
            if self.accept(&[s1, s2], &[traded1, traded2], base) {
                return true;
            }
        }
        false
    }

    /// Exchange the tails of two plans.
    fn two_opt_star(&mut self, s1: usize, pos1: usize, s2: usize, pos2: usize) -> bool {
        if s1 == s2 {
            return false;
        }
        let seq1 = self.individual.plans[s1].sequence.clone();
        let seq2 = self.individual.plans[s2].sequence.clone();
        if pos1 == 0 || pos1 > seq1.len() || pos2 == 0 || pos2 > seq2.len() {
            return false;
        }
        let base = self.context.value(&self.individual.plans[s1])
            + self.context.value(&self.individual.plans[s2]);
        let candidate1: Vec<usize> = seq1[..pos1].iter().chain(&seq2[pos2..]).copied().collect();
        let candidate2: Vec<usize> = seq2[..pos2].iter().chain(&seq1[pos1..]).copied().collect();
        self.accept(&[s1, s2], &[candidate1, candidate2], base)
    }

    // ---------------------- Local-search execution ---------------------- //

    /// Local search run.
    pub fn run(mut self) {
        let mut sweep = 0usize;
        let mut improved = true;
        while improved {
            improved = false;
            sweep += 1;
            let mut order = self.context.debris.clone();
            order.shuffle(&mut self.context.rng);

            for u in order {
                let last_tested = self.tested[u];
                self.tested[u] = self.clock;

                // Offer u to each of its neighbors, from a random entry point.
                let peers = self.context.neighbors[u].len();
                let mut done = false;
                if peers > 0 {
                    let start = self.context.rng.random_range(0..peers);
                    for offset in 0..peers {
                        let v = self.context.neighbors[u][(start + offset) % peers];
                        let (su, sv) = (self.holder[u], self.holder[v]);
                        if su == sv {
                            continue;
                        }
                        if self.touched[su].max(self.touched[sv]) <= last_tested {
                            continue;
                        }
                        let (pu, pv) = (self.rank[u], self.rank[v]);
                        if self.inter_relocate(su, pu, sv, pv + 1)
                            || (pu == 1 && self.inter_relocate(sv, pv, su, pu))
                            || self.two_opt_star(su, pu, sv, pv + 1)
                        {
                            self.clock += 1;
                            self.touched[su] = self.clock;
                            self.touched[sv] = self.clock;
                            self.register(su);
                            self.register(sv);
                            improved = true;
                            done = true;
                            break;
                        }
                    }
                }
                if done {
                    continue;
                }

                // Once the plans have settled, try waking an idle chaser up.
                if sweep > 1 {
                    let idle = self
                        .individual
                        .plans
                        .iter()
                        .position(|p| p.sequence.len() <= 1);
                    let (su, pu) = (self.holder[u], self.rank[u]);
                    if let Some(idle) = idle {
                        if idle != su
                            && (self.two_opt_star(su, pu, idle, 1)
                                || self.inter_relocate(su, pu, idle, 1))
                        {
                            self.clock += 1;
                            self.touched[su] = self.clock;
                            self.touched[idle] = self.clock;
                            self.register(su);
                            self.register(idle);
                            improved = true;
                            continue;
                        }
                    }
                }

                // Rearrange u's own plan, but only if that plan has moved on.
                if self.touched[self.holder[u]] > last_tested {
                    for move_index in 0..3 {
                        let (su, pu) = (self.holder[u], self.rank[u]);
                        let applied = match move_index {
                            0 => self.intra_relocate(su, pu),
                            1 => self.intra_swap(su, pu),
                            _ => self.two_opt(su, pu),
                        };
                        if applied {
                            self.clock += 1;
                            self.touched[su] = self.clock;
                            self.register(su);
                            improved = true;
                        }
                    }
                }
            }
        }
        self.individual.refresh(&self.context.problem);
    }
}

// ========================================================================= //
// Context
// ========================================================================= //

/// Algorithmic trace.
pub struct LogEntry {
    /// Time in seconds since the start of the run.
    pub time: f64,
    /// Iteration number.
    pub iter: usize,
    /// Number of iterations without improvement.
    pub nimp: usize,
    /// Penalized cost of the incumbent.
    pub cost: f64,
    /// Number of feasible individuals in the population.
    pub num_feasible: usize,
    /// Number of infeasible individuals in the population.
    pub num_infeasible: usize,
    /// Current penalties for the load and time limits.
    pub penalty_load: f64,
    /// Current penalties for the load and time limits.
    pub penalty_time: f64,
}

/// Algorithmic trace, one entry per iteration.
pub type Trace = Vec<LogEntry>;

/// Context manager for hybrid genetic search.
pub struct Context<S: ScheduleLayer> {
    /// The instance being solved.
    pub problem: Rc<Problem>,
    /// The layer pricing a chaser plan along a fixed sequence.
    pub schedule: S,
    /// The parameters of the run.
    pub config: HgsConfig,
    /// The random source, seeded from the configuration.
    pub rng: SmallRng,
    /// Indices of the debris to collect.
    pub debris: Vec<usize>,
    /// The penalized objective.
    pub metric: Metric,
    /// The individuals currently held.
    pub population: Population,
    /// For each debris, its `nb_neighbors` closest peers.
    pub neighbors: Vec<Vec<usize>>,
    /// Start time
    pub start_time: Instant,
    /// Number of iterations
    pub iteration: usize,
    /// Number of iterations without improvement
    pub stalled: usize,
    /// Algorithmic trace
    pub trace: Trace,
}

impl<S: ScheduleLayer> Context<S> {
    /// Prepare a run over `problem`.
    pub fn new(problem: Rc<Problem>, schedule: S, config: HgsConfig) -> Self {
        let debris: Vec<usize> = problem.debris().collect();
        let metric = Metric::new(&problem, &schedule, config);
        Self {
            problem,
            schedule,
            config,
            rng: SmallRng::seed_from_u64(config.seed),
            debris,
            metric,
            population: Population::new(config),
            neighbors: Vec::new(),
            start_time: Instant::now(),
            iteration: 0,
            stalled: 0,
            trace: Trace::new(),
        }
    }

    // ------------------------------ Helpers ----------------------------- //

    /// Schedule a chaser over `bucket`, starting from the departure orbit.
    pub fn plan(&self, bucket: &[usize]) -> Rc<ChaserPlan> {
        let mut sequence = Vec::with_capacity(bucket.len() + 1);
        sequence.push(DEPOT);
        sequence.extend_from_slice(bucket);
        self.schedule.evaluate(&sequence)
    }

    /// Penalized cost of a chaser plan.
    pub fn value(&self, plan: &ChaserPlan) -> f64 {
        self.metric.plan_value(&self.problem, plan)
    }

    /// The cheapest the transfer from `i` to `j` can ever be.
    pub fn proximity(&self, i: usize, j: usize) -> f64 {
        self.schedule.evaluate_lb(&[i, j])
    }

    /// Turn debris buckets into a mission plan, one bucket per chaser.
    pub fn assemble(&self, buckets: &[Vec<usize>]) -> Individual {
        assert!(
            buckets.len() <= self.problem.num_chasers,
            "{} buckets for {} chasers",
            buckets.len(),
            self.problem.num_chasers
        );
        let empty: Vec<usize> = Vec::new();
        let plans = (0..self.problem.num_chasers)
            .map(|c| self.plan(buckets.get(c).unwrap_or(&empty)))
            .collect();
        Individual::new(&self.problem, plans)
    }

    /// Improve an individual by local search, until no move helps.
    pub fn improve(&mut self, individual: &mut Individual) {
        Search::new(self, individual).run();
    }

    /// Insert an individual into the population, retuning the penalties.
    pub fn insert(&mut self, individual: Individual) {
        let old = self.best_value();
        self.population.add(&self.debris, &mut self.metric, individual);
        let new = self.best_value();
        if new < old { self.stalled = 0; } else { self.stalled += 1; }
        self.iteration += 1;
    }

    /// The cheapest individual held, feasible if any is.
    pub fn best(&self) -> Option<&Individual> {
        self.population.best(&self.metric)
    }

    /// Penalized cost of the incumbent, infinite while none exists.
    pub fn best_value(&self) -> f64 {
        self.best()
            .map(|individual| self.metric.value(individual))
            .unwrap_or(f64::INFINITY)
    }

    /// Whether the run should stop
    pub fn terminate(&self) -> bool {
        self.iteration >= self.config.max_iter
            || self.stalled >= self.config.max_nimp
            || self.start_time.elapsed().as_secs_f64() >= self.config.max_time
    }

    // ---------------------------- Generation ---------------------------- //

    /// Build one initial individual, by whichever strategy is configured.
    pub fn generate(&mut self) -> Individual {
        match self.config.generation {
            Generation::Random => self.generate_random(),
            Generation::Greedy => self.generate_greedy(),
        }
    }

    /// Grow each chaser plan by repeatedly appending the cheapest debris.
    pub fn generate_greedy(&mut self) -> Individual {
        let mut available = vec![false; self.problem.states.len()];
        for &d in &self.debris {
            available[d] = true;
        }

        // Debris ordered by how cheaply the swarm reaches them, then shuffled
        // within a short window so repeated calls seed different plans.
        let mut ordered = self.debris.clone();
        ordered.sort_by(|&a, &b| {
            self.proximity(DEPOT, a)
                .total_cmp(&self.proximity(DEPOT, b))
        });
        self.window_shuffle(&mut ordered, 4);

        let mut buckets: Vec<Vec<usize>> = Vec::new();
        for start in ordered {
            if !available[start] || buckets.len() >= self.problem.num_chasers {
                continue;
            }
            available[start] = false;
            let mut bucket = vec![start];
            while let Some(next) = self.cheapest_extension(&bucket, &available) {
                available[next] = false;
                bucket.push(next);
            }
            buckets.push(bucket);
        }

        let leftovers: Vec<usize> = self
            .debris
            .iter()
            .copied()
            .filter(|&d| available[d])
            .collect();
        Self::place_leftovers(&mut buckets, &leftovers);
        self.assemble(&buckets)
    }

    /// Fill each chaser plan with random debris while it stays feasible.
    pub fn generate_random(&mut self) -> Individual {
        let mut remaining = self.debris.clone();
        remaining.shuffle(&mut self.rng);

        let mut buckets: Vec<Vec<usize>> = Vec::new();
        while !remaining.is_empty() && buckets.len() < self.problem.num_chasers {
            let mut bucket: Vec<usize> = Vec::new();
            let mut rejected: Vec<usize> = Vec::new();
            while let Some(candidate) = remaining.pop() {
                bucket.push(candidate);
                if !self.admissible(&bucket) {
                    bucket.pop();
                    rejected.push(candidate);
                }
            }
            remaining = rejected;
            if bucket.is_empty() {
                break; // nothing fits anywhere; stop rather than spin
            }
            buckets.push(bucket);
        }

        Self::place_leftovers(&mut buckets, &remaining);
        self.assemble(&buckets)
    }

    /// Perturb an ordering locally, keeping its overall structure.
    fn window_shuffle(&mut self, order: &mut [usize], width: usize) {
        let n = order.len();
        for i in 0..n.saturating_sub(1) {
            let end = (i + width).min(n - 1);
            let j = i + self.rng.random_range(0..=end - i);
            order.swap(i, j);
        }
    }

    /// Whether a chaser can fly this bucket within the operational limits.
    fn admissible(&self, bucket: &[usize]) -> bool {
        bucket.len() <= self.problem.max_load
            && self.problem.is_plan_feasible(&self.plan(bucket))
    }

    /// The cheapest debris that can still be appended to `bucket`.
    fn cheapest_extension(&self, bucket: &[usize], available: &[bool]) -> Option<usize> {
        if bucket.len() >= self.problem.max_load {
            return None;
        }
        let current = *bucket.last()?;
        let mut candidates: Vec<usize> = self
            .debris
            .iter()
            .copied()
            .filter(|&d| available[d])
            .collect();
        candidates.sort_by(|&a, &b| {
            self.proximity(current, a)
                .total_cmp(&self.proximity(current, b))
        });

        let mut trial = bucket.to_vec();
        for candidate in candidates {
            trial.push(candidate);
            if self.admissible(&trial) {
                return Some(candidate);
            }
            trial.pop();
        }
        None
    }

    /// Append the debris no plan could take, so that none is ever dropped.
    fn place_leftovers(buckets: &mut Vec<Vec<usize>>, leftovers: &[usize]) {
        if leftovers.is_empty() {
            return;
        }
        let mut sorted = leftovers.to_vec();
        sorted.sort_unstable();
        if buckets.is_empty() {
            buckets.push(Vec::new());
        }
        buckets.last_mut().expect("just ensured non-empty").extend(sorted);
    }

    // ---------------------------- Crossover ----------------------------- //

    /// Concatenate the chaser plans into a single permutation of the debris.
    pub fn giant_tour(&self, individual: &Individual) -> Vec<usize> {
        let active: Vec<Rc<ChaserPlan>> = individual
            .plans
            .iter()
            .filter(|p| p.load() > 0)
            .cloned()
            .collect();
        let ordered = match self.config.ordering {
            Ordering::Raan => self.order_by_raan(active),
            Ordering::Nearest => self.order_by_nearest(active),
            Ordering::Index => active,
        };
        ordered
            .iter()
            .flat_map(|p| p.sequence[1..].iter().copied())
            .collect()
    }

    /// Order the plans by the circular mean RAAN of the debris they visit.
    fn order_by_raan(&self, mut plans: Vec<Rc<ChaserPlan>>) -> Vec<Rc<ChaserPlan>> {
        plans.sort_by(|a, b| self.mean_raan(a).total_cmp(&self.mean_raan(b)));
        plans
    }

    /// Circular mean of the RAAN of a plan's debris, at their visiting times.
    fn mean_raan(&self, plan: &ChaserPlan) -> f64 {
        let states = &self.problem.states;
        let (mut sin, mut cos, mut any) = (0.0, 0.0, false);
        for (&d, &t) in plan.sequence[1..].iter().zip(&plan.depart[1..]) {
            let angle = states[d].r + states[d].dr_ * t;
            sin += angle.sin();
            cos += angle.cos();
            any = true;
        }
        if !any {
            return 0.0;
        }
        let mean = sin.atan2(cos);
        if mean < 0.0 {
            mean + TAU
        } else {
            mean
        }
    }

    /// Chain the plans greedily, each following the one it is cheapest to reach.
    fn order_by_nearest(&self, plans: Vec<Rc<ChaserPlan>>) -> Vec<Rc<ChaserPlan>> {
        if plans.len() <= 1 {
            return plans;
        }
        let head_of = |i: usize| plans[i].sequence[1];
        let mut pending: Vec<usize> = (0..plans.len()).collect();

        let first = *pending
            .iter()
            .min_by(|&&a, &&b| {
                self.proximity(DEPOT, head_of(a))
                    .total_cmp(&self.proximity(DEPOT, head_of(b)))
            })
            .expect("pending is non-empty");
        pending.retain(|&i| i != first);

        let mut order = vec![first];
        while !pending.is_empty() {
            let tail = *plans[*order.last().expect("non-empty")]
                .sequence
                .last()
                .expect("a plan always visits the depot");
            let next = *pending
                .iter()
                .min_by(|&&a, &&b| {
                    self.proximity(tail, head_of(a))
                        .total_cmp(&self.proximity(tail, head_of(b)))
                })
                .expect("pending is non-empty");
            pending.retain(|&i| i != next);
            order.push(next);
        }
        order.into_iter().map(|i| Rc::clone(&plans[i])).collect()
    }

    /// Recombine two parents drawn from the population into a child.
    pub fn crossover(&mut self) -> Individual {
        let (first, second) = match (
            self.population.pick(&mut self.rng),
            self.population.pick(&mut self.rng),
        ) {
            (Some(a), Some(b)) => (a, b),
            _ => return self.generate(),
        };

        if self.config.crossover == Crossover::Srex {
            let buckets = self.srex(&first, &second);
            return self.assemble(&buckets);
        }

        let tour_a = self.giant_tour(&first);
        let tour_b = self.giant_tour(&second);
        if tour_a.is_empty() {
            let buckets = self.srex(&first, &second);
            return self.assemble(&buckets);
        }

        let child = match self.config.crossover {
            Crossover::Erx => self.erx(&tour_a, &tour_b),
            _ => self.ox(&tour_a, &tour_b),
        };

        // Occasionally allow one more chaser than the parent used, so that the
        // number of active chasers can grow as well as shrink.
        let extra = usize::from(self.rng.random::<f64>() < 0.1);
        let goal = (first.active.max(1) + extra).min(self.problem.num_chasers);
        let buckets = self.split(&child, goal);
        self.assemble(&buckets)
    }

    /// Order crossover: keep a block of the first parent, fill from the second.
    pub fn ox(&mut self, first: &[usize], second: &[usize]) -> Vec<usize> {
        let n = first.len();
        if n < 2 {
            return first.to_vec();
        }
        let start = self.rng.random_range(0..n);
        let mut end = self.rng.random_range(0..n);
        while end == start {
            end = self.rng.random_range(0..n);
        }

        let mut child = vec![usize::MAX; n];
        let mut taken = vec![false; self.problem.states.len()];
        let mut i = start;
        loop {
            child[i] = first[i];
            taken[first[i]] = true;
            if i == end {
                break;
            }
            i = (i + 1) % n;
        }

        // The second parent is read from a fixed offset while the write cursor
        // advances independently, so every position the block left free is
        // filled exactly once. Coupling the two skips positions and returns a
        // child shorter than its parents.
        let stop = (end + 1) % n;
        let mut pos = stop;
        for offset in 0..n {
            let debris = second[(stop + offset) % n];
            if !taken[debris] {
                child[pos] = debris;
                taken[debris] = true;
                pos = (pos + 1) % n;
            }
        }
        child
    }

    /// Edge recombination: prefer successions both parents agree on.
    pub fn erx(&mut self, first: &[usize], second: &[usize]) -> Vec<usize> {
        let n = first.len();
        if n < 2 {
            return first.to_vec();
        }
        // Sorted sets, so the tie-break depends on the seed alone and never on
        // an iteration order.
        let mut adjacency: Vec<BTreeSet<usize>> =
            vec![BTreeSet::new(); self.problem.states.len()];
        for tour in [first, second] {
            for pair in tour.windows(2) {
                adjacency[pair[0]].insert(pair[1]);
                adjacency[pair[1]].insert(pair[0]);
            }
        }

        let mut child: Vec<usize> = Vec::with_capacity(n);
        let mut visited = vec![false; self.problem.states.len()];
        let mut current = first[0];

        while child.len() < n {
            child.push(current);
            visited[current] = true;
            let peers: Vec<usize> = adjacency[current].iter().copied().collect();
            for peer in &peers {
                adjacency[*peer].remove(&current);
            }
            let live: Vec<usize> = peers.into_iter().filter(|&d| !visited[d]).collect();
            adjacency[current].clear();
            if child.len() == n {
                break;
            }
            current = if !live.is_empty() {
                let fewest = live
                    .iter()
                    .map(|&d| adjacency[d].len())
                    .min()
                    .expect("live is non-empty");
                let tied: Vec<usize> = live
                    .into_iter()
                    .filter(|&d| adjacency[d].len() == fewest)
                    .collect();
                *tied.choose(&mut self.rng).expect("tied is non-empty")
            } else {
                let free: Vec<usize> = first.iter().copied().filter(|&d| !visited[d]).collect();
                match free.choose(&mut self.rng) {
                    Some(&d) => d,
                    None => break,
                }
            };
        }
        child
    }

    /// Replace a block of the first parent's plans by plans of the second.
    pub fn srex(&mut self, first: &Individual, second: &Individual) -> Vec<Vec<usize>> {
        let active = |ind: &Individual| -> Vec<Vec<usize>> {
            ind.plans
                .iter()
                .filter(|p| p.load() > 0)
                .map(|p| p.sequence[1..].to_vec())
                .collect()
        };
        let left = active(first);
        let right = active(second);
        if left.is_empty() {
            return right;
        }
        if right.is_empty() {
            return left;
        }

        let count = 1 + self.rng.random_range(0..left.len().min(right.len()));
        let start_left = self.rng.random_range(0..left.len());
        let start_right = self.rng.random_range(0..right.len());

        let n_states = self.problem.states.len();
        let mut removed = vec![false; left.len()];
        let mut injected = vec![false; n_states];
        let mut buckets: Vec<Vec<usize>> = Vec::new();

        for t in 0..count {
            removed[(start_left + t) % left.len()] = true;
            let donor = &right[(start_right + t) % right.len()];
            for &d in donor {
                injected[d] = true;
            }
            buckets.push(donor.clone());
        }

        let mut served = injected.clone();
        for (i, bucket) in left.iter().enumerate() {
            if removed[i] {
                continue;
            }
            let kept: Vec<usize> = bucket.iter().copied().filter(|&d| !injected[d]).collect();
            for &d in &kept {
                served[d] = true;
            }
            buckets.push(kept);
        }

        let mut orphans: Vec<usize> = Vec::new();
        for i in 0..left.len() {
            if !removed[i] {
                continue;
            }
            for &d in &left[i] {
                if !served[d] {
                    served[d] = true;
                    orphans.push(d);
                }
            }
        }
        while buckets.len() < self.problem.num_chasers {
            buckets.push(Vec::new());
        }
        for debris in orphans {
            self.cheapest_insert(&mut buckets, debris);
        }
        buckets
    }

    /// Split a giant tour into individual chaser plans.
    pub fn split(&self, tour: &[usize], goal: usize) -> Vec<Vec<usize>> {
        let n = tour.len();
        if n == 0 {
            return Vec::new();
        }
        let k = goal.max(1);
        let factor = self.config.threshold_factor;

        let mut memo = vec![vec![f64::INFINITY; n + 1]; k + 1];
        let mut pred = vec![vec![0usize; n + 1]; k + 1];
        memo[0][0] = 0.0;

        for s in 1..=k {
            for i in (s - 1)..n {
                let base = memo[s - 1][i];
                if !base.is_finite() {
                    continue;
                }
                for j in i..n {
                    let plan = self.plan(&tour[i..=j]);
                    let candidate = base + self.value(&plan);
                    if candidate < memo[s][j + 1] {
                        memo[s][j + 1] = candidate;
                        pred[s][j + 1] = i;
                    }
                    if plan.load() as f64 > factor * self.problem.max_load as f64
                        || plan.time() > factor * self.problem.max_time
                    {
                        break;
                    }
                }
            }
        }

        let best_k = (1..=k)
            .min_by(|&a, &b| memo[a][n].total_cmp(&memo[b][n]))
            .expect("k is at least one");
        if !memo[best_k][n].is_finite() {
            return vec![tour.to_vec()];
        }

        let mut bounds = Vec::with_capacity(best_k);
        let mut j = n;
        for s in (1..=best_k).rev() {
            let i = pred[s][j];
            bounds.push((i, j));
            j = i;
        }
        bounds.reverse();
        bounds.into_iter().map(|(a, b)| tour[a..b].to_vec()).collect()
    }

    /// Place a debris where it costs the least, over all plans and positions.
    fn cheapest_insert(&self, buckets: &mut [Vec<usize>], debris: usize) {
        let mut best: Option<(usize, usize)> = None;
        let mut best_delta = f64::INFINITY;

        for (b, bucket) in buckets.iter().enumerate() {
            let base = self.value(&self.plan(bucket));
            let mut candidate = Vec::with_capacity(bucket.len() + 1);
            for position in 0..=bucket.len() {
                candidate.clear();
                candidate.extend_from_slice(&bucket[..position]);
                candidate.push(debris);
                candidate.extend_from_slice(&bucket[position..]);
                let delta = self.value(&self.plan(&candidate)) - base;
                if delta < best_delta {
                    best = Some((b, position));
                    best_delta = delta;
                }
            }
        }
        match best {
            Some((b, position)) => buckets[b].insert(position, debris),
            None => buckets[0].push(debris),
        }
    }

    // --------------------------- Neighborhood --------------------------- //

    /// Rank the potential successors of every debris by proximity.
    pub fn build_neighbors(&mut self) {
        let size = self.config.nb_neighbors;
        let mut neighbors = vec![Vec::new(); self.problem.states.len()];
        for &i in &self.debris {
            let mut others: Vec<usize> =
                self.debris.iter().copied().filter(|&j| j != i).collect();
            others.sort_by(|&a, &b| {
                self.proximity(i, a).total_cmp(&self.proximity(i, b))
            });
            others.truncate(size);
            neighbors[i] = others;
        }
        self.neighbors = neighbors;
    }

}

// ========================================================================= //
// Hybrid genetic search layer
// ========================================================================= //

/// Sequence layer driving the hybrid genetic search.
pub struct HgsSequence<S: ScheduleLayer> {
    /// Scheduling layer (handed to context after initialization).
    schedule: Option<S>,
    /// Parameters of the search.
    pub config: HgsConfig,

    /// Context manager holding all pieces of the algorithm.
    context: Option<Context<S>>,
}

impl<S: ScheduleLayer> HgsSequence<S> {
    /// Build the layer over a scheduling layer, with the default parameters.
    pub fn new(schedule: S) -> Self {
        Self::with_config(schedule, HgsConfig::default())
    }

    /// Build the layer over a scheduling layer, with explicit parameters.
    pub fn with_config(schedule: S, config: HgsConfig) -> Self {
        Self {
            schedule: Some(schedule),
            config,
            context: None,
        }
    }

    /// Build the layer over a scheduling layer, with preset.
    pub fn with_preset(schedule: S, preset: usize) -> Self {
        Self::with_config(schedule, HgsConfig::preset(preset))
    }

    /// Get the trace of the last run, if any.
    pub fn get_trace(&self) -> Option<&Trace> {
        self.context.as_ref().map(|c| &c.trace)
    }

    // ------------------------------ Helpers ----------------------------- //

    /// The scheduling layer underneath.
    pub fn schedule(&self) -> &S {
        match (&self.context, &self.schedule) {
            (Some(context), _) => &context.schedule,
            (None, Some(schedule)) => schedule,
            _ => unreachable!("the scheduling layer is always held somewhere"),
        }
    }

    /// The context holder, available once the layer has been initialized.
    pub fn context(&self) -> Option<&Context<S>> {
        self.context.as_ref()
    }

    /// Reclaim the scheduling layer from wherever it currently lives.
    fn take_schedule(&mut self) -> S {
        if let Some(schedule) = self.schedule.take() {
            return schedule;
        }
        self.context
            .take()
            .expect("the scheduling layer is always held somewhere")
            .schedule
    }

    /// Turn the best individual into a mission plan, one entry per chaser.
    fn finalize(context: &Context<S>) -> MissionPlan {
        let plans = context
            .best()
            .map(|individual| individual.plans.clone())
            .expect("the search ended with an empty population");

        let mut out: Vec<ChaserPlan> = plans.iter().map(|p| (**p).clone()).collect();
        while out.len() < context.problem.num_chasers {
            out.push(ChaserPlan::empty());
        }
        MissionPlan::new(out)
    }

    // ------------------------------ Logging ----------------------------- //

    /// Log the initialization start.
    fn log_init(_context: &Context<S>) {
        println!("Constructing initial population...");
        println!(
            " {:>7} {:>7} {:>7} {:>13} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "time", "iter", "nimp", "cost", "pop-f", "pop-i", "pop-r", "pen-l", "pen-t"
        );
    }

    /// Log the initialization end and the start of the search.
    fn log_head(_context: &Context<S>) {
        println!("Starting genetic search...");
    }

    /// Whether a log must be triggered
    fn log_trigger(context: &Context<S>) -> bool {
        context.config.verbose && 
        context.iteration % context.config.log_iter.max(1) == 0
    }

    /// Log a row of the search progress table.
    fn log_iter(context: &mut Context<S>) {
        if !Self::log_trigger(context) { return; }

        let best = context.best();
        let cost = match best {
            Some(individual) => format!("{:.1}", individual.cost),
            None => "--".to_string(),
        };
        println!(
            " {:7.1} {:7} {:>7} {:>13} {:>7} {:>7} {:>7.2} {:>7.2e} {:>7.2e}",
            context.start_time.elapsed().as_secs_f64(),
            context.iteration,
            context.stalled,
            cost,
            context.population.feasible.len(),
            context.population.infeasible.len(),
            context.population.feasible.len() as f64 / context.population.len() as f64,
            context.metric.penalties[0],
            context.metric.penalties[1]
        );

        context.trace.push(
            LogEntry {
                time: context.start_time.elapsed().as_secs_f64(),
                iter: context.iteration,
                nimp: context.stalled,
                cost: best.map(|i| i.cost).unwrap_or(f64::INFINITY),
                num_feasible: context.population.feasible.len(),
                num_infeasible: context.population.infeasible.len(),
                penalty_load: context.metric.penalties[0],
                penalty_time: context.metric.penalties[1],
            }
        );
    }

    /// Log the search end.
    fn log_foot(context: &Context<S>) {
        if context.stalled >= context.config.max_nimp {
            println!("Number of iterations without improvement reached");
        }
        if context.iteration >= context.config.max_iter {
            println!("Maximum number of iterations reached");
        }
        if context.start_time.elapsed().as_secs_f64() >= context.config.max_time {
            println!("Maximum time limit reached");
        }
    }
}

impl<S: ScheduleLayer> SequenceLayer for HgsSequence<S> {
    fn initialize(&mut self, problem: &Rc<Problem>) {
        let mut schedule = self.take_schedule();
        schedule.initialize(problem);

        let mut context = Context::new(Rc::clone(problem), schedule, self.config);
        context.build_neighbors();
        self.context = Some(context);
    }

    /// Evaluate the search.
    fn evaluate(&mut self) -> MissionPlan {

        // Working variables
        let context = self.context
            .as_mut()
            .expect("initialize() always installs a context");
        let verbose = context.config.verbose;

        // Refresh context
        context.start_time = Instant::now();
        context.trace.clear();
        
        if verbose { Self::log_init(context); }

        // Population initialization
        for _ in 0..context.config.pop_init {
            let mut child = context.generate();
            context.improve(&mut child);
            context.insert(child);
            if verbose { Self::log_iter(context); }
        }

        if verbose { Self::log_head(context); }

        // Refresh iteration count
        context.iteration = 0;
        context.stalled = 0;

        // Genetic selection
        while !context.terminate() {
            let mut child = context.crossover();
            context.improve(&mut child);
            context.insert(child);
            if verbose { Self::log_iter(context); }
        }

        if verbose {
            Self::log_iter(context);
            Self::log_foot(context);
        }

        Self::finalize(context)
    }
}
