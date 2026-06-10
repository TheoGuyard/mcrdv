use std::collections::VecDeque;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

use crate::inner::InnerLoop;
use crate::middle::{MiddleBound, MiddleLoop, MiddleOutput};
use crate::outer::{Trace, OuterLoop, OuterOutput};
use crate::problem::Problem;

// =============================== Parameters ============================== //

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMethod {
    /// Randomly assign debris to chaser routes
    Random,
    /// Nearest-neighbor greedy assignment
    Greedy,
}

#[derive(Debug, Clone)]
pub struct HgsParams {
    // --- Termination ---
    /// Max number of iterations
    pub max_iter: usize,
    /// Max number of iterations without improving the best individual
    pub max_nimp: usize,
    /// Time limit [s]
    pub time_limit: f64,
    /// Iterations between two log display
    pub log_iter: usize,
    /// Random seed
    pub seed: u64,

    // --- Generator ---
    /// Initial population construction method
    pub generation: GenerationMethod,
    /// Initial population size
    pub pop_init: usize,

    // --- Population ---
    /// Minimum subpopulation size after survivor selection
    pub pop_min: usize,
    /// Maximum subpopulation size triggering survivor selection
    pub pop_max: usize,
    /// Number of nearest individuals used in the diversity measure
    pub nb_close: usize,
    /// Number of elite individuals ranked on cost only
    pub nb_elite: usize,
    /// Iterations between two penalty adaptations
    pub adapt_iter: usize,
    /// Broken-pair distance threshold below which two individuals are clones
    pub clone_eps: f64,

    // --- Crossover ---
    /// Number of candidates per parent selection tournament
    pub nb_pick: usize,
    /// Relaxation factor on the load limit for split pruning
    pub load_threshold_factor: f64,
    /// Relaxation factor on the time limit for split pruning
    pub time_threshold_factor: f64,
    /// Relaxation factor on the fuel limit for split pruning
    pub fuel_threshold_factor: f64,
    /// Whether to use cheap middle-loop lower bounds
    pub use_bound_pruning: bool,
    /// Whether to order chaser plans by average RAAN during crossover
    pub raan_ordering: bool,

    // --- Search ---
    /// Number of neighbors considered in local-search moves
    pub nb_neighbors: usize,
    /// Number of samples in the orbital-distance proxy evaluation
    pub neighbor_samples: usize,

    // --- Metric ---
    /// Initial load penalty for infeasibility
    pub penalty_load_init: f64,
    /// Multiplicative increase of load penalty
    pub penalty_load_increase: f64,
    /// Multiplicative decrease of load penalty
    pub penalty_load_decrease: f64,
    /// Minimum load penalty
    pub penalty_load_min: f64,
    /// Maximum load penalty
    pub penalty_load_max: f64,
    /// Initial time penalty for infeasibility
    pub penalty_time_init: f64,
    /// Multiplicative increase of time penalty
    pub penalty_time_increase: f64,
    /// Multiplicative decrease of time penalty
    pub penalty_time_decrease: f64,
    /// Minimum time penalty
    pub penalty_time_min: f64,
    /// Maximum time penalty
    pub penalty_time_max: f64,
    /// Initial fuel penalty for infeasibility
    pub penalty_fuel_init: f64,
    /// Multiplicative increase of fuel penalty
    pub penalty_fuel_increase: f64,
    /// Multiplicative decrease of fuel penalty
    pub penalty_fuel_decrease: f64,
    /// Minimum fuel penalty
    pub penalty_fuel_min: f64,
    /// Maximum fuel penalty
    pub penalty_fuel_max: f64,
    /// Target fraction of feasible individuals for penalty adaptation
    pub target_ratio: f64,
}

impl HgsParams {
    pub fn base() -> Self {        
        Self {
            max_iter: 500,
            max_nimp: 100,
            time_limit: f64::INFINITY,
            log_iter: 10,
            seed: Instant::now().elapsed().as_nanos() as u64,

            generation: GenerationMethod::Greedy,
            pop_init: 10,

            pop_min: 5,
            pop_max: 10,
            nb_close: 2,
            nb_elite: 2,
            adapt_iter: 10,
            clone_eps: 1e-6,

            nb_pick: 2,
            load_threshold_factor: 1.5,
            time_threshold_factor: 1.5,
            fuel_threshold_factor: 1.5,
            use_bound_pruning: false,
            raan_ordering: true,

            nb_neighbors: 30,
            neighbor_samples: 16,

            penalty_load_init: 1.0,
            penalty_load_increase: 2.0,
            penalty_load_decrease: 0.5,
            penalty_load_min: 1e-9,
            penalty_load_max: 1e9,
            penalty_time_init: 1.0,
            penalty_time_increase: 2.0,
            penalty_time_decrease: 0.5,
            penalty_time_min: 1e-9,
            penalty_time_max: 1e9,
            penalty_fuel_init: 1.0,
            penalty_fuel_increase: 2.0,
            penalty_fuel_decrease: 0.5,
            penalty_fuel_min: 1e-9,
            penalty_fuel_max: 1e9,
            target_ratio: 0.9,
        }
    }

    /// Build parameters for a preset depth `level`. Levels above the deepest
    /// preset are clamped to it rather than rejected.
    pub fn preset(level: usize) -> Self {
        let mut base = Self::base();
        match level.min(6) {
            0 => {
                // Local search on a single initial individual
                base.max_iter = 0;
                base.max_nimp = 0;
                base.pop_init = 1;
                base.pop_min = 2;
                base.pop_max = 2;
                base.nb_close = 1;
                base.nb_elite = 1;
            }
            1 => {
                // Local search on multiple initial individuals
                base.max_iter = 0;
                base.max_nimp = 0;
                base.log_iter = 1;
                base.pop_init = 5;
                base.pop_min = 2;
                base.pop_max = 3;
                base.nb_close = 1;
                base.nb_elite = 1;
            }
            2 => {
                // Very shallow genetic selection
                base.max_iter = 50;
                base.max_nimp = 10;
                base.log_iter = 1;
                base.pop_init = 6;
                base.pop_min = 3;
                base.pop_max = 6;
                base.nb_close = 1;
                base.nb_elite = 1;
            }
            3 => {
                // Shallow genetic selection
                base.max_iter = 500;
                base.max_nimp = 100;
                base.pop_init = 10;
                base.pop_min = 5;
                base.pop_max = 10;
                base.nb_close = 2;
                base.nb_elite = 2;
            }
            4 => {
                // Medium genetic selection
                base.max_iter = 5_000;
                base.max_nimp = 500;
                base.pop_init = 20;
                base.pop_min = 10;
                base.pop_max = 20;
                base.nb_close = 2;
                base.nb_elite = 3;
            }
            5 => {
                // Deep genetic selection
                base.max_iter = 50_000;
                base.max_nimp = 5_000;
                base.pop_init = 24;
                base.pop_min = 12;
                base.pop_max = 32;
                base.nb_close = 3;
                base.nb_elite = 4;
            }
            6 => {
                // Very deep genetic selection
                base.max_iter = 200_000;
                base.max_nimp = 10_000;
                base.pop_init = 50;
                base.pop_min = 25;
                base.pop_max = 65;
                base.nb_close = 3;
                base.nb_elite = 8;
            }
            // Unreachable: `level.min(6)` keeps the match exhaustive over 0..=6.
            _ => {}
        }
        base
    }
}

impl Default for HgsParams {
    fn default() -> Self {
        Self::preset(5)
    }
}


// =============================== Individual ============================== //

#[derive(Clone)]
pub struct Individual {
    pub routes: Vec<MiddleOutput>,
    pub cost: f64,
    pub load_excess: usize,
    pub time_excess: f64,
    pub fuel_excess: f64,
    pub nb_active: usize,
    pub pred: Vec<usize>,
    pub succ: Vec<usize>,
}

impl Individual {
    pub fn new(problem: &Problem, routes: Vec<MiddleOutput>) -> Self {
        let n = problem.num_debris();
        let mut ind = Self {
            routes,
            cost: 0.0,
            load_excess: 0,
            time_excess: 0.0,
            fuel_excess: 0.0,
            nb_active: 0,
            pred: vec![n; n],
            succ: vec![n; n],
        };
        ind.build(problem);
        ind
    }

    pub fn build(&mut self, problem: &Problem) {
        let depot = problem.num_debris(); // sentinel for "home"
        self.cost = 0.0;
        self.load_excess = 0;
        self.time_excess = 0.0;
        self.fuel_excess = 0.0;
        self.nb_active = 0;
        self.pred.iter_mut().for_each(|v| *v = depot);
        self.succ.iter_mut().for_each(|v| *v = depot);

        for route in &self.routes {
            self.cost += route.cost;
            self.load_excess += route.total_load.saturating_sub(problem.max_load);
            self.time_excess += (route.mission_time() - problem.max_time).max(0.0);
            self.fuel_excess += (route.total_fuel - problem.max_fuel).max(0.0);

            if !route.is_empty() {
                self.nb_active += 1;
                let sequence = &route.sequence;
                let n = sequence.len();
                // Open route `[home, d₁…dₖ]`: every position but 0 is a debris;
                // the last debris ends the route, so its successor is the depot.
                for i in 1..n {
                    let d = sequence[i];
                    self.pred[d] = if i == 1 { depot } else { sequence[i - 1] };
                    self.succ[d] = if i == n - 1 { depot } else { sequence[i + 1] };
                }
            }
        }
    }

    #[inline]
    pub fn is_feasible(&self) -> bool {
        self.routes.iter().all(|r| r.feasible)
    }

    #[inline]
    pub fn max_time(&self) -> f64 {
        self.routes.iter().map(|r| r.mission_time()).fold(0.0, f64::max)
    }

    #[inline]
    pub fn max_fuel(&self) -> f64 {
        self.routes.iter().map(|r| r.total_fuel).fold(0.0, f64::max)
    }

    #[inline]
    pub fn max_load(&self) -> usize {
        self.routes.iter().map(|r| r.total_load).fold(0, usize::max)
    }
}

/// Evaluate a chaser route list through the middle and inner loops
fn evaluate(
    chaser: usize,
    debris: &[usize],
    inner: &dyn InnerLoop,
    middle: &dyn MiddleLoop,
    problem: &Problem,
) -> MiddleOutput {
    // Open route: depart from the shared home (the chaser node) and end the
    // mission at the last debris — no return leg.
    let mut route = Vec::with_capacity(debris.len() + 1);
    route.push(chaser);
    route.extend_from_slice(debris);
    middle.solve(&route, inner, problem)
}

/// Cheap middle-loop lower bound on a chaser route covering `debris`. Mirrors
/// `evaluate` but calls `MiddleLoop::solve_lb` instead of the costly
/// `MiddleLoop::solve`, for branch-and-bound style pruning.
fn evaluate_lb(
    chaser: usize,
    debris: &[usize],
    inner: &dyn InnerLoop,
    middle: &dyn MiddleLoop,
    problem: &Problem,
) -> MiddleBound {
    let mut route = Vec::with_capacity(debris.len() + 1);
    route.push(chaser);
    route.extend_from_slice(debris);
    middle.solve_lb(&route, inner, problem)
}

/// Assemble all chaser routes, padding unused ones with empty routes
fn assemble(
    buckets: &[Vec<usize>],
    inner: &dyn InnerLoop,
    middle: &dyn MiddleLoop,
    problem: &Problem,
) -> Individual {
    let k = problem.num_chasers;
    let mut routes: Vec<MiddleOutput> = Vec::with_capacity(k);
    for chaser in 0..k {
        let debris = buckets.get(chaser).map(Vec::as_slice).unwrap_or(&[]);
        routes.push(evaluate(chaser, debris, inner, middle, problem));
    }
    Individual::new(problem, routes)
}


// ================================= Metric ================================ //

pub struct Metric {
    params: HgsParams,
    pub penalty_load: f64,
    pub penalty_time: f64,
    pub penalty_fuel: f64,
}

impl Metric {
    pub fn new(params: &HgsParams) -> Self {
        Self {
            penalty_load: params.penalty_load_init,
            penalty_time: params.penalty_time_init,
            penalty_fuel: params.penalty_fuel_init,
            params: params.clone(),
        }
    }

    #[inline]
    pub fn route_penalty(&self, problem: &Problem, route: &MiddleOutput) -> f64 {
        let load_viol = route.total_load.saturating_sub(problem.max_load) as f64;
        let time_viol = (route.mission_time() - problem.max_time).max(0.0);
        let fuel_viol = (route.total_fuel - problem.max_fuel).max(0.0);
        self.penalty_load * load_viol + self.penalty_time * time_viol + self.penalty_fuel * fuel_viol
    }

    #[inline]
    pub fn route_value(&self, problem: &Problem, route: &MiddleOutput) -> f64 {
        route.cost + self.route_penalty(problem, route)
    }

    /// Penalized value of a route lower bound. Mirrors `route_value`/`route_penalty`
    /// but on the lower-bound quantities, so `bound_value <= route_value` for the
    /// route the bound was built from — a valid branch-and-bound gate.
    #[inline]
    pub fn bound_value(&self, problem: &Problem, bound: &MiddleBound) -> f64 {
        let load_viol = bound.total_load.saturating_sub(problem.max_load) as f64;
        let time_viol = (bound.total_time - problem.max_time).max(0.0);
        let fuel_viol = (bound.total_fuel - problem.max_fuel).max(0.0);
        bound.cost
            + self.penalty_load * load_viol
            + self.penalty_time * time_viol
            + self.penalty_fuel * fuel_viol
    }

    #[inline]
    pub fn penalty(&self, individual: &Individual) -> f64 {
        self.penalty_load * (individual.load_excess as f64)
            + self.penalty_time * individual.time_excess
            + self.penalty_fuel * individual.fuel_excess
    }

    #[inline]
    pub fn value(&self, individual: &Individual) -> f64 {
        individual.cost + self.penalty(individual)
    }

    pub fn update(
        &mut self, 
        load_feasible_ratio: f64, 
        time_feasible_ratio: f64, 
        fuel_feasible_ratio: f64
    ) {

        self.penalty_load *= if load_feasible_ratio < self.params.target_ratio {
            self.params.penalty_load_increase
        } else {
            self.params.penalty_load_decrease
        };
        self.penalty_load = self.penalty_load.clamp(
            self.params.penalty_load_min,
            self.params.penalty_load_max
        );

        self.penalty_time *= if time_feasible_ratio < self.params.target_ratio {
            self.params.penalty_time_increase
        } else {
            self.params.penalty_time_decrease
        };
        self.penalty_time = self.penalty_time.clamp(
            self.params.penalty_time_min,
            self.params.penalty_time_max
        );

        self.penalty_fuel *= if fuel_feasible_ratio < self.params.target_ratio {
            self.params.penalty_fuel_increase
        } else {
            self.params.penalty_fuel_decrease
        };
        self.penalty_fuel = self.penalty_fuel.clamp(
            self.params.penalty_fuel_min,
            self.params.penalty_fuel_max
        );
    }

    pub fn distance(a: &Individual, b: &Individual) -> f64 {
    
        debug_assert_eq!(a.pred.len(), b.pred.len());
        
        let n = a.pred.len(); // = num_debris
        if n == 0 { return 0.0; }
        let depot = n; // sentinel

        // Compute broken-pairs distance
        let mut dist = 0usize;
        for j in 0..n {
            if a.succ[j] != b.succ[j] && a.succ[j] != b.pred[j] {
                dist += 1;
            }
            if a.pred[j] == depot && b.pred[j] != depot && b.succ[j] != depot {
                dist += 1;
            }
        }

        // Return normalized distance (0.0 = identical, 1.0 = completely different)
        dist as f64 / n as f64
    }
}


// =============================== Generator =============================== //

pub struct Generator {
    params: HgsParams,
}

impl Generator {
    pub fn new(params: &HgsParams) -> Self {
        Self { params: params.clone() }
    }

    pub fn generate(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        problem: &Problem,
        rng: &mut SmallRng,
    ) -> Individual {
        match self.params.generation {
            GenerationMethod::Random => self.random(inner, middle, problem, rng),
            GenerationMethod::Greedy => self.greedy(inner, middle, problem, rng),
        }
    }

    /// Random individuals generation: fill chaser routes up to the load, time,
    /// and fuel limits with randomly selected debris, then assign any leftover
    /// debris (if any) to the last route.
    fn random(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        problem: &Problem,
        rng: &mut SmallRng,
    ) -> Individual {
        let n = problem.num_debris();
        let k = problem.num_chasers;
        let home = problem.num_debris(); // node index of the shared home

        let mut remaining: Vec<usize> = (0..n).collect();
        remaining.shuffle(rng);

        let mut buckets: Vec<Vec<usize>> = Vec::new();
        while !remaining.is_empty() && buckets.len() < k {
            let mut route: Vec<usize> = Vec::new();
            let mut prev: usize = home;
            let mut time = 0.0;
            let mut fuel = 0.0;

            let mut i = 0;
            while i < remaining.len() {
                if route.len() >= problem.max_load {
                    break;
                }

                let node = remaining[i];

                // Compute time and fuel to go from prev -> node (open route: no return)
                let leg_prev = inner.solve(prev, node, time, problem);

                // Skip the debris if it is not feasible to add it to the route
                if time + leg_prev.time > problem.max_time ||
                   fuel + leg_prev.fuel > problem.max_fuel {
                    i += 1;
                    continue;
                }

                route.push(remaining[i]);
                time += leg_prev.time;
                fuel += leg_prev.fuel;
                prev = node;
                remaining.swap_remove(i); // brings a new element to index i
            }

            buckets.push(route);
        }

        // Leftover debris (if any) are assigned to the last route
        if !remaining.is_empty() {
            if buckets.is_empty() {
                buckets.push(Vec::new());
            }
            buckets.last_mut().unwrap().append(&mut remaining);
        }

        assemble(&buckets, inner, middle, problem)
    }

    /// Greedy nearest-neighbor: seed routes from debris ordered by distance to
    /// home (with small random windows for diversity), extend each with the
    /// cheapest feasible neighbor, then assign any leftover debris (if any) to
    /// the last route.
    fn greedy(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        problem: &Problem,
        rng: &mut SmallRng,
    ) -> Individual {
        let n = problem.num_debris();
        let k = problem.num_chasers;
        let home = problem.num_debris(); // node index of the shared home

        let mut available = vec![true; n];

        // Order debris by problem cost to home at initial time
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_unstable_by(|&i, &j| {
            let leg_i = inner.solve(home, i, 0.0, problem);
            let leg_j = inner.solve(home, j, 0.0, problem);
            let cost_i = problem.cost(leg_i.time, leg_i.fuel);
            let cost_j = problem.cost(leg_j.time, leg_j.fuel);
            cost_i.total_cmp(&cost_j)
        });

        // Shuffle within small windows so seeds differ across calls
        let len = sorted.len();
        for i in 0..len.saturating_sub(1) {
            let end = (i + 4).min(len - 1);
            let j = rng.gen_range(i..=end);
            sorted.swap(i, j);
        }

        let mut buckets: Vec<Vec<usize>> = Vec::new();
        for &start in &sorted {
            
            // Pick a new debris to start a route among the available ones
            if !available[start] || buckets.len() >= k { continue; }
            available[start] = false;

            let leg_init = inner.solve(home, start, 0.0, problem);
            let mut route = vec![start];
            let mut time = leg_init.time;
            let mut fuel = leg_init.fuel;

            // Fill the route with the cheapest debris reachable
            while route.len() < problem.max_load {

                let current = *route.last().unwrap();
                let mut best_i: Option<usize> = None;
                let mut best_cost = f64::INFINITY;
                let mut best_fuel = f64::INFINITY;
                let mut best_time = f64::INFINITY;

                for &i in &sorted {
                    if !available[i] { continue; }
                    let node = i;

                    // Compute time and fuel to go from current -> node (open route: no return)
                    let leg_prev = inner.solve(current, node, time, problem);

                    // Skip the debris if it is not feasible to add it to the route
                    if time + leg_prev.time > problem.max_time ||
                        fuel + leg_prev.fuel > problem.max_fuel {
                        continue;
                    }
                    
                    // Keep track of the best candidate
                    let cost = problem.cost(leg_prev.time, leg_prev.fuel);
                    if cost < best_cost {
                        best_i = Some(i);
                        best_cost = cost;
                        best_time = leg_prev.time;
                        best_fuel = leg_prev.fuel;
                    }
                }

                match best_i {
                    Some(i) => {
                        available[i] = false;
                        time += best_time;
                        fuel += best_fuel;
                        route.push(i);
                    }
                    None => break,
                }
            }

            buckets.push(route);
        }

        // Leftover debris (if any) are assigned to the last route
        let leftover: Vec<usize> = (0..n).filter(|&d| available[d]).collect();
        if !leftover.is_empty() {
            if buckets.is_empty() {
                buckets.push(Vec::new());
            }
            buckets.last_mut().unwrap().extend(leftover);
        }

        assemble(&buckets, inner, middle, problem)
    }

}

// ============================== Population =============================== //

#[derive(Default)]
struct Subpopulation {
    /// Population individuals
    individuals: Vec<Individual>,
    /// `proxi[i]` = sorted `(distance, j)` to the other individuals
    proxi: Vec<Vec<(f64, usize)>>,
    /// `score[i]` = combined cost and diversity rank (smaller is better)
    score: Vec<f64>,
    /// Individual indices sorted by increasing penalized value
    order: Vec<usize>,
}

impl Subpopulation {
    /// Insert new individual at index `j` into proximity list and build its own
    fn update_proxi(&mut self, j: usize) {
        debug_assert_eq!(self.proxi.len(), j);
        let m = self.individuals.len();
        self.proxi.push(Vec::with_capacity(m.saturating_sub(1)));

        for i in 0..j {
            let dist = Metric::distance(&self.individuals[j], &self.individuals[i]);
            let pos = self.proxi[i].partition_point(|(dd, _)| *dd <= dist);
            self.proxi[i].insert(pos, (dist, j));
            self.proxi[j].push((dist, i));
        }
        self.proxi[j].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }

    /// Update individual order by increasing penalized value
    fn update_order(&mut self, metric: &Metric) {
        self.order.clear();
        self.order.extend(0..self.individuals.len());
        let values: Vec<f64> = self.individuals
            .iter()
            .map(|s| metric.value(s))
            .collect();
        self.order
            .sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    }

    /// Recompute individual score from order and proximity
    fn update_score(&mut self, nb_close: usize, nb_elite: usize) {
        let n = self.individuals.len();
        self.score.resize(n, 0.0);
        
        // Handle trivial cases
        if n == 0 { return; }
        if n == 1 {
            self.score[0] = 0.0;
            return;
        }

        let nb_close = nb_close.min(n - 1);
        let denom = (n - 1) as f64;

        // Average distance to the nb_close nearest neighbors
        let mut avg_dist = vec![0.0; n];
        for (i, avg) in avg_dist.iter_mut().enumerate() {
            let neighbors = &self.proxi[i];
            let kk = nb_close.min(neighbors.len());
            let sum: f64 = neighbors.iter().take(kk).map(|(d, _)| *d).sum();
            *avg = if kk > 0 { sum / kk as f64 } else { 0.0 };
        }

        // Diversity ranking (larger neighbors distance = better)
        let mut div_pairs: Vec<(f64, usize)> = (0..n).map(|i| (-avg_dist[i], i)).collect();
        div_pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut div_rank = vec![0.0; n];
        for (pos, &(_, idx)) in div_pairs.iter().enumerate() {
            div_rank[idx] = pos as f64 / denom;
        }

        // Cost ranking (lower value = better)
        let mut cost_pos = vec![0usize; n];
        for (pos, &idx) in self.order.iter().enumerate() {
            cost_pos[idx] = pos;
        }
        let ord_rank: Vec<f64> = cost_pos
            .iter()
            .map(|&p| p as f64 / denom)
            .collect();

        // Assign scores (elite individuals ranked on cost alone)
        let scale = 1.0 - (nb_elite as f64) / (n as f64);
        for i in 0..n {
            self.score[i] = if cost_pos[i] < nb_elite {
                ord_rank[i]
            } else {
                ord_rank[i] + scale * div_rank[i]
            };
        }
    }

    /// Remove the individual at index `j`, repairing proximity lists
    fn remove_at(&mut self, j: usize) {
        let last = self.individuals.len() - 1;
        self.individuals.swap_remove(j);
        self.proxi.swap_remove(j);
        self.score.swap_remove(j);
        for list in self.proxi.iter_mut() {
            list.retain(|&(_, i)| i != j);
            if last != j {
                for pair in list.iter_mut() {
                    if pair.1 == last {
                        pair.1 = j;
                    }
                }
            }
        }
    }

    /// Index of the worst individual, prioritizing clones if any
    fn worst_index(&self, clone_eps: f64) -> usize {
        let mut worst_idx = 0;
        let mut worst_is_clone = self.proxi[0].first().is_some_and(|&(d, _)| d <= clone_eps);
        let mut worst_rank = self.score[0];

        for i in 1..self.individuals.len() {
            let is_clone = self.proxi[i].first().is_some_and(|&(d, _)| d <= clone_eps);
            let score = self.score[i];
            if (is_clone && !worst_is_clone) || (is_clone == worst_is_clone && score > worst_rank) {
                worst_is_clone = is_clone;
                worst_rank = score;
                worst_idx = i;
            }
        }
        worst_idx
    }

    /// Trim down to `pop_min` individuals by repeatedly dropping worst individual
    fn survivors_selection(
        &mut self,
        metric: &Metric,
        pop_min: usize,
        nb_close: usize,
        nb_elite: usize,
        clone_eps: f64,
    ) {
        while self.individuals.len() > pop_min {
            let idx = self.worst_index(clone_eps);
            self.remove_at(idx);
            self.update_order(metric);
            self.update_score(nb_close, nb_elite);
        }
    }
}


pub struct Population {
    /// Parameters
    params: HgsParams,
    /// Feasible subpopulation
    feasible: Subpopulation,
    /// Infeasible subpopulation
    infeasible: Subpopulation,
    /// Rolling window with load feasibility of recent individuals
    load_window: VecDeque<bool>,
    /// Rolling window with time feasibility of recent individuals
    time_window: VecDeque<bool>,
    /// Rolling window with fuel feasibility of recent individuals
    fuel_window: VecDeque<bool>,
}

impl Population {
    pub fn new(params: &HgsParams) -> Self {
        Self {
            params: params.clone(),
            feasible: Subpopulation::default(),
            infeasible: Subpopulation::default(),
            load_window: VecDeque::new(),
            time_window: VecDeque::new(),
            fuel_window: VecDeque::new(),
        }
    }

    /// Insert a individual in the population
    pub fn add(&mut self, ind: Individual, metric: &mut Metric) {
        // Record per-constraint feasibility in the rolling windows.
        self.load_window.push_back(ind.load_excess == 0);
        self.time_window.push_back(ind.time_excess == 0.0);
        self.fuel_window.push_back(ind.fuel_excess == 0.0);

        // Insert individual into the appropriate subpopulation 
        let sub = if ind.is_feasible() {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };
        let idx = sub.individuals.len();
        sub.individuals.push(ind);

        // Update subpopulation
        sub.update_proxi(idx);
        sub.update_order(metric);
        sub.update_score(self.params.nb_close, self.params.nb_elite);

        // Trigger survivor selection if population exceeds maximum size
        if sub.individuals.len() > self.params.pop_max {
            sub.survivors_selection(
                metric,
                self.params.pop_min,
                self.params.nb_close,
                self.params.nb_elite,
                self.params.clone_eps,
            );
        }

        // Update penalties periodically based on recent feasibility ratios
        let adapt_iter = self.params.adapt_iter;
        if adapt_iter != 0 && self.load_window.len().is_multiple_of(adapt_iter) {
            
            // Helper to compute the feasibility fraction of recent individuals
            let frac = |w: &VecDeque<bool>| {
                w.iter()
                    .rev()
                    .take(adapt_iter)
                    .filter(|&&b| b)
                    .count() as f64 / adapt_iter as f64
            };

            // Compute feasibility fractions and update penalties accordingly
            let fl = frac(&self.load_window);
            let ft = frac(&self.time_window);
            let ff = frac(&self.fuel_window);
            metric.update(fl, ft, ff);

            // Update subpopulation after penalties change
            self.feasible.update_order(metric);
            self.feasible.update_score(self.params.nb_close, self.params.nb_elite);
            self.infeasible.update_order(metric);
            self.infeasible.update_score(self.params.nb_close, self.params.nb_elite);
        }
    }

    /// Tournament selection over `nb_pick` candidates from both subpopulations
    pub fn pick<'a>(&'a self, nb_pick: usize, rng: &mut SmallRng) -> &'a Individual {
        let fpop_n = self.feasible.individuals.len();
        let ipop_n = self.infeasible.individuals.len();
        let tpop_n = fpop_n + ipop_n;
        assert!(tpop_n >= 1, "tournament needs at least one individual");
        let k = nb_pick.max(1).min(tpop_n);

        // Helper to draw a random candidate from the combined subpopulation
        let draw = |rng: &mut SmallRng| -> (bool, usize, f64) {
            let g = rng.gen_range(0..tpop_n);
            if g < fpop_n {
                (true, g, self.feasible.score[g])
            } else {
                let i = g - fpop_n;
                (false, i, self.infeasible.score[i])
            }
        };

        // Draw `k` candidates and return the index of the one with best score
        let (mut best_f, mut best_i, mut best_s) = draw(rng);
        for _ in 1..k {
            let (f, i, s) = draw(rng);
            if s < best_s {
                best_f = f;
                best_i = i;
                best_s = s;
            }
        }

        if best_f {
            &self.feasible.individuals[best_i]
        } else {
            &self.infeasible.individuals[best_i]
        }
    }

    /// Return best feasible individual, if any
    pub fn best_feasible(&self) -> Option<&Individual> {
        self.feasible.individuals.get(*self.feasible.order.first()?)
    }

    /// Return best infeasible individual, if any
    pub fn best_infeasible(&self) -> Option<&Individual> {
        self.infeasible.individuals.get(*self.infeasible.order.first()?)
    }

    /// Return best individual, prioritizing feasible ones
    pub fn best(&self) -> &Individual {
        self.best_feasible()
            .or_else(|| self.best_infeasible())
            .expect("population is empty")
    }
}


// ============================== Crossover ================================ //

pub struct Crossover {
    params: HgsParams,
}

impl Crossover {
    pub fn new(params: &HgsParams) -> Self {
        Self { params: params.clone() }
    }

    /// Produce one offspring by crossover
    pub fn generate(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        population: &Population,
        problem: &Problem,
        rng: &mut SmallRng,
    ) -> Individual {
        // Pick two parent individuals through tournament selection
        let p1 = population.pick(self.params.nb_pick, rng).clone();
        let p2 = population.pick(self.params.nb_pick, rng).clone();

        // Build giant tours by flattening parent individuals
        let t1 = self.giant_tour(problem, &p1);
        let t2 = self.giant_tour(problem, &p2);

        // Occasionally allow one more route to encourage diversity
        let extra = if rng.gen_bool(0.1) { 1 } else { 0 };
        let k_goal = (p1.nb_active.max(1) + extra).min(problem.num_chasers);

        // Crossover on giant tours
        let tour = self.crossover_ox(&t1, &t2, rng);

        // Split the offspring giant tour
        let buckets = self.split(inner, middle, metric, problem, &tour, k_goal);
        
        assemble(&buckets, inner, middle, problem)
    }

    /// Flatten a individual into a since debris sequence, concatenating routes
    /// by average RAAN and keeping their individual visit order
    fn giant_tour(&self, problem: &Problem, ind: &Individual) -> Vec<usize> {
        let mut keys: Vec<(f64, usize)> = Vec::new();
        
        // Sort routes by average RAAN of visited debris. With `raan_ordering`
        // disabled (ablation), keep the routes in their solution order so the
        // giant tour carries no orbital structure.
        for (i, route) in ind.routes.iter().enumerate() {
            if route.is_empty() { continue; }
            let key = if self.params.raan_ordering {
                let debris = &route.sequence[1..];
                debris.iter().map(|&d| problem.debris[d].r).sum::<f64>() / debris.len() as f64
            } else {
                i as f64
            };
            keys.push((key, i));
        }
        keys.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Concatenate debris in route order
        let mut tour = Vec::new();
        for &(_, idx) in &keys {
            let route = &ind.routes[idx].sequence;
            tour.extend_from_slice(&route[1..]);
        }

        tour
    }

    /// Order crossover (OX) of two equal-length giant tours.
    fn crossover_ox(&self, parent1: &[usize], parent2: &[usize], rng: &mut SmallRng) -> Vec<usize> {

        let n = parent1.len();
        debug_assert!(n > 1);
        debug_assert_eq!(n, parent2.len());

        let mut child = vec![0usize; n];
        let max_index = *parent1.iter().max().unwrap_or(&0);
        let mut added = vec![false; max_index + 1];

        // Select a non-empty segment [start, end]
        let start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        while end == start {
            end = rng.gen_range(0..n);
        }

        // Copy the segment [start, end] from parent1
        let stop = (end + 1) % n;
        let mut j = start;
        loop {
            let pos = j % n;
            let idx = parent1[pos];
            child[pos] = idx;
            added[idx] = true;
            j += 1;
            if j % n == stop {
                break;
            }
        }

        // Fill the rest from parent2 in order, skipping already-used debris
        let mut pos = stop;
        for t in 0..n {
            let idx = parent2[(stop + t) % n];
            if !added[idx] {
                child[pos] = idx;
                added[idx] = true;
                pos = (pos + 1) % n;
            }
        }
        child
    }

    /// Split a giant tour into at most `k_goal` routes, minimizing the total
    /// penalized cost. Each candidate segment is scored through the
    /// middle loop. Returns the debris list per segment.
    fn split(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        giant: &[usize],
        k_goal: usize,
    ) -> Vec<Vec<usize>> {
        let n = giant.len();
        let k = k_goal.max(1);
       
        // Handle trivial case of empty giant tour
        if n == 0 { return Vec::new(); }

        // Slack-bounded infinity to keep additions from overflowing
        let inf = f64::INFINITY / 4.0;

        // memo[s][j] = best penalized cost to cover j debris with s routes
        // pred[s][j] = the split point achieving it
        let mut memo = vec![vec![inf; n + 1]; k + 1];
        let mut pred = vec![vec![0usize; n + 1]; k + 1];
        memo[0][0] = 0.0;

        // Relaxed thresholds to prune over-long candidate segments
        let load_threshold = self.params.load_threshold_factor * problem.max_load as f64;
        let time_threshold = self.params.time_threshold_factor * problem.max_time;
        let fuel_threshold = self.params.fuel_threshold_factor * problem.max_fuel;

        // Fill the memoization table
        for s in 1..=k {
            for i in (s - 1)..n {
                let base = memo[s - 1][i];

                // Skip if the previous cost is already infinite
                if base >= inf { continue; }

                for j in i..n {

                    // Lower-bound gate: skip the costly middle-loop solve when
                    // even the cheap middle-loop lower bound cannot improve this
                    // cell.
                    if self.params.use_bound_pruning {
                        let bound = evaluate_lb(0, &giant[i..=j], inner, middle, problem);
                        if base + metric.bound_value(problem, &bound) >= memo[s][j + 1] {
                            // Cannot improve this cell. Stop extending once the
                            // exact load alone exceeds the relaxed threshold.
                            if bound.total_load as f64 > load_threshold { break; }
                            continue;
                        }
                    }

                    // Candidate route covering giant[i..=j]
                    let route = evaluate(0, &giant[i..=j], inner, middle, problem);
                    let c = base + metric.route_value(problem, &route);
                    if c < memo[s][j + 1] {
                        memo[s][j + 1] = c;
                        pred[s][j + 1] = i;
                    }

                    // Check if the route can be pruned from relaxed thresholds
                    if route.total_load as f64 > load_threshold
                        || route.mission_time() > time_threshold
                        || route.total_fuel > fuel_threshold
                    {
                        break;
                    }
                }
            }
        }

        // Choose the cheapest number of routes to split
        let mut best_k = k;
        let mut best_c = memo[k][n];
         for (s, memo_s) in memo.iter().enumerate().take(k).skip(1) {
            if memo_s[n] < best_c {
                best_c = memo_s[n];
                best_k = s;
            }
        }

        // Fallback with all debris in one route if no feasible split found
        if best_c >= inf { return vec![giant.to_vec()]; }

        // Backtrack to build debris list per route
        let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(best_k);
        let mut j = n;
        for s in (1..=best_k).rev() {
            let i = pred[s][j];
            bounds.push((i, j));
            j = i;
        }
        bounds.reverse();

        bounds
            .into_iter()
            .map(|(start, end)| giant[start..end].to_vec())
            .collect()
    }
}


// ================================ Search ================================= //


pub struct Search {
    params: HgsParams,
    /// Nearest-neighbor lists for each debris computed statically
    neighbors: Vec<Vec<usize>>,
}

impl Search {
    pub fn new(params: &HgsParams) -> Self {
        Self { params: params.clone(), neighbors: Vec::new() }
    }

    /// Precompute the nearest-neighbor lists from the orbital-distance proxy.
    pub fn initialize(&mut self, inner: &dyn InnerLoop, problem: &Problem) {
        let n = problem.num_debris();
        self.neighbors = vec![Vec::new(); n];
        for i in 0..n {
            let mut proxi: Vec<(f64, usize)> = Vec::with_capacity(n.saturating_sub(1));
            for j in 0..n {
                if j == i {
                    continue;
                }
                proxi.push((self.orbital_distance(i, j, inner, problem), j));
            }
            proxi.sort_by(|a, b| a.0.total_cmp(&b.0));
            let keep = self.params.nb_neighbors.min(proxi.len());
            self.neighbors[i] = proxi[..keep].iter().map(|&(_, j)| j).collect();
        }
    }

    /// Orbital distance proxy
    fn orbital_distance(
        &self,
        i: usize,
        j: usize,
        inner: &dyn InnerLoop,
        problem: &Problem,
    ) -> f64 {
        let m = self.params.neighbor_samples.max(1);
        let horizon = problem.max_time;
        let mut acc = 0.0;
        for s in 0..m {
            let t = horizon * (s as f64 + 0.5) / m as f64; // midpoint of s-th cell
            let leg = inner.solve(i, j, t, problem);
            acc += problem.cost(leg.time, leg.fuel);
        }
        acc / m as f64
    }

    /// Run local search on a individual
    pub fn improve(
        &self,
        ind: &mut Individual,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        rng: &mut SmallRng,
    ) {
        let n = problem.num_debris();
        let k = ind.routes.len();

        let mut state_seq = vec![0usize; n]; // debris -> route index
        let mut state_pos = vec![0usize; n]; // debris -> position index in the route
        let mut when_last_tested = vec![0usize; n];
        let mut when_last_modified = vec![0usize; k];
        let mut num_moves: usize = 1;

        Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);

        let mut improved = true;
        let mut loop_num = 0usize;
        while improved {
            improved = false;
            loop_num += 1;

            let mut clients: Vec<usize> = (0..n).collect();
            clients.shuffle(rng);

            for &u in &clients {
                let last_tested = when_last_tested[u];
                when_last_tested[u] = num_moves;
                let su = state_seq[u];
                let pu = state_pos[u];

                // ==================== Inter-route moves ====================
                let neigh_len = self.neighbors[u].len();
                if neigh_len > 0 {
                    let start = rng.gen_range(0..neigh_len);
                    for off in 0..neigh_len {
                        let v = self.neighbors[u][(start + off) % neigh_len];
                        let sv = state_seq[v];
                        if su == sv {
                            continue;
                        }
                        // Skip if neither route changed since u was last tested.
                        if when_last_modified[su].max(when_last_modified[sv]) <= last_tested {
                            continue;
                        }
                        let pv = state_pos[v];

                        if self.move_inter_relocate(inner, middle, metric, problem, ind, su, pu, sv, pv + 1) {
                            num_moves += 1;
                            when_last_modified[su] = num_moves;
                            when_last_modified[sv] = num_moves;
                            Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                            improved = true;
                            break;
                        }
                        if pu == 1
                            && self.move_inter_relocate(inner, middle, metric, problem, ind, sv, pv, su, pu)
                        {
                            num_moves += 1;
                            when_last_modified[su] = num_moves;
                            when_last_modified[sv] = num_moves;
                            Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                            improved = true;
                            break;
                        }
                        if self.move_2opt_star(inner, middle, metric, problem, ind, su, pu, sv, pv + 1) {
                            num_moves += 1;
                            when_last_modified[su] = num_moves;
                            when_last_modified[sv] = num_moves;
                            Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                            improved = true;
                            break;
                        }
                    }
                }

                // ==================== Empty-route moves ====================
                let su = state_seq[u];
                let pu = state_pos[u];
                if loop_num > 1 {
                    if let Some(empty_s) = ind.routes.iter().position(|s| s.is_empty()) {
                        if empty_s != su {
                            if self.move_2opt_star(inner, middle, metric, problem, ind, su, pu, empty_s, 1) {
                                num_moves += 1;
                                when_last_modified[su] = num_moves;
                                when_last_modified[empty_s] = num_moves;
                                Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                                improved = true;
                                continue;
                            }
                            if self.move_inter_relocate(inner, middle, metric, problem, ind, su, pu, empty_s, 1) {
                                num_moves += 1;
                                when_last_modified[su] = num_moves;
                                when_last_modified[empty_s] = num_moves;
                                Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                                improved = true;
                                continue;
                            }
                        }
                    }
                }

                // ==================== Intra-route moves ====================
                let su = state_seq[u];
                let pu = state_pos[u];
                if when_last_modified[su] > last_tested {
                    if self.move_intra_relocate(inner, middle, metric, problem, ind, su, pu) {
                        num_moves += 1;
                        when_last_modified[su] = num_moves;
                        Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                        improved = true;
                    }

                    let su = state_seq[u];
                    let pu = state_pos[u];
                    if self.move_intra_swap(inner, middle, metric, problem, ind, su, pu) {
                        num_moves += 1;
                        when_last_modified[su] = num_moves;
                        Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                        improved = true;
                    }

                    let su = state_seq[u];
                    let pu = state_pos[u];
                    if self.move_2opt(inner, middle, metric, problem, ind, su, pu) {
                        num_moves += 1;
                        when_last_modified[su] = num_moves;
                        Self::rebuild_mappings(ind, &mut state_seq, &mut state_pos);
                        improved = true;
                    }
                }
            }
        }

        ind.build(problem);
    }

    /// Refresh the debris to (route, position) mappings
    fn rebuild_mappings(ind: &Individual, state_seq: &mut [usize], state_pos: &mut [usize]) {
        for (s, mo) in ind.routes.iter().enumerate() {
            // Open route: position 0 is the home; every other position is a debris.
            for (p, &idx) in mo.sequence.iter().enumerate() {
                if p != 0 {
                    state_seq[idx] = s;
                    state_pos[idx] = p;
                }
            }
        }
    }

    // ------------------------- Intra-route moves --------------------------

    /// Relocate the debris at `pos` to the best position in its own route
    #[allow(clippy::too_many_arguments)]
    fn move_intra_relocate(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        ind: &mut Individual,
        s: usize,
        pos: usize,
    ) -> bool {
        let len = ind.routes[s].sequence.len();
        if len < 3 || pos == 0 || pos >= len {
            return false;
        }

        let old_value = metric.route_value(problem, &ind.routes[s]);
        let state = ind.routes[s].sequence[pos];

        let mut best_value = old_value;
        let mut best: Option<MiddleOutput> = None;
        // Open route: debris occupy positions 1..len; re-insert anywhere in that
        // range (incl. the end, which makes the relocated debris the terminal).
        for insert_at in 1..len {
            if insert_at == pos {
                continue;
            }
            let mut cand = ind.routes[s].sequence.clone();
            cand.remove(pos);
            cand.insert(insert_at, state);
            // Lower-bound gate: skip the costly solve when the candidate cannot
            // improve on the best route value found so far.
            if self.params.use_bound_pruning
                && metric.bound_value(problem, &middle.solve_lb(&cand, inner, problem)) >= best_value
            {
                continue;
            }
            let mo = middle.solve(&cand, inner, problem);
            let v = metric.route_value(problem, &mo);
            if v < best_value {
                best_value = v;
                best = Some(mo);
            }
        }

        match best {
            Some(mo) => {
                ind.routes[s] = mo;
                true
            }
            None => false,
        }
    }

    /// Swap the debris at `pos` with the best other debris in its route
    #[allow(clippy::too_many_arguments)]
    fn move_intra_swap(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        ind: &mut Individual,
        s: usize,
        pos: usize,
    ) -> bool {
        let len = ind.routes[s].sequence.len();
        if len < 3 || pos == 0 || pos >= len {
            return false;
        }

        let old_value = metric.route_value(problem, &ind.routes[s]);
        let mut best_value = old_value;
        let mut best: Option<MiddleOutput> = None;
        for t in (pos + 1)..len {
            let mut cand = ind.routes[s].sequence.clone();
            cand.swap(pos, t);
            if self.params.use_bound_pruning
                && metric.bound_value(problem, &middle.solve_lb(&cand, inner, problem)) >= best_value
            {
                continue;
            }
            let mo = middle.solve(&cand, inner, problem);
            let v = metric.route_value(problem, &mo);
            if v < best_value {
                best_value = v;
                best = Some(mo);
            }
        }

        match best {
            Some(mo) => {
                ind.routes[s] = mo;
                true
            }
            None => false,
        }
    }

    /// Reverse the best interior subsegment starting at `pos`
    #[allow(clippy::too_many_arguments)]
    fn move_2opt(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        ind: &mut Individual,
        s: usize,
        pos: usize,
    ) -> bool {
        let len = ind.routes[s].sequence.len();
        if len < 3 || pos == 0 || pos >= len - 1 {
            return false;
        }

        let old_value = metric.route_value(problem, &ind.routes[s]);
        let mut best_value = old_value;
        let mut best: Option<MiddleOutput> = None;
        // Open route: the reversible segment may extend to the terminal debris.
        for end in (pos + 1)..len {
            let mut cand = ind.routes[s].sequence.clone();
            cand[pos..=end].reverse();
            if self.params.use_bound_pruning
                && metric.bound_value(problem, &middle.solve_lb(&cand, inner, problem)) >= best_value
            {
                continue;
            }
            let mo = middle.solve(&cand, inner, problem);
            let v = metric.route_value(problem, &mo);
            if v < best_value {
                best_value = v;
                best = Some(mo);
            }
        }

        match best {
            Some(mo) => {
                ind.routes[s] = mo;
                true
            }
            None => false,
        }
    }

    // ------------------------- Inter-route moves --------------------------

    /// Relocate debris `u` in route `s1` at pos `pos1` into route `s2` at
    /// `pos2`, or exchange it with the debris at `pos2`, whichever helps most
    #[allow(clippy::too_many_arguments)]
    fn move_inter_relocate(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        ind: &mut Individual,
        s1: usize,
        pos1: usize,
        s2: usize,
        pos2: usize,
    ) -> bool {
        debug_assert_ne!(s1, s2);
        let len1 = ind.routes[s1].sequence.len();
        let len2 = ind.routes[s2].sequence.len();
        if pos1 == 0 || pos1 >= len1 {
            return false;
        }
        if pos2 == 0 || pos2 > len2 {
            return false;
        }

        let old_value = metric.route_value(problem, &ind.routes[s1])
            + metric.route_value(problem, &ind.routes[s2]);
        let state_u = ind.routes[s1].sequence[pos1];

        let mut best_value = old_value;
        let mut best: Option<(MiddleOutput, MiddleOutput)> = None;

        // Relocate u into s2 at pos2.
        {
            let mut new1 = ind.routes[s1].sequence.clone();
            new1.remove(pos1);
            let mut new2 = ind.routes[s2].sequence.clone();
            // Open route: allow inserting after the last debris (append = new terminal).
            let ins = pos2.min(new2.len());
            new2.insert(ins, state_u);
            // Lower-bound gate on the combined value of the two changed routes.
            if !(self.params.use_bound_pruning
                && metric.bound_value(problem, &middle.solve_lb(&new1, inner, problem))
                    + metric.bound_value(problem, &middle.solve_lb(&new2, inner, problem))
                    >= best_value)
            {
                let mo1 = middle.solve(&new1, inner, problem);
                let mo2 = middle.solve(&new2, inner, problem);
                let v = metric.route_value(problem, &mo1) + metric.route_value(problem, &mo2);
                if v < best_value {
                    best_value = v;
                    best = Some((mo1, mo2));
                }
            }
        }

        // Exchange u with a debris of s2 at pos2 (any real stop, incl. the last).
        if pos2 < len2 {
            let state_v = ind.routes[s2].sequence[pos2];
            let mut new1 = ind.routes[s1].sequence.clone();
            new1[pos1] = state_v;
            let mut new2 = ind.routes[s2].sequence.clone();
            new2[pos2] = state_u;
            if !(self.params.use_bound_pruning
                && metric.bound_value(problem, &middle.solve_lb(&new1, inner, problem))
                    + metric.bound_value(problem, &middle.solve_lb(&new2, inner, problem))
                    >= best_value)
            {
                let mo1 = middle.solve(&new1, inner, problem);
                let mo2 = middle.solve(&new2, inner, problem);
                let v = metric.route_value(problem, &mo1) + metric.route_value(problem, &mo2);
                if v < best_value {
                    best = Some((mo1, mo2));
                }
            }
        }

        match best {
            Some((mo1, mo2)) => {
                ind.routes[s1] = mo1;
                ind.routes[s2] = mo2;
                true
            }
            None => false,
        }
    }

    /// Swap the tails of routes `s1` and `s2` at `pos1`/`pos2`. Open routes keep
    /// their shared home as the head; the terminal debris is what gets swapped.
    #[allow(clippy::too_many_arguments)]
    fn move_2opt_star(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        metric: &Metric,
        problem: &Problem,
        ind: &mut Individual,
        s1: usize,
        pos1: usize,
        s2: usize,
        pos2: usize,
    ) -> bool {
        debug_assert_ne!(s1, s2);
        let len1 = ind.routes[s1].sequence.len();
        let len2 = ind.routes[s2].sequence.len();
        if pos1 == 0 || pos1 > len1 {
            return false;
        }
        if pos2 == 0 || pos2 > len2 {
            return false;
        }

        let old_value = metric.route_value(problem, &ind.routes[s1])
            + metric.route_value(problem, &ind.routes[s2]);

        let r1 = &ind.routes[s1].sequence;
        let r2 = &ind.routes[s2].sequence;

        // Open routes: swap the tails after each cut. Position 0 (the shared
        // home) stays as each route's head; there is no trailing home to keep.
        let mut new1: Vec<usize> = r1[..pos1].to_vec();
        new1.extend_from_slice(&r2[pos2..len2]);
        let mut new2: Vec<usize> = r2[..pos2].to_vec();
        new2.extend_from_slice(&r1[pos1..len1]);

        // Lower-bound gate on the combined value of the two changed routes.
        if self.params.use_bound_pruning
            && metric.bound_value(problem, &middle.solve_lb(&new1, inner, problem))
                + metric.bound_value(problem, &middle.solve_lb(&new2, inner, problem))
                >= old_value
        {
            return false;
        }

        let mo1 = middle.solve(&new1, inner, problem);
        let mo2 = middle.solve(&new2, inner, problem);
        let new_value = metric.route_value(problem, &mo1) + metric.route_value(problem, &mo2);

        if new_value < old_value {
            ind.routes[s1] = mo1;
            ind.routes[s2] = mo2;
            true
        } else {
            false
        }
    }
}


// ============================== HGS engine =============================== //

pub struct Hgs {
    params: HgsParams,
}

impl Hgs {
    pub fn new(params: &HgsParams) -> Self {
        Self { params: params.clone() }
    }

    fn log_init(&self) {
        println!("Solver start");
        println!("Generating initial population...")
    }

    fn log_head(&self) {
        println!("Starting generic selection...");
        println!(
            "{:>10} | {:>10} | {:>10} | {:>10} | {:>10} |",
            "time",
            "iter",
            "nimp",
            "cost",
            "feas"
        );
    }

    fn log_iter(
        &self,
        t0  : Instant,
        iter: usize,
        nimp: usize,
        population: &Population,
    ) {
        if self.params.log_iter != 0 && iter.is_multiple_of(self.params.log_iter) {
            let best = population.best();
            println!(
                "{:>10.2} | {:>10} | {:>10} | {:>10.1} | {:>10} |",
                t0.elapsed().as_secs_f64(),
                iter,
                nimp,
                best.cost,
                best.is_feasible()
            );
        }
    }

    fn log_foot(
        &self,
        t0  : Instant,
        iter: usize,
        nimp: usize,
        population: &Population,
    ) {
        self.log_iter(t0, iter, nimp, population);
        println!("Terminated in {:.2}s and {} iterations ({} non-improving).", t0.elapsed().as_secs_f64(), iter, nimp);
    }

    fn log_trace(
        &self, 
        trace: &mut Trace, 
        t0: Instant,
        iter: usize, 
        population: &Population
    ) {
        if self.params.log_iter != 0 && iter.is_multiple_of(self.params.log_iter) {
            let best = population.best();
            trace.push(
                (
                    t0.elapsed().as_secs_f64(),
                    iter,
                    best.cost,
                    best.is_feasible(),
                )
            );
        }
    }
}

impl OuterLoop for Hgs {
    fn solve(
        &self,
        inner: &dyn InnerLoop,
        middle: &dyn MiddleLoop,
        problem: &Problem,
    ) -> OuterOutput {

        let t0 = Instant::now();

        self.log_init();

        // Initialize components
        let mut rng = SmallRng::seed_from_u64(self.params.seed);
        let mut metric = Metric::new(&self.params);
        let mut population = Population::new(&self.params);
        let generator = Generator::new(&self.params);
        let crossover = Crossover::new(&self.params);
        let mut search = Search::new(&self.params);

        search.initialize(inner, problem);

        // Generate initial population
        for _ in 0..self.params.pop_init.max(1) {
            let mut individual = generator.generate(inner, middle, problem, &mut rng);
            search.improve(&mut individual, inner, middle, &metric, problem, &mut rng);
            population.add(individual, &mut metric);
        }

        self.log_head();

        // Main loop
        let mut cost = metric.value(population.best());
        let mut iter = 0usize;
        let mut nimp = 0usize;

        // Algorithm trace
        let mut trace: Trace = Vec::new();
        self.log_trace(&mut trace, t0, iter, &population);
        
        loop {
            // Stopping criteria
            if !problem.has_objective() && population.best_feasible().is_some() { break; }
            if iter >= self.params.max_iter { break; }
            if nimp >= self.params.max_nimp { break; }
            if t0.elapsed().as_secs_f64() >= self.params.time_limit { break; }

            // Construct a new individual by crossover and local search
            let mut individual = crossover.generate(inner, middle, &metric, &population, problem, &mut rng);
            search.improve(&mut individual, inner, middle, &metric, problem, &mut rng);
            population.add(individual, &mut metric);
    
            // Update working variables 
            let curr = metric.value(population.best());
            if curr < cost {
                cost = curr;
                nimp = 0;
                iter += 1;
            } else {
                nimp += 1;
                iter += 1;
            }
            
            self.log_iter(t0, iter, nimp, &population);
            self.log_trace(&mut trace, t0, iter, &population);
        }

        self.log_foot(t0, iter, nimp, &population);
        self.log_trace(&mut trace, t0, iter, &population);

        // Recover best individual and build output
        let best = population.best();

        let out = OuterOutput {
            routes: best.routes.clone(),
            cost: best.routes.iter().map(|s| s.cost).sum(),
            feasible: best.routes.iter().all(|s| s.feasible),
            trace: trace,
        };

        out
    }
}
