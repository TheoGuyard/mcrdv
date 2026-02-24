use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::optim::crossover::Crossover;
use crate::optim::generator::Generator;
use crate::optim::metric::Metric;
use crate::optim::params::Params;
use crate::optim::polisher::Polisher;
use crate::optim::population::Population;
use crate::optim::search::Search;
use crate::optim::solution::Solution;
use crate::orbit::Oracle;
use crate::problem::Problem;

/// Length of log lines in terminal output
const LOG_LENGTH: usize = 99;

/// Parameters for Solver
#[derive(Clone)]
pub struct SolverParams {
    /// Limit of iterations with no improvement until termination
    pub limit_nimp: usize,
    /// Limit of iterations until termination
    pub limit_iter: usize,
    /// Limit of time in seconds until termination
    pub limit_time: f64,
    /// Number of iterations between log traces
    pub log_iter: usize,
    /// Whether to save log traces
    pub log_trace: bool,
    /// Seed for random operations
    pub seed: u64,
}

/// Trace for Solver
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SolverTrace {
    pub iter: Vec<usize>,
    pub nimp: Vec<usize>,
    pub time: Vec<f64>,
    pub best_cost: Vec<f64>,
    pub best_time: Vec<f64>,
    pub n_feasible: Vec<usize>,
    pub n_infeasible: Vec<usize>,
    pub ratio_feasible: Vec<f64>,
    pub penalty_time: Vec<f64>,
    pub penalty_load: Vec<f64>,
}

/// Solver for space debris remediation routing problem
pub struct Solver {
    /// Algorithm parameters
    pub params: SolverParams,
    /// Algorithm trace
    pub trace: SolverTrace,
    /// Metric to compare different solutions
    pub metric: Metric,
    /// Generator for initial solutions
    pub generator: Generator,
    /// Population of solutions
    pub population: Population,
    /// Crossover operator
    pub crossover: Crossover,
    /// Local search operator
    pub search: Search,
    /// Polishing operator for meeting times
    pub polisher: Polisher,
}

impl Solver {
    /// Instantiate solver with given parameters
    pub fn new(all_params: Params) -> Self {
        // Unpack parameters for each component
        let params = all_params.solver_params.clone();
        let metric = Metric::new(all_params.metric_params);
        let generator = Generator::new(all_params.generator_params);
        let population = Population::new(all_params.population_params);
        let crossover = Crossover::new(all_params.crossover_params);
        let search = Search::new(all_params.search_params);
        let polisher = Polisher::new(all_params.polisher_params);

        // Initialize trace
        let trace = SolverTrace::default();

        Self {
            params,
            trace,
            metric,
            population,
            generator,
            crossover,
            search,
            polisher,
        }
    }

    /// Instantiate new solver with parameters from given preset level
    pub fn preset(level: usize) -> Self {
        let all_params = Params::preset(level);
        Self::new(all_params)
    }

    /// Run the solver and return the best solution constructed
    pub fn solve(&mut self, problem: &Problem, oracle: &Oracle) -> (Solution, SolverTrace) {
        self.print_head();

        let t0 = Instant::now();
        let mut rng = SmallRng::seed_from_u64(self.params.seed);

        // ========== Initialization ========== //

        // Initialize local search engine
        self.search.initialize(problem, oracle);

        // Initialize population
        self.generator.initialize(
            problem,
            oracle,
            &mut self.metric,
            &mut self.population,
            &self.search,
            &self.polisher,
            &mut rng,
        );

        // ========== Main genetic selection loop ========== //

        let mut best = f64::INFINITY;
        let mut iter: usize = 0; // total iterations
        let mut nimp: usize = 0; // iterations since last improvement

        if !self.population.best().is_feasible() {
            eprintln!(
                "Warning: Initial population is infeasible. Consider relaxing \
                problem constraints or adapting penalties."
            );
        }

        self.print_iter(iter, nimp, t0);

        loop {
            // Termination criteria checks
            if t0.elapsed().as_secs_f64() > self.params.limit_time {
                break;
            }
            if nimp >= self.params.limit_nimp {
                break;
            }
            if iter >= self.params.limit_iter {
                break;
            }

            // Generate new solution by crossover
            let mut solution =
                self.crossover
                    .generate(problem, oracle, &self.metric, &self.population, &mut rng);

            // Improve new solution by local search
            self.search
                .run(&mut solution, problem, oracle, &self.metric, &mut rng);

            // Polish solution if condition is met
            self.polisher
                .polish(&mut solution, "feasible", problem, oracle, &self.metric);

            // Add new solution to population
            self.population.add(solution, &mut self.metric);

            // Track best solution
            let curr = self.metric.value(self.population.best());
            if curr < best {
                best = curr;
                nimp = 0;
            } else {
                nimp += 1;
            }

            iter += 1;

            // Logging
            if iter.is_multiple_of(self.params.log_iter) {
                self.print_iter(iter, nimp, t0);
            }
        }

        // Final polishing of best solution
        self.polisher.polish(
            self.population.best_mut(),
            "last",
            problem,
            oracle,
            &self.metric,
        );

        self.print_iter(iter, nimp, t0);
        self.print_tail(iter, t0);

        if !self.population.best().is_feasible() {
            eprintln!(
                "Warning: Final population is infeasible. Consider relaxing \
                problem constraints or adapting penalties."
            );
        }

        (self.population.best().clone(), self.trace.clone())
    }

    // =================== Logging and printing utilities ===================

    fn print_head(&self) {
        println!("Running solver...");
        println!("{}", "-".repeat(LOG_LENGTH));
        println!(
            "{:>8} {:>8} {:>8} | {:>12} {:>12} | {:>6} {:>6} {:>6} {:>10} {:>10}",
            "iter",
            "(last)",
            "time",
            "cost [m/s]",
            "time [s]",
            "feas.",
            "inf.",
            "ratio",
            "p-time",
            "p-load",
        );
        println!("{}", "-".repeat(LOG_LENGTH));
    }

    fn print_iter(&mut self, iter: usize, nimp: usize, t0: Instant) {
        let mut best_cost = "--".to_string();
        let mut best_time = "--".to_string();

        if let Some(sol) = self.population.best_feasible() {
            best_cost = format!("{:.2}", sol.total_cost);
            best_time = format!("{:.2}", sol.total_time());
        }

        let n_fea = self.population.feasible.solutions.len();
        let n_inf = self.population.infeasible.solutions.len();
        let ratio = if n_inf + n_fea > 0 {
            n_fea as f64 / (n_inf + n_fea) as f64
        } else {
            0.0
        };

        println!(
            "{:8} {:8} {:>8.2} | {:>12} {:>12} | {:>6} {:>6} {:>6.2} {:>10.2e} {:>10.2e}",
            iter,
            nimp,
            t0.elapsed().as_secs_f64(),
            best_cost,
            best_time,
            n_fea,
            n_inf,
            ratio,
            self.metric.penalty_time,
            self.metric.penalty_load,
        );

        // Save trace if enabled
        if self.params.log_trace {
            self.trace.iter.push(iter);
            self.trace.nimp.push(nimp);
            self.trace.time.push(t0.elapsed().as_secs_f64());
            self.trace
                .best_cost
                .push(best_cost.parse::<f64>().unwrap_or(f64::INFINITY));
            self.trace
                .best_time
                .push(best_time.parse::<f64>().unwrap_or(f64::INFINITY));
            self.trace.n_feasible.push(n_fea);
            self.trace.n_infeasible.push(n_inf);
            self.trace.ratio_feasible.push(ratio);
            self.trace.penalty_time.push(self.metric.penalty_time);
            self.trace.penalty_load.push(self.metric.penalty_load);
        }
    }

    fn print_tail(&self, iter: usize, t0: Instant) {
        println!("{}", "-".repeat(LOG_LENGTH));
        println!(
            "Terminated in {} iterations and {:.2} seconds",
            iter,
            t0.elapsed().as_secs_f64()
        );
    }
}
