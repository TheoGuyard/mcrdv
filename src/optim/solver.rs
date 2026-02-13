use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

use crate::optim::crossover::Crossover;
use crate::optim::generator::Generator;
use crate::optim::metric::Metric;
use crate::optim::params::Params;
use crate::optim::population::Population;
use crate::optim::search::Search;
use crate::optim::solution::Solution;
use crate::orbit::Oracle;
use crate::problem::Problem;

/// Length of log lines in terminal output
const LOG_LENGTH: usize = 99;

/// Solver for space debris remediation routing problem
pub struct Solver {
    /// Algorithm parameters
    pub params: Params,
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
}

impl Solver {
    /// Instantiate solver and its components with given parameters
    pub fn new(params: Params) -> Self {
        let metric = Metric::new(
            params.penalty_load_init,
            params.penalty_load_increase,
            params.penalty_load_decrease,
            params.penalty_load_min,
            params.penalty_load_max,
            params.penalty_time_init,
            params.penalty_time_increase,
            params.penalty_time_decrease,
            params.penalty_time_min,
            params.penalty_time_max,
            params.target_ratio,
        );

        let generator = Generator::new(params.pop_init, params.generation_method.clone());

        let population = Population::new(
            params.pop_min,
            params.pop_max,
            params.nb_close,
            params.nb_elite,
            params.adapt_iter,
        );

        let crossover = Crossover::new(
            params.nb_pick,
            params.load_threshold_factor,
            params.time_threshold_factor,
        );

        let search = Search::new(params.nb_neighbors);

        Self {
            params,
            metric,
            population,
            generator,
            crossover,
            search,
        }
    }

    /// Instantiate new solver with parameters from given preset level
    pub fn preset(level: usize, limit_time: f64, seed: u64) -> Self {
        let params = Params::preset(level, limit_time, seed);
        Self::new(params)
    }

    /// Run the solver and return the best solution constructed
    pub fn solver(&mut self, problem: &Problem, oracle: &Oracle) -> Solution {
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
            &mut rng,
        );

        // ========== Main genetic selection loop ========== //

        let mut best = self.population.best().clone();
        let mut iter: usize = 0; // total iterations
        let mut nimp: usize = 0; // iterations since last improvement

        if !best.is_feasible() {
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

            // Add new solution to population
            self.population.add(solution, &mut self.metric);

            // Track best solution
            let curr = self.population.best();
            if self.metric.better_than(curr, &best) {
                best = curr.clone();
                nimp = 0;
            } else {
                nimp += 1;
            }

            iter += 1;

            // Logging
            if iter % self.params.log_iter == 0 {
                self.print_iter(iter, nimp, t0);
            }
        }

        if !best.is_feasible() {
            eprintln!(
                "Warning: Final population is infeasible. Consider relaxing \
                problem constraints or adapting penalties."
            );
        }

        self.print_tail(iter, t0);

        return best;
    }

    // ======================== Printing utilities ========================

    fn print_head(&self) {
        println!("Running solver...");
        println!("{}", "-".repeat(LOG_LENGTH));
        println!(
            "{:>8} {:>8} {:>8} | {:>12} {:>12} | {:>6} {:>6} {:>6} {:>10} {:>10}",
            "iter",
            "(last)",
            "time",
            "cost [km/s]",
            "time [d]",
            "feas.",
            "inf.",
            "ratio",
            "p-time",
            "p-load",
        );
        println!("{}", "-".repeat(LOG_LENGTH));
    }

    fn print_iter(&self, iter: usize, nimp: usize, t0: Instant) {
        let mut best_cost = "--".to_string();
        let mut best_time = "--".to_string();

        match self.population.best_feasible() {
            Some(sol) => {
                best_cost = format!("{:.2}", sol.total_cost / 1000.0); // convert m/s to km/s
                best_time = format!("{:.2}", sol.total_time() / (24. * 60. * 60.));
                // convert s to d
            }
            None => {}
        };

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
