use crate::optim::crossover::CrossoverParams;
use crate::optim::generator::GeneratorParams;
use crate::optim::metric::MetricParams;
use crate::optim::polisher::PolisherParams;
use crate::optim::population::PopulationParams;
use crate::optim::search::SearchParams;
use crate::optim::solver::SolverParams;

/// Global parameters handler for all components of the Solver struct
#[derive(Clone)]
pub struct Params {
    pub solver_params: SolverParams,
    pub metric_params: MetricParams,
    pub generator_params: GeneratorParams,
    pub population_params: PopulationParams,
    pub crossover_params: CrossoverParams,
    pub search_params: SearchParams,
    pub polisher_params: PolisherParams,
}

impl Params {
    /// Create parameters from baseline values
    pub fn base() -> Self {
        Self {
            solver_params: SolverParams {
                limit_nimp: 100,
                limit_iter: 500,
                limit_time: f64::INFINITY,
                log_iter: 10,
                log_trace: true,
                seed: 42,
            },
            metric_params: MetricParams {
                penalty_load_init: 1.0,
                penalty_load_increase: 2.0,
                penalty_load_decrease: 0.5,
                penalty_load_min: 1e-4,
                penalty_load_max: 1e4,
                penalty_time_init: 1.0,
                penalty_time_increase: 2.0,
                penalty_time_decrease: 0.5,
                penalty_time_min: 1e-6,
                penalty_time_max: 1e2,
                target_ratio: 0.9,
            },
            generator_params: GeneratorParams {
                pop_init: 10,
                generation_method: "random".to_string(),
            },
            population_params: PopulationParams {
                pop_min: 5,
                pop_max: 10,
                nb_close: 2,
                nb_elite: 2,
                adapt_iter: 10,
                clone_eps: 1e-6,
            },
            crossover_params: CrossoverParams {
                nb_pick: 2,
                load_threshold_factor: 1.5,
                time_threshold_factor: 1.5,
            },
            search_params: SearchParams { nb_neighbors: 30 },
            polisher_params: PolisherParams {
                polish_condition: "last".to_string(),
                polish_method: "dp".to_string(),
                dp_time_step: 86_400.0,
            },
        }
    }

    /// Create parameters from level preset
    pub fn preset(level: usize) -> Self {
        let mut base = Self::base();
        match level {
            0 => {
                // Local search on a single initial solution
                base.solver_params.limit_nimp = 0;
                base.solver_params.limit_iter = 0;
                base.solver_params.log_iter = 1;
                base.generator_params.pop_init = 1;
                base.population_params.pop_min = 2;
                base.population_params.pop_max = 2;
                base.population_params.nb_close = 1;
                base.population_params.nb_elite = 1;
            }
            1 => {
                // Local search on multiple initial solutions
                base.solver_params.limit_nimp = 0;
                base.solver_params.limit_iter = 0;
                base.solver_params.log_iter = 1;
                base.generator_params.pop_init = 5;
                base.population_params.pop_min = 2;
                base.population_params.pop_max = 3;
                base.population_params.nb_close = 1;
                base.population_params.nb_elite = 1;
            }
            2 => {
                // Very shallow genetic selection
                base.solver_params.limit_nimp = 10;
                base.solver_params.limit_iter = 50;
                base.solver_params.log_iter = 1;
                base.generator_params.pop_init = 6;
                base.population_params.pop_min = 3;
                base.population_params.pop_max = 6;
                base.population_params.nb_close = 1;
                base.population_params.nb_elite = 1;
            }
            3 => {
                // Shallow genetic selection
                base.solver_params.limit_nimp = 100;
                base.solver_params.limit_iter = 500;
                base.solver_params.log_iter = 20;
                base.generator_params.pop_init = 10;
                base.population_params.pop_min = 5;
                base.population_params.pop_max = 10;
                base.population_params.nb_close = 2;
                base.population_params.nb_elite = 2;
            }
            4 => {
                // Medium genetic selection
                base.solver_params.limit_nimp = 500;
                base.solver_params.limit_iter = 5_000;
                base.solver_params.log_iter = 100;
                base.generator_params.pop_init = 20;
                base.population_params.pop_min = 10;
                base.population_params.pop_max = 20;
                base.population_params.nb_close = 2;
                base.population_params.nb_elite = 3;
            }
            5 => {
                // Deep genetic selection
                base.solver_params.limit_nimp = 5_000;
                base.solver_params.limit_iter = 50_000;
                base.solver_params.log_iter = 200;
                base.generator_params.pop_init = 24;
                base.population_params.pop_min = 12;
                base.population_params.pop_max = 32;
                base.population_params.nb_close = 3;
                base.population_params.nb_elite = 4;
            }
            6 => {
                // Very deep genetic selection
                base.solver_params.limit_nimp = 10_000;
                base.solver_params.limit_iter = 200_000;
                base.solver_params.log_iter = 500;
                base.generator_params.pop_init = 50;
                base.population_params.pop_min = 25;
                base.population_params.pop_max = 65;
                base.population_params.nb_close = 3;
                base.population_params.nb_elite = 8;
            }
            _ => {
                panic!("Warning: level {} out of range {{0,...,6}}", level);
            }
        }

        base
    }
}
