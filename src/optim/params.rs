/// Solver parameters
#[derive(Clone)]
pub struct Params {

    // ===== Top level parameters

    /// Limit of iterations with no improvement until termination
    pub limit_nimp: usize,
    
    /// Limit of iterations until termination
    pub limit_iter: usize,
    
    /// Limit of time in seconds until termination
    pub limit_time: f64,
    
    /// Number of iterations between log traces
    pub log_iter: usize,
    
    /// Seed for random operations
    pub seed: u64,


    // ===== Metric parameters

    /// Load violation penalty initial value
    pub penalty_load_init: f64,
    
    /// Load violation penalty increase factor
    pub penalty_load_increase: f64,
    
    /// Load violation penalty decrease factor
    pub penalty_load_decrease: f64,
    
    /// Load violation penalty minimum value
    pub penalty_load_min: f64,
    
    /// Load violation penalty maximum value
    pub penalty_load_max: f64,
    
    /// Time violation penalty initial value
    pub penalty_time_init: f64,
    
    /// Time violation penalty increase factor
    pub penalty_time_increase: f64,
    
    /// Time violation penalty decrease factor
    pub penalty_time_decrease: f64,
    
    /// Time violation penalty minimum value
    pub penalty_time_min: f64,
    
    /// Time violation penalty maximum value
    pub penalty_time_max: f64,
    
    /// Target ratio of feasible solutions for penalty adaptations
    pub target_ratio: f64,

    
    // ===== Generator parameters

    /// Initial population size
    pub pop_init: usize,

    /// Initial population generation method ("random" or "greedy")
    pub generation_method: String,


    // ===== Population parameters
    
    /// Minimum population size after survival selection
    pub pop_min: usize,
    
    /// Maximum population size until survival selection
    pub pop_max: usize,
    
    /// Number of closest solutions used for diversity measurement
    pub nb_close: usize,
    
    /// Number of elite solutions preserved during survival selection
    pub nb_elite: usize,

    /// Number of solutions added between two penalty adaptations
    pub adapt_iter: usize,


    // ===== Crossover parameters

    /// Number of competitors selected in tournament for parent selection
    pub nb_pick: usize,

    /// Factor to relax load constraint for split dynamic programming pruning
    pub load_threshold_factor: f64,

    /// Factor to relax time constraint for split dynamic programming pruning
    pub time_threshold_factor: f64,


    // ===== Local search parameters

    /// Number of neighbors to consider in local search operations
    pub nb_neighbors: usize,
}

impl Params {
    /// Create parameters with baseline values
    pub fn base(limit_time: f64, seed: u64) -> Self {
        Self {
            // Top level parameters
            limit_nimp                  : 0,
            limit_iter                  : 0,
            limit_time,
            log_iter                    : 100,
            seed,
            // Metric parameters
            penalty_load_init           : 1.0,
            penalty_load_increase       : 2.0,
            penalty_load_decrease       : 0.5,
            penalty_load_min            : 1e-4,
            penalty_load_max            : 1e4,
            penalty_time_init           : 1.0,
            penalty_time_increase       : 2.0,
            penalty_time_decrease       : 0.5,
            penalty_time_min            : 1e-6,
            penalty_time_max            : 1e2,
            target_ratio                : 0.9,
            // Generator parameters
            pop_init                    : 0,
            generation_method           : "random".to_string(),
            // Population parameters
            pop_min                     : 0,
            pop_max                     : 0,
            nb_close                    : 1,
            nb_elite                    : 1,
            adapt_iter                  : 10,
            // Crossover parameters
            nb_pick                     : 2,
            load_threshold_factor       : 1.5,
            time_threshold_factor       : 1.5,
            // Local search parameters
            nb_neighbors                : 30,
        }
    }

    /// Create parameters from exploration level preset
    pub fn preset(level: usize, limit_time: f64, seed: u64) -> Self {
        let base = Self::base(limit_time, seed);
        match level {
            0 => Self { // Local search on single solution
                limit_nimp          : 0,
                limit_iter          : 0,
                log_iter            : 1,
                pop_init            : 1,
                pop_min             : 2,
                pop_max             : 2,
                nb_close            : 1,
                nb_elite            : 1,
                ..base
            },
            1 => Self { // Local search on multiple solutions
                limit_nimp          : 0,
                limit_iter          : 0,
                log_iter            : 1,
                pop_init            : 5,
                pop_min             : 2,
                pop_max             : 3,
                nb_close            : 1,
                nb_elite            : 1,
                ..base
            },
            2 => Self { // Very shallow genetic selection
                limit_nimp          : 10,
                limit_iter          : 50,
                log_iter            : 1,
                pop_init            : 6,
                pop_min             : 3,
                pop_max             : 6,
                nb_close            : 1,
                nb_elite            : 1,
                ..base
            },
            3 => Self { // Shallow generic selection
                limit_nimp          : 100,
                limit_iter          : 500,
                log_iter            : 20,
                pop_init            : 10,
                pop_min             : 5,
                pop_max             : 10,
                nb_close            : 2,
                nb_elite            : 2,
                ..base
            },
            4 => Self { // Medium genetic selection
                limit_nimp          : 500,
                limit_iter          : 5_000,
                log_iter            : 100,
                pop_init            : 20,
                pop_min             : 10,
                pop_max             : 20,
                nb_close            : 2,
                nb_elite            : 3,
                ..base
            },
            5 => Self { // Deep genetic selection
                limit_nimp          : 5_000,
                limit_iter          : 50_000,
                log_iter            : 200,
                pop_init            : 24,
                pop_min             : 12,
                pop_max             : 32,
                nb_close            : 3,
                nb_elite            : 4,
                ..base
            },
            6 => Self { // Very deep genetic selection
                limit_nimp          : 10_000,
                limit_iter          : 200_000,
                log_iter            : 500,
                pop_init            : 50,
                pop_min             : 25,
                pop_max             : 65,
                nb_close            : 3,
                nb_elite            : 8,
                ..base
            },
            _ => {
                panic!("Warning: level {} out of range {{0,...,6}}", level);
            }
        }
    }
}
