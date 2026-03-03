use chrono::Local;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use uuid::Uuid;

use scorpion::Problem;
use scorpion::io::Loader;
use scorpion::optim::{Params, Solution, Solver, SolverTrace};
use scorpion::orbit::Oracle;

#[derive(Serialize, Deserialize)]
struct ExperimentResult {
    run_id: String,
    config: serde_json::Value,
    solution: Solution,
    trace: SolverTrace,
}

fn update_params(params: &mut Params, key: &str, value: &serde_json::Value) {
    match key {
        // SolverParams
        "limit_nimp" => params.solver_params.limit_nimp = value.as_u64().unwrap() as usize,
        "limit_iter" => params.solver_params.limit_iter = value.as_u64().unwrap() as usize,
        "limit_time" => params.solver_params.limit_time = value.as_f64().unwrap(),
        "log_iter" => params.solver_params.log_iter = value.as_u64().unwrap() as usize,
        "log_trace" => params.solver_params.log_trace = value.as_bool().unwrap(),
        "seed" => params.solver_params.seed = value.as_u64().unwrap(),
        // MetricParams
        "penalty_load_init" => params.metric_params.penalty_load_init = value.as_f64().unwrap(),
        "penalty_load_increase" => {
            params.metric_params.penalty_load_increase = value.as_f64().unwrap()
        }
        "penalty_load_decrease" => {
            params.metric_params.penalty_load_decrease = value.as_f64().unwrap()
        }
        "penalty_load_min" => params.metric_params.penalty_load_min = value.as_f64().unwrap(),
        "penalty_load_max" => params.metric_params.penalty_load_max = value.as_f64().unwrap(),
        "penalty_time_init" => params.metric_params.penalty_time_init = value.as_f64().unwrap(),
        "penalty_time_increase" => {
            params.metric_params.penalty_time_increase = value.as_f64().unwrap()
        }
        "penalty_time_decrease" => {
            params.metric_params.penalty_time_decrease = value.as_f64().unwrap()
        }
        "penalty_time_min" => params.metric_params.penalty_time_min = value.as_f64().unwrap(),
        "penalty_time_max" => params.metric_params.penalty_time_max = value.as_f64().unwrap(),
        "target_ratio" => params.metric_params.target_ratio = value.as_f64().unwrap(),
        // GeneratorParams
        "pop_init" => params.generator_params.pop_init = value.as_u64().unwrap() as usize,
        "generation_method" => {
            params.generator_params.generation_method = value.as_str().unwrap().to_string()
        }
        // PopulationParams
        "pop_min" => params.population_params.pop_min = value.as_u64().unwrap() as usize,
        "pop_max" => params.population_params.pop_max = value.as_u64().unwrap() as usize,
        "nb_close" => params.population_params.nb_close = value.as_u64().unwrap() as usize,
        "nb_elite" => params.population_params.nb_elite = value.as_u64().unwrap() as usize,
        "adapt_iter" => params.population_params.adapt_iter = value.as_u64().unwrap() as usize,
        "clone_eps" => params.population_params.clone_eps = value.as_f64().unwrap(),
        // CrossoverParams
        "nb_pick" => params.crossover_params.nb_pick = value.as_u64().unwrap() as usize,
        "load_threshold_factor" => {
            params.crossover_params.load_threshold_factor = value.as_f64().unwrap()
        }
        "time_threshold_factor" => {
            params.crossover_params.time_threshold_factor = value.as_f64().unwrap()
        }
        // SearchParams
        "nb_neighbors" => params.search_params.nb_neighbors = value.as_u64().unwrap() as usize,
        // PolisherParams
        "polish_condition" => {
            params.polisher_params.polish_condition = value.as_str().unwrap().to_string()
        }
        "polish_method" => {
            params.polisher_params.polish_method = value.as_str().unwrap().to_string()
        }
        "dp_time_step" => params.polisher_params.dp_time_step = value.as_f64().unwrap(),
        // level preset overrides
        "level" => {
            let preset_level = value.as_u64().unwrap() as usize;
            let preset_params = Params::preset(preset_level);
            *params = preset_params;
        }
        _ => eprintln!("Warning: Unknown solver arg '{}', skipping", key),
    }
}

#[derive(Parser)]
pub struct Args {
    /// Path to .yml configuration file for the experiment
    pub config_path: String,
    /// Whether to save results
    #[arg(long, default_value_t = false)]
    pub save: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Running experiment...");
    println!("  config: {}", args.config_path);
    println!();

    // ===== Unpack configuration =====

    let stream = fs::read_to_string(&args.config_path)?;
    let config: serde_json::Value = serde_json::from_str(&stream)?;

    let datafile = &format!(
        "data/{}.csv",
        config["problem"]["dataset"]
            .as_str()
            .expect("Missing config.problem.dataset keyword")
    );
    let debris = Loader::load(datafile);

    let problem = Problem::new(
        &debris,
        config["problem"]["chaser_start"]
            .as_str()
            .expect("Missing config.problem.chaser_start keyword")
            .to_string(),
        config["problem"]["max_chasers"]
            .as_u64()
            .expect("Missing config.problem.max_chasers keyword") as usize,
        config["problem"]["max_load"]
            .as_u64()
            .expect("Missing config.problem.max_load keyword") as usize,
        config["problem"]["mission_time"]
            .as_f64()
            .expect("Missing config.problem.mission_time keyword"),
    );

    let oracle = Oracle::new(
        config["oracle"]["strategy"]
            .as_str()
            .expect("Missing oracle.strategy keyword")
            .to_string(),
        config["oracle"]["max_thrust"]
            .as_f64()
            .expect("Missing oracle.max_thrust keyword"),
    );

    // ===== Solve problem =====

    println!("Problem parameters");
    println!("  chaser start : {}", problem.chaser_start);
    println!("  max chasers  : {}", problem.max_chasers);
    println!("  max load     : {}", problem.max_load);
    println!("  max time     : {:.0} sec", problem.mission_time);
    println!();

    println!("Oracle parameters");
    println!("  strategy     : {}", oracle.strategy);
    println!("  max thrust   : {} m/s^2", oracle.max_thrust);
    println!();

    let argmap = config["solver"]
        .as_object()
        .expect("Missing solver keyword");

    let mut params = match argmap.get("level") {
        Some(value) => Params::preset(value.as_u64().unwrap() as usize),
        None => Params::base(),
    };

    for (key, value) in argmap {
        if key != "level" {
            update_params(&mut params, key, value);
        }
    }

    let mut solver = Solver::new(params);
    let (solution, trace) = solver.solve(&problem, &oracle);

    // ===== Save results =====

    if args.save {
        let run_time = Local::now().format("%Y%m%d-%H%M%S");
        let run_uuid = Uuid::new_v4().to_string();
        let run_path = format!("experiments/results/{}_{}.json", run_time, run_uuid);

        println!("Saving results to {}...", run_path);

        let result = ExperimentResult {
            run_id: run_uuid.clone(),
            config: config.clone(),
            solution,
            trace,
        };

        let mut file = fs::File::create(run_path)?;
        serde_json::to_writer(&mut file, &result)?;
    }

    Ok(())
}
