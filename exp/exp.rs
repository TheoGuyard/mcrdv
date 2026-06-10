/// Run as: cargo run --release --example run -- exp/config.yaml


use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::iter;

use rand::Rng;
use serde::{Deserialize, Serialize};

use scorpion::inner::qlaw::QlawParams;
use scorpion::inner::InnerParams;
use scorpion::io::load_states;
use scorpion::middle::dynprog::DynProgParams;
use scorpion::middle::MiddleParams;
use scorpion::orbit::centroid;
use scorpion::outer::hgs::HgsParams;
use scorpion::outer::{OuterParams, Trace};
use scorpion::problem::Problem;
use scorpion::solver::{Params, Solution, Solver};


// ================================ Solvers ================================ //

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorpionConfig {
    pub inner: ScorpionInner,
    pub middle: ScorpionMiddle,
    pub outer: ScorpionOuter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ScorpionInner {
    Qlaw(ScorpionQlaw),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorpionQlaw {
    pub max_thrust: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ScorpionMiddle {
    DynProg(ScorpionDynProg),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorpionDynProg {
    pub allow_waiting: bool,
    pub grid_steps: usize,
    pub departure_hints: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ScorpionOuter {
    Hgs(ScorpionHgs),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorpionHgs {
    pub preset: usize,
    pub time_limit: Option<f64>,
}


// ================================ Helpers ================================ //

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub experiment: ConfigExperiment,
    pub problem: ConfigProblem,
    pub solver: ConfigSolver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigExperiment {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigProblem {
    pub path: String,
    pub max_time: Option<f64>,
    pub max_fuel: Option<f64>,
    pub max_load: usize,
    pub factor_time: f64,
    pub factor_fuel: f64,
    pub chaser_init: String,
    pub chaser_load: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ConfigSolver {
    Scorpion(ScorpionConfig),
}

#[derive(Debug, Serialize)]
struct ExperimentResult<'a> {
    config: &'a Config,
    solution: &'a Solution,
    trace: &'a Trace,
}


fn build_problem(cfg: &ConfigProblem) -> Result<Problem, Box<dyn Error>> {

    let debris = load_states(&cfg.path)?;

    let chasers = match cfg.chaser_init.as_str() {
        "centroid" => centroid(&debris)?,
        _ => return Err(format!("unknown chaser_init '{}'", cfg.chaser_init).into()),
    };

    let fac_chasers = cfg.chaser_load;
    let min_chasers = ((debris.len() as f64) / (cfg.max_load as f64)).ceil();
    let num_chasers = (fac_chasers * min_chasers) as usize;
    
    Problem::new(
        debris,
        chasers,
        num_chasers,
        cfg.max_time.unwrap_or(f64::INFINITY),
        cfg.max_fuel.unwrap_or(f64::INFINITY),
        cfg.max_load,
        cfg.factor_time,
        cfg.factor_fuel,
    )
}

fn build_solver(cfg: &ConfigSolver) -> Result<Solver, Box<dyn Error>> {

    let ConfigSolver::Scorpion(scorpion) = cfg;

    // Inner loop
    let inner = match &scorpion.inner {
        ScorpionInner::Qlaw(p) => {
            let mut params = QlawParams::default();
            params.max_thrust = p.max_thrust;
            InnerParams::Qlaw(params)
        }
    };

    // Middle loop
    let middle = match &scorpion.middle {
        ScorpionMiddle::DynProg(p) => {
            let mut params = DynProgParams::default();
            params.allow_waiting = p.allow_waiting;
            params.grid_steps = p.grid_steps;
            params.departure_hints = p.departure_hints;
            MiddleParams::DynProg(params)
        }
    };

    // Outer loop
    let outer = match &scorpion.outer {
        ScorpionOuter::Hgs(p) => {
            let mut params = HgsParams::preset(p.preset);
            if let Some(time_limit) = p.time_limit {
                params.time_limit = time_limit;
            }
            OuterParams::Hgs(params)
        }
    };

    let params = Params { inner, middle, outer };
    Ok(Solver::new(&params))
}

fn build_uuid(len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let mut rng = rand::thread_rng();
    let one_char = || CHARSET[rng.gen_range(0..CHARSET.len())] as char;
    iter::repeat_with(one_char).take(len).collect()
}


// ================================ Runners ================================ //

fn main() -> Result<(), Box<dyn Error>> {

    // Parse config
    let path = std::env::args().nth(1).ok_or("usage: cargo run --release --example run -- exp/config.json [--save]",)?;
    let save = std::env::args().any(|arg| arg == "--save");
    let text = fs::read_to_string(&path).map_err(|e| format!("reading '{path}': {e}"))?;
    let config: Config = serde_json::from_str(&text).map_err(|e| format!("parsing '{path}': {e}"))?;

    println!("====================== EXPERIMENT START ======================");
    let problem = build_problem(&config.problem)?;
    let solver = build_solver(&config.solver)?;
    let (solution, trace) = solver.solve(&problem);
    println!("{solution}");
    let result = ExperimentResult {
        config: &config,
        solution: &solution,
        trace: &trace,
    };
    println!("====================== EXPERIMENT FINISH =====================");

    // Save results
    if save {
        let out_dir = PathBuf::from("exp/results");
        fs::create_dir_all(&out_dir).map_err(|e| format!("creating '{}': {e}", out_dir.display()))?;
        let out_uuid = build_uuid(12);
        let out_path = out_dir.join(format!("{}_{}.json", config.experiment.name, out_uuid));
        let json = serde_json::to_string_pretty(&result)?;
        fs::write(&out_path, json).map_err(|e| format!("writing '{}': {e}", out_path.display()))?;
        println!("Saved results to {}", out_path.display());
    }

    return Ok(());
}
