//! Run with `cargo run --release --experiments simple`.

use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::iter;

use mcrdv::MissionPlan;
use rand::Rng;
use serde::{Deserialize, Serialize};

use mcrdv::io::{read_debris,format_mission};
use mcrdv::orbit::centroid;
use mcrdv::problem::Problem;
use mcrdv::sequence::{HgsConfig,HgsSequence};
use mcrdv::schedule::DpSchedule;
use mcrdv::solver::{SequenceLayer,Solver};
use mcrdv::transfer::qlaw::QlawTransfer;


// ================================ Solvers ================================ //

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMcrdv {
    pub transfer: ConfigMcrdvTransfer,
    pub schedule: ConfigMcrdvSchedule,
    pub sequence: ConfigMcrdvSequence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ConfigMcrdvTransfer {
    Qlaw(ConfigMcrdvQlaw),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMcrdvQlaw {
    #[serde(default = "ConfigMcrdvQlaw::default_thrust")]
    pub thrust: f64,
    #[serde(default = "ConfigMcrdvQlaw::default_factor")]
    pub factor: f64,
    #[serde(default = "ConfigMcrdvQlaw::default_num_samples")]
    pub num_samples: usize,
    #[serde(default = "ConfigMcrdvQlaw::default_num_hints")]
    pub num_hints: usize,
    #[serde(default = "ConfigMcrdvQlaw::default_cost_factor_time")]
    pub cost_factor_time: f64,
    #[serde(default = "ConfigMcrdvQlaw::default_cost_factor_fuel")]
    pub cost_factor_fuel: f64,
}

impl ConfigMcrdvQlaw {
    fn default_thrust() -> f64 { 3e-3 }
    fn default_factor() -> f64 { 1.5 }
    fn default_num_samples() -> usize { 512 }
    fn default_num_hints() -> usize { 16 }
    fn default_cost_factor_time() -> f64 { 0.0 }
    fn default_cost_factor_fuel() -> f64 { 1.0 }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ConfigMcrdvSchedule {
    Dp(ConfigMcrdvDp),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMcrdvDp {
    #[serde(default = "ConfigMcrdvDp::allow_waiting")]
    pub allow_waiting: bool,
    #[serde(default = "ConfigMcrdvDp::grid_size")]
    pub grid_size: usize,
    #[serde(default = "ConfigMcrdvDp::use_hints")]
    pub use_hints: bool,
    #[serde(default = "ConfigMcrdvDp::use_cache")]
    pub use_cache: bool,
}

impl ConfigMcrdvDp {
    fn allow_waiting() -> bool { true }
    fn grid_size() -> usize { 32 }
    fn use_hints() -> bool { true }
    fn use_cache() -> bool { true }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ConfigMcrdvSequence {
    Hgs(ConfigMcrdvHgs),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMcrdvHgs {
    #[serde(default = "ConfigMcrdvHgs::default_seed")]
    pub seed: u64,
    #[serde(default = "ConfigMcrdvHgs::default_preset")]
    pub preset: usize,
    #[serde(default = "ConfigMcrdvHgs::default_max_time")]
    pub max_time: f64,
}

impl ConfigMcrdvHgs {
    fn default_seed() -> u64 { 0 }
    fn default_preset() -> usize { 5 }
    fn default_max_time() -> f64 { 60.0 }
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
    pub pos_chasers: String,
    pub num_chasers: usize,
    pub max_time: f64,
    pub max_load: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "args", rename_all = "lowercase")]
pub enum ConfigSolver {
    Mcrdv(ConfigMcrdv),
}

#[derive(Debug, Serialize)]
struct ExperimentResult<'a> {
    config: &'a Config,
    result: &'a MissionPlan,
}


fn build_problem(cfg: &ConfigProblem) -> Result<Problem, Box<dyn Error>> {

    let debris = read_debris(&cfg.path);
    let mut states = match cfg.pos_chasers.as_str() {
        "centroid" => vec![centroid(&debris)],
        _ => return Err("unknown chaser_init".into())
    };
    states.extend(debris);
    
    let problem = Problem::new(
        states,
        cfg.num_chasers,
        cfg.max_time,
        cfg.max_load
    );
    
    Ok(problem)
}

fn build_solver(cfg: &ConfigSolver) -> Result<Solver<impl SequenceLayer>, Box<dyn Error>> {

    let ConfigSolver::Mcrdv(mcrdv) = cfg;

    let transfer = match &mcrdv.transfer {
        ConfigMcrdvTransfer::Qlaw(p) => {
            QlawTransfer::with_options(
                p.thrust,
                p.factor,
                p.num_samples,
                p.num_hints,
                p.cost_factor_time,
                p.cost_factor_fuel
            )
        }
    };

    let schedule = match &mcrdv.schedule {
        ConfigMcrdvSchedule::Dp(p) => {
            DpSchedule::with_options(
                transfer,
                p.allow_waiting,
                p.grid_size,
                p.use_hints,
                p.use_cache
            )
        }
    };

    let sequence = match &mcrdv.sequence {
        ConfigMcrdvSequence::Hgs(p) => {
            let hgs_cfg = HgsConfig::preset(p.preset)
                .with_seed(p.seed)
                .with_max_time(p.max_time);
            HgsSequence::with_config(schedule, hgs_cfg)
        }
    };

    Ok(Solver::new(sequence))
}

fn build_uuid(len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    let mut rng = rand::rng();
    let one_char = || CHARSET[rng.random_range(0..CHARSET.len())] as char;
    iter::repeat_with(one_char).take(len).collect()
}


// ================================ Runners ================================ //

fn main() -> Result<(), Box<dyn Error>> {

    // Parse config
    let path = std::env::args().nth(1).ok_or("usage: cargo run --release --example run -- exp/config.json [--save]",)?;
    let save = std::env::args().any(|arg| arg == "--save");
    let text = fs::read_to_string(&path).map_err(|e| format!("reading '{path}': {e}"))?;
    let config: Config = serde_json::from_str(&text).map_err(|e| format!("parsing '{path}': {e}"))?;

    println!("====================== EXPERIMENT STARTS =====================");
    let problem = build_problem(&config.problem)?;
    let mut solver = build_solver(&config.solver)?;
    let result = solver.solve(&problem);
    println!("{}", format_mission(&problem, &result));
    let result = ExperimentResult {
        config: &config,
        result: &result,
    };
    println!("====================== EXPERIMENT FINISH =====================");

    // Save results
    if save {
        let out_dir = PathBuf::from("experiments/results");
        fs::create_dir_all(&out_dir).map_err(|e| format!("creating '{}': {e}", out_dir.display()))?;
        let out_uuid = build_uuid(12);
        let out_path = out_dir.join(format!("{}_{}.json", config.experiment.name, out_uuid));
        let json = serde_json::to_string_pretty(&result)?;
        fs::write(&out_path, json).map_err(|e| format!("writing '{}': {e}", out_path.display()))?;
        println!("Saved results to {}", out_path.display());
    }

    Ok(())
}
