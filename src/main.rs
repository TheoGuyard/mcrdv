use clap::Parser;

use scorpion::Problem;
use scorpion::io::{Loader, Writer};
use scorpion::optim::Solver;
use scorpion::orbit::Oracle;

/// Command-line arguments parser
#[derive(Parser)]
pub struct Args {
    // =============== I/O files =============== //
    /// Path to .csv file input containing debris orbital data
    pub input_path: String,

    /// Path to .txt solution file output (if not provided, no file is written)
    #[arg(long)]
    pub output_path: Option<String>,

    /// Starting position of chasers ("first": all chasers start at the
    /// position of the first debris; "barycenter": all chasers start at the
    /// barycenter of all debris)
    #[arg(long, default_value = "barycenter")]
    pub chaser_start: String,

    // =============== Problem parameters =============== //
    /// Maximum number of chasers
    #[arg(long, default_value_t = 5)]
    pub max_chasers: usize,

    /// Maximum debris load per chaser
    #[arg(long, default_value_t = 5)]
    pub max_load: usize,

    /// Maximum mission duration [seconds]
    #[arg(long, default_value_t = 31_536_000.0)]
    pub mission_time: f64,

    // =============== Oracle parameters =============== //
    /// Type of transfer strategy to use for maneuvers ("direct": direct
    /// transfer using maximum thrust; "drift": drift transfer using
    /// intermediate orbit; "best": select best strategy among all available)
    #[arg(long, default_value = "direct")]
    pub strategy: String,

    /// Maximum chaser thrust [m/s^2]
    #[arg(long, default_value_t = 0.003)]
    pub max_thrust: f64,

    // =============== Solver parameters =============== //
    /// Level of aggressiveness in the solver (0-6, higher = more aggressive)
    #[arg(long, default_value_t = 3)]
    pub level: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load debris from input file
    println!("Loading debris states...");
    let debris = Loader::load(&args.input_path);
    println!("  loaded {} debris", debris.len());
    println!();

    // Set problem data
    println!("Problem parameters");
    println!("  chaser start : {}", args.chaser_start);
    println!("  max chasers  : {}", args.max_chasers);
    println!("  max load     : {}", args.max_load);
    println!("  max time     : {:.0} sec", args.mission_time);
    let problem = Problem::new(
        &debris,
        args.chaser_start,
        args.max_chasers,
        args.max_load,
        args.mission_time,
    );
    println!();

    // Set oracle (used to optimize maneuvers between states)
    println!("Oracle parameters");
    println!("  strategy     : {}", args.strategy);
    println!("  max thrust   : {} m/s^2", args.max_thrust);
    println!();
    let oracle = Oracle::new(args.strategy, args.max_thrust);

    // Set solver (used to optimize chaser routes)
    println!("Solver parameters");
    println!("  preset level : {}", args.level);
    println!();
    let mut solver = Solver::preset(args.level);

    // Run solver
    let (solution, _) = solver.solve(&problem, &oracle);
    println!("\n{}", solution);

    // Write solution to file if output path provided
    if let Some(output_path) = &args.output_path {
        Writer::write(output_path, &solution);
    }

    Ok(())
}
