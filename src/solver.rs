use std::fmt;

use serde::Serialize;
use tracing::{info, warn};

use crate::inner::{InnerLoop, InnerParams};
use crate::middle::{MiddleLoop, MiddleParams};
use crate::outer::{OuterLoop, OuterOutput, OuterParams};
use crate::problem::Problem;

/// Solver parameters partitioned by inner, middle, and outer loop
#[derive(Debug, Clone, Default)]
pub struct Params {
    pub inner: InnerParams,
    pub middle: MiddleParams,
    pub outer: OuterParams,
}

/// Solver solution details
#[derive(Debug, Clone, Serialize)]
pub struct Solution {
    /// Per-chaser route `[chaser, debris_1, ..., debris_n]`
    pub sequences: Vec<Vec<usize>>,
    /// Per-chaser waiting times [s]
    pub waits: Vec<Vec<f64>>,
    /// Per-chaser transfer times [s]
    pub times: Vec<Vec<f64>>,
    /// Per-chaser meeting times [s]
    pub meets: Vec<Vec<f64>>,
    /// Per-chaser fuels [m/s]
    pub fuels: Vec<Vec<f64>>,
    /// Per-chaser total waiting time [s]
    pub total_waits: Vec<f64>,
    /// Per-chaser total transfer time [s]
    pub total_times: Vec<f64>,
    /// Per-chaser total fuel [m/s]
    pub total_fuels: Vec<f64>,
    /// Per-chaser total load [debris]
    pub total_loads: Vec<usize>,
    /// Per-chaser objective costs
    pub costs: Vec<f64>,
    /// Per-chaser feasibility
    pub feasibilities: Vec<bool>,
    /// Mission cost
    pub cost: f64,
    /// Mission feasibility
    pub feasible: bool,
    /// Number of active chasers
    pub active_chasers: usize,
}

impl Solution {
    pub fn new(outer: &OuterOutput) -> Self {
        let k = outer.routes.len();
        let mut sequences = Vec::with_capacity(k);
        let mut waits = Vec::with_capacity(k);
        let mut times = Vec::with_capacity(k);
        let mut meets = Vec::with_capacity(k);
        let mut fuels = Vec::with_capacity(k);
        let mut total_waits = Vec::with_capacity(k);
        let mut total_times = Vec::with_capacity(k);
        let mut total_fuels = Vec::with_capacity(k);
        let mut total_loads = Vec::with_capacity(k);
        let mut costs = Vec::with_capacity(k);
        let mut feasibilities = Vec::with_capacity(k);
        let mut active_chasers = 0;

        for mo in &outer.routes {
            if !mo.is_empty() {
                active_chasers += 1;
            }
            sequences.push(mo.sequence.clone());
            waits.push(mo.wait.clone());
            times.push(mo.time.clone());
            meets.push(mo.meet.clone());
            fuels.push(mo.fuel.clone());
            total_waits.push(mo.total_wait);
            total_times.push(mo.total_time);
            total_fuels.push(mo.total_fuel);
            total_loads.push(mo.total_load);
            costs.push(mo.cost);
            feasibilities.push(mo.feasible);
        }

        Self {
            sequences,
            waits,
            times,
            meets,
            fuels,
            total_waits,
            total_times,
            total_fuels,
            total_loads,
            costs,
            feasibilities,
            cost: outer.cost,
            feasible: outer.feasible,
            active_chasers,
        }
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "============================ Mission plan ============================")?;
        writeln!(f, "Mission feasible : {:>10}", self.feasible)?;
        writeln!(f, "Mission cost     : {:>10.1}", self.cost)?;
        writeln!(f, "Mission chasers  : {:>10}", self.active_chasers)?;
        
        // Per-chaser [min | avg | max| tot] over active chasers
        let active: Vec<usize> = (0..self.total_loads.len())
            .filter(|&i| self.total_loads[i] > 0)
            .collect();
        let stats = |vals: &[f64]| -> (f64, f64, f64, f64) {
            if active.is_empty() {
                return (0.0, 0.0, 0.0, 0.0);
            }
            let (mut min, mut max, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
            for &i in &active {
                min = min.min(vals[i]);
                max = max.max(vals[i]);
                sum += vals[i];
            }
            (min, sum / active.len() as f64, max, sum)
        };
        // Wall-clock mission time per chaser = final meeting epoch.
        let mission_times: Vec<f64> =
            self.meets.iter().map(|m| *m.last().unwrap_or(&0.0)).collect();
        let (t_min, t_avg, t_max, t_tot) = stats(&mission_times);
        let (f_min, f_avg, f_max, f_tot) = stats(&self.total_fuels);
        let loads_f64: Vec<f64> = self.total_loads.iter().map(|&l| l as f64).collect();
        let (l_min, l_avg, l_max, l_tot) = stats(&loads_f64);

        writeln!(
            f,
            "Mission specs    : [{:>10} | {:>10} | {:>10} | {:>10}]",
            "min",
            "avg",
            "max",
            "tot"
        )?;
        writeln!(
            f,
            "     time [  s]  : [{:>10.1} | {:>10.1} | {:>10.1} | {:>10.1}]",
            t_min,
            t_avg,
            t_max,
            t_tot
        )?;
        writeln!(
            f,
            "     fuel [m/s]  : [{:>10.1} | {:>10.1} | {:>10.1} | {:>10.1}]",
            f_min,
            f_avg,
            f_max,
            f_tot
        )?;
        writeln!(
            f,
            "     load [deb]  : [{:>10.0} | {:>10.1} | {:>10.0} | {:>10.0}]",
            l_min,
            l_avg,
            l_max,
            l_tot
        )?;

        writeln!(f)?;

        for id in 0..self.sequences.len() {

            let seq = &self.sequences[id];

            if seq.len() <= 1 {
                continue;
            }

            let n = seq.len();
            writeln!(
                f,
                "Route {} (feasible: {}, cost: {:.1})", 
                id, 
                self.feasibilities[id], 
                self.costs[id]
            )?;
            writeln!(
                f,
                "  {:<8} {:<8} {:>12} {:>12} {:>12}",
                "src",
                "dst",
                "depart [s]",
                "arrive [s]",
                "fuel [m/s]",
            )?;

            let mut cum_fuel = 0.0;

            for i in 1..n {
                let src = if i == 1 {
                    "C   ".to_string()
                } else {
                    format!("D{:<3}", seq[i - 1])
                };
                // Open route: every destination is a debris (no return home).
                let dst = format!("D{:<3}", seq[i]);

                // depart from previous stop = meeting epoch − this leg's transfer.
                let depart = self.meets[id][i] - self.times[id][i];
                let arrive = self.meets[id][i];
                cum_fuel += self.fuels[id][i];
                writeln!(
                    f,
                    "  {:<8} {:<8} {:>12.1} {:>12.1} {:>12.2}",
                    src,
                    dst,
                    depart,
                    arrive,
                    cum_fuel,
                )?;
            }
        }
        writeln!(f, "======================================================================")?;
        
        Ok(())
    }
}


/// Multi-chaser active debris removal solver
pub struct Solver {
    inner: Box<dyn InnerLoop>,
    middle: Box<dyn MiddleLoop>,
    outer: Box<dyn OuterLoop>,
}

impl Solver {
    pub fn new(params: &Params) -> Self {
        Self {
            inner: params.inner.build(),
            middle: params.middle.build(),
            outer: params.outer.build(),
        }
    }

    /// Solve a problem instance
    pub fn solve(&self, problem: &Problem) -> Solution {
        info!("Problem summary");
        info!("  debris         : {}", problem.num_debris());
        info!("  chasers        : {}", problem.num_chasers);
        info!("  max time [  s] : {:.1}", problem.max_time);
        info!("  max fuel [m/s] : {:.2}", problem.max_fuel);
        info!("  max load [deb] : {}", problem.max_load);
        info!("  factor time    : {:.3}", problem.factor_time);
        info!("  factor fuel    : {:.3}", problem.factor_fuel);

        if !problem.has_objective() {
            info!("No objective is set, only searching a feasible solution.");
        }

        let out = self
            .outer
            .solve(self.inner.as_ref(), self.middle.as_ref(), problem);

        if !out.feasible { warn!("no feasible solution found"); }

        Solution::new(&out)
    }
}
