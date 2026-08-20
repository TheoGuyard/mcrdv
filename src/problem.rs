//! Active debris removal mission planning problem definition.

use std::any::Any;
use std::error::Error;

use crate::orbit::Catalog;

// ========================================================================= //
// Mission representation
// ========================================================================= //

/// Mission plan for a single chaser.
#[derive(Debug, Clone, PartialEq)]
pub struct ChaserPlan {
    /// Visited node indices, starting at the depot `0`.
    pub nodes_indices: Vec<usize>,
    /// Meeting epoch at each node [s].
    pub meeting_times: Vec<f64>,
    /// Waiting time held at each node before departure [s].
    pub waiting_times: Vec<f64>,
    /// Cost incurred on the leg arriving at each node.
    pub per_leg_costs: Vec<f64>,
    /// Total plan cost.
    pub cost: f64,
    /// Whether the plan satisfies all constraints.
    pub feasible: bool,
}

impl ChaserPlan {
    /// Build a chaser plan from its fields.
    pub fn new(
        nodes_indices: Vec<usize>,
        meeting_times: Vec<f64>,
        waiting_times: Vec<f64>,
        per_leg_costs: Vec<f64>,
        cost: f64,
        feasible: bool,
    ) -> Self {
        Self {
            nodes_indices,
            meeting_times,
            waiting_times,
            per_leg_costs,
            cost,
            feasible,
        }
    }

    /// Number of collected debris (every visited node but the depot start).
    pub fn load(&self) -> usize {
        self.nodes_indices.len() - 1
    }

    /// Wall-clock mission time, i.e. the final meeting epoch [s].
    pub fn mission_time(&self) -> f64 {
        self.meeting_times.last().copied().unwrap_or(0.0)
    }

    /// Departure epoch from each node (`meeting + waiting`) [s].
    pub fn departure_times(&self) -> Vec<f64> {
        self.meeting_times
            .iter()
            .zip(&self.waiting_times)
            .map(|(m, w)| m + w)
            .collect()
    }

    /// The trivial plan that stays at the depot.
    pub fn empty() -> Self {
        Self {
            nodes_indices: vec![0],
            meeting_times: vec![0.0],
            waiting_times: vec![0.0],
            per_leg_costs: vec![0.0],
            cost: 0.0,
            feasible: true,
        }
    }
}

/// Mission plan for a swarm of chasers.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionPlan {
    /// Per-chaser plans.
    pub chaser_plans: Vec<ChaserPlan>,
    /// Mission cost.
    pub cost: f64,
    /// Mission feasibility.
    pub feasible: bool,
}

impl MissionPlan {
    /// Build a mission plan from its fields.
    pub fn new(chaser_plans: Vec<ChaserPlan>, cost: f64, feasible: bool) -> Self {
        Self {
            chaser_plans,
            cost,
            feasible,
        }
    }

    /// Sorted, de-duplicated collected debris nodes (the depot `0` excluded).
    pub fn collected(&self) -> Vec<usize> {
        let mut nodes: Vec<usize> = self
            .chaser_plans
            .iter()
            .flat_map(|p| p.nodes_indices.iter().copied())
            .filter(|&node| node != 0)
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }
}

// ========================================================================= //
// Constraints
// ========================================================================= //

/// Default tolerance used when checking constraint satisfaction.
pub const DEFAULT_TOL: f64 = 1e-9;

/// Interface for an operational constraint on a chaser plan.
pub trait Constraint {
    /// Short constraint name.
    fn name(&self) -> &'static str {
        "cst"
    }

    /// Unit of the monitored metric.
    fn unit(&self) -> &'static str {
        "---"
    }

    /// The amount by which the plan violates the constraint (>0 if violated).
    fn violation(&self, plan: &ChaserPlan) -> f64;

    /// The monitored plan quantity that this constraint bounds.
    fn metric(&self, plan: &ChaserPlan) -> f64;

    /// Whether the plan satisfies the constraint within tolerance `tol`.
    fn satisfied(&self, plan: &ChaserPlan, tol: f64) -> bool {
        self.violation(plan) <= tol
    }

    /// Downcast hook for the few places that need a concrete constraint type
    /// (the `MaxTime` time-horizon deduction and the `MaxCost` cost cap). This
    /// keeps the trait free of bound-specific methods.
    fn as_any(&self) -> &dyn Any;
}

/// Constraint on the maximum chaser load.
#[derive(Debug, Clone, Copy)]
pub struct MaxLoad {
    pub qmax: usize,
}

impl MaxLoad {
    pub fn new(qmax: usize) -> Self {
        Self { qmax }
    }
}

impl Constraint for MaxLoad {
    fn name(&self) -> &'static str {
        "load"
    }
    fn unit(&self) -> &'static str {
        "num"
    }
    fn metric(&self, plan: &ChaserPlan) -> f64 {
        plan.load() as f64
    }
    fn violation(&self, plan: &ChaserPlan) -> f64 {
        plan.load() as f64 - self.qmax as f64
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Constraint on the maximum chaser mission time.
#[derive(Debug, Clone, Copy)]
pub struct MaxTime {
    pub tmax: f64,
}

impl MaxTime {
    pub fn new(tmax: f64) -> Self {
        Self { tmax }
    }
}

impl Constraint for MaxTime {
    fn name(&self) -> &'static str {
        "time"
    }
    fn unit(&self) -> &'static str {
        "sec"
    }
    fn metric(&self, plan: &ChaserPlan) -> f64 {
        plan.mission_time()
    }
    fn violation(&self, plan: &ChaserPlan) -> f64 {
        plan.mission_time() - self.tmax
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Constraint on the maximum chaser cost.
#[derive(Debug, Clone, Copy)]
pub struct MaxCost {
    pub cmax: f64,
}

impl MaxCost {
    pub fn new(cmax: f64) -> Self {
        Self { cmax }
    }
}

impl Constraint for MaxCost {
    fn name(&self) -> &'static str {
        "cost"
    }
    fn unit(&self) -> &'static str {
        "val"
    }
    fn metric(&self, plan: &ChaserPlan) -> f64 {
        plan.cost
    }
    fn violation(&self, plan: &ChaserPlan) -> f64 {
        plan.cost - self.cmax
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ========================================================================= //
// Objective
// ========================================================================= //

/// Interface for aggregating per-chaser costs into the mission cost.
pub trait Objective {
    /// Human-readable objective name.
    fn name(&self) -> &'static str {
        "obj"
    }

    /// Combine the per-chaser plan costs into a single objective value.
    fn aggregate(&self, costs: &[f64]) -> f64;
}

/// Mission cost expressed as the sum of chaser costs.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectiveSum;

impl Objective for ObjectiveSum {
    fn name(&self) -> &'static str {
        "sum chaser cost"
    }
    fn aggregate(&self, costs: &[f64]) -> f64 {
        costs.iter().sum()
    }
}

/// Mission cost expressed as the maximum of chaser costs.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectiveMax;

impl Objective for ObjectiveMax {
    fn name(&self) -> &'static str {
        "max chaser cost"
    }
    fn aggregate(&self, costs: &[f64]) -> f64 {
        if costs.is_empty() {
            0.0
        } else {
            costs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
    }
}

// ========================================================================= //
// Problem
// ========================================================================= //

/// Multi-chaser active debris removal problem instance.
pub struct Problem {
    /// Orbital states catalog (depot at index `0`).
    pub catalog: Catalog,
    /// Number of available chasers.
    pub num_chasers: usize,
    /// Number of debris to collect.
    pub num_debris: usize,
    /// Mission time horizon [s].
    pub time_horizon: f64,
    /// Operational constraints applied to every chaser plan.
    pub constraints: Vec<Box<dyn Constraint>>,
    /// Mission cost aggregating per-chaser costs.
    pub objective: Box<dyn Objective>,
}

impl Problem {
    /// Build a problem instance.
    ///
    /// The time horizon is taken from the `time_horizon` argument and/or from a
    /// `MaxTime` constraint (the minimum of the two when both are present). An
    /// error is returned if neither defines it.
    pub fn new(
        catalog: Catalog,
        num_chasers: usize,
        time_horizon: Option<f64>,
        constraints: Vec<Box<dyn Constraint>>,
        objective: Option<Box<dyn Objective>>,
    ) -> Result<Self, Box<dyn Error>> {
        let num_debris = catalog.num_debris();
        let objective: Box<dyn Objective> = objective.unwrap_or_else(|| Box::new(ObjectiveSum));

        let time_horizon_from_constraint = constraints
            .iter()
            .filter_map(|c| c.as_any().downcast_ref::<MaxTime>())
            .map(|c| c.tmax)
            .fold(None, |acc: Option<f64>, t| Some(acc.map_or(t, |a| a.min(t))));

        let time_horizon = match (time_horizon, time_horizon_from_constraint) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => { return Err("time horizon not provided".into()) }
        };

        if !time_horizon.is_finite() {
            return Err("time horizon is not finite".into());
        }

        Ok(Self {
            catalog,
            num_chasers,
            num_debris,
            time_horizon,
            constraints,
            objective,
        })
    }

    /// Debris indices in the catalog (depot excluded).
    pub fn debris(&self) -> Vec<usize> {
        (self.catalog.num_depots()..self.catalog.n).collect()
    }

    /// Tightest `MaxCost` upper bound on a feasible plan's cost across the
    /// constraints, or `+inf` if none bounds the cost.
    pub fn cost_cap(&self) -> f64 {
        self.constraints
            .iter()
            .filter_map(|c| c.as_any().downcast_ref::<MaxCost>())
            .map(|c| c.cmax)
            .fold(f64::INFINITY, f64::min)
    }

    /// Per-constraint violations of a plan.
    pub fn plan_violations(&self, plan: &ChaserPlan) -> Vec<f64> {
        self.constraints.iter().map(|c| c.violation(plan)).collect()
    }

    /// Whether a plan satisfies all constraints within tolerance `tol`.
    pub fn is_feasible_tol(&self, plan: &ChaserPlan, tol: f64) -> bool {
        self.constraints.iter().all(|c| c.satisfied(plan, tol))
    }

    /// Whether a plan satisfies all constraints within the default tolerance.
    pub fn is_feasible(&self, plan: &ChaserPlan) -> bool {
        self.is_feasible_tol(plan, DEFAULT_TOL)
    }
}

// ========================================================================= //
// Utilities
// ========================================================================= //

/// Helper for problem construction from basic parameters.
pub fn make_problem(
    debris: &Catalog,
    num_chasers: usize,
    max_time: Option<f64>,
    max_load: Option<usize>,
    max_cost: Option<f64>,
    objective: &str,
) -> Result<Problem, Box<dyn Error>> {
    
    let mut catalog = Catalog::new(
        debris.a.clone(),
        debris.e.clone(),
        debris.i.clone(),
        debris.r.clone(),
        debris.o.clone(),
        debris.t.clone(),
    );
    catalog.add_depot();

    let mut constraints: Vec<Box<dyn Constraint>> = Vec::new();
    if let Some(t) = max_time {
        constraints.push(Box::new(MaxTime::new(t)));
    }
    if let Some(c) = max_cost {
        constraints.push(Box::new(MaxCost::new(c)));
    }
    if let Some(q) = max_load {
        if debris.len() > q * num_chasers {
            return Err(format!("load constraint is trivially infeasible").into());
        }
        constraints.push(Box::new(MaxLoad::new(q)));
    }

    let objective: Box<dyn Objective> = match objective {
        "sum" => Box::new(ObjectiveSum),
        "max" => Box::new(ObjectiveMax),
        other => return Err(format!("unknown objective: {other:?}").into()),
    };

    Problem::new(catalog, num_chasers, None, constraints, Some(objective))
}
