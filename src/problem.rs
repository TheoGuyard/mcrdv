//! The problem instance and the objects a solution is made of.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::orbit::{KepState, DAY};

/// Index of the swarm in a problem state list.
pub const DEPOT: usize = 0;

// ========================================================================= //
// Solution representation
// ========================================================================= //

/// The mission plan of a single chaser.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ChaserPlan {
    /// State indices visited, starting at [`DEPOT`].
    pub sequence: Vec<usize>,
    /// Departure time from each visited state \[s\].
    pub depart: Vec<f64>,
    /// Arrival time at each visited state \[s\].
    pub arrive: Vec<f64>,
    /// Cost of the leg arriving at each visited state.
    pub costs: Vec<f64>,
}

impl ChaserPlan {
    /// Build a chaser plan.
    pub fn new(
        sequence: Vec<usize>,
        depart: Vec<f64>,
        arrive: Vec<f64>,
        costs: Vec<f64>,
    ) -> Self {
        assert!(
            sequence.len() == depart.len()
                && sequence.len() == arrive.len()
                && sequence.len() == costs.len(),
            "chaser plan attributes must have equal length"
        );
        Self {
            sequence,
            depart,
            arrive,
            costs,
        }
    }

    /// Empty chaser plan visiting no debris.
    pub fn empty() -> Self {
        Self {
            sequence: vec![DEPOT],
            depart: vec![0.0],
            arrive: vec![0.0],
            costs: vec![0.0],
        }
    }

    /// Number of states visited, including the depot one.
    #[inline]
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Number of debris collected.
    #[inline]
    pub fn load(&self) -> usize {
        self.sequence.len() - 1
    }

    /// Total cost of the plan.
    #[inline]
    pub fn cost(&self) -> f64 {
        self.costs.iter().sum()
    }

    /// Time at which the chaser reaches its last debris \[s\].
    #[inline]
    pub fn time(&self) -> f64 {
        self.arrive.last().copied().unwrap_or(0.0)
    }

    /// Whether the plan collects no debris.
    ///
    /// A plan always holds the object it starts from, so this asks about the
    /// load rather than about the length of the sequence.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.load() == 0
    }
}

/// Mission plan containing all individual chaser plans.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct MissionPlan {
    /// The individual chaser plans (idle plans included).
    pub plans: Vec<ChaserPlan>,
}

impl MissionPlan {
    /// Build a mission plan.
    pub fn new(plans: Vec<ChaserPlan>) -> Self {
        Self { plans }
    }

    /// Total cost of the mission.
    pub fn cost(&self) -> f64 {
        self.plans.iter().map(ChaserPlan::cost).sum()
    }
}

// ========================================================================= //
// Problem
// ========================================================================= //

/// Problem instance, with depot as first element of the states by convention.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    /// Problem states `[depot, debris_1, ..., debris_n]`.
    pub states: Vec<KepState>,
    /// Number of chasers available in the depot.
    pub num_chasers: usize,
    /// Maximum mission time per chaser \[s\].
    pub max_time: f64,
    /// Maximum mission load per chaser \[num of debris\].
    pub max_load: usize,
}

impl Problem {
    /// Tolerance when judging time feasibility \[s\].
    pub const TIME_TOLERANCE: f64 = 1.0;

    /// Build a problem instance.
    pub fn new(
        states: Vec<KepState>,
        num_chasers: usize,
        max_time: f64,
        max_load: usize,
    ) -> Self {
        if states.len() <= 1 { panic!("'states' must be of length >= 2"); }
        if num_chasers == 0 { panic!("'num_chasers' must be positive"); }
        if max_time <= 0.0 { panic!("'max_time' must be positive"); }
        if !max_time.is_finite() { panic!("'max_time' must be finite"); }
        if max_load == 0 { panic!("'max_load' must be positive"); }
        
        let debris = states.len().saturating_sub(1);
        let totcap = num_chasers.saturating_mul(max_load);
        if totcap < debris { panic!("insufficient capacity over the swarm"); }

        Self {
            states,
            num_chasers,
            max_time,
            max_load,
        }
    }

    /// Number of debris to collect (depot excluded).
    #[inline]
    pub fn num_debris(&self) -> usize {
        self.states.len().saturating_sub(1)
    }

    /// Debris indices.
    pub fn debris(&self) -> std::ops::Range<usize> {
        1..self.states.len()
    }

    /// Whether one chaser plan is respects operational limits.
    pub fn is_plan_feasible(&self, plan: &ChaserPlan) -> bool {
        plan.load() <= self.max_load
            && plan.time() <= self.max_time + Self::TIME_TOLERANCE
            && plan.cost().is_finite()
    }

    /// Whether every plan of a mission respects operational limits.
    pub fn is_feasible(&self, mission: &MissionPlan) -> bool {
        mission.plans.iter().all(|p| self.is_plan_feasible(p))
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Problem")?;
        writeln!(f, "  num_chasers: {}", self.num_chasers)?;
        writeln!(f, "  num_states : {} (including swarm)", self.states.len())?;
        writeln!(f, "  max_time   : {:.2} days", self.max_time / DAY)?;
        writeln!(f, "  max_load   : {} debris", self.max_load)
    }
}
