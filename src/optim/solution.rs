use serde::{Deserialize, Serialize};
use std::fmt;

use crate::optim::sequence::Sequence;
use crate::problem::Problem;

/// Solution in the population corresponding to a candidate solution
///
/// An solution is a collection of one sequence per available chaser,
/// possibly empty (depot-to-depot) if the chaser is unused.
#[derive(Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Sequence for each chaser
    pub sequences: Vec<Sequence>,
    /// Total cost across all sequences
    pub total_cost: f64,
    /// Total chaser capacity violation over all sequences
    pub load_excess: usize,
    /// Total chaser time violation over all sequences
    pub time_excess: f64,
    /// Number of non-empty routes
    pub nb_sequences: usize,
    /// Predecessor of each state: `pred[i]` = state index visited before `i`
    pub pred: Vec<usize>,
    /// Successor of each state: `succ[i]` = state index visited after `i`
    pub succ: Vec<usize>,
}

impl Solution {
    pub fn new(problem: &Problem, sequences: Vec<Sequence>) -> Self {
        let n = problem.states.len();
        let mut sol = Self {
            sequences,
            total_cost: 0.0,
            load_excess: 0,
            time_excess: 0.0,
            nb_sequences: 0,
            pred: vec![0; n],
            succ: vec![0; n],
        };
        sol.recompute_metrics(problem);

        sol
    }

    /// Recompute all solution metrics (except penalized cost)
    pub fn recompute_metrics(&mut self, problem: &Problem) {
        self.total_cost = 0.0;
        self.load_excess = 0;
        self.time_excess = 0.0;
        self.nb_sequences = 0;

        for v in self.pred.iter_mut() {
            *v = 0;
        }
        for v in self.succ.iter_mut() {
            *v = 0;
        }

        for seq in &self.sequences {
            self.total_cost += seq.cost;
            self.load_excess += seq.load.saturating_sub(problem.max_load); // avoids usize overflow
            self.time_excess += (seq.time - problem.mission_time).max(0.0);
            if !seq.is_empty() {
                self.nb_sequences += 1;
                for i in 1..seq.indices.len() - 1 {
                    let node = seq.indices[i];
                    if node == 0 {
                        continue;
                    }
                    self.pred[node] = seq.indices[i - 1];
                    self.succ[node] = seq.indices[i + 1];
                }
            }
        }
    }

    /// Whether the solution satisfies time and load constraints
    #[inline]
    pub fn is_feasible(&self) -> bool {
        self.load_excess == 0 && self.time_excess == 0.0
    }

    /// Sum of number of debris collected over all chasers
    #[inline]
    pub fn total_load(&self) -> usize {
        if self.nb_sequences > 0 {
            self.sequences.iter().map(|s| s.load).sum()
        } else {
            0
        }
    }

    /// Maximum load collected among chasers
    #[inline]
    pub fn max_load(&self) -> usize {
        self.sequences.iter().map(|s| s.load).max().unwrap_or(0)
    }

    /// Minimum load collected among chasers
    #[inline]
    pub fn min_load(&self) -> usize {
        self.sequences.iter().map(|s| s.load).min().unwrap_or(0)
    }

    /// Average load per used chaser
    #[inline]
    pub fn avg_load(&self) -> f64 {
        if self.nb_sequences > 0 {
            self.total_load() as f64 / self.nb_sequences as f64
        } else {
            0.0
        }
    }

    /// Sum of mission time over all chasers
    #[inline]
    pub fn total_time(&self) -> f64 {
        if self.nb_sequences > 0 {
            self.sequences.iter().map(|s| s.time).sum()
        } else {
            0.0
        }
    }

    /// Maximum mission time among chasers
    #[inline]
    pub fn max_time(&self) -> f64 {
        self.sequences
            .iter()
            .map(|s| s.time)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Minimum mission time among chasers
    #[inline]
    pub fn min_time(&self) -> f64 {
        self.sequences
            .iter()
            .map(|s| s.time)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Average mission time per used chaser
    #[inline]
    pub fn avg_time(&self) -> f64 {
        if self.nb_sequences > 0 {
            self.total_time() / self.nb_sequences as f64
        } else {
            0.0
        }
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mission plan")?;
        writeln!(f, "  feasible   : {}", self.is_feasible())?;
        writeln!(f, "  chasers    : {}", self.nb_sequences)?;
        writeln!(f, "  total cost : {:.2}", self.total_cost)?;
        writeln!(f, "  total time : {:.0} sec", self.total_time())?;
        writeln!(f, "  total load : {}", self.total_load())?;
        for (i, seq) in self.sequences.iter().enumerate() {
            if !seq.is_empty() {
                // Go back to initial indices without depot
                write!(f, "  Chaser {} → [", i + 1)?;
                for (pos, &idx) in seq.indices.iter().enumerate() {
                    if idx == 0 {
                        write!(f, "depot")?;
                    } else {
                        write!(f, "{}", idx + 1)?;
                    }
                    if pos < seq.indices.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                writeln!(f, "]")?;

                writeln!(f, "     cost : {:.2}", seq.cost)?;
                writeln!(f, "     time : {:.0} sec", seq.time)?;
                if i != self.sequences.iter().len() - 1 {
                    writeln!(f, "     load : {}", seq.load)?;
                } else {
                    write!(f, "     load : {}", seq.load)?;
                }
            }
        }
        Ok(())
    }
}
