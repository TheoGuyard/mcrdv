use crate::optim::solution::Solution;
use crate::optim::sequence::Sequence;
use crate::problem::Problem;


/// Metric to compare two different solutions (feasible or infeasible)
pub struct Metric {
    /// Load violation penalty value
    pub penalty_load: f64,
    /// Load violation penalty increase factor
    pub penalty_load_increase: f64,
    /// Load violation penalty decrease factor
    pub penalty_load_decrease: f64,
    /// Load violation penalty minimum value
    pub penalty_load_min: f64,
    /// Load violation penalty maximum value
    pub penalty_load_max: f64,
    
    /// Time violation penalty value
    pub penalty_time: f64,
    /// Time violation penalty increase factor
    pub penalty_time_increase: f64,
    /// Time violation penalty decrease factor
    pub penalty_time_decrease: f64,
    /// Time violation penalty minimum value
    pub penalty_time_min: f64,
    /// Time violation penalty maximum value
    pub penalty_time_max: f64,
    
    /// Target fraction of feasible solutions for penalty adaptation
    pub target_ratio: f64,
}

impl Metric {

    #![allow(clippy::too_many_arguments)]
    pub fn new(
        penalty_load: f64,
        penalty_load_increase: f64,
        penalty_load_decrease: f64,
        penalty_load_min: f64,
        penalty_load_max: f64,
        penalty_time: f64,
        penalty_time_increase: f64,
        penalty_time_decrease: f64,
        penalty_time_min: f64,
        penalty_time_max: f64,
        target_ratio: f64
    ) -> Self {
        Self {
            penalty_load,
            penalty_load_increase,
            penalty_load_decrease,
            penalty_load_min,
            penalty_load_max,
            penalty_time,
            penalty_time_increase,
            penalty_time_decrease,
            penalty_time_min,
            penalty_time_max,
            target_ratio,
        }
    }

    /// Penalty of a sequence for constraint violations
    #[inline]
    pub fn sequence_penalty(&self, problem: &Problem, seq: &Sequence) -> f64 {
        let load_viol = seq.load.saturating_sub(problem.max_load) as f64;
        let time_viol = (seq.time - problem.mission_time).max(0.0);
        self.penalty_load * load_viol + self.penalty_time * time_viol
    }

    /// Value of a sequence (raw cost + penalty for constraint violations)
    #[inline]
    pub fn sequence_value(&self, problem: &Problem, seq: &Sequence) -> f64 {
        seq.cost + self.sequence_penalty(problem, seq)
    }

    /// Feasibility status of a sequence
    #[inline]
    pub fn sequence_is_feasible(&self, problem: &Problem, seq: &Sequence) -> bool {
        self.sequence_penalty(problem, seq) == 0.0
    }

    /// Penalty of an solution for constraint violations
    #[inline]
    pub fn penalty(&self, sol: &Solution) -> f64 {
        self.penalty_load * (sol.load_excess as f64) + self.penalty_time * sol.time_excess
    }

    /// Value of an solution (raw cost + penalty for constraint violations)
    #[inline]
    pub fn value(&self, sol: &Solution) -> f64 {
        sol.total_cost + self.penalty(sol)
    }

    /// Feasibility status of a sequence
    #[inline]
    pub fn is_feasible(&self, sol: &Solution) -> bool {
        self.penalty(sol) == 0.0
    }

    /// Check whether `a` is strictly better than `b` (feasibility then value)
    #[inline]
    pub fn better_than(&self, a: &Solution, b: &Solution) -> bool {
        let fa = a.is_feasible();
        let fb = b.is_feasible();
        match (fa, fb) {
            (true, false) => true,
            (false, true) => false,
            _ => {
                // If both feasible or infeasible: tie break on value
                self.value(a) < self.value(b)
            }
        }
    }

    /// Adapt penalties based on the fraction of feasible solutions
    pub fn adapt_penalties(
        &mut self,
        frac_load_feasible: f64,
        frac_time_feasible: f64
    ) {
        // Load penalty
        if frac_load_feasible < self.target_ratio {
            self.penalty_load *= self.penalty_load_increase;
        } else {
            self.penalty_load *= self.penalty_load_decrease;
        }
        self.penalty_load = self.penalty_load.clamp(self.penalty_load_min, self.penalty_load_max);

        // Time penalty
        if frac_time_feasible < self.target_ratio {
            self.penalty_time *= self.penalty_time_increase;
        } else {
            self.penalty_time *= self.penalty_time_decrease;
        }
        self.penalty_time = self.penalty_time.clamp(self.penalty_time_min, self.penalty_time_max);
    }

    /// Broken-pairs distance between two solutions
    pub fn broken_pairs_distance(a: &Solution, b: &Solution) -> f64 {
        debug_assert_eq!(a.pred.len(), b.pred.len());
        
        let num_debris = a.pred.len() - 1;
        if num_debris == 0 { return 0.0; }

        let mut differences = 0usize;
        for j in 1..=num_debris {
            // Check if successor of j in a exists in b
            if a.succ[j] != b.succ[j] && a.succ[j] != b.pred[j] {
                differences += 1;
            }
            // Check depot adjacency differences
            if a.pred[j] == 0 && b.pred[j] != 0 && b.succ[j] != 0 {
                differences += 1;
            }
        }
        (differences as f64) / (num_debris as f64)
    }
}
