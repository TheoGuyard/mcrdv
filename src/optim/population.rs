use rand::Rng;
use rand::rngs::SmallRng;
use std::collections::VecDeque;

use crate::optim::metric::Metric;
use crate::optim::solution::Solution;

#[derive(Clone)]
pub struct PopulationParams {
    /// Minimum population size after survival selection
    pub pop_min: usize,
    /// Maximum population size until survival selection
    pub pop_max: usize,
    /// Number of closest solutions used for diversity measurement
    pub nb_close: usize,
    /// Number of elite solutions preserved during survival selection
    pub nb_elite: usize,
    /// Number of insertions before penalty adaptation
    pub adapt_iter: usize,
    /// Threshold for clone detection in survivor selection
    pub clone_eps: f64,
}

/// Feasible or infeasible subpopulation with score and diversity metrics
#[derive(Default)]
pub struct Subpopulation {
    /// Subpopulation solutions
    pub solutions: Vec<Solution>,
    /// Sorted list of (distance, index) to neighbors for each solution
    pub proxi: Vec<Vec<(f64, usize)>>,
    /// Solution score based on cost and diversity (smaller is better)
    pub score: Vec<f64>,
    /// Solution indices sorted by increasing value (smaller is better)
    pub order: Vec<usize>,
}

impl Subpopulation {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Population of solutions with insertion and survival selection mechanisms
pub struct Population {
    /// Parameters
    pub params: PopulationParams,
    /// Feasible solutions
    pub feasible: Subpopulation,
    /// Infeasible solutions
    pub infeasible: Subpopulation,
    /// Rolling window of load feasibility status for new solutions
    pub load_window: VecDeque<bool>,
    /// Rolling window of time feasibility status for new solutions
    pub time_window: VecDeque<bool>,
}

impl Population {
    pub fn new(params: PopulationParams) -> Self {
        Self {
            params,
            feasible: Subpopulation::new(),
            infeasible: Subpopulation::new(),
            load_window: VecDeque::new(),
            time_window: VecDeque::new(),
        }
    }

    /// Add solution to population
    pub fn add(&mut self, sol: Solution, metric: &mut Metric) {
        // Select appropriate subpopulation
        let sub = if sol.is_feasible() {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };

        // Record solution feasibility status in rolling windows
        self.load_window.push_back(sol.load_excess == 0);
        self.time_window.push_back(sol.time_excess == 0.0);

        // Insert solution and update subpopulation attributes
        let idx = sub.solutions.len();
        sub.solutions.push(sol);
        Self::update_proxi(sub, idx);
        Self::update_order(sub, metric);
        Self::update_score(sub, self.params.nb_close, self.params.nb_elite);

        // Trigger survivor selection
        if sub.solutions.len() > self.params.pop_max {
            Self::survivors_selection(
                sub,
                metric,
                self.params.pop_min,
                self.params.nb_close,
                self.params.nb_elite,
                self.params.clone_eps,
            );
        }

        // Trigger penalty adaptation and reorder subpopulations
        debug_assert_eq!(self.load_window.len(), self.time_window.len());
        if self
            .load_window
            .len()
            .is_multiple_of(self.params.adapt_iter)
        {
            let frac_load_feasible = self
                .load_window
                .iter()
                .rev()
                .take(self.params.adapt_iter)
                .filter(|&&b| b)
                .count() as f64
                / self.params.adapt_iter as f64;
            let frac_time_feasible = self
                .time_window
                .iter()
                .rev()
                .take(self.params.adapt_iter)
                .filter(|&&b| b)
                .count() as f64
                / self.params.adapt_iter as f64;
            metric.adapt_penalties(frac_load_feasible, frac_time_feasible);
            Self::update_order(sub, metric);
            Self::update_order(sub, metric);
        }
    }

    /// Pick an solution via tournament over nb_pick candidates
    pub fn pick<'a>(&'a self, nb_pick: usize, rng: &mut SmallRng) -> &'a Solution {
        let fpop_n = self.feasible.solutions.len();
        let ipop_n = self.infeasible.solutions.len();
        let tpop_n = fpop_n + ipop_n;

        assert!(tpop_n >= 1, "Need at least 1 solution for tournament");
        let k = nb_pick.max(1).min(tpop_n);

        // Utility to select a random solution from the combined population
        let draw = |rng: &mut SmallRng| -> (bool, usize, f64) {
            let g = rng.gen_range(0..tpop_n);
            if g < fpop_n {
                (true, g, self.feasible.score[g])
            } else {
                let i = g - fpop_n;
                (false, i, self.infeasible.score[i])
            }
        };

        // Select best among k random candidates, prioritizing feasibility
        // first and diversity score second
        let (mut best_f, mut best_i, mut best_r) = draw(rng);
        for _ in 1..k {
            let (f, i, r) = draw(rng);
            if r < best_r {
                best_f = f;
                best_i = i;
                best_r = r;
            }
        }

        // Return reference to best candidate
        if best_f {
            &self.feasible.solutions[best_i]
        } else {
            &self.infeasible.solutions[best_i]
        }
    }

    /// Return reference to the best solution, prioritizing feasible ones
    pub fn best(&self) -> &Solution {
        if !self.feasible.solutions.is_empty() {
            &self.feasible.solutions[self.feasible.order[0]]
        } else if !self.infeasible.solutions.is_empty() {
            &self.infeasible.solutions[self.infeasible.order[0]]
        } else {
            panic!("Population is empty");
        }
    }

    /// Return mut reference to the best solution, prioritizing feasible ones
    pub fn best_mut(&mut self) -> &mut Solution {
        if !self.feasible.solutions.is_empty() {
            &mut self.feasible.solutions[self.feasible.order[0]]
        } else if !self.infeasible.solutions.is_empty() {
            &mut self.infeasible.solutions[self.infeasible.order[0]]
        } else {
            panic!("Population is empty");
        }
    }

    /// Return reference to the best feasible solution, if any
    pub fn best_feasible(&self) -> Option<&Solution> {
        if !self.feasible.solutions.is_empty() {
            Some(&self.feasible.solutions[self.feasible.order[0]])
        } else {
            None
        }
    }

    /// Return reference to the best infeasible solution, if any
    pub fn best_infeasible(&self) -> Option<&Solution> {
        if !self.infeasible.solutions.is_empty() {
            Some(&self.infeasible.solutions[self.infeasible.order[0]])
        } else {
            None
        }
    }

    // ======================== Internal helpers ========================

    /// Reduce subpopulation via survivor selection
    fn survivors_selection(
        sub: &mut Subpopulation,
        metric: &Metric,
        pop_min: usize,
        nb_close: usize,
        nb_elite: usize,
        clone_eps: f64,
    ) {
        while sub.solutions.len() > pop_min {
            let idx = Self::worst_index(sub, clone_eps);
            Self::remove_at(sub, idx);
            Self::update_order(sub, metric);
            Self::update_score(sub, nb_close, nb_elite);
        }
    }

    /// Find the solution with worst score, prioritizing clones
    fn worst_index(sub: &Subpopulation, clone_eps: f64) -> usize {
        let mut worst_idx = 0;
        let mut worst_is_clone = sub.proxi[0].first().is_some_and(|&(d, _)| d <= clone_eps);
        let mut worst_rank = sub.score[0];

        for i in 1..sub.solutions.len() {
            let is_clone = sub.proxi[i].first().is_some_and(|&(d, _)| d <= clone_eps);
            let score = sub.score[i];

            if (is_clone && !worst_is_clone) || (is_clone == worst_is_clone && score > worst_rank) {
                worst_is_clone = is_clone;
                worst_rank = score;
                worst_idx = i;
            }
        }
        worst_idx
    }

    /// Update proximity metric when a new solution is added to subpopulation
    fn update_proxi(sub: &mut Subpopulation, new_idx: usize) {
        // Should always the called with the last index
        debug_assert_eq!(sub.proxi.len(), new_idx);

        let m = sub.solutions.len();
        sub.proxi.push(Vec::with_capacity(m.saturating_sub(1)));

        // Insert new solution in proximity lists of existing solutions
        for i in 0..new_idx {
            let dist = Metric::broken_pairs_distance(&sub.solutions[new_idx], &sub.solutions[i]);
            let pos = sub.proxi[i].partition_point(|(dd, _)| *dd <= dist);
            sub.proxi[i].insert(pos, (dist, new_idx));

            // Directly update new solution proximity to avoid double loop
            sub.proxi[new_idx].push((dist, i));
        }

        // Sort proximity list of new solution
        sub.proxi[new_idx].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    /// Remove solution at index `idx` and fix proximity/ranking lists
    fn remove_at(sub: &mut Subpopulation, idx: usize) {
        let last = sub.solutions.len() - 1;
        sub.solutions.swap_remove(idx);
        sub.proxi.swap_remove(idx);
        sub.score.swap_remove(idx);
        for list in sub.proxi.iter_mut() {
            list.retain(|&(_, j)| j != idx);
            if last != idx {
                for pair in list.iter_mut() {
                    if pair.1 == last {
                        pair.1 = idx;
                    }
                }
            }
        }
    }

    /// Update solution ordering based on penalized cost
    fn update_order(sub: &mut Subpopulation, metric: &Metric) {
        sub.order.clear();
        sub.order.extend(0..sub.solutions.len());

        let values: Vec<f64> = sub
            .solutions
            .iter()
            .map(|solution| metric.value(solution))
            .collect();

        sub.order
            .sort_unstable_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
    }

    /// Update solution ranking based on value order and diversity
    fn update_score(sub: &mut Subpopulation, nb_close: usize, nb_elite: usize) {
        let n = sub.solutions.len();
        sub.score.resize(n, 0.0);

        if n == 0 {
            return;
        }
        if n == 1 {
            sub.score[0] = 0.0;
            return;
        }

        let nb_close = nb_close.min(n - 1);
        let denom = (n - 1) as f64;

        // Average distance to nb_close nearest neighbors
        let mut avg_dist = vec![0.0; n];
        for (i, avg_dist_i) in avg_dist.iter_mut().enumerate().take(n) {
            let neighbors = &sub.proxi[i];
            let k = nb_close.min(neighbors.len());
            let mut sum = 0.0;
            for (dist, _) in neighbors.iter().take(k) {
                sum += dist;
            }
            *avg_dist_i = if k > 0 { sum / k as f64 } else { 0.0 };
        }

        // Diversity ranking (larger distance to neighbors ==> better ranking)
        let mut div_pairs: Vec<(f64, usize)> = (0..n).map(|i| (-avg_dist[i], i)).collect();
        div_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut div_rank = vec![0.0; n];
        for (pos, &(_, idx)) in div_pairs.iter().enumerate() {
            div_rank[idx] = pos as f64 / denom;
        }

        // Cost ordering ranking (smaller cost ==> better ranking)
        let mut cost_pos = vec![0usize; n];
        for (pos, &idx) in sub.order.iter().enumerate() {
            cost_pos[idx] = pos;
        }
        let ord_rank: Vec<f64> = cost_pos.iter().map(|&p| p as f64 / denom).collect();

        // Ranking: elite only use cost score, others mix it with diversity score
        let scale = 1.0 - (nb_elite as f64) / (n as f64);
        for i in 0..n {
            if cost_pos[i] < nb_elite {
                sub.score[i] = ord_rank[i];
            } else {
                sub.score[i] = ord_rank[i] + scale * div_rank[i];
            }
        }
    }
}
