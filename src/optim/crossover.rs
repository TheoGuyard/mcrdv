use rand::Rng;
use rand::rngs::SmallRng;
use std::cmp::min;

use crate::optim::metric::Metric;
use crate::optim::population::Population;
use crate::optim::sequence::Sequence;
use crate::optim::solution::Solution;
use crate::orbit::Oracle;
use crate::problem::Problem;

/// Parameters for Crossover operator
#[derive(Clone)]
pub struct CrossoverParams {
    /// Number of solutions to consider for parent tournament selection
    pub nb_pick: usize,
    /// Factor to relax load constraint for split dynamic programming pruning
    pub load_threshold_factor: f64,
    /// Factor to relax time constraint for split dynamic programming pruning
    pub time_threshold_factor: f64,
}

/// Crossover operator based on giant-tour representation
pub struct Crossover {
    pub params: CrossoverParams,
}

impl Crossover {
    pub fn new(params: CrossoverParams) -> Self {
        Self { params }
    }

    /// Build a giant tour flattening an solution, ordering all sequences by
    /// their average RAAN. Within each sequence, debris indices appear in
    /// their visit order.
    fn build_giant_tour(&self, problem: &Problem, sol: &Solution) -> Vec<usize> {
        // For each sequence: (average RAAN, sequence index)
        let mut keys: Vec<(f64, usize)> = Vec::new();

        for (i, seq) in sol.sequences.iter().enumerate() {
            if seq.is_empty() {
                continue;
            }
            let mut raan = 0.0;
            let mut nseq = 0usize;
            for &idx in &seq.indices[1..seq.indices.len() - 1] {
                if idx != 0 {
                    raan += problem.states[idx].O;
                    nseq += 1;
                }
            }
            let avg_raan = if nseq > 0 { raan / nseq as f64 } else { 0.0 };
            keys.push((avg_raan, i));
        }

        // Sort sequence keys by average RAAN
        keys.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut tour = Vec::new();
        for &(_, r_idx) in &keys {
            let seq = &sol.sequences[r_idx];
            for &idx in &seq.indices[1..seq.indices.len() - 1] {
                debug_assert_ne!(idx, 0); // depot should only appear at start/end
                tour.push(idx);
            }
        }
        tour
    }

    /// Split a giant tour into an solution using dynamic programming to
    /// minimize the overall penalized cost
    fn split_giant_tour(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        giant: &[usize],
        k_goal: usize,
    ) -> Vec<Sequence> {
        let n = giant.len();
        let k = k_goal; // number of sequences to split into

        if n == 0 {
            return (0..k).map(|_| Sequence::empty()).collect();
        }

        // Infinite value with some slack to avoid overflow in addition
        let inf = f64::INFINITY / 4.0;

        // memo[s][j] = best penalized cost to collect first j debris with s sequences
        let mut memo = vec![vec![inf; n + 1]; k + 1];

        // pred[s][j] = index i of last split point for best solution in memo[s][j]
        let mut pred = vec![vec![0usize; n + 1]; k + 1];

        memo[0][0] = 0.0;

        // Constraint thresholds to prune infeasible sequences and reduce search space
        let load_threshold = (self.params.load_threshold_factor * problem.max_load as f64) as usize;
        let time_threshold = self.params.time_threshold_factor * problem.mission_time;

        for s in 1..=k {
            for i in (s - 1)..n {
                let base = memo[s - 1][i];
                if base >= inf {
                    continue;
                }

                // Build candidate depot -> giant[i] -> giant[i+1] -> ... -> depot
                let mut seq = Sequence::open();

                for j in i..n {
                    seq.push(giant[j], problem, oracle);

                    // Add trailing depot before evaluation
                    seq.push(0, problem, oracle);

                    let c = base + metric.sequence_value(problem, &seq);
                    if c < memo[s][j + 1] {
                        memo[s][j + 1] = c;
                        pred[s][j + 1] = i;
                    }

                    // Remove trailing depot after evaluation
                    seq.pop_last();

                    // Prune if constraints are violated above pruning thresholds
                    if seq.load > load_threshold {
                        break;
                    }
                    if seq.time > time_threshold {
                        break;
                    }
                }
            }
        }

        // Find the best number of sequences (allow fewer than k if lower cost)
        let mut best_k = k;
        let mut best_c = memo[k][n];
        for (s, memo_s) in memo.iter().enumerate().take(k).skip(1) {
            if memo_s[n] < best_c {
                best_c = memo_s[n];
                best_k = s;
            }
        }

        // Fallback if split fails: put all debris in one sequence. This can
        // happen, but means that problem constraints are tight.
        if best_c >= inf {
            let mut indices = vec![0];
            indices.extend_from_slice(giant);
            indices.push(0);
            let seq = Sequence::from_indices(indices, problem, oracle);
            let mut result = vec![seq];
            while result.len() < k {
                result.push(Sequence::empty());
            }
            return result;
        }

        // Backtrack to recover sequences start and end points
        let mut bounds: Vec<(usize, usize)> = Vec::with_capacity(best_k);
        let mut j = n;
        for s in (1..=best_k).rev() {
            let i = pred[s][j];
            bounds.push((i, j));
            j = i;
        }
        bounds.reverse();

        // Build sequences from start and end points in giant tour
        let mut sequences: Vec<Sequence> = Vec::with_capacity(k);
        for (start, end) in bounds {
            let mut indices = vec![0];
            indices.extend_from_slice(&giant[start..end]);
            indices.push(0);
            let seq = Sequence::from_indices(indices, problem, oracle);
            sequences.push(seq);
        }

        // Pad with empty sequences if not all k sequences are used
        while sequences.len() < k {
            sequences.push(Sequence::empty());
        }

        sequences
    }

    /// Order crossover (OX) on two giant tours
    fn crossover_ox(&self, parent1: &[usize], parent2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
        let n = parent1.len();

        debug_assert!(n > 1);
        debug_assert_eq!(n, parent2.len());

        let mut child = vec![0usize; n];
        let max_index = *parent1.iter().max().unwrap_or(&0);
        let mut added = vec![false; max_index + 1];

        // Pick two cut points (ensure they are different)
        let start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        while end == start {
            end = rng.gen_range(0..n);
        }

        // Copy indices in segment [start, end] or [end, start] from parent1
        let stop = (end + 1) % n;
        let mut j = start;
        loop {
            let pos = j % n;
            let idx = parent1[pos];
            child[pos] = idx;
            added[idx] = true;
            j += 1;
            if j % n == stop {
                break;
            }
        }

        // Fill remaining indices from parent2 in order, skipping already used ones
        let mut pos = stop;
        for t in 0..n {
            let idx = parent2[(stop + t) % n];
            if !added[idx] {
                child[pos] = idx;
                added[idx] = true;
                pos = (pos + 1) % n;
            }
        }

        child
    }

    pub fn generate(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        population: &Population,
        rng: &mut SmallRng,
    ) -> Solution {
        // Select two parents (clone to release borrow on population)
        let p1 = population.pick(self.params.nb_pick, rng).clone();
        let p2 = population.pick(self.params.nb_pick, rng).clone();

        // Extract giant tours
        let t1 = self.build_giant_tour(problem, &p1);
        let t2 = self.build_giant_tour(problem, &p2);

        // Add extra sequence with some probability to encourage diversity
        let extra: usize = if rng.gen_bool(0.1) { 1 } else { 0 };
        let k_goal = min(p1.nb_sequences.max(1) + extra, problem.max_chasers);

        // Perform crossover on giant tours
        let tour = self.crossover_ox(&t1, &t2, rng);

        // Split back the giant tour into sequences for the solution
        let sequences = self.split_giant_tour(problem, oracle, metric, &tour, k_goal);

        Solution::new(problem, sequences)
    }
}
