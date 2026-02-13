use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::optim::solution::Solution;
use crate::optim::metric::Metric;
use crate::optim::sequence::Sequence;
use crate::orbit::Oracle;
use crate::problem::Problem;


/// Local search operator to improve solutions
///
/// Moves implemented:
/// - Intra-route: relocate, swap, 2-opt
/// - Inter-route: relocate/exchange, 2-opt*
pub struct Search {
    /// For each client state i, its nearest neighbors (by static distance proxy).
    neighbors: Vec<Vec<usize>>,
    /// Number of neighbors to consider for proximity-based moves
    nb_neighbors: usize,
}

/// Type of relocate/exchange move performed
#[derive(Clone, Copy)]
enum InterMove { Relocate, Exchange }

impl Search {

    pub fn new (nb_neighbors: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            nb_neighbors,
        }
    }
    
    /// Initialize the local search
    pub fn initialize(&mut self, problem: &Problem, oracle: &Oracle) {
        let n = problem.states.len();
        
        self.neighbors = vec![Vec::new(); n];

        // Evaluate time-independent proximity between states
        for i in 1..n {
            let mut proxi: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
            for j in 1..n {
                if j == i { continue; }
                let d = oracle.distance(&problem.states[i], &problem.states[j]);
                proxi.push((d, j));
            }
            proxi.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let keep = self.nb_neighbors.min(proxi.len());
            self.neighbors[i] = proxi[..keep].iter().map(|&(_, j)| j).collect();
        }
    }

    /// Run local search on an solution until no more improving move found
    pub fn run(
        &self,
        sol: &mut Solution,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        rng: &mut SmallRng,
    ) {
        let n = problem.states.len();
        let k = sol.sequences.len();

        // Track which sequences each state belongs to 
        let mut state_seq = vec![0usize; n];
        let mut state_pos = vec![0usize; n];

        // Track when state was last tests and sequence was last modified
        let mut when_last_tested = vec![0usize; n];
        let mut when_last_modified = vec![0usize; k];
        
        let mut num_moves: usize = 1;

        // Initialize mappings
        Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);

        let mut improved = true;
        let mut loop_id = 0usize;

        while improved {
            improved = false;
            loop_id += 1;

            let mut clients: Vec<usize> = (1..n).collect();
            clients.shuffle(rng);

            for &u in &clients {
                let last_tested = when_last_tested[u];
                when_last_tested[u] = num_moves;

                let su = state_seq[u];
                let pu = state_pos[u];

                // ==================== Inter-route moves ====================
                let neigh_len = self.neighbors[u].len();
                if neigh_len > 0 {
                    let start = rng.gen_range(0..neigh_len);
                    for off in 0..neigh_len {
                        let v = self.neighbors[u][(start + off) % neigh_len];
                        let sv = state_seq[v];
                        if su == sv {
                            continue;
                        }

                        // Skip if both sequences unchanged since last test
                        if when_last_modified[su].max(when_last_modified[sv]) <= last_tested {
                            continue;
                        }

                        let pv = state_pos[v];

                        // Inter-route relocate: move u to after v
                        if self.move_inter_relocate(problem, oracle, metric, sol, su, pu, sv, pv + 1)
                        {
                            num_moves += 1;
                            when_last_modified[su] = num_moves;
                            when_last_modified[sv] = num_moves;
                            Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                            improved = true;
                            break;
                        }

                        // Special case: relocate u when it's right after depot
                        if pu == 1
                            && self.move_inter_relocate(
                                problem, oracle, metric, sol, sv, pv, su, pu
                            )
                        {
                            num_moves += 1;
                            when_last_modified[su] = num_moves;
                            when_last_modified[sv] = num_moves;
                            Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                            improved = true;
                            break;
                        }

                        // 2-opt*
                        if self.move_2opt_star(problem, oracle, metric, sol, su, pu, sv, pv + 1) {
                            num_moves += 1;
                            when_last_modified[su] = num_moves;
                            when_last_modified[sv] = num_moves;
                            Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                            improved = true;
                            break;
                        }
                    }
                }

                // ==================== Empty route moves ====================
                let su = state_seq[u];
                let pu = state_pos[u];
                if loop_id > 1 {
                    if let Some(empty_s) = sol
                        .sequences
                        .iter()
                        .position(|s| s.is_empty())
                    {
                        if empty_s != su {
                            // 2-opt* with empty route
                            if self.move_2opt_star(problem, oracle, metric, sol, su, pu, empty_s, 1) {
                                num_moves += 1;
                                when_last_modified[su] = num_moves;
                                when_last_modified[empty_s] = num_moves;
                                Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                                improved = true;
                                continue;
                            }

                            // Relocate to empty route
                            if self.move_inter_relocate(problem, oracle, metric, sol, su, pu, empty_s, 1) {
                                num_moves += 1;
                                when_last_modified[su] = num_moves;
                                when_last_modified[empty_s] = num_moves;
                                Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                                improved = true;
                                continue;
                            }
                        }
                    }
                }

                // ==================== Intra-route moves ====================
                let su = state_seq[u];
                let pu = state_pos[u];
                if when_last_modified[su] > last_tested {
                    if self.move_intra_relocate(problem, oracle, metric, sol, su, pu) {
                        num_moves += 1;
                        when_last_modified[su] = num_moves;
                        Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                        improved = true;
                    }

                    let pu = state_pos[u];
                    let su = state_seq[u];
                    if self.move_intra_swap(problem, oracle, metric, sol, su, pu) {
                        num_moves += 1;
                        when_last_modified[su] = num_moves;
                        Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                        improved = true;
                    }

                    let pu = state_pos[u];
                    let su = state_seq[u];
                    if self.move_2opt(problem, oracle, metric, sol, su, pu) {
                        num_moves += 1;
                        when_last_modified[su] = num_moves;
                        Self::rebuild_mappings(sol, &mut state_seq, &mut state_pos);
                        improved = true;
                    }
                }
            }
        }

        // Final re-evaluation
        sol.recompute_metrics(problem);
    }

    /// Update state in sequence mappings
    fn rebuild_mappings(
        sol: &Solution,
        state_seq: &mut [usize],
        state_pos: &mut [usize],
    ) {
        for (s, seq) in sol.sequences.iter().enumerate() {
            for (p, &idx) in seq.indices.iter().enumerate() {
                if idx != 0 {
                    state_seq[idx] = s;
                    state_pos[idx] = p;
                }
            }
        }
    }

    // ======================== Intra-route moves ========================

    /// Try to relocate state at pos at every position in its sequence
    fn move_intra_relocate(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        sol: &mut Solution,
        s: usize,
        pos: usize,
    ) -> bool {
        let seq = &sol.sequences[s];
        let len = seq.len();
        if len < 4 || pos == 0 || pos >= len - 1 {
            return false;
        }

        let old_value = metric.sequence_value(problem, seq);
        let mut best_value = old_value;
        let mut best_pos: Option<usize> = None;

        let state = seq.indices[pos];

        for t in 1..len - 1 {
            if t == pos {
                continue;
            }

            // Build candidate: remove state from pos, insert at t
            let mut cand_indices = seq.indices.clone();
            cand_indices.remove(pos);
            let insert_at = if t > pos { t - 1 } else { t };
            cand_indices.insert(insert_at, state);

            let cand = Sequence::from_indices(cand_indices, problem, oracle);
            let cand_value = metric.sequence_value(problem, &cand);

            if cand_value < best_value {
                best_value = cand_value;
                best_pos = Some(t);
            }
        }

        if let Some(t) = best_pos {
            let state = sol.sequences[s].indices.remove(pos);
            let insert_at = if t > pos { t - 1 } else { t };
            sol.sequences[s].indices.insert(insert_at, state);
            sol.sequences[s].evaluate(problem, oracle);
            true
        } else {
            false
        }
    }

    /// Try swapping state at pos with every other state in its sequence
    fn move_intra_swap(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        sol: &mut Solution,
        s: usize,
        pos: usize,
    ) -> bool {
        let seq = &sol.sequences[s];
        let len = seq.len();
        if len < 4 || pos == 0 || pos >= len - 1 {
            return false;
        }

        let old_value = metric.sequence_value(problem, seq);
        let mut best_value = old_value;
        let mut best_pos: Option<usize> = None;

        for t in (pos + 1)..len - 1 {
            let mut cand_indices = seq.indices.clone();
            cand_indices.swap(pos, t);

            let cand = Sequence::from_indices(cand_indices, problem, oracle);
            let cand_value = metric.sequence_value(problem, &cand);

            if cand_value < best_value {
                best_value = cand_value;
                best_pos = Some(t);
            }
        }

        if let Some(t) = best_pos {
            sol.sequences[s].indices.swap(pos, t);
            sol.sequences[s].evaluate(problem, oracle);
            true
        } else {
            false
        }
    }

    /// Try reversing subsegment [i,...,end] for every i starting at pos
    fn move_2opt(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        sol: &mut Solution,
        s: usize,
        pos: usize,
    ) -> bool {
        let seq = &sol.sequences[s];
        let len = seq.len();
        if len < 4 || pos == 0 || pos >= len - 2 {
            return false;
        }

        let old_value = metric.sequence_value(problem, seq);
        let mut best_value = old_value;
        let mut best_end: Option<usize> = None;

        for end in (pos + 1)..len - 1 {
            let mut cand_indices = seq.indices.clone();
            cand_indices[pos..=end].reverse();

            let cand = Sequence::from_indices(cand_indices, problem, oracle);
            let cand_value = metric.sequence_value(problem, &cand);

            if cand_value < best_value {
                best_value = cand_value;
                best_end = Some(end);
            }
        }

        if let Some(end) = best_end {
            sol.sequences[s].indices[pos..=end].reverse();
            sol.sequences[s].evaluate(problem, oracle);
            true
        } else {
            false
        }
    }

    // ======================== Inter-route moves ========================

    /// Try moving state in sequence s1 at position pos1 with state in sequence
    /// s2 at position pos2 or to swap the two states (take best of both moves)
    #[allow(clippy::too_many_arguments)]
    fn move_inter_relocate(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        sol: &mut Solution,
        s1: usize,
        pos1: usize,
        s2: usize,
        pos2: usize,
    ) -> bool {
        debug_assert_ne!(s1, s2);
        let seq1 = &sol.sequences[s1];
        let seq2 = &sol.sequences[s2];

        if pos1 == 0 || pos1 >= seq1.len() - 1 {
            return false;
        }
        if pos2 == 0 || pos2 > seq2.len() - 1 {
            return false;
        }

        let old_value = metric.sequence_value(problem, seq1)
            + metric.sequence_value(problem, seq2);

        let state_u = seq1.indices[pos1];
        let mut best_value = old_value;
        let mut best_move: Option<InterMove> = None;

        // Relocate u from s1 to position pos2 in s2
        {
            let mut new_s1 = seq1.indices.clone();
            new_s1.remove(pos1);
            let mut new_s2 = seq2.indices.clone();
            let ins = pos2.min(new_s2.len() - 1);
            new_s2.insert(ins, state_u);

            let c1 = Sequence::from_indices(new_s1, problem, oracle);
            let c2 = Sequence::from_indices(new_s2, problem, oracle);
            let val = metric.sequence_value(problem, &c1)
                + metric.sequence_value(problem, &c2);

            if val < best_value {
                best_value = val;
                best_move = Some(InterMove::Relocate);
            }
        }

        // Exchange u with the state at pos2 in s2 (if valid and not depot)
        if pos2 < seq2.len() - 1 && seq2.indices[pos2] != 0 {
            let state_v = seq2.indices[pos2];
            let mut new_s1 = seq1.indices.clone();
            new_s1[pos1] = state_v;
            let mut new_s2 = seq2.indices.clone();
            new_s2[pos2] = state_u;

            let c1 = Sequence::from_indices(new_s1, problem, oracle);
            let c2 = Sequence::from_indices(new_s2, problem, oracle);
            let val = metric.sequence_value(problem, &c1)
                + metric.sequence_value(problem, &c2);

            if val < best_value {
                #[allow(unused_assignments)]
                { best_value = val; }
                best_move = Some(InterMove::Exchange);
            }
        }

        match best_move {
            Some(InterMove::Relocate) => {
                let state = sol.sequences[s1].indices.remove(pos1);
                let ins = pos2.min(sol.sequences[s2].indices.len() - 1);
                sol.sequences[s2].indices.insert(ins, state);
                sol.sequences[s1].evaluate(problem, oracle);
                sol.sequences[s2].evaluate(problem, oracle);
                true
            }
            Some(InterMove::Exchange) => {
                let state_u = sol.sequences[s1].indices[pos1];
                let state_v = sol.sequences[s2].indices[pos2];
                sol.sequences[s1].indices[pos1] = state_v;
                sol.sequences[s2].indices[pos2] = state_u;
                sol.sequences[s1].evaluate(problem, oracle);
                sol.sequences[s2].evaluate(problem, oracle);
                true
            }
            None => false,
        }
    }

    /// Swap tails starting at pos1 and pos2 between sequences s1 and s2
    #[allow(clippy::too_many_arguments)]
    fn move_2opt_star(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
        sol: &mut Solution,
        s1: usize,
        pos1: usize,
        s2: usize,
        pos2: usize,
    ) -> bool {
        debug_assert_ne!(s1, s2);
        let seq1 = &sol.sequences[s1];
        let seq2 = &sol.sequences[s2];

        if pos1 == 0 || pos1 >= seq1.len() {
            return false;
        }
        if pos2 == 0 || pos2 >= seq2.len() {
            return false;
        }

        let old_value = metric.sequence_value(problem, seq1)
            + metric.sequence_value(problem, seq2);

        // Build new routes by swapping tails
        let mut new_s1_indices: Vec<usize> = seq1.indices[..pos1].to_vec();
        new_s1_indices.extend_from_slice(&seq2.indices[pos2..]);

        let mut new_s2_indices: Vec<usize> = seq2.indices[..pos2].to_vec();
        new_s2_indices.extend_from_slice(&seq1.indices[pos1..]);

        let c1 = Sequence::from_indices(new_s1_indices.clone(), problem, oracle);
        let c2 = Sequence::from_indices(new_s2_indices.clone(), problem, oracle);
        let new_value = metric.sequence_value(problem, &c1)
            + metric.sequence_value(problem, &c2);

        if new_value < old_value {
            sol.sequences[s1] = c1;
            sol.sequences[s2] = c2;
            true
        } else {
            false
        }
    }
}
