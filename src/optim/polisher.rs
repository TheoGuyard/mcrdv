use crate::optim::metric::Metric;
use crate::optim::sequence::Sequence;
use crate::optim::solution::Solution;
use crate::orbit::Oracle;
use crate::problem::Problem;

/// Parameters for Polisher operator
#[derive(Clone)]
pub struct PolisherParams {
    /// When to polish solutions:
    /// - "never": no polishing
    /// - "feasible": polish all feasible solutions created
    /// - "last": polish last best solution among all iterations
    pub polish_condition: String,
    /// Method to polish solutions:
    /// - "dp": dynamic programming on time grid
    pub polish_method: String,
    /// Time step in dynamic-programming polishing method
    pub dp_time_step: f64,
}

/// Polishing operator for meeting times in solutions
pub struct Polisher {
    pub params: PolisherParams,
}

impl Polisher {
    pub fn new(params: PolisherParams) -> Self {
        Self { params }
    }

    /// Polish meeting times in a solution
    pub fn polish(
        &self,
        solution: &mut Solution,
        condition: &str,
        problem: &Problem,
        oracle: &Oracle,
        metric: &Metric,
    ) {
        if !metric.is_feasible(solution) {
            return;
        }
        if condition != self.params.polish_condition {
            return;
        }
        let mut improved = false;
        for sequence in &mut solution.sequences {
            if sequence.is_empty() {
                continue;
            }
            improved |= match self.params.polish_method.as_str() {
                "dp" => self.polish_dp(sequence, problem, oracle),
                _ => panic!("Unknown polish method: {}", self.params.polish_method),
            };
        }
        if improved {
            solution.recompute_metrics(problem);
        }
    }

    /// Polish meeting times of a sequence using dynamic programming.
    fn polish_dp(&self, seq: &mut Sequence, problem: &Problem, oracle: &Oracle) -> bool {
        let n = seq.len();
        if n <= 2 {
            return false;
        }

        let dt = self.params.dp_time_step;
        let max_time = problem.mission_time;
        let num_steps = (max_time / dt).floor() as usize + 1;

        // Initial sequence cost used for pruning and final comparison
        let initial_cost = seq.cost;

        // memo[i][k] = minimum cum. cost to reach debris i at time step k
        let mut memo = vec![vec![f64::INFINITY; num_steps]; n];

        // back[i][l] = departure time step k at debris i-1
        let mut back = vec![vec![usize::MAX; num_steps]; n];

        // Depart from depot at time 0: cost = 0.0
        memo[0][0] = 0.0;

        // Forward DP pass
        for i in 0..n - 1 {
            let src_idx = seq.indices[i];
            let dst_idx = seq.indices[i + 1];

            for k in 0..num_steps {
                let cost_so_far = memo[i][k];
                if cost_so_far.is_infinite() {
                    continue;
                } // pruning: unreachable
                if cost_so_far >= initial_cost {
                    continue;
                } // pruning: can't improve

                let t_depart = k as f64 * dt;
                let (cost, min_time) = oracle.evaluate(
                    &problem.states[src_idx],
                    &problem.states[dst_idx],
                    t_depart,
                    problem.mission_time,
                );

                let l_min = ((t_depart + min_time) / dt).ceil() as usize;
                if l_min >= num_steps {
                    continue;
                } // pruning: arrival out of range

                let new_cost = cost_so_far + cost;
                if new_cost < memo[i + 1][l_min] {
                    memo[i + 1][l_min] = new_cost;
                    back[i + 1][l_min] = k;
                }
            }

            // Propagation: waiting at intermediate debris positions is free.
            // Skip for the last position (depot return) so that the time step
            // accurately reflects actual arrival time for penalty computation.
            if i + 1 < n - 1 {
                for l in 1..num_steps {
                    if memo[i + 1][l - 1] < memo[i + 1][l] {
                        memo[i + 1][l] = memo[i + 1][l - 1];
                        back[i + 1][l] = back[i + 1][l - 1];
                    }
                }
            }
        }

        // Find the best ending time step at the last position
        let mut best_step = usize::MAX;
        let mut best_cost = f64::INFINITY;
        for (step, cost) in memo[n - 1].iter().enumerate() {
            if cost.is_infinite() {
                continue;
            }
            if *cost < best_cost {
                best_cost = *cost;
                best_step = step;
            }
        }

        if best_step == usize::MAX {
            return false;
        } // no reachable DP state

        // Backtrack to recover optimal departure time steps at each position
        let mut opt_steps = vec![0usize; n];
        opt_steps[n - 1] = best_step;
        for i in (1..n).rev() {
            opt_steps[i - 1] = back[i][opt_steps[i]];
        }

        // Recompute exact times and costs from the optimal departure schedule
        let mut new_times = vec![0.0; n];
        let mut new_costs = vec![0.0; n];
        let mut new_cost = 0.0;

        for i in 1..n {
            let start_time = opt_steps[i - 1] as f64 * dt;
            let (cost, time) = oracle.evaluate(
                &problem.states[seq.indices[i - 1]],
                &problem.states[seq.indices[i]],
                start_time,
                problem.mission_time,
            );
            new_cost += cost;
            new_times[i] = start_time + time;
            new_costs[i] = new_cost;
        }

        let new_time = *new_times.last().unwrap();
        let improved = new_cost < initial_cost;

        debug_assert!(new_time <= problem.mission_time);

        // Update the sequence only if a strictly better cost was found
        if improved {
            seq.times = new_times;
            seq.costs = new_costs;
            seq.cost = new_cost;
            seq.time = new_time;
        }

        improved
    }
}
