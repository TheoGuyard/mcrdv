use core::f64;

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::optim::metric::Metric;
use crate::optim::polisher::Polisher;
use crate::optim::population::Population;
use crate::optim::search::Search;
use crate::optim::sequence::Sequence;
use crate::optim::solution::Solution;
use crate::orbit::Oracle;
use crate::problem::Problem;

/// Parameters for Generator operator
#[derive(Clone)]
pub struct GeneratorParams {
    /// Initial population size
    pub pop_init: usize,
    /// Method to generate initial solutions:
    /// - "random": randomly assign debris to chaser routes
    /// - "greedy": assign debris to chaser routes using nearest-neighbor
    pub generation_method: String,
}

/// Generating operator for creating initial solutions
pub struct Generator {
    pub params: GeneratorParams,
}

impl Generator {
    pub fn new(params: GeneratorParams) -> Self {
        Self { params }
    }

    /// Generate an solution by randomly assigning debris to chaser routes
    fn random_solution(&self, problem: &Problem, oracle: &Oracle, rng: &mut SmallRng) -> Solution {
        let n = problem.states.len();

        // TEMP FIX: Modify oracle strategy to "direct" for init to avoid drift from eating up all the time
        let new_oracle = Oracle::new("direct".to_string(), oracle.max_thrust, oracle.min_sma);
        let oracle = &new_oracle;

        // All debris state indices, except chaser start state at index 0
        let mut remaining: Vec<usize> = (1..n).collect();
        remaining.shuffle(rng);

        let mut sequences: Vec<Sequence> = Vec::new();

        while !remaining.is_empty() {
            let mut indices = vec![0usize];
            let mut time = 0.0;
            let mut load = 0usize;

            // Add debris to the sequence until constraint violation
            let mut i = 0;
            while i < remaining.len() {
                // Check load constraint (whether one more debris can be visited)
                if load >= problem.max_load {
                    break;
                }

                let node = remaining[i];
                let prev = *indices.last().unwrap();

                // Check time constraint (whether we can go to debris and return to depot)
                let (_, tof_to) = oracle.evaluate(
                    &problem.states[prev],
                    &problem.states[node],
                    time,
                    problem.mission_time,
                );
                let (_, tof_back) = oracle.evaluate(
                    &problem.states[node],
                    &problem.states[0],
                    time + tof_to,
                    problem.mission_time,
                );

                if time + tof_to + tof_back > problem.mission_time {
                    i += 1; // Continue checking if other debris can be visited
                    continue;
                }

                indices.push(node);
                time += tof_to;
                load += 1;
                remaining.swap_remove(i);
                // don't increment i: swap_remove brought a new element here
            }

            // Return to depot
            indices.push(0);

            // Create a new sequence
            let seq = Sequence::from_indices(indices, problem, oracle);
            sequences.push(seq);

            // Break if all chasers have been used
            if sequences.len() >= problem.max_chasers {
                break;
            }
        }

        // If there are unassigned debris, dump all of them into the last chaser
        // route and re-evaluate it (the solution will be infeasible)
        if !remaining.is_empty() {
            let last = sequences.last_mut().unwrap();
            let depot = last.indices.pop().unwrap(); // remove trailing depot
            for &node in &remaining {
                last.indices.push(node);
            }
            last.indices.push(depot); // add back depot
            last.evaluate(problem, oracle);
            remaining.clear();
        }

        // If not all chasers were used, pad solution with empty sequences
        while sequences.len() < problem.max_chasers {
            sequences.push(Sequence::empty());
        }

        Solution::new(problem, sequences)
    }

    /// Generate an solution using a greedy nearest-neighbor heuristic
    fn greedy_solution(&self, problem: &Problem, oracle: &Oracle, rng: &mut SmallRng) -> Solution {
        let n = problem.states.len();

        // Modify oracle strategy to "direct" for init
        let new_oracle = Oracle::new("direct".to_string(), oracle.max_thrust, oracle.min_sma);
        let oracle = &new_oracle;

        // All debris state indices, except chaser start state at index 0
        let mut available = vec![true; n];
        available[0] = false;

        // Sort nodes by distance to depot
        let mut sorted_indices: Vec<usize> = (1..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            oracle
                .distance(&problem.states[0], &problem.states[a])
                .partial_cmp(&oracle.distance(&problem.states[0], &problem.states[b]))
                .unwrap()
        });

        // Randomize nodes with similar distance to depot for diversity
        let len = sorted_indices.len();
        for i in 0..len.saturating_sub(1) {
            let end = (i + 4).min(len - 1);
            let j = rng.gen_range(i..=end);
            sorted_indices.swap(i, j);
        }

        let mut sequences: Vec<Sequence> = Vec::new();

        for &start_index in &sorted_indices {
            if !available[start_index] {
                continue;
            }

            // Break if all chasers have been used
            if sequences.len() >= problem.max_chasers {
                break;
            }

            available[start_index] = false;
            let mut indices = vec![0, start_index];
            let (_, mut time) = oracle.evaluate(
                &problem.states[0],
                &problem.states[start_index],
                0.0,
                problem.mission_time,
            );
            let mut load = 1usize;

            // Greedily extend the route with nearest feasible neighbor
            loop {
                // Check load constraint (whether one more debris can be visited)
                if load >= problem.max_load {
                    break;
                }

                let current = *indices.last().unwrap();
                let mut best_index: Option<usize> = None;
                let mut best_cost = f64::INFINITY;
                let mut best_time = f64::INFINITY;

                for &node in &sorted_indices {
                    if !available[node] {
                        continue;
                    }

                    // Check time constraint (whether we can go to debris and return to depot)
                    let (fuel, tof) = oracle.evaluate(
                        &problem.states[current],
                        &problem.states[node],
                        time,
                        problem.mission_time,
                    );
                    let (_, tof_back) = oracle.evaluate(
                        &problem.states[node],
                        &problem.states[0],
                        time + tof,
                        problem.mission_time,
                    );
                    if time + tof + tof_back > problem.mission_time {
                        continue;
                    }

                    // Update best candidate
                    if fuel < best_cost {
                        best_index = Some(node);
                        best_cost = fuel;
                        best_time = tof;
                    }
                }

                // Extend route or break if no feasible neighbor found
                match best_index {
                    Some(node) => {
                        available[node] = false;
                        time += best_time;
                        load += 1;
                        indices.push(node);
                    }
                    None => break,
                }
            }

            // Return to depot
            indices.push(0);

            // Create a new sequence
            let seq = Sequence::from_indices(indices, problem, oracle);
            sequences.push(seq);
        }

        // If there are unassigned debris, dump all of them into the last chaser
        // route and re-evaluate it (the solution will be infeasible)
        let unassigned: Vec<usize> = (1..n).filter(|&i| available[i]).collect();
        if !unassigned.is_empty() {
            let last = sequences.last_mut().unwrap();
            let depot = last.indices.pop().unwrap(); // remove trailing depot
            for &node in &unassigned {
                last.indices.push(node);
            }
            last.indices.push(depot); // add back depot
            last.evaluate(problem, oracle);
        }

        // If not all chasers were used, pad solution with empty sequences
        while sequences.len() < problem.max_chasers {
            sequences.push(Sequence::empty());
        }

        Solution::new(problem, sequences)
    }

    /// Generate an initial solution for the population
    fn generate_solution(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        rng: &mut SmallRng,
    ) -> Solution {
        match self.params.generation_method.as_str() {
            "random" => self.random_solution(problem, oracle, rng),
            "greedy" => self.greedy_solution(problem, oracle, rng),
            _ => panic!(
                "Unknown population initialization method: {}",
                self.params.generation_method
            ),
        }
    }

    /// Generate an initial population of solutions
    #[allow(clippy::too_many_arguments)]
    pub fn initialize(
        &self,
        problem: &Problem,
        oracle: &Oracle,
        metric: &mut Metric,
        population: &mut Population,
        search: &Search,
        polish: &Polisher,
        rng: &mut SmallRng,
    ) {
        for _ in 0..self.params.pop_init {
            let mut solution = self.generate_solution(problem, oracle, rng);
            search.run(&mut solution, problem, oracle, metric, rng);
            polish.polish(&mut solution, "feasible", problem, oracle, metric);
            population.add(solution, metric);
        }
    }
}
