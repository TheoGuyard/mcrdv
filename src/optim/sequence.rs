use crate::orbit::Oracle;
use crate::problem::Problem;

/// Sequence representing the debris visited by a chaser
#[derive(Clone, Debug)]
pub struct Sequence {
    /// Ordered list of debris indices (first/last are 0: the chaser depot)
    pub indices: Vec<usize>,
    /// Cumulative arrival time at each debris [seconds]
    pub times: Vec<f64>,
    /// Transfer cost to reach each debris from its predecessor
    pub costs: Vec<f64>,
    /// Cost of the sequence
    pub cost: f64,
    /// Time of the sequence [seconds]
    pub time: f64,
    /// Number of debris visited (excluding chaser depot visits)
    pub load: usize,
}

impl Sequence {
    /// Empty sequence with depot-to-depot trip only
    pub fn empty() -> Self {
        Self {
            indices: vec![0, 0],
            times: vec![0.0, 0.0],
            costs: vec![0.0, 0.0],
            cost: 0.0,
            time: 0.0,
            load: 0,
        }
    }

    /// Open sequence starting from depot (no closing depot yet)
    pub fn open() -> Self {
        Self {
            indices: vec![0],
            times: vec![0.0],
            costs: vec![0.0],
            cost: 0.0,
            time: 0.0,
            load: 0,
        }
    }

    /// Extend the sequence by appending a debris index
    pub fn push(&mut self, idx: usize, problem: &Problem, oracle: &Oracle) {
        let prev = *self.indices.last().unwrap();
        let (cost, time) = oracle.evaluate(&problem.states[prev], &problem.states[idx], self.time);
        self.cost += cost;
        self.time += time;
        self.indices.push(idx);
        self.times.push(self.time);
        self.costs.push(self.cost);
        if idx != 0 {
            self.load += 1;
        }
    }

    /// Remove the last debris from the sequence
    pub fn pop_last(&mut self) {
        if let Some(idx) = self.indices.pop() {
            self.times.pop();
            self.costs.pop();
            self.cost = *self.costs.last().unwrap();
            self.time = *self.times.last().unwrap();
            if idx != 0 {
                self.load -= 1;
            }
        }
    }

    /// Sequence built from a list of debris indices (must start/end at depot index 0)
    pub fn from_indices(indices: Vec<usize>, problem: &Problem, oracle: &Oracle) -> Self {
        debug_assert!(indices.len() >= 2);
        debug_assert_eq!(indices[0], 0);
        debug_assert_eq!(*indices.last().unwrap(), 0);

        let mut seq = Self {
            indices,
            times: Vec::new(),
            costs: Vec::new(),
            cost: 0.0,
            time: 0.0,
            load: 0,
        };
        seq.evaluate(problem, oracle);
        seq
    }

    /// Number of positions visited in the sequence (including depot visits)
    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the sequence is empty (depot-to-depot only)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() <= 2
    }

    /// Evaluate the sequence attributes from scratch
    pub fn evaluate(&mut self, problem: &Problem, oracle: &Oracle) {
        let n = self.indices.len();
        self.costs = vec![0.0; n];
        self.times = vec![0.0; n];
        self.cost = 0.0;
        self.time = 0.0;
        self.load = 0;

        for i in 1..n {
            let src = self.indices[i - 1];
            let dst = self.indices[i];
            let (cost, time) =
                oracle.evaluate(&problem.states[src], &problem.states[dst], self.time);

            self.time += time;
            self.cost += cost;
            self.times[i] = self.time;
            self.costs[i] = self.cost;

            if dst != 0 {
                self.load += 1;
            }
        }
    }
}
