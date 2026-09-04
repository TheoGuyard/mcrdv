//! Sequence layer partitioning the catalogue by right ascension in one sweep.

use std::rc::Rc;

use crate::orbit::TAU;
use crate::problem::{ChaserPlan, MissionPlan, Problem, DEPOT};
use crate::solver::{ScheduleLayer, SequenceLayer};

/// Sequence layer based on a capacitated clustering of the debris RAAN.
#[derive(Debug, Clone)]
pub struct ClusterSequence<S: ScheduleLayer> {
    schedule: S,
    problem: Option<Rc<Problem>>,
}

impl<S: ScheduleLayer> ClusterSequence<S> {
    /// Build the layer over a scheduling layer.
    pub fn new(schedule: S) -> Self {
        Self {
            schedule,
            problem: None,
        }
    }

    /// The scheduling layer underneath.
    pub fn schedule(&self) -> &S {
        &self.schedule
    }

    /// The scheduling layer underneath, mutably.
    pub fn schedule_mut(&mut self) -> &mut S {
        &mut self.schedule
    }

    /// Shortest angular distance between the orbital planes of two states \[rad\].
    fn raan_distance(problem: &Rc<Problem>, i: usize, j: usize) -> f64 {
        let d = (problem.states[i].r - problem.states[j].r).abs() % TAU;
        d.min(TAU - d)
    }

    /// Seed `k` medoids spread evenly along the RAAN sweep.
    fn medoids(by_raan: &[usize], k: usize) -> Vec<usize> {
        let n = by_raan.len();
        let mut positions: Vec<usize> = (0..k)
            .map(|s| {
                if s + 1 == k {
                    n - 1
                } else {
                    round_half_even(
                        (n - 1) as f64 * (s as f64) / 
                        ((k - 1).max(1) as f64)
                    )
                }
            })
            .collect();
        positions.sort_unstable();
        positions.dedup();
        positions.into_iter().map(|p| by_raan[p]).collect()
    }

    /// Assign each debris to its nearest medoid, respecting the load limit.
    fn assign(problem: &Rc<Problem>, debris: &[usize], medoids: &[usize]) -> Vec<Vec<usize>> {
        let mut counts = vec![0usize; medoids.len()];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); medoids.len()];
        let mut assigned = vec![false; problem.states.len()];

        // Cheapest (distance, debris, medoid) triples first; ordering by the
        // two indices after the distance keeps ties resolved deterministically.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(debris.len() * medoids.len());
        for &d in debris {
            for (mi, &m) in medoids.iter().enumerate() {
                pairs.push((Self::raan_distance(problem, d, m), d, mi));
            }
        }
        pairs.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });

        for &(_, d, mi) in &pairs {
            if assigned[d] || counts[mi] >= problem.max_load {
                continue;
            }
            assigned[d] = true;
            counts[mi] += 1;
            buckets[mi].push(d);
        }

        // A debris that found every cluster full falls back on the nearest one
        // with room left, or simply the nearest one when none has any.
        for &d in debris {
            if assigned[d] {
                continue;
            }
            let mi = (0..medoids.len())
                .min_by(|&x, &y| {
                    let full = |m: usize| counts[m] >= problem.max_load;
                    full(x)
                        .cmp(&full(y))
                        .then(
                            Self::raan_distance(problem, d, medoids[x])
                                .total_cmp(&Self::raan_distance(problem, d, medoids[y])),
                        )
                })
                .unwrap_or(0);
            assigned[d] = true;
            counts[mi] += 1;
            buckets[mi].push(d);
        }
        buckets
    }

    /// Return exactly `k` buckets, without ever dropping a debris.
    fn pad(mut buckets: Vec<Vec<usize>>, k: usize) -> Vec<Vec<usize>> {
        buckets.retain(|b| !b.is_empty());
        if buckets.len() > k && k > 0 {
            let overflow: Vec<usize> = buckets.split_off(k).into_iter().flatten().collect();
            buckets[k - 1].extend(overflow);
        }
        while buckets.len() < k {
            buckets.push(Vec::new());
        }
        buckets
    }

    /// Schedule every bucket, prepending the departure orbit to each.
    fn finalize(&self, buckets: &[Vec<usize>]) -> MissionPlan {
        let plans = buckets
            .iter()
            .map(|bucket| {
                let mut sequence = Vec::with_capacity(bucket.len() + 1);
                sequence.push(DEPOT);
                sequence.extend_from_slice(bucket);
                (*self.schedule.evaluate(&sequence)).clone()
            })
            .collect();
        MissionPlan::new(plans)
    }
}

impl<S: ScheduleLayer> SequenceLayer for ClusterSequence<S> {
    fn initialize(&mut self, problem: &Rc<Problem>) {
        self.schedule.initialize(problem);
        self.problem = Some(Rc::clone(problem));
    }

    fn evaluate(&mut self) -> MissionPlan {
        let problem = self
            .problem
            .as_ref()
            .expect("initialize() must be called before evaluate()");
        let debris: Vec<usize> = problem.debris().collect();
        if debris.is_empty() {
            return MissionPlan::new(vec![ChaserPlan::empty()]);
        }

        let mut by_raan = debris.clone();
        by_raan.sort_by(|&a, &b| problem.states[a].r.total_cmp(&problem.states[b].r));

        let buckets = if problem.num_chasers >= by_raan.len() {
            by_raan.iter().map(|&d| vec![d]).collect()
        } else {
            let medoids = Self::medoids(&by_raan, problem.num_chasers);
            let mut buckets = Self::assign(problem, &by_raan, &medoids);
            for bucket in &mut buckets {
                bucket.sort_by(|&a, &b| problem.states[a].r.total_cmp(&problem.states[b].r));
            }
            buckets
        };

        self.finalize(&Self::pad(buckets, problem.num_chasers))
    }
}

/// Round to the nearest integer, halves going to the even neighbor.
fn round_half_even(x: f64) -> usize {
    let floor = x.floor();
    let diff = x - floor;
    let up = diff > 0.5 || (diff == 0.5 && (floor as i64) % 2 != 0);
    let n = if up { floor + 1.0 } else { floor };
    n.max(0.0) as usize
}
