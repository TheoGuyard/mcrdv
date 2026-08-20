//! One-shot clustering sequencer (capacitated clustering by RAAN).

use std::any::Any;
use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::problem::{ChaserPlan, MaxLoad, MissionPlan, Problem};
use crate::solver::{ScheduleLayer, SequenceLayer, TransferLayer};
use crate::transfer::qlaw::linspace;

/// Assign debris to chasers by clustering on RAAN, one cluster per chaser.
pub struct ClusterSequencer<'p> {
    problem: &'p Problem,
    #[allow(dead_code)]
    transfer: &'p dyn TransferLayer,
    schedule: &'p dyn ScheduleLayer,
}

impl<'p> ClusterSequencer<'p> {
    pub fn new(
        problem: &'p Problem,
        transfer: &'p dyn TransferLayer,
        schedule: &'p dyn ScheduleLayer,
    ) -> Self {
        Self { problem, transfer, schedule }
    }

    /// The depot node index.
    fn depot(&self) -> usize {
        0
    }

    /// The tightest `MaxLoad` cap, or `usize::MAX` when the load is unbounded.
    fn max_load(&self) -> usize {
        self.problem
            .constraints
            .iter()
            .filter_map(|c| c.as_any().downcast_ref::<MaxLoad>())
            .map(|c| c.qmax)
            .min()
            .unwrap_or(usize::MAX)
    }

    /// Retrieve RAAN from state index.
    fn raan(&self, i: usize) -> f64 {
        self.problem.catalog.r[i]
    }

    /// Shortest angular distance between two state RAAN values [rad].
    fn raan_distance(&self, i: usize, j: usize) -> f64 {
        let d = (self.raan(i) - self.raan(j)).abs().rem_euclid(TAU);
        d.min(TAU - d)
    }

    /// Debris indices sorted by RAAN, tie based on debris order.
    fn sweep(&self) -> Vec<usize> {
        let mut debris = self.problem.debris();
        debris.sort_by(|&a, &b| self.raan(a).total_cmp(&self.raan(b)));
        debris
    }

    /// Seed `k` medoids evenly along the RAAN sweep.
    fn medoids(&self, by_raan: &[usize], k: usize) -> Vec<usize> {
        let n = by_raan.len();
        let mut grid = linspace(0.0, (n - 1) as f64, k);
        if let Some(last) = grid.last_mut() {
            *last = (n - 1) as f64;
        }
        let mut idx: Vec<usize> = grid
            .into_iter()
            .map(|x| round_half_even(x) as usize)
            .collect();
        idx.sort_unstable();
        idx.dedup();
        idx.into_iter().map(|i| by_raan[i]).collect()
    }

    /// Nearest-medoid assignment in RAAN, respecting the load capacity.
    fn assign(&self, debris: &[usize], medoids: &[usize]) -> Vec<Vec<usize>> {
        let cap = self.max_load();
        let mut counts = vec![0usize; medoids.len()];
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); medoids.len()];
        let mut assigned: HashMap<usize, usize> = HashMap::new();

        // All (distance, debris, medoid) triples, cheapest first. Ties break on
        // the debris index then the medoid index, matching Python's tuple sort.
        let mut pairs: Vec<(f64, usize, usize)> = debris
            .iter()
            .flat_map(|&d| {
                medoids
                    .iter()
                    .enumerate()
                    .map(move |(mi, &m)| (d, mi, m))
            })
            .map(|(d, mi, m)| (self.raan_distance(d, m), d, mi))
            .collect();
        pairs.sort_by(|a, b| {
            a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
        });

        for (_, d, mi) in pairs {
            if assigned.contains_key(&d) || counts[mi] >= cap {
                continue;
            }
            assigned.insert(d, mi);
            counts[mi] += 1;
            buckets[mi].push(d);
        }

        // Debris that hit every capacity: drop into the nearest medoid with room.
        for &d in debris {
            if assigned.contains_key(&d) {
                continue;
            }
            let mi = (0..medoids.len())
                .min_by(|&a, &b| {
                    let (fa, fb) = (counts[a] >= cap, counts[b] >= cap);
                    fa.cmp(&fb).then(
                        self.raan_distance(d, medoids[a])
                            .total_cmp(&self.raan_distance(d, medoids[b])),
                    )
                })
                .expect("`assign` requires at least one medoid");
            counts[mi] += 1;
            buckets[mi].push(d);
        }

        buckets
    }

    /// Exactly `k` buckets, never dropping a debris (overflow merged into the
    /// last kept bucket).
    fn pad(&self, buckets: Vec<Vec<usize>>, k: usize) -> Vec<Vec<usize>> {
        let mut buckets: Vec<Vec<usize>> =
            buckets.into_iter().filter(|b| !b.is_empty()).collect();
        if buckets.len() > k && k > 0 {
            let overflow: Vec<usize> = buckets.split_off(k).into_iter().flatten().collect();
            buckets[k - 1].extend(overflow);
        }
        while buckets.len() < k {
            buckets.push(Vec::new());
        }
        buckets
    }

    /// Cost every bucket through the schedule layer and aggregate the mission.
    fn finalize(&self, buckets: &[Vec<usize>]) -> MissionPlan {
        let plans: Vec<ChaserPlan> = buckets
            .iter()
            .map(|b| {
                let mut seq = Vec::with_capacity(b.len() + 1);
                seq.push(self.depot());
                seq.extend_from_slice(b);
                (*self.schedule.evaluate(&seq)).clone()
            })
            .collect();
        let costs: Vec<f64> = plans.iter().map(|p| p.cost).collect();
        let cost = self.problem.objective.aggregate(&costs);
        let feasible = plans.iter().all(|p| p.feasible);
        MissionPlan::new(plans, cost, feasible)
    }
}

impl<'p> SequenceLayer for ClusterSequencer<'p> {
    fn initialize(&self) {}

    fn evaluate(&self) -> MissionPlan {
        if self.problem.num_debris == 0 {
            return MissionPlan::new(
                vec![ChaserPlan::empty()],
                0.0,
                true
            );
        }

        if self.problem.num_chasers == 0 {
            return MissionPlan::new(
                Vec::new(),
                self.problem.objective.aggregate(&[]), 
                false
            );
        }

        let by_raan = self.sweep();

        let buckets = if self.problem.num_chasers >= by_raan.len() {
            by_raan.iter().map(|&d| vec![d]).collect()
        } else {
            let medoids = self.medoids(&by_raan, self.problem.num_chasers);
            let mut buckets = self.assign(&by_raan, &medoids);
            for bucket in &mut buckets {
                bucket.sort_by(|&a, &b| self.raan(a).total_cmp(&self.raan(b)));
            }
            buckets
        };

        self.finalize(&self.pad(buckets, self.problem.num_chasers))
    }

    fn statistics(&self) -> HashMap<String, Box<dyn Any>> {
        HashMap::new()
    }
}

/// Round to the nearest integer, ties to the even neighbour.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let frac = x - floor;
    if frac > 0.5 {
        floor + 1.0
    } else if frac < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}
