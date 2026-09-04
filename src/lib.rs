//! `mcrdv` -- Multi-Chaser Rendezvous.
//!
//! Planning software for multi-chaser rendezvous mission.

#![warn(missing_docs)]
#![allow(clippy::manual_is_multiple_of)]

pub mod io;
pub mod orbit;
pub mod problem;
pub mod schedule;
pub mod sequence;
pub mod solver;
pub mod transfer;

pub use io::{format_mission, read_debris, write_mission};
pub use orbit::{centroid, circular_mean, KepState, DAY, MU, TAU};
pub use problem::{ChaserPlan, MissionPlan, Problem, DEPOT};
pub use schedule::DpSchedule;
pub use sequence::{ClusterSequence, Crossover, Generation, HgsConfig, HgsSequence, Ordering};
pub use solver::{ScheduleLayer, SequenceLayer, Solver, TransferLayer};
pub use transfer::QlawTransfer;
