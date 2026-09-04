//! Sequence layer: which debris each chaser collects, and in which order.

pub mod cluster;
pub mod hgs;

pub use cluster::ClusterSequence;
pub use hgs::{Crossover, Generation, HgsConfig, HgsSequence, Ordering};
