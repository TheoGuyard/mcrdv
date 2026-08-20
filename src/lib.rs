//! scorpion -- Active debris removal mission planner in Rust.
//!
//! - [`orbit`]: orbital state representation and utilities.
//! - [`problem`]: problem definition with generic objective and constraints.
//! - [`solver`]: abstract layer interfaces (transfer / schedule / sequence).
//! - [`transfer`]: concrete transfer layer (Q-law estimator).
//! - [`schedule`]: concrete schedule layer (dynamic-programming scheduler).
//! - [`sequence`]: concrete sequence layer (Hybrid Genetic Search).
//! - [`io`]: helpers to read debris catalogs and write/format mission plans.

pub mod io;
pub mod orbit;
pub mod problem;
pub mod schedule;
pub mod sequence;
pub mod solver;
pub mod transfer;
