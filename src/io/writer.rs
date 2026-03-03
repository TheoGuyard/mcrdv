use crate::optim::Solution;
use std::fs;

/// Write solution to output .txt file
pub struct Writer;

impl Writer {
    pub fn write(path: &str, solution: &Solution) {
        fs::write(path, solution.to_string()).unwrap();
    }
}
