use std::fs;
use crate::optim::Solution;

/// Write solution to output .txt file
pub struct Writer;

impl Writer {
    
    pub fn write(path: &str, solution: &Solution) {
        let mut content = String::new();

        content.push_str("Solution\n");
        content.push_str(&format!("    feasible        : {}\n", solution.is_feasible()));
        content.push_str(&format!("    total cost      : {}\n", solution.total_cost));
        content.push_str(&format!("    total time      : {}\n", solution.total_time()));
        content.push_str(&format!("    total load      : {}\n", solution.total_load()));
        content.push_str(&format!("    total chasers   : {}\n", solution.sequences.len()));

        for (i, seq) in solution.sequences.iter().enumerate() {
            content.push_str(&format!("Chaser {}\n", i + 1));
            content.push_str(&format!("    cost            : {}\n", seq.cost));
            content.push_str(&format!("    time            : {}\n", seq.time));
            content.push_str(&format!("    load            : {}\n", seq.load));
            content.push_str("    Sequence\n");
            for (j, &idx) in seq.indices.iter().enumerate() {
                if idx == 0 {
                    content.push_str("        depot\n");
                } else { 
                    content.push_str(&format!("        debris {}\n", idx + 1)); 
                }
                content.push_str(&format!("            meet time : {}\n", seq.times[j]));
                content.push_str(&format!("            meet cost : {}\n", seq.costs[j]));
            }
        }

        fs::write(path, content).unwrap();
    }
}