use crate::optim::Solution;
use std::fs;

/// Write solution to output .txt file
pub struct Writer;

impl Writer {
    pub fn write(path: &str, solution: &Solution) {
        let mut content = String::new();

        content.push_str("Solution\n");
        content.push_str(&format!(
            "    feasible        : {}\n",
            solution.is_feasible()
        ));
        content.push_str(&format!("    total cost      : {}\n", solution.total_cost));
        content.push_str(&format!(
            "    total chasers   : {}\n",
            solution.sequences.len()
        ));
        content.push_str(&format!("    max time        : {}\n", solution.max_time()));
        content.push_str(&format!("    min time        : {}\n", solution.min_time()));
        content.push_str(&format!("    avg time        : {}\n", solution.avg_time()));
        content.push_str(&format!("    max load        : {}\n", solution.max_load()));
        content.push_str(&format!("    min load        : {}\n", solution.min_load()));
        content.push_str(&format!("    avg load        : {}\n", solution.avg_load()));

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
