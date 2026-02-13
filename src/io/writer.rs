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
        content.push_str(&format!(
            "    total cost      : {:>6.2} km/s\n",
            solution.total_cost / 1000.
        ));
        content.push_str(&format!(
            "    total time      : {:>6.1} days\n",
            solution.total_time() / 86400.
        ));
        content.push_str(&format!(
            "    total load      : {}\n",
            solution.total_load()
        ));
        content.push_str(&format!(
            "    total chasers   : {}\n",
            solution.sequences.len()
        ));

        for (i, seq) in solution.sequences.iter().enumerate() {
            content.push_str(&format!("Chaser {}\n", i + 1));
            content.push_str(&format!(
                "    cost            : {:>6.2} km/s\n",
                seq.cost / 1000.
            ));
            content.push_str(&format!(
                "    time            : {:>6.1} days\n",
                seq.time / 86400.
            ));
            content.push_str(&format!("    load            : {}\n", seq.load));
            content.push_str("    Sequence\n");
            for (j, &idx) in seq.indices.iter().enumerate() {
                if idx == 0 {
                    content.push_str("        depot\n");
                } else {
                    content.push_str(&format!("        debris {}\n", idx + 1));
                }
                content.push_str(&format!(
                    "            meet time : {:>6.2} days\n",
                    seq.times[j] / 86400.
                ));
                content.push_str(&format!(
                    "            meet cost : {:>6.2} m/s\n",
                    seq.costs[j]
                ));
            }
        }

        fs::write(path, content).unwrap();
    }
}
