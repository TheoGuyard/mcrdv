//! IO utilities for orbital states and mission plans.

use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::orbit::KepState;
use crate::problem::{MissionPlan, Problem};

/// Header format of CSV state file.
pub const CSV_HEADER: &str = "a[m],e[prop],i[rad],r[rad],o[rad],t[rad]";

/// Read a CSV state file, each row given one state in Keplerian elements.
pub fn read_debris<P: AsRef<Path>>(path: P) -> Vec<KepState> {
    let path = path.as_ref();
    let show = path.display();
    let file = fs::File::open(path)
        .unwrap_or_else(|e| panic!("cannot open '{show}': {e}"));
    
    let mut lines = BufReader::new(file).lines().enumerate();

    let header = match lines.next() {
        Some((_, header)) => header
            .unwrap_or_else(|e| panic!("cannot read '{show}': {e}")),
        None => String::new(),
    };
    if header.trim() != CSV_HEADER {
        panic!("invalid csv header, expected: {CSV_HEADER}");
    }

    let mut debris = Vec::new();
    for (index, line) in lines {
        let line = line.unwrap_or_else(|e| panic!("cannot read '{show}': {e}"));
        if line.trim().is_empty() { continue; }
        let number = index + 1;
        let mut values = Vec::with_capacity(6);
        for field in line.split(',').take(6) {
            let field = field.trim();
            match field.parse::<f64>() {
                Ok(value) => values.push(value),
                Err(_) => panic!(
                    "{show}, line {number}: cannot parse '{field}' as a number"
                ),
            }
        }
        if values.len() < 6 {
            panic!(
                "{show}, line {number}: expected 6 columns, found {}", 
                values.len()
            );
        }
        debris.push(
            KepState::new(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
            )
        );
    }
    debris
}

/// Write a mission plan as one row per visited object.
pub fn write_mission<P: AsRef<Path>>(mission: &MissionPlan, path: P, precision: usize) {
    let mut text = String::from("chaser,node,depart,arrive,cost\n");
    for (k, plan) in mission.plans.iter().enumerate() {
        for i in 0..plan.len() {
            let _ = writeln!(
                text,
                "{},{},{:.*},{:.*},{:.*}",
                k,
                plan.sequence[i],
                precision,
                plan.depart[i],
                precision,
                plan.arrive[i],
                precision,
                plan.costs[i]
            );
        }
    }
    let path = path.as_ref();
    fs::write(path, text)
        .unwrap_or_else(|e| panic!("cannot write '{}': {e}", path.display()));
}

/// Render a mission plan as a human-readable string.
pub fn format_mission(problem: &Problem, mission: &MissionPlan) -> String {
    const WIDTH: usize = 58;
    let side = "=".repeat(WIDTH / 2 - 7);
    let used = mission.plans.iter().filter(|p| p.load() > 0).count();

    let mut lines = Vec::new();
    lines.push(format!("{side} Mission plan {side}"));
    lines.push(format!("Feasible     : {}", problem.is_feasible(mission)));
    lines.push(format!("Total cost   : {:.1}", mission.cost()));
    lines.push(format!("Chasers used : {}/{}", used, problem.num_chasers));

    for (p, plan) in mission.plans.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!(
            "Chaser {} (feasible: {}, cost: {:.1})",
            p,
            problem.is_plan_feasible(plan),
            plan.cost()
        ));
        lines.push(format!(
            "  {:<7}{:<9}{:>13}{:>13}{:>14}",
            "src", "dst", "arrive", "depart", "leg cost"
        ));

        let last = plan.len() - 1;
        for i in 0..plan.len() {
            let (src, dst, arrive, depart, cost) = if i == last {
                (
                    plan.sequence[i].to_string(),
                    "--".to_string(), 
                    format!("{:.1}", plan.arrive[i]), 
                    "--".to_string(),
                    plan.costs[i],
                )
            } else if i == 0 {
                (
                    plan.sequence[i].to_string(),
                    plan.sequence[1].to_string(), 
                    "--".to_string(),
                    format!("{:.1}", plan.depart[0]),
                    plan.costs[i],
                )
            } else {
                (
                    plan.sequence[i].to_string(),
                    plan.sequence[i + 1].to_string(),
                    format!("{:.1}", plan.arrive[i]),
                    format!("{:.1}", plan.depart[i]),
                    plan.costs[i],
                )
            };
            lines.push(format!(
                "  {:<7}{:<9}{:>13}{:>13}{:>14.2}",
                src, dst, arrive, depart, cost
            ));
        }
    }
    lines.push("=".repeat(WIDTH));
    lines.join("\n")
}
