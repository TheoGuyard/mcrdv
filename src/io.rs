//! Input-output helpers to read debris catalogs and write mission plans.

use std::error::Error;
use std::path::Path;

use crate::orbit::Catalog;
use crate::problem::{MissionPlan, Problem};

/// Expected header of a debris catalog csv file.
const HEADER: [&str; 6] = ["a[m]", "e[prop]", "i[rad]", "r[rad]", "o[rad]", "t[rad]"];

/// Read a debris catalog from a csv file (depot not included).
pub fn read_debris(path: impl AsRef<Path>) -> Result<Catalog, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path.as_ref())?;

    let header: Vec<String> = reader.headers()?.iter().map(str::to_string).collect();
    if header.as_slice() != HEADER {
        return Err(format!("Invalid header, expected: {HEADER:?}.").into());
    }

    let (mut a, mut e, mut i, mut r, mut o, mut t) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for record in reader.records() {
        let record = record?;
        if record.iter().all(|c| c.trim().is_empty()) {
            continue;
        }
        let v: Vec<f64> = record
            .iter()
            .take(6)
            .map(|c| c.trim().parse::<f64>())
            .collect::<Result<_, _>>()?;
        if v.len() < 6 {
            return Err("row has fewer than 6 columns".into());
        }
        a.push(v[0]);
        e.push(v[1]);
        i.push(v[2]);
        r.push(v[3]);
        o.push(v[4]);
        t.push(v[5]);
    }
    Ok(Catalog::new(a, e, i, r, o, t))
}

/// Write a mission plan to a csv file.
pub fn write_mission(
    mission: &MissionPlan,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path.as_ref())?;
    writer.write_record(["chaser", "node", "meet", "wait", "cost"])?;
    for (p, plan) in mission.chaser_plans.iter().enumerate() {
        for k in 0..plan.nodes_indices.len() {
            writer.write_record([
                p.to_string(),
                plan.nodes_indices[k].to_string(),
                format!("{:.*}", 2, plan.meeting_times[k]),
                format!("{:.*}", 2, plan.waiting_times[k]),
                format!("{:.*}", 2, plan.per_leg_costs[k]),
            ])?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Return a human-readable report of a mission plan.
pub fn format_mission(mission: &MissionPlan, problem: &Problem) -> String {
    let width: usize = 70;

    let banner = |title: &str| -> String {
        let pad = width.saturating_sub(title.len() + 2);
        let left = pad / 2;
        let right = pad - left;
        format!("{} {} {}", "=".repeat(left), title, "=".repeat(right))
    };
    let tag = |node: usize| -> String {
        if node == 0 {
            "C".to_string()
        } else {
            format!("D{}", node - 1)
        }
    };

    let used: Vec<&crate::problem::ChaserPlan> =
        mission.chaser_plans.iter().filter(|p| p.load() > 0).collect();

    let mut lines: Vec<String> = vec![banner("Mission plan")];
    lines.push(format!(
        "{:<16} : {:>10}",
        "Mission feasible",
        if mission.feasible { "true" } else { "false" }
    ));
    lines.push(format!("{:<16} : {:>10.1}", "Mission cost", mission.cost));
    lines.push(format!("{:<16} : {:>10}", "Mission chasers", used.len()));

    if !problem.constraints.is_empty() && !used.is_empty() {
        lines.push(format!(
            "{:<16} :  {:>10} | {:>10} | {:>10}",
            "Constraints", "min", "avg", "max"
        ));
        for c in &problem.constraints {
            let vals: Vec<f64> = used.iter().map(|p| c.metric(p)).collect();
            let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let av = vals.iter().sum::<f64>() / vals.len() as f64;
            let head = format!("{} [{:>3}]", c.name(), c.unit());
            let cell = format!(" {mn:>10.1} | {av:>10.1} | {mx:>10.1} ");
            lines.push(format!("{head:>16} : {cell}"));
        }
    }

    for (k, plan) in used.iter().enumerate() {
        let feas = problem.is_feasible(plan);
        lines.push(String::new());
        lines.push(format!(
            "Route {k} (feasible: {}, cost: {:.1})",
            if feas { "true" } else { "false" },
            plan.cost
        ));
        lines.push(format!(
            "  {:<7}{:<9}{:>13}{:>13}{:>14}",
            "src", "dst", "depart", "arrive", "cum. cost"
        ));
        let depart = plan.departure_times();
        let mut cum = 0.0;
        for i in 1..plan.nodes_indices.len() {
            cum += plan.per_leg_costs[i];
            lines.push(format!(
                "  {:<7}{:<9}{:>13.1}{:>13.1}{:>14.2}",
                tag(plan.nodes_indices[i - 1]),
                tag(plan.nodes_indices[i]),
                depart[i - 1],
                plan.meeting_times[i],
                cum
            ));
        }
    }
    lines.push("=".repeat(width));
    lines.join("\n")
}

/// Print a human-readable report of a mission plan to stdout.
pub fn print_mission(mission: &MissionPlan, problem: &Problem) {
    println!("{}", format_mission(mission, problem));
}
