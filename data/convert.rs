use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

use scorpion::orbit::MU;

#[derive(Deserialize)]
struct CelestrakRow {
    #[serde(rename = "MEAN_MOTION")]
    mean_motion: f64, // [rev/day]
    #[serde(rename = "ECCENTRICITY")]
    eccentricity: f64, // [prop]
    #[serde(rename = "INCLINATION")]
    inclination: f64, // [deg]
    #[serde(rename = "RA_OF_ASC_NODE")]
    ra_of_asc_node: f64, // [deg]
    #[serde(rename = "ARG_OF_PERICENTER")]
    arg_of_pericenter: f64, // [deg]
    #[serde(rename = "MEAN_ANOMALY")]
    mean_anomaly: f64, // [deg]
}

struct FormattedFloat {
    value: f64,
    precision: usize,
    scientific: bool,
}

impl Serialize for FormattedFloat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = if self.scientific {
            let raw = format!("{:.*e}", self.precision, self.value);
            if let Some(pos) = raw.find('e') {
                let mantissa = &raw[..pos];
                let exponent: i32 = raw[pos + 1..].parse().unwrap();
                format!("{}e{:+03}", mantissa, exponent)
            } else {
                raw
            }
        } else {
            format!("{:.*}", self.precision, self.value)
        };
        serializer.serialize_str(&s)
    }
}

#[derive(Serialize)]
struct KeplerianRow {
    #[serde(rename = "a[m]")]
    semi_major_axis: FormattedFloat, // [m]
    #[serde(rename = "e[prop]")]
    eccentricity: FormattedFloat, // [prop]
    #[serde(rename = "i[rad]")]
    inclination: FormattedFloat, // [rad]
    #[serde(rename = "Omega[rad]")]
    raan: FormattedFloat, // [rad]
    #[serde(rename = "omega[rad]")]
    arg_of_periapsis: FormattedFloat, // [rad]
    #[serde(rename = "theta[rad]")]
    true_anomaly: FormattedFloat, // [rad]
}

fn mean_motion_to_sma(rev_per_day: f64) -> f64 {
    let rad_per_sec = rev_per_day * 2.0 * std::f64::consts::PI / 86400.0;
    (MU / rad_per_sec.powi(2)).powf(1.0 / 3.0)
}

fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

pub fn convert<P: AsRef<Path>>(
    src_path: P,
    dst_path: P,
    precision: usize,
    scientific: bool,
) -> Result<(), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(src_path)?;
    let mut writer = csv::Writer::from_path(dst_path)?;

    let fmt = |value| FormattedFloat {
        value,
        precision,
        scientific,
    };

    for result in reader.deserialize() {
        let input: CelestrakRow = result?;
        let output = KeplerianRow {
            semi_major_axis: fmt(mean_motion_to_sma(input.mean_motion)),
            eccentricity: fmt(input.eccentricity),
            inclination: fmt(deg_to_rad(input.inclination)),
            raan: fmt(deg_to_rad(input.ra_of_asc_node)),
            arg_of_periapsis: fmt(deg_to_rad(input.arg_of_pericenter)),
            true_anomaly: fmt(deg_to_rad(input.mean_anomaly)),
        };
        writer.serialize(output)?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Path to input .csv file in celestrak format
    let src_path = "<input_csv_path>".to_string();

    // Path to output .csv file in scorpion format
    let dst_path = "<output_csv_path>".to_string();

    // Number of decimal places to include in output
    let precision = 9;

    // Whether to use scientific notation in output
    let scientific = true;

    println!("Converting data...");
    println!("  src path  : {}", src_path);
    println!("  dst path  : {}", dst_path);
    println!("  precision : {}", precision);
    println!("  sci. fmt. : {}", scientific);
    convert(src_path, dst_path, precision, scientific)?;

    Ok(())
}
