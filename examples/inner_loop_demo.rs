use scorpion::orbit::Oracle;
use scorpion::orbit::State;

fn main() {
    // Dimensional Keplerian elements
    let ch_kep = State {
        a: 7164040.5518,
        e: 0.0019,
        i: 1.5079,
        O: 2.8765,
        o: 0.8909,
        t: 5.3923,
    };
    let tg_kep = State {
        a: 6989199.3166,
        e: 0.003,
        i: 1.5081,
        O: 2.6417,
        o: 2.1655,
        t: 1.8687,
    };

    let src_time = 0.0;
    let dst_time = 100.0 * 86400.0;
    let spacecraft_accel = 3e-3; // m/s^2
    let min_sma = 6578e3;

    println!("Running transfer optimization demo...");
    println!("Time of flight: {:.3} days", dst_time / 86400.0);
    println!("Spacecraft acceleration: {} m/s^2", spacecraft_accel);
    println!("Minimum SMA: {} km", min_sma / 1000.0);
    println!("------------------------------------");

    // --- Best strategy ---
    let mut oracle = Oracle::new("best".to_string(), spacecraft_accel, min_sma);
    let (dv, dt) = oracle.evaluate(&ch_kep, &tg_kep, src_time, dst_time);

    println!("[Best Strategy]");
    println!("  Delta-V: {:.2} m/s", dv);
    println!("  Delta-t: {:.3} days", dt / 86400.0);
    println!("------------------------------------");

    // --- Drift-only strategy ---
    oracle.strategy = "drift".to_string();
    let (dv, dt) = oracle.evaluate(&ch_kep, &tg_kep, src_time, dst_time);
    println!("[Drift-Only Strategy]");
    println!("  Delta-V: {:.2} m/s", dv);
    println!("  Delta-t: {:.3} days", dt / 86400.0);
    println!("------------------------------------");

    // --- Direct-only strategy ---
    oracle.strategy = "direct".to_string();
    let (dv, dt) = oracle.evaluate(&ch_kep, &tg_kep, src_time, dst_time);
    println!("[Direct-Only Strategy]");
    println!("  Delta-V: {:.2} m/s", dv);
    println!("  Delta-t: {:.3} days", dt / 86400.0);
    println!("------------------------------------");
}
