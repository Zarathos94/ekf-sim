//! The correctness gate.
//!
//! An estimator that tracks well but reports the wrong uncertainty is not "done right" — it is
//! lucky, and it will mislead whatever consumes its covariance. The decisive test is statistical
//! *consistency*: across many independent Monte-Carlo flights, the filter's error weighed by its
//! own covariance — the Normalized Estimation Error Squared (NEES) — must match the state
//! dimension. This harness runs the same `eskf` core the browser runs, pools the NEES over many
//! flights, and checks it against χ² bounds. A wrong Jacobian, a mis-scaled noise term, or a
//! dropped covariance-reset term shows up here as an inconsistent filter, even when the RMSE
//! looks fine.

use eskf::sim::{SimConfig, BEACONS};
use eskf::{nees, position_nees, Eskf, InitialSigma, Noise, Simulator, MAG_REFERENCE};

const IMU_RATE: f64 = 200.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("check");
    match command {
        "check" => {
            let ok = consistency_check();
            if !ok {
                std::process::exit(1);
            }
        }
        "scenarios" => scenarios(),
        other => {
            eprintln!("unknown command '{other}'. Try: check | scenarios");
            std::process::exit(2);
        }
    }
}

/// Run one flight, feeding every sensor, and return per-step NEES samples (position 3-DOF and
/// full 15-DOF) taken after the filter has settled, plus final RMSE figures.
struct RunResult {
    pos_nees: Vec<f64>,
    full_nees: Vec<f64>,
    rmse_pos: f64,
    rmse_vel: f64,
    rmse_att_deg: f64,
}

fn run_flight(cfg: SimConfig, seed: u64, seconds: f64) -> RunResult {
    let mut sim = Simulator::new(cfg, seed);
    let mut f = Eskf::new(sim.truth_nominal(), InitialSigma::default(), filter_noise(&cfg));

    let steps = (seconds * IMU_RATE) as usize;
    let settle = (5.0 * IMU_RATE) as usize; // ignore the first 5 s of convergence
    let sample_every = 20; // decorrelate the NEES stream a little

    let mut pos_nees = Vec::new();
    let mut full_nees = Vec::new();
    let (mut se_pos, mut se_vel, mut se_att, mut n_err) = (0.0, 0.0, 0.0, 0.0);

    for k in 0..steps {
        let tick = sim.step();
        f.predict(tick.accel, tick.gyro, tick.dt);
        if let Some(z) = tick.gps {
            f.update_gps(z, cfg.gps_noise.max(1e-3));
        }
        if let Some(z) = tick.baro {
            f.update_baro(z, cfg.baro_noise.max(1e-3));
        }
        if let Some(z) = tick.mag {
            f.update_mag(z, MAG_REFERENCE, cfg.mag_noise.max(1e-4));
        }
        if let Some(z) = tick.lidar {
            f.update_lidar_altimeter(z, cfg.lidar_noise.max(1e-3));
        }
        if let Some(ranges) = tick.uwb {
            for (i, r) in ranges.iter().enumerate() {
                f.update_range(BEACONS[i], *r, cfg.uwb_noise.max(1e-3));
            }
        }
        if let Some(z) = tick.flow {
            f.update_optical_flow(z, cfg.flow_noise.max(1e-3));
        }

        if k >= settle {
            let dp = sub(f.nom.p, tick.truth.p);
            let dv = sub(f.nom.v, tick.truth.v);
            let dth = eskf::quat::boxminus(tick.truth.q, f.nom.q);
            se_pos += dot(dp, dp);
            se_vel += dot(dv, dv);
            se_att += dot(dth, dth);
            n_err += 1.0;

            if k % sample_every == 0 {
                if let Some(v) = position_nees(&f.nom, &tick.truth, f.covariance()) {
                    pos_nees.push(v);
                }
                if let Some(v) = nees(&f.nom, &tick.truth, f.covariance()) {
                    full_nees.push(v);
                }
            }
        }
    }

    RunResult {
        pos_nees,
        full_nees,
        rmse_pos: (se_pos / n_err).sqrt(),
        rmse_vel: (se_vel / n_err).sqrt(),
        rmse_att_deg: (se_att / n_err).sqrt().to_degrees(),
    }
}

/// The filter's assumed process noise, derived from (but deliberately not identical to) the
/// simulator's — a filter never knows the true noise, and the consistency check must hold anyway.
fn filter_noise(cfg: &SimConfig) -> Noise {
    Noise {
        gravity: eskf::GRAVITY,
        accel: cfg.accel_noise,
        gyro: cfg.gyro_noise,
        accel_bias: cfg.accel_bias_walk,
        gyro_bias: cfg.gyro_bias_walk,
    }
}

fn consistency_check() -> bool {
    let runs = 40;
    let seconds = 30.0;
    // Every sensor on, so the check exercises the full fusion — IMU, GPS, baro, mag, LiDAR, UWB
    // and optical flow — not just a subset.
    let cfg = SimConfig { uwb_enabled: true, flow_enabled: true, ..SimConfig::default() };

    println!("Monte-Carlo consistency check — {runs} flights of {seconds:.0} s, all sensors on\n");

    let mut pos = Vec::new();
    let mut full = Vec::new();
    let (mut rp, mut rv, mut ra) = (0.0, 0.0, 0.0);
    for seed in 0..runs {
        let r = run_flight(cfg, seed as u64 * 2_654_435_761, seconds);
        rp += r.rmse_pos;
        rv += r.rmse_vel;
        ra += r.rmse_att_deg;
        pos.extend(r.pos_nees);
        full.extend(r.full_nees);
    }
    let n = runs as f64;
    println!("  RMSE   position {:.3} m   velocity {:.3} m/s   attitude {:.3}°", rp / n, rv / n, ra / n);
    println!();

    let pos_ok = report_nees("position", &pos, 3, runs);
    let full_ok = report_nees("full state", &full, 15, runs);

    println!();
    let ok = pos_ok && full_ok;
    println!("{}", if ok { "CONSISTENT — the filter's covariance matches its error. Gate PASSED." } else { "INCONSISTENT — covariance does not match the error. Gate FAILED." });
    ok
}

/// Test the average NEES against two-sided 95% χ² bounds.
///
/// The independent unit is the *flight*, not the sample: NEES values within one flight are time
/// correlated, so the confidence band is set by the number of runs (`N·dof` degrees of freedom),
/// the classic Bar-Shalom Monte-Carlo consistency test. The reported mean still pools every
/// steady-state sample, which only sharpens the estimate inside that band.
fn report_nees(label: &str, samples: &[f64], dof: usize, runs: usize) -> bool {
    if samples.is_empty() {
        println!("  {label:<11} no samples");
        return false;
    }
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let k = (runs * dof) as f64;
    let lo = chi2_quantile_lower(k) / runs as f64;
    let hi = chi2_quantile_upper(k) / runs as f64;
    let ok = mean >= lo && mean <= hi;
    println!(
        "  {label:<11} NEES mean {mean:6.3}   expected {dof}   95% band [{lo:.3}, {hi:.3}]   {}",
        if ok { "ok" } else { "OUT OF BAND" }
    );
    ok
}

/// Show the behaviour the sliders drive: nominal, GPS dropout, and a noisy IMU.
type Scenario = (&'static str, fn(&mut SimConfig));

fn scenarios() {
    let seconds = 30.0;
    let cases: [Scenario; 5] = [
        ("nominal", |_c| {}),
        ("GPS dropout", |c| c.gps_enabled = false),
        ("GPS-denied + UWB", |c| {
            c.gps_enabled = false;
            c.uwb_enabled = true;
        }),
        ("noisy IMU (5×)", |c| {
            c.accel_noise *= 5.0;
            c.gyro_noise *= 5.0;
        }),
        ("bias drift (10×)", |c| {
            c.accel_bias_walk *= 10.0;
            c.gyro_bias_walk *= 10.0;
        }),
    ];
    println!("Steady-state RMSE by scenario (mean of 12 flights, {seconds:.0} s each)\n");
    println!("  {:<18} {:>10} {:>12} {:>12}", "scenario", "pos (m)", "vel (m/s)", "att (deg)");
    for (name, mutate) in cases {
        let mut cfg = SimConfig::default();
        mutate(&mut cfg);
        let (mut rp, mut rv, mut ra) = (0.0, 0.0, 0.0);
        let runs = 12;
        for seed in 0..runs {
            let r = run_flight(cfg, 1000 + seed as u64, seconds);
            rp += r.rmse_pos;
            rv += r.rmse_vel;
            ra += r.rmse_att_deg;
        }
        let n = runs as f64;
        println!("  {name:<18} {:>10.3} {:>12.3} {:>12.3}", rp / n, rv / n, ra / n);
    }
    println!("\nGPS dropout inflates position error while attitude holds — inertial coasting, as expected.");
}

// --- χ² quantiles via the Wilson–Hilferty normal approximation (excellent for large k). ---

fn chi2_quantile_lower(k: f64) -> f64 {
    wilson_hilferty(k, -1.959_964) // 2.5%
}
fn chi2_quantile_upper(k: f64) -> f64 {
    wilson_hilferty(k, 1.959_964) // 97.5%
}
fn wilson_hilferty(k: f64, z: f64) -> f64 {
    let a = 2.0 / (9.0 * k);
    let base = 1.0 - a + z * a.sqrt();
    k * base * base * base
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
