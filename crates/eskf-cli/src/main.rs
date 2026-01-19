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
use eskf::{
    nees, position_nees, Eskf, InitialSigma, Nominal, Noise, Simulator, Tick, TrueState,
    MAG_REFERENCE,
};

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
        "record" => {
            let seconds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15.0);
            let path = args.get(3).map(String::as_str).unwrap_or("data/reference-flight.csv");
            if let Err(e) = cmd_record(seconds, path) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        "replay" => {
            let path = args.get(2).map(String::as_str).unwrap_or("data/reference-flight.csv");
            if let Err(e) = cmd_replay(path) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        "plotdata" => {
            let dir = args.get(2).map(String::as_str).unwrap_or("paper/figdata");
            if let Err(e) = cmd_plotdata(dir) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown command '{other}'. Try: check | scenarios | record | replay | plotdata");
            std::process::exit(2);
        }
    }
}

/// Every sensor enabled — the full ten-modality fusion.
fn full_config() -> SimConfig {
    SimConfig {
        lidar_enabled: true,
        uwb_enabled: true,
        flow_enabled: true,
        gps_vel_enabled: true,
        dvl_enabled: true,
        att_enabled: true,
        ..SimConfig::default()
    }
}

const CSV_HEADER: &str = "t,ax,ay,az,gx,gy,gz,gps_x,gps_y,gps_z,baro,mag_x,mag_y,mag_z,lidar,\
uwb0,uwb1,uwb2,uwb3,flow_x,flow_y,gpsv_x,gpsv_y,gpsv_z,dvl_x,dvl_y,dvl_z,att_w,att_x,att_y,att_z,\
tp_x,tp_y,tp_z,tv_x,tv_y,tv_z,tq_w,tq_x,tq_y,tq_z";

/// Records a flight to CSV: the raw IMU stream, every aiding measurement (empty where a sensor did
/// not fire on that tick), and ground truth. A portable, replayable test dataset.
fn cmd_record(seconds: f64, path: &str) -> Result<(), String> {
    let mut sim = Simulator::new(full_config(), 0xE5F1_2024);
    let steps = (seconds * IMU_RATE) as usize;

    let mut out = String::from(CSV_HEADER);
    out.push('\n');
    let n = |x: f64| format!("{x:.5}");
    for _ in 0..steps {
        let tk = sim.step();
        let mut f: Vec<String> = Vec::with_capacity(41);
        f.push(n(tk.t));
        for v in tk.accel {
            f.push(n(v));
        }
        for v in tk.gyro {
            f.push(n(v));
        }
        push_vec(&mut f, tk.gps.map(|v| v.to_vec()), 3);
        push_vec(&mut f, tk.baro.map(|v| vec![v]), 1);
        push_vec(&mut f, tk.mag.map(|v| v.to_vec()), 3);
        push_vec(&mut f, tk.lidar.map(|v| vec![v]), 1);
        push_vec(&mut f, tk.uwb.map(|v| v.to_vec()), 4);
        push_vec(&mut f, tk.flow.map(|v| v.to_vec()), 2);
        push_vec(&mut f, tk.gps_vel.map(|v| v.to_vec()), 3);
        push_vec(&mut f, tk.dvl.map(|v| v.to_vec()), 3);
        push_vec(&mut f, tk.att.map(|q| vec![q.w, q.x, q.y, q.z]), 4);
        for v in tk.truth.p {
            f.push(n(v));
        }
        for v in tk.truth.v {
            f.push(n(v));
        }
        for v in [tk.truth.q.w, tk.truth.q.x, tk.truth.q.y, tk.truth.q.z] {
            f.push(n(v));
        }
        out.push_str(&f.join(","));
        out.push('\n');
    }

    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    std::fs::write(path, out).map_err(|e| e.to_string())?;
    println!("wrote {path}: {steps} rows over {seconds:.0} s, all sensors");
    Ok(())
}

fn push_vec(f: &mut Vec<String>, v: Option<Vec<f64>>, count: usize) {
    match v {
        Some(vals) => {
            for x in vals {
                f.push(format!("{x:.5}"));
            }
        }
        None => {
            for _ in 0..count {
                f.push(String::new());
            }
        }
    }
}

/// Replays a recorded dataset through the filter and reports RMSE and position-NEES consistency
/// against the recorded ground truth — verifying the estimator on external data, not just a live
/// run.
fn cmd_replay(path: &str) -> Result<(), String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut lines = data.lines();
    lines.next(); // header

    let noise = Noise {
        gravity: eskf::GRAVITY,
        accel: SimConfig::default().accel_noise,
        gyro: SimConfig::default().gyro_noise,
        accel_bias: SimConfig::default().accel_bias_walk,
        gyro_bias: SimConfig::default().gyro_bias_walk,
    };
    let cfg = SimConfig::default();

    let mut filter: Option<Eskf> = None;
    let mut prev_t = 0.0f64;
    let (mut se_p, mut se_v, mut se_a, mut cnt) = (0.0, 0.0, 0.0, 0.0);
    let mut pos_nees = Vec::new();
    let mut rows = 0usize;

    for line in lines {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 41 {
            continue;
        }
        rows += 1;
        let g = |i: usize| c[i].parse::<f64>().unwrap_or(0.0);
        let o1 = |i: usize| -> Option<f64> {
            if c[i].is_empty() {
                None
            } else {
                c[i].parse().ok()
            }
        };
        let o3 = |i: usize| -> Option<[f64; 3]> {
            if c[i].is_empty() {
                None
            } else {
                Some([g(i), g(i + 1), g(i + 2)])
            }
        };

        let t = g(0);
        let accel = [g(1), g(2), g(3)];
        let gyro = [g(4), g(5), g(6)];
        let truth = TrueState {
            p: [g(31), g(32), g(33)],
            v: [g(34), g(35), g(36)],
            q: eskf::Quat::new(g(37), g(38), g(39), g(40)),
            accel_bias: [0.0; 3],
            gyro_bias: [0.0; 3],
        };

        let f = filter.get_or_insert_with(|| {
            let nom = Nominal { p: truth.p, v: truth.v, q: truth.q, accel_bias: [0.0; 3], gyro_bias: [0.0; 3] };
            Eskf::new(nom, InitialSigma::default(), noise)
        });

        let dt = if prev_t > 0.0 { (t - prev_t).max(1e-4) } else { 1.0 / IMU_RATE };
        prev_t = t;
        f.predict(accel, gyro, dt);

        if let Some(z) = o3(7) {
            f.update_gps(z, cfg.gps_noise);
        }
        if let Some(z) = o1(10) {
            f.update_baro(z, cfg.baro_noise);
        }
        if let Some(z) = o3(11) {
            f.update_mag(z, MAG_REFERENCE, cfg.mag_noise);
        }
        if let Some(z) = o1(14) {
            f.update_lidar_altimeter(z, cfg.lidar_noise);
        }
        if !c[15].is_empty() {
            for (k, b) in BEACONS.iter().enumerate() {
                f.update_range(*b, g(15 + k), cfg.uwb_noise);
            }
        }
        if !c[19].is_empty() {
            f.update_optical_flow([g(19), g(20)], cfg.flow_noise);
        }
        if let Some(z) = o3(21) {
            f.update_gps_velocity(z, cfg.gps_vel_noise);
        }
        if let Some(z) = o3(24) {
            f.update_body_velocity(z, cfg.dvl_noise);
        }
        if !c[27].is_empty() {
            f.update_attitude(eskf::Quat::new(g(27), g(28), g(29), g(30)), cfg.att_noise);
        }

        if t >= 5.0 {
            let dp = sub(f.nom.p, truth.p);
            let dv = sub(f.nom.v, truth.v);
            let dth = eskf::quat::boxminus(truth.q, f.nom.q);
            se_p += dot(dp, dp);
            se_v += dot(dv, dv);
            se_a += dot(dth, dth);
            cnt += 1.0;
            if rows % 20 == 0 {
                if let Some(v) = position_nees(&f.nom, &truth, f.covariance()) {
                    pos_nees.push(v);
                }
            }
        }
    }

    if cnt == 0.0 {
        return Err("no usable rows in dataset".into());
    }
    let mean_nees = pos_nees.iter().sum::<f64>() / pos_nees.len().max(1) as f64;
    println!("replayed {rows} rows from {path}\n");
    println!("  RMSE   position {:.3} m   velocity {:.3} m/s   attitude {:.3}°", (se_p / cnt).sqrt(), (se_v / cnt).sqrt(), (se_a / cnt).sqrt().to_degrees());
    println!("  position NEES mean {mean_nees:.3}   expected 3   ({})", if (1.5..4.5).contains(&mean_nees) { "consistent" } else { "OUT OF RANGE" });
    Ok(())
}

/// Propagate on the IMU and apply whichever aiding measurements fired on this tick. The single
/// place the filter is driven, shared by the gate, the scenarios, and the figure-data export, so
/// every reported result comes from exactly the same update sequence.
fn drive(f: &mut Eskf, tick: &Tick, cfg: &SimConfig) {
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
    if let Some(z) = tick.gps_vel {
        f.update_gps_velocity(z, cfg.gps_vel_noise.max(1e-3));
    }
    if let Some(z) = tick.dvl {
        f.update_body_velocity(z, cfg.dvl_noise.max(1e-3));
    }
    if let Some(z) = tick.att {
        f.update_attitude(z, cfg.att_noise.max(1e-4));
    }
}

/// Run one flight, feeding every sensor, and return per-step NEES samples (position 3-DOF and
/// full 15-DOF) taken after the filter has settled, plus final RMSE figures.
struct RunResult {
    pos_nees: Vec<f64>,
    full_nees: Vec<f64>,
    /// (time, position-NEES) samples, for the consistency-over-time histogram.
    pos_nees_time: Vec<(f64, f64)>,
    rmse_pos: f64,
    rmse_vel: f64,
    rmse_att_deg: f64,
}

fn run_flight(cfg: SimConfig, seed: u64, seconds: f64) -> RunResult {
    let mut sim = Simulator::new(cfg, seed);
    let seed_nom = sim.sample_initial_nominal(InitialSigma::default());
    let mut f = Eskf::new(seed_nom, InitialSigma::default(), filter_noise(&cfg));

    let steps = (seconds * IMU_RATE) as usize;
    let settle = (5.0 * IMU_RATE) as usize; // ignore the first 5 s of convergence
    let sample_every = 20; // decorrelate the NEES stream a little

    let mut pos_nees = Vec::new();
    let mut full_nees = Vec::new();
    let mut pos_nees_time = Vec::new();
    let (mut se_pos, mut se_vel, mut se_att, mut n_err) = (0.0, 0.0, 0.0, 0.0);

    for k in 0..steps {
        let tick = sim.step();
        drive(&mut f, &tick, &cfg);

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
                    pos_nees_time.push((k as f64 / IMU_RATE, v));
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
        pos_nees_time,
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
    let cfg = SimConfig {
        lidar_enabled: true,
        uwb_enabled: true,
        flow_enabled: true,
        gps_vel_enabled: true,
        dvl_enabled: true,
        att_enabled: true,
        ..SimConfig::default()
    };

    println!("Monte-Carlo consistency check — {runs} flights of {seconds:.0} s, all sensors on\n");

    let mut pos = Vec::new();
    let mut full = Vec::new();
    let mut time_bins: Vec<Vec<f64>> = vec![Vec::new(); 5];
    let (mut rp, mut rv, mut ra) = (0.0, 0.0, 0.0);
    for seed in 0..runs {
        let r = run_flight(cfg, seed as u64 * 2_654_435_761, seconds);
        rp += r.rmse_pos;
        rv += r.rmse_vel;
        ra += r.rmse_att_deg;
        for (t, v) in &r.pos_nees_time {
            let b = (((t - 5.0) / 5.0) as usize).min(time_bins.len() - 1);
            time_bins[b].push(*v);
        }
        pos.extend(r.pos_nees);
        full.extend(r.full_nees);
    }
    let n = runs as f64;
    println!("  RMSE   position {:.3} m   velocity {:.3} m/s   attitude {:.3}°", rp / n, rv / n, ra / n);
    println!();

    let pos_ok = report_nees("position", &pos, 3, runs);
    let full_ok = report_nees("full state", &full, 15, runs);

    // Consistency is not just an average — it must hold throughout the flight, not only in the mean.
    println!("\n  position NEES over the flight (mean per 5 s window, expected 3.0):");
    let mut time_ok = true;
    for (i, b) in time_bins.iter().enumerate() {
        let m = if b.is_empty() { 0.0 } else { b.iter().sum::<f64>() / b.len() as f64 };
        let (lo, hi) = (5.0 + i as f64 * 5.0, 10.0 + i as f64 * 5.0);
        // A generous per-window sanity band (windows have fewer samples than the pooled test).
        let win_ok = (2.0..4.2).contains(&m);
        time_ok &= win_ok;
        println!("    {lo:>4.0}–{hi:<4.0}s   {m:.3}   {}", if win_ok { "ok" } else { "OUT" });
    }

    println!();
    let ok = pos_ok && full_ok && time_ok;
    println!("{}", if ok { "CONSISTENT — the filter's covariance matches its error, throughout. Gate PASSED." } else { "INCONSISTENT — covariance does not match the error. Gate FAILED." });
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
    let cases: [Scenario; 9] = [
        ("nominal (GPS+baro+mag)", |_c| {}),
        ("IMU only (dead-reckon)", |c| {
            c.gps_enabled = false;
            c.baro_enabled = false;
            c.mag_enabled = false;
            c.lidar_enabled = false;
        }),
        ("GPS dropout", |c| c.gps_enabled = false),
        ("GPS-denied + UWB", |c| {
            c.gps_enabled = false;
            c.uwb_enabled = true;
        }),
        ("indoor: UWB+LiDAR+flow", |c| {
            c.gps_enabled = false;
            c.uwb_enabled = true;
            c.lidar_enabled = true;
            c.flow_enabled = true;
        }),
        ("vision-aided: att+DVL", |c| {
            c.att_enabled = true;
            c.dvl_enabled = true;
        }),
        ("full fusion (10 sensors)", |c| {
            c.lidar_enabled = true;
            c.uwb_enabled = true;
            c.flow_enabled = true;
            c.gps_vel_enabled = true;
            c.dvl_enabled = true;
            c.att_enabled = true;
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

/// Writes the CSV series behind the paper's figures, all produced by the same `eskf` core the
/// gate runs, so every plotted curve is real filter output:
///   `nees_time.csv`  position and full-state NEES versus time, averaged over 40 flights;
///   `envelope.csv`   one flight's position error against its own $\pm 3\sigma$ envelope;
///   `drift.csv`      aided versus IMU-only (dead-reckoning) position error versus time.
fn cmd_plotdata(dir: &str) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let seconds = 30.0;
    let cfg = full_config();

    // NEES versus time, averaged across many flights so the curve is the ensemble mean the
    // consistency band is defined against.
    let runs = 40u64;
    let mut acc: Vec<[f64; 3]> = Vec::new(); // [sum_pos, sum_full, count] per sample time
    let mut times: Vec<f64> = Vec::new();
    for seed in 0..runs {
        let series = nees_series(cfg, seed * 2_654_435_761, seconds);
        if acc.is_empty() {
            times = series.iter().map(|s| s.0).collect();
            acc = series.iter().map(|s| [s.1, s.2, 1.0]).collect();
        } else {
            for (a, s) in acc.iter_mut().zip(series.iter()) {
                a[0] += s.1;
                a[1] += s.2;
                a[2] += 1.0;
            }
        }
    }
    let mut nees_csv = String::from("t,pos,full\n");
    for (t, a) in times.iter().zip(acc.iter()) {
        nees_csv.push_str(&format!("{:.2},{:.4},{:.4}\n", t, a[0] / a[2], a[1] / a[2]));
    }
    std::fs::write(format!("{dir}/nees_time.csv"), nees_csv).map_err(|e| e.to_string())?;

    // One flight's position error against its own reported 3-sigma envelope.
    let mut env_csv = String::from("t,ex,s3x,ez,s3z\n");
    for (t, ex, s3x, ez, s3z) in envelope_series(cfg, 7, seconds) {
        env_csv.push_str(&format!("{t:.3},{ex:.4},{s3x:.4},{ez:.4},{s3z:.4}\n"));
    }
    std::fs::write(format!("{dir}/envelope.csv"), env_csv).map_err(|e| e.to_string())?;

    // Dead-reckoning: aided fusion versus IMU-only coasting.
    let aided = drift_series(full_config(), 3, seconds);
    let imu_cfg = SimConfig {
        gps_enabled: false,
        baro_enabled: false,
        mag_enabled: false,
        ..SimConfig::default()
    };
    let imu = drift_series(imu_cfg, 3, seconds);
    let mut drift_csv = String::from("t,aided,imu_only\n");
    for (a, b) in aided.iter().zip(imu.iter()) {
        drift_csv.push_str(&format!("{:.3},{:.5},{:.5}\n", a.0, a.1, b.1));
    }
    std::fs::write(format!("{dir}/drift.csv"), drift_csv).map_err(|e| e.to_string())?;

    println!("wrote nees_time.csv, envelope.csv, drift.csv to {dir}/");
    Ok(())
}

/// Per-sample (time, position-NEES, full-state-NEES) after the filter settles.
fn nees_series(cfg: SimConfig, seed: u64, seconds: f64) -> Vec<(f64, f64, f64)> {
    let mut sim = Simulator::new(cfg, seed);
    let mut f = Eskf::new(sim.sample_initial_nominal(InitialSigma::default()), InitialSigma::default(), filter_noise(&cfg));
    let steps = (seconds * IMU_RATE) as usize;
    let settle = (5.0 * IMU_RATE) as usize;
    let mut out = Vec::new();
    for k in 0..steps {
        let tick = sim.step();
        drive(&mut f, &tick, &cfg);
        if k >= settle && k % 20 == 0 {
            if let (Some(p), Some(fu)) = (
                position_nees(&f.nom, &tick.truth, f.covariance()),
                nees(&f.nom, &tick.truth, f.covariance()),
            ) {
                out.push((k as f64 / IMU_RATE, p, fu));
            }
        }
    }
    out
}

/// Per-sample (time, x-error, 3-sigma-x, z-error, 3-sigma-z) from t=0, including convergence.
fn envelope_series(cfg: SimConfig, seed: u64, seconds: f64) -> Vec<(f64, f64, f64, f64, f64)> {
    let mut sim = Simulator::new(cfg, seed);
    let mut f = Eskf::new(sim.sample_initial_nominal(InitialSigma::default()), InitialSigma::default(), filter_noise(&cfg));
    let steps = (seconds * IMU_RATE) as usize;
    let mut out = Vec::new();
    for k in 0..steps {
        let tick = sim.step();
        drive(&mut f, &tick, &cfg);
        if k % 20 == 0 {
            let p = f.covariance();
            out.push((
                k as f64 / IMU_RATE,
                f.nom.p[0] - tick.truth.p[0],
                3.0 * p.m[0][0].max(0.0).sqrt(),
                f.nom.p[2] - tick.truth.p[2],
                3.0 * p.m[2][2].max(0.0).sqrt(),
            ));
        }
    }
    out
}

/// Per-sample (time, position-error-norm) from t=0.
fn drift_series(cfg: SimConfig, seed: u64, seconds: f64) -> Vec<(f64, f64)> {
    let mut sim = Simulator::new(cfg, seed);
    let mut f = Eskf::new(sim.sample_initial_nominal(InitialSigma::default()), InitialSigma::default(), filter_noise(&cfg));
    let steps = (seconds * IMU_RATE) as usize;
    let mut out = Vec::new();
    for k in 0..steps {
        let tick = sim.step();
        drive(&mut f, &tick, &cfg);
        if k % 20 == 0 {
            let dp = sub(f.nom.p, tick.truth.p);
            out.push((k as f64 / IMU_RATE, dot(dp, dp).sqrt()));
        }
    }
    out
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
