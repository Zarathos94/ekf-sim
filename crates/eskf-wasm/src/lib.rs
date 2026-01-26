//! wasm-bindgen surface for the sensor-fusion playground.
//!
//! Holds a live [`Simulator`] and [`Eskf`] and advances them together, one browser frame at a
//! time. The rolling estimated and true trajectories are exposed as pointers into linear memory so
//! the renderer reads them without a per-point copy; the small per-frame scalars — pose, the 3×3
//! position covariance for the ellipsoid, the error metrics, the estimated biases and a decaying
//! per-sensor activity pulse — come back as one short `Float32Array`.
//!
//! The same `eskf` crate that passes the native NEES consistency gate runs here unchanged.

use eskf::sim::BEACONS;
use eskf::{
    error_state, nees, position_nees, Eskf, InitialSigma, Noise, SimConfig, Simulator, TrueState,
    MAG_REFERENCE,
};
use wasm_bindgen::prelude::*;

/// Trajectory history: 60 s at 50 Hz — two full orbits, so the divergence between the estimated
/// and true paths persists on screen. The renderer fades the older end so it stays legible.
const TRAIL_MAX: usize = 3000;
const TRAIL_STRIDE: usize = 4; // push a point every 4 IMU steps (≈50 Hz at 200 Hz IMU)
const SNAPSHOT_LEN: usize = 48;

// Sensor-pulse indices.
const S_GPS: usize = 0;
const S_BARO: usize = 1;
const S_MAG: usize = 2;
const S_LIDAR: usize = 3;
const S_UWB: usize = 4;
const S_FLOW: usize = 5;
const S_GPS_VEL: usize = 6;
const S_DVL: usize = 7;
const S_ATT: usize = 8;
const N_SENSORS: usize = 9;

/// The most recent correction diagnostics for one sensor, mirrored from the filter's `last_update`
/// so the analytics view can show a live per-sensor innovation and NIS.
#[derive(Clone, Copy, Default)]
struct SensorStat {
    dim: f32,
    nis: f32,
    innov: f32,
    accepted: f32,
}

#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// The fixed UWB beacon positions, flattened `[x,y,z, …]`, so the renderer can draw them.
#[wasm_bindgen]
pub fn beacon_positions() -> Vec<f32> {
    let mut out = Vec::with_capacity(BEACONS.len() * 3);
    for b in BEACONS {
        out.push(b[0] as f32);
        out.push(b[1] as f32);
        out.push(b[2] as f32);
    }
    out
}

#[wasm_bindgen]
pub struct Session {
    cfg: SimConfig,
    sim: Simulator,
    filter: Eskf,
    seed: u64,
    last_truth: TrueState,

    est_trail: Vec<f32>,
    truth_trail: Vec<f32>,
    step_count: usize,

    ema_pos: f64,
    ema_att: f64,
    ema_nees: f64,
    inst_pos: f64,
    pulse: [f32; N_SENSORS],
    sensor_stats: [SensorStat; N_SENSORS],

    snapshot: Vec<f32>,
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> Session {
        let cfg = SimConfig::default();
        let mut sim = Simulator::new(cfg, seed);
        let filter = build_filter(&mut sim, &cfg);
        let last_truth = initial_truth(&sim);
        Session {
            cfg,
            sim,
            filter,
            seed,
            last_truth,
            est_trail: Vec::with_capacity(TRAIL_MAX * 3),
            truth_trail: Vec::with_capacity(TRAIL_MAX * 3),
            step_count: 0,
            ema_pos: 0.0,
            ema_att: 0.0,
            ema_nees: 3.0,
            inst_pos: 0.0,
            pulse: [0.0; N_SENSORS],
            sensor_stats: [SensorStat::default(); N_SENSORS],
            snapshot: vec![0.0; SNAPSHOT_LEN],
        }
    }

    /// Restart from a fresh flight, keeping the current slider settings.
    pub fn reset(&mut self) {
        self.seed = self.seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.sim = Simulator::new(self.cfg, self.seed);
        self.filter = build_filter(&mut self.sim, &self.cfg);
        self.last_truth = initial_truth(&self.sim);
        self.est_trail.clear();
        self.truth_trail.clear();
        self.step_count = 0;
        self.ema_pos = 0.0;
        self.ema_att = 0.0;
        self.ema_nees = 3.0;
        self.inst_pos = 0.0;
        self.pulse = [0.0; N_SENSORS];
        self.sensor_stats = [SensorStat::default(); N_SENSORS];
    }

    // --- Sensor noise (the simulator's actual noise; the filter's assumed noise tracks the IMU
    // terms so the uncertainty ellipsoid always reflects the real uncertainty). ---

    pub fn set_accel_noise(&mut self, v: f64) {
        self.cfg.accel_noise = v;
        self.sim.cfg.accel_noise = v;
        self.filter.noise.accel = v;
    }
    pub fn set_gyro_noise(&mut self, v: f64) {
        self.cfg.gyro_noise = v;
        self.sim.cfg.gyro_noise = v;
        self.filter.noise.gyro = v;
    }
    pub fn set_accel_bias_walk(&mut self, v: f64) {
        self.cfg.accel_bias_walk = v;
        self.sim.cfg.accel_bias_walk = v;
        self.filter.noise.accel_bias = v;
    }
    pub fn set_gyro_bias_walk(&mut self, v: f64) {
        self.cfg.gyro_bias_walk = v;
        self.sim.cfg.gyro_bias_walk = v;
        self.filter.noise.gyro_bias = v;
    }
    pub fn set_gps_noise(&mut self, v: f64) {
        self.cfg.gps_noise = v;
        self.sim.cfg.gps_noise = v;
    }
    pub fn set_baro_noise(&mut self, v: f64) {
        self.cfg.baro_noise = v;
        self.sim.cfg.baro_noise = v;
    }
    pub fn set_mag_noise(&mut self, v: f64) {
        self.cfg.mag_noise = v;
        self.sim.cfg.mag_noise = v;
    }
    pub fn set_lidar_noise(&mut self, v: f64) {
        self.cfg.lidar_noise = v;
        self.sim.cfg.lidar_noise = v;
    }
    pub fn set_uwb_noise(&mut self, v: f64) {
        self.cfg.uwb_noise = v;
        self.sim.cfg.uwb_noise = v;
    }
    pub fn set_flow_noise(&mut self, v: f64) {
        self.cfg.flow_noise = v;
        self.sim.cfg.flow_noise = v;
    }
    pub fn set_gps_vel_noise(&mut self, v: f64) {
        self.cfg.gps_vel_noise = v;
        self.sim.cfg.gps_vel_noise = v;
    }
    pub fn set_dvl_noise(&mut self, v: f64) {
        self.cfg.dvl_noise = v;
        self.sim.cfg.dvl_noise = v;
    }
    pub fn set_att_noise(&mut self, v: f64) {
        self.cfg.att_noise = v;
        self.sim.cfg.att_noise = v;
    }

    // --- Per-sensor enables. ---

    pub fn set_gps_enabled(&mut self, on: bool) {
        self.cfg.gps_enabled = on;
        self.sim.cfg.gps_enabled = on;
    }
    pub fn set_baro_enabled(&mut self, on: bool) {
        self.cfg.baro_enabled = on;
        self.sim.cfg.baro_enabled = on;
    }
    pub fn set_mag_enabled(&mut self, on: bool) {
        self.cfg.mag_enabled = on;
        self.sim.cfg.mag_enabled = on;
    }
    pub fn set_lidar_enabled(&mut self, on: bool) {
        self.cfg.lidar_enabled = on;
        self.sim.cfg.lidar_enabled = on;
    }
    pub fn set_uwb_enabled(&mut self, on: bool) {
        self.cfg.uwb_enabled = on;
        self.sim.cfg.uwb_enabled = on;
    }
    pub fn set_flow_enabled(&mut self, on: bool) {
        self.cfg.flow_enabled = on;
        self.sim.cfg.flow_enabled = on;
    }
    pub fn set_gps_vel_enabled(&mut self, on: bool) {
        self.cfg.gps_vel_enabled = on;
        self.sim.cfg.gps_vel_enabled = on;
    }
    pub fn set_dvl_enabled(&mut self, on: bool) {
        self.cfg.dvl_enabled = on;
        self.sim.cfg.dvl_enabled = on;
    }
    pub fn set_att_enabled(&mut self, on: bool) {
        self.cfg.att_enabled = on;
        self.sim.cfg.att_enabled = on;
    }

    /// Advance by `dt` seconds of wall-clock time, running the matching number of IMU steps
    /// (capped so a stalled tab cannot trigger a catch-up spiral).
    pub fn step(&mut self, dt: f64) {
        let want = (dt * self.cfg.imu_rate).round() as usize;
        let n = want.clamp(1, 8);
        for _ in 0..n {
            self.advance_one();
        }
        self.write_snapshot();
    }

    fn advance_one(&mut self) {
        for p in self.pulse.iter_mut() {
            *p *= 0.94; // fade the activity indicators
        }

        let tick = self.sim.step();
        self.filter.predict(tick.accel, tick.gyro, tick.dt);
        if let Some(z) = tick.gps {
            self.filter.update_gps(z, self.cfg.gps_noise.max(1e-3));
            self.mark(S_GPS);
        }
        if let Some(z) = tick.baro {
            self.filter.update_baro(z, self.cfg.baro_noise.max(1e-3));
            self.mark(S_BARO);
        }
        if let Some(z) = tick.mag {
            self.filter.update_mag(z, MAG_REFERENCE, self.cfg.mag_noise.max(1e-4));
            self.mark(S_MAG);
        }
        if let Some(z) = tick.lidar {
            self.filter.update_lidar_altimeter(z, self.cfg.lidar_noise.max(1e-3));
            self.mark(S_LIDAR);
        }
        if let Some(ranges) = tick.uwb {
            for (i, r) in ranges.iter().enumerate() {
                self.filter.update_range(BEACONS[i], *r, self.cfg.uwb_noise.max(1e-3));
            }
            self.mark(S_UWB);
        }
        if let Some(z) = tick.flow {
            self.filter.update_optical_flow(z, self.cfg.flow_noise.max(1e-3));
            self.mark(S_FLOW);
        }
        if let Some(z) = tick.gps_vel {
            self.filter.update_gps_velocity(z, self.cfg.gps_vel_noise.max(1e-3));
            self.mark(S_GPS_VEL);
        }
        if let Some(z) = tick.dvl {
            self.filter.update_body_velocity(z, self.cfg.dvl_noise.max(1e-3));
            self.mark(S_DVL);
        }
        if let Some(z) = tick.att {
            self.filter.update_attitude(z, self.cfg.att_noise.max(1e-4));
            self.mark(S_ATT);
        }
        self.last_truth = tick.truth;

        let dp = sub(self.filter.nom.p, tick.truth.p);
        let dth = eskf::quat::boxminus(tick.truth.q, self.filter.nom.q);
        let a = 0.02;
        self.inst_pos = norm(dp);
        self.ema_pos += a * (self.inst_pos - self.ema_pos);
        self.ema_att += a * (norm(dth).to_degrees() - self.ema_att);
        if let Some(v) = eskf::position_nees(&self.filter.nom, &tick.truth, self.filter.covariance()) {
            self.ema_nees += a * (v - self.ema_nees);
        }

        if self.step_count % TRAIL_STRIDE == 0 {
            push_trail(&mut self.est_trail, self.filter.nom.p);
            push_trail(&mut self.truth_trail, tick.truth.p);
        }
        self.step_count += 1;
    }

    /// Flag a sensor as having just fired and mirror the filter's last-correction diagnostics into
    /// its analytics slot.
    fn mark(&mut self, sensor: usize) {
        self.pulse[sensor] = 1.0;
        if let Some(u) = self.filter.last_update {
            self.sensor_stats[sensor] = SensorStat {
                dim: u.dim as f32,
                nis: u.nis as f32,
                innov: u.innovation_norm as f32,
                accepted: if u.accepted { 1.0 } else { 0.0 },
            };
        }
    }

    /// A richer, on-demand payload for the analytics view: the nominal state, the ground truth, the
    /// 15-vector estimation error, the full 15×15 error covariance, the position and full-state
    /// NEES, and each sensor's latest innovation/NIS. Computed only when the view asks for it, so
    /// the 3D path pays nothing for it. Layout (all `f32`):
    ///   [0..16]    nominal p(3) v(3) q(4) accel_bias(3) gyro_bias(3)
    ///   [16..26]   truth   p(3) v(3) q(4)
    ///   [26..41]   error   δp(3) δv(3) δθ(3) δa_b(3) δω_b(3)
    ///   [41..266]  covariance P, row-major 15×15
    ///   [266]      position NEES     [267] full-state NEES     [268] time (s)
    ///   [269..305] per sensor ×9: dim, nis, innovation-norm, accepted(1/0)
    pub fn analytics(&self) -> Vec<f32> {
        let n = &self.filter.nom;
        let t = &self.last_truth;
        let mut o: Vec<f32> = Vec::with_capacity(305);
        let push3 = |v: [f64; 3], o: &mut Vec<f32>| {
            o.push(v[0] as f32);
            o.push(v[1] as f32);
            o.push(v[2] as f32);
        };
        push3(n.p, &mut o);
        push3(n.v, &mut o);
        o.extend_from_slice(&[n.q.w as f32, n.q.x as f32, n.q.y as f32, n.q.z as f32]);
        push3(n.accel_bias, &mut o);
        push3(n.gyro_bias, &mut o);
        push3(t.p, &mut o);
        push3(t.v, &mut o);
        o.extend_from_slice(&[t.q.w as f32, t.q.x as f32, t.q.y as f32, t.q.z as f32]);

        let e = error_state(n, t);
        for i in 0..15 {
            o.push(e.m[i][0] as f32);
        }
        let p = self.filter.covariance();
        for row in &p.m {
            for &v in row {
                o.push(v as f32);
            }
        }
        o.push(position_nees(n, t, p).unwrap_or(0.0) as f32);
        o.push(nees(n, t, p).unwrap_or(0.0) as f32);
        o.push(self.sim.time() as f32);
        for st in &self.sensor_stats {
            o.push(st.dim);
            o.push(st.nis);
            o.push(st.innov);
            o.push(st.accepted);
        }
        o
    }

    fn write_snapshot(&mut self) {
        let s = &mut self.snapshot;
        let n = &self.filter.nom;
        let put3 = |s: &mut [f32], i: usize, v: [f64; 3]| {
            s[i] = v[0] as f32;
            s[i + 1] = v[1] as f32;
            s[i + 2] = v[2] as f32;
        };
        put3(s, 0, n.p);
        put3(s, 3, n.v);
        s[6] = n.q.w as f32;
        s[7] = n.q.x as f32;
        s[8] = n.q.y as f32;
        s[9] = n.q.z as f32;
        let truth = self.last_truth;
        put3(s, 10, truth.p);
        s[13] = truth.q.w as f32;
        s[14] = truth.q.x as f32;
        s[15] = truth.q.y as f32;
        s[16] = truth.q.z as f32;
        let cov = self.filter.position_covariance();
        for r in 0..3 {
            for c in 0..3 {
                s[17 + r * 3 + c] = cov.m[r][c] as f32;
            }
        }
        s[26] = self.ema_pos as f32;
        s[27] = self.ema_att as f32;
        s[28] = self.ema_nees as f32;
        s[29] = self.sim.time() as f32;
        put3(s, 30, n.accel_bias);
        put3(s, 33, n.gyro_bias);
        for (i, p) in self.pulse.iter().enumerate() {
            s[36 + i] = *p; // nine sensor pulses at [36..45]
        }
        // 1σ position uncertainty (RMS of the three axes) — the ellipsoid's scale, for the plot.
        let trace = cov.m[0][0] + cov.m[1][1] + cov.m[2][2];
        s[45] = (trace / 3.0).max(0.0).sqrt() as f32;
        s[46] = self.inst_pos as f32;
    }

    /// The per-frame scalars: pose, truth pose, position covariance, metrics, biases, activity.
    pub fn snapshot(&self) -> Vec<f32> {
        self.snapshot.clone()
    }

    pub fn est_trail_ptr(&self) -> *const f32 {
        self.est_trail.as_ptr()
    }
    pub fn est_trail_len(&self) -> usize {
        self.est_trail.len()
    }
    pub fn truth_trail_ptr(&self) -> *const f32 {
        self.truth_trail.as_ptr()
    }
    pub fn truth_trail_len(&self) -> usize {
        self.truth_trail.len()
    }
}

fn build_filter(sim: &mut Simulator, cfg: &SimConfig) -> Eskf {
    let noise = Noise {
        gravity: eskf::GRAVITY,
        accel: cfg.accel_noise,
        gyro: cfg.gyro_noise,
        accel_bias: cfg.accel_bias_walk,
        gyro_bias: cfg.gyro_bias_walk,
    };
    // Start with a real initial error to converge from, not omnisciently at truth.
    let seed_nom = sim.sample_initial_nominal(InitialSigma::default());
    Eskf::new(seed_nom, InitialSigma::default(), noise)
}

fn initial_truth(sim: &Simulator) -> TrueState {
    let t = sim.truth_nominal();
    TrueState { p: t.p, v: t.v, q: t.q, accel_bias: [0.0; 3], gyro_bias: [0.0; 3] }
}

fn push_trail(trail: &mut Vec<f32>, p: [f64; 3]) {
    if trail.len() >= TRAIL_MAX * 3 {
        trail.drain(0..3); // keeps capacity, so the base pointer stays put
    }
    trail.push(p[0] as f32);
    trail.push(p[1] as f32);
    trail.push(p[2] as f32);
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
