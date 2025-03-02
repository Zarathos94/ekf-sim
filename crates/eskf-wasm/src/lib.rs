//! wasm-bindgen surface for the sensor-fusion playground.
//!
//! Holds a live [`Simulator`] and [`Eskf`] and advances them together, one browser frame at a
//! time. The heavy state — the rolling estimated and true trajectories — is exposed as pointers
//! into linear memory so the renderer reads them without a per-point copy across the boundary;
//! the small per-frame scalars (pose, the 3×3 position covariance for the ellipsoid, and the
//! error metrics) come back as one short `Float32Array`.
//!
//! The same `eskf` crate that passes the native NEES consistency gate runs here unchanged.

use eskf::{Eskf, InitialSigma, Noise, SimConfig, Simulator, TrueState, MAG_REFERENCE};
use wasm_bindgen::prelude::*;

/// Trajectory history: 12 s at 50 Hz.
const TRAIL_MAX: usize = 600;
const TRAIL_STRIDE: usize = 4; // push a point every 4 IMU steps (≈50 Hz at 200 Hz IMU)

#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
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

    // Exponential moving averages, so the readouts respond to a slider without a noisy jump.
    ema_pos: f64,
    ema_att: f64,
    ema_nees: f64,

    snapshot: Vec<f32>,
}

#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> Session {
        let cfg = SimConfig::default();
        let sim = Simulator::new(cfg, seed);
        let filter = build_filter(&sim, &cfg);
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
            snapshot: vec![0.0; 32],
        }
    }

    /// Restart from a fresh flight, keeping the current slider settings.
    pub fn reset(&mut self) {
        self.seed = self.seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.sim = Simulator::new(self.cfg, self.seed);
        self.filter = build_filter(&self.sim, &self.cfg);
        self.last_truth = initial_truth(&self.sim);
        self.est_trail.clear();
        self.truth_trail.clear();
        self.step_count = 0;
        self.ema_pos = 0.0;
        self.ema_att = 0.0;
        self.ema_nees = 3.0;
    }

    // --- Slider-driven configuration. The simulator's actual noise and the filter's assumed
    // noise track together, so the uncertainty ellipsoid always reflects the real uncertainty. ---

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
    pub fn set_gps_dropout(&mut self, dropped: bool) {
        self.cfg.gps_dropout = dropped;
        self.sim.cfg.gps_dropout = dropped;
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
        let tick = self.sim.step();
        self.filter.predict(tick.accel, tick.gyro, tick.dt);
        if let Some(z) = tick.gps {
            self.filter.update_gps(z, self.cfg.gps_noise.max(1e-3));
        }
        if let Some(z) = tick.baro {
            self.filter.update_baro(z, self.cfg.baro_noise.max(1e-3));
        }
        if let Some(z) = tick.mag {
            self.filter.update_mag(z, MAG_REFERENCE, self.cfg.mag_noise.max(1e-4));
        }
        self.last_truth = tick.truth;

        // Error metrics (EMA).
        let dp = sub(self.filter.nom.p, tick.truth.p);
        let dth = eskf::quat::boxminus(tick.truth.q, self.filter.nom.q);
        let a = 0.02;
        self.ema_pos += a * (norm(dp) - self.ema_pos);
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
        s[29] = if self.cfg.gps_dropout { 0.0 } else { 1.0 };
        s[30] = self.sim.time() as f32;
        s[31] = norm(n.gyro_bias).to_degrees() as f32; // estimated gyro-bias magnitude, °/s
    }

    /// The per-frame scalars: est pose, truth pose, 3×3 position covariance, and metrics.
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

fn build_filter(sim: &Simulator, cfg: &SimConfig) -> Eskf {
    let noise = Noise {
        gravity: eskf::GRAVITY,
        accel: cfg.accel_noise,
        gyro: cfg.gyro_noise,
        accel_bias: cfg.accel_bias_walk,
        gyro_bias: cfg.gyro_bias_walk,
    };
    Eskf::new(sim.truth_nominal(), InitialSigma::default(), noise)
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
