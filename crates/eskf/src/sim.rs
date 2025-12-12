//! A flight simulator that produces the sensor streams the filter consumes, plus the ground
//! truth to score it against.
//!
//! The trajectory is analytic — a banking helical orbit — so position, velocity and acceleration
//! are exact and there is no integration error contaminating the "truth". From it we synthesise a
//! strapdown IMU (specific force and body rate, with bias and white noise), and lower-rate GPS,
//! barometer and magnetometer measurements. Every noise term, the bias random walk, and GPS
//! dropout are knobs — the point of the playground is to turn them and watch the filter degrade
//! and recover.

use crate::filter::Nominal;
use crate::linalg::{v3, V3};
use crate::quat::{boxminus, Quat};

/// World magnetic reference direction (unit): mostly north (+x) and downward, ~65° inclination.
pub const MAG_REFERENCE: V3 = [0.422_618, 0.0, -0.906_308];
pub const GRAVITY: V3 = [0.0, 0.0, -9.806_65];

/// Everything a viewer can turn. Rates are in hertz; noise terms are 1-σ.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub imu_rate: f64,
    pub gps_rate: f64,
    pub baro_rate: f64,
    pub mag_rate: f64,

    /// Accelerometer white noise, m/s².
    pub accel_noise: f64,
    /// Gyroscope white noise, rad/s.
    pub gyro_noise: f64,
    /// Accelerometer bias random-walk step density, (m/s²)/√s.
    pub accel_bias_walk: f64,
    /// Gyroscope bias random-walk step density, (rad/s)/√s.
    pub gyro_bias_walk: f64,

    /// GPS horizontal/vertical noise, m.
    pub gps_noise: f64,
    /// Barometer noise, m.
    pub baro_noise: f64,
    /// Magnetometer noise, unit-vector components.
    pub mag_noise: f64,

    /// LiDAR (downward laser altimeter) noise, m, and its rate.
    pub lidar_noise: f64,
    pub lidar_rate: f64,
    /// UWB / radio ranging noise, m, and its rate (one sweep of every beacon per tick).
    pub uwb_noise: f64,
    pub uwb_rate: f64,
    /// Optical-flow (body-frame horizontal velocity) noise, m/s, and its rate.
    pub flow_noise: f64,
    pub flow_rate: f64,

    /// Per-sensor enables. Turning GPS off is the urban-canyon dropout; turning UWB on is the
    /// radio-ranging fallback that keeps the estimate alive without it.
    pub gps_enabled: bool,
    pub baro_enabled: bool,
    pub mag_enabled: bool,
    pub lidar_enabled: bool,
    pub uwb_enabled: bool,
    pub flow_enabled: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            imu_rate: 200.0,
            gps_rate: 5.0,
            baro_rate: 20.0,
            mag_rate: 40.0,
            accel_noise: 0.06,
            gyro_noise: 0.004,
            accel_bias_walk: 0.002,
            gyro_bias_walk: 2.0e-4,
            gps_noise: 0.8,
            baro_noise: 0.6,
            mag_noise: 0.02,
            lidar_noise: 0.15,
            lidar_rate: 25.0,
            uwb_noise: 0.35,
            uwb_rate: 10.0,
            flow_noise: 0.15,
            flow_rate: 30.0,
            gps_enabled: true,
            baro_enabled: true,
            mag_enabled: true,
            lidar_enabled: true,
            uwb_enabled: false,
            flow_enabled: false,
        }
    }
}

/// Fixed radio-ranging beacons (UWB anchors) at known world positions, m.
pub const BEACONS: [V3; 4] = [
    [-55.0, -55.0, 0.0],
    [58.0, -50.0, 6.0],
    [52.0, 56.0, 34.0],
    [-50.0, 54.0, 18.0],
];

/// Ground truth at an instant.
#[derive(Clone, Copy, Debug)]
pub struct TrueState {
    pub p: V3,
    pub v: V3,
    pub q: Quat,
    pub accel_bias: V3,
    pub gyro_bias: V3,
}

/// What one IMU tick produces: an IMU sample always, and the aiding sensors when their turn
/// comes round.
#[derive(Clone, Copy, Debug)]
pub struct Tick {
    pub t: f64,
    pub dt: f64,
    pub truth: TrueState,
    pub accel: V3,
    pub gyro: V3,
    pub gps: Option<V3>,
    pub baro: Option<f64>,
    pub mag: Option<V3>,
    pub lidar: Option<f64>,
    pub uwb: Option<[f64; 4]>,
    pub flow: Option<[f64; 2]>,
}

pub struct Simulator {
    pub cfg: SimConfig,
    rng: Rng,
    t: f64,
    accel_bias: V3,
    gyro_bias: V3,
    next_gps: f64,
    next_baro: f64,
    next_mag: f64,
    next_lidar: f64,
    next_uwb: f64,
    next_flow: f64,
}

impl Simulator {
    pub fn new(cfg: SimConfig, seed: u64) -> Self {
        Simulator {
            cfg,
            rng: Rng::new(seed),
            t: 0.0,
            // Small non-zero initial biases so there is something for the filter to learn.
            accel_bias: [0.05, -0.03, 0.02],
            gyro_bias: [0.001, -0.0015, 0.0008],
            next_gps: 0.0,
            next_baro: 0.0,
            next_mag: 0.0,
            next_lidar: 0.0,
            next_uwb: 0.0,
            next_flow: 0.0,
        }
    }

    pub fn time(&self) -> f64 {
        self.t
    }

    /// The ideal filter seed: the true initial state, so a run scores the filter, not the guess.
    pub fn truth_nominal(&self) -> Nominal {
        let s = truth_at(0.0);
        Nominal { p: s.p, v: s.v, q: s.q, accel_bias: [0.0; 3], gyro_bias: [0.0; 3] }
    }

    /// Advance one IMU period and emit the samples.
    pub fn step(&mut self) -> Tick {
        let dt = 1.0 / self.cfg.imu_rate;
        let t = self.t;

        // Bias random walk.
        let aw = self.cfg.accel_bias_walk * dt.sqrt();
        let gw = self.cfg.gyro_bias_walk * dt.sqrt();
        for k in 0..3 {
            self.accel_bias[k] += self.rng.gaussian() * aw;
            self.gyro_bias[k] += self.rng.gaussian() * gw;
        }

        let s = truth_at(t);
        let omega = body_rate(t);

        // Strapdown IMU: specific force f = Rᵀ(a − g); body rate = ω.
        let f_world = v3::sub(accel_at(t), GRAVITY);
        let f_body = s.q.rotate_inv(f_world);
        let accel = [
            f_body[0] + self.accel_bias[0] + self.rng.gaussian() * self.cfg.accel_noise,
            f_body[1] + self.accel_bias[1] + self.rng.gaussian() * self.cfg.accel_noise,
            f_body[2] + self.accel_bias[2] + self.rng.gaussian() * self.cfg.accel_noise,
        ];
        let gyro = [
            omega[0] + self.gyro_bias[0] + self.rng.gaussian() * self.cfg.gyro_noise,
            omega[1] + self.gyro_bias[1] + self.rng.gaussian() * self.cfg.gyro_noise,
            omega[2] + self.gyro_bias[2] + self.rng.gaussian() * self.cfg.gyro_noise,
        ];

        // Each aiding sensor fires on its own schedule when enabled. A disabled sensor still keeps
        // its clock current, so re-enabling it resumes at the right cadence rather than in a burst.
        let due = |t: f64, next: &mut f64, rate: f64, enabled: bool| -> bool {
            if t < *next {
                return false;
            }
            *next += 1.0 / rate;
            enabled
        };

        let gps = if due(t, &mut self.next_gps, self.cfg.gps_rate, self.cfg.gps_enabled) {
            Some([
                s.p[0] + self.rng.gaussian() * self.cfg.gps_noise,
                s.p[1] + self.rng.gaussian() * self.cfg.gps_noise,
                s.p[2] + self.rng.gaussian() * self.cfg.gps_noise,
            ])
        } else {
            None
        };

        let baro = if due(t, &mut self.next_baro, self.cfg.baro_rate, self.cfg.baro_enabled) {
            Some(s.p[2] + self.rng.gaussian() * self.cfg.baro_noise)
        } else {
            None
        };

        let mag = if due(t, &mut self.next_mag, self.cfg.mag_rate, self.cfg.mag_enabled) {
            let b = s.q.rotate_inv(MAG_REFERENCE);
            Some([
                b[0] + self.rng.gaussian() * self.cfg.mag_noise,
                b[1] + self.rng.gaussian() * self.cfg.mag_noise,
                b[2] + self.rng.gaussian() * self.cfg.mag_noise,
            ])
        } else {
            None
        };

        // Downward LiDAR altimeter: slant range to the ground plane, only when the beam actually
        // reaches it (not banked past horizontal, above the plane).
        let lidar = if due(t, &mut self.next_lidar, self.cfg.lidar_rate, self.cfg.lidar_enabled) {
            let r22 = s.q.to_matrix().m[2][2];
            if r22 > 0.2 && s.p[2] > 0.0 {
                Some(s.p[2] / r22 + self.rng.gaussian() * self.cfg.lidar_noise)
            } else {
                None
            }
        } else {
            None
        };

        // UWB: one range to every beacon per sweep.
        let uwb = if due(t, &mut self.next_uwb, self.cfg.uwb_rate, self.cfg.uwb_enabled) {
            let mut r = [0.0; 4];
            for (i, b) in BEACONS.iter().enumerate() {
                r[i] = v3::norm(v3::sub(s.p, *b)) + self.rng.gaussian() * self.cfg.uwb_noise;
            }
            Some(r)
        } else {
            None
        };

        // Optical flow: body-frame horizontal velocity.
        let flow = if due(t, &mut self.next_flow, self.cfg.flow_rate, self.cfg.flow_enabled) {
            let vb = s.q.rotate_inv(s.v);
            Some([
                vb[0] + self.rng.gaussian() * self.cfg.flow_noise,
                vb[1] + self.rng.gaussian() * self.cfg.flow_noise,
            ])
        } else {
            None
        };

        let truth = TrueState {
            p: s.p,
            v: s.v,
            q: s.q,
            accel_bias: self.accel_bias,
            gyro_bias: self.gyro_bias,
        };
        self.t += dt;
        Tick { t, dt, truth, accel, gyro, gps, baro, mag, lidar, uwb, flow }
    }
}

// --- The analytic trajectory: a banking helical orbit. ---

const ORBIT_RADIUS: f64 = 40.0;
const ORBIT_PERIOD: f64 = 30.0;
const CLIMB_AMPLITUDE: f64 = 8.0;
const CLIMB_PERIOD: f64 = 17.0;
const BASE_ALTITUDE: f64 = 20.0;

fn omega_orbit() -> f64 {
    core::f64::consts::TAU / ORBIT_PERIOD
}
fn omega_climb() -> f64 {
    core::f64::consts::TAU / CLIMB_PERIOD
}

fn position_at(t: f64) -> V3 {
    let w = omega_orbit();
    [
        ORBIT_RADIUS * (w * t).cos(),
        ORBIT_RADIUS * (w * t).sin(),
        BASE_ALTITUDE + CLIMB_AMPLITUDE * (omega_climb() * t).sin(),
    ]
}

fn velocity_at(t: f64) -> V3 {
    let w = omega_orbit();
    let wz = omega_climb();
    [
        -ORBIT_RADIUS * w * (w * t).sin(),
        ORBIT_RADIUS * w * (w * t).cos(),
        CLIMB_AMPLITUDE * wz * (wz * t).cos(),
    ]
}

fn accel_at(t: f64) -> V3 {
    let w = omega_orbit();
    let wz = omega_climb();
    [
        -ORBIT_RADIUS * w * w * (w * t).cos(),
        -ORBIT_RADIUS * w * w * (w * t).sin(),
        -CLIMB_AMPLITUDE * wz * wz * (wz * t).sin(),
    ]
}

/// Orientation of a coordinated, banking aircraft: nose along velocity, rolled into the turn by
/// the bank angle that balances lateral acceleration against gravity.
fn orientation_at(t: f64) -> Quat {
    let v = velocity_at(t);
    let a = accel_at(t);
    let fwd = v3::normalize(v); // body x → world
    let mut left0 = v3::cross([0.0, 0.0, 1.0], fwd); // horizontal left
    if v3::norm(left0) < 1e-6 {
        left0 = [0.0, 1.0, 0.0];
    }
    let left0 = v3::normalize(left0);
    let up0 = v3::cross(fwd, left0);
    // Bank so the lift vector leans into the turn.
    let lateral = v3::dot(a, left0);
    let phi = lateral.atan2(9.806_65);
    let (s, c) = phi.sin_cos();
    let left = v3::add(v3::scale(left0, c), v3::scale(up0, s));
    let up = v3::add(v3::scale(left0, -s), v3::scale(up0, c));
    // Columns of the body→world matrix are the body axes expressed in the world frame.
    let m = crate::linalg::Mat::<3, 3>::from_rows([
        [fwd[0], left[0], up[0]],
        [fwd[1], left[1], up[1]],
        [fwd[2], left[2], up[2]],
    ]);
    Quat::from_matrix(&m)
}

fn truth_at(t: f64) -> TrueState {
    TrueState {
        p: position_at(t),
        v: velocity_at(t),
        q: orientation_at(t),
        accel_bias: [0.0; 3],
        gyro_bias: [0.0; 3],
    }
}

/// Body angular rate from the orientation path, by central difference of the quaternion — the
/// body-frame ω the gyroscope would read.
fn body_rate(t: f64) -> V3 {
    let h = 1.0e-4;
    let earlier = orientation_at(t - h);
    let later = orientation_at(t + h);
    v3::scale(boxminus(later, earlier), 1.0 / (2.0 * h))
}

/// A small deterministic PRNG (SplitMix64) with a Box–Muller Gaussian. No dependency, and
/// reproducible per seed so a Monte-Carlo run can be replayed exactly.
struct Rng {
    state: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed ^ 0x9e37_79b9_7f4a_7c15, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn uniform(&mut self) -> f64 {
        // 53-bit mantissa, shifted off zero.
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn gaussian(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let mag = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (core::f64::consts::TAU * u2).sin_cos();
        self.spare = Some(mag * s);
        mag * c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_is_the_derivative_of_position() {
        // Central difference of the analytic position must match the analytic velocity.
        let h = 1e-6;
        for t in [0.0, 3.3, 11.7, 25.0] {
            let num = v3::scale(v3::sub(position_at(t + h), position_at(t - h)), 1.0 / (2.0 * h));
            let ana = velocity_at(t);
            for i in 0..3 {
                assert!((num[i] - ana[i]).abs() < 1e-4, "t={t} axis {i}: {} vs {}", num[i], ana[i]);
            }
        }
    }

    #[test]
    fn acceleration_is_the_derivative_of_velocity() {
        let h = 1e-6;
        for t in [0.5, 7.0, 19.4] {
            let num = v3::scale(v3::sub(velocity_at(t + h), velocity_at(t - h)), 1.0 / (2.0 * h));
            let ana = accel_at(t);
            for i in 0..3 {
                assert!((num[i] - ana[i]).abs() < 1e-3, "t={t} axis {i}");
            }
        }
    }

    #[test]
    fn gaussian_is_roughly_standard_normal() {
        let mut rng = Rng::new(42);
        let n = 100_000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let g = rng.gaussian();
            sum += g;
            sq += g * g;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    #[test]
    fn gps_dropout_suppresses_fixes() {
        let cfg = SimConfig { gps_enabled: false, ..Default::default() };
        let mut sim = Simulator::new(cfg, 1);
        for _ in 0..400 {
            assert!(sim.step().gps.is_none(), "GPS leaked through a dropout");
        }
    }
}
