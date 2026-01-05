//! The error-state Kalman filter.
//!
//! Two states in play. The **nominal** state (position, velocity, orientation quaternion, and
//! the two IMU biases) holds the full estimate and is integrated directly from the IMU. The
//! **error** state — a 15-vector `[δp, δv, δθ, δa_b, δω_b]`, always zero between steps — is what
//! the Kalman machinery actually estimates: it stays small, so its 15×15 covariance is well
//! conditioned and the orientation error is a minimal 3-vector rather than an over-parameterised
//! quaternion. Each correction is computed in the error state and then *injected* into the
//! nominal state, and the error state is reset. Formulation follows Solà, "Quaternion kinematics
//! for the error-state Kalman filter" (2017), with the local (body-frame) angular error.

use crate::linalg::{col3, skew, v3, Mat, V3};
use crate::quat::{boxminus, Quat};

// Error-state block offsets.
const IP: usize = 0; // position
const IV: usize = 3; // velocity
const ITH: usize = 6; // orientation error
const IAB: usize = 9; // accelerometer bias
const IWB: usize = 12; // gyroscope bias
pub const N: usize = 15;

/// The full estimate, integrated directly from the IMU.
#[derive(Clone, Copy, Debug)]
pub struct Nominal {
    pub p: V3,
    pub v: V3,
    pub q: Quat,
    pub accel_bias: V3,
    pub gyro_bias: V3,
}

impl Nominal {
    pub fn at_rest(p: V3, q: Quat) -> Self {
        Nominal { p, v: [0.0; 3], q, accel_bias: [0.0; 3], gyro_bias: [0.0; 3] }
    }
}

/// Process-noise densities (continuous-time) and gravity. These are what the filter *believes*
/// about the IMU; the simulator's actual noise is separate, which is the whole point of the
/// consistency check.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    /// Gravity vector, world frame (z up), m/s².
    pub gravity: V3,
    /// Accelerometer white-noise density, (m/s²)/√Hz.
    pub accel: f64,
    /// Gyroscope white-noise density, (rad/s)/√Hz.
    pub gyro: f64,
    /// Accelerometer bias random-walk density, (m/s²)√Hz.
    pub accel_bias: f64,
    /// Gyroscope bias random-walk density, (rad/s)√Hz.
    pub gyro_bias: f64,
}

impl Default for Noise {
    fn default() -> Self {
        Noise {
            gravity: [0.0, 0.0, -9.80665],
            accel: 0.08,
            gyro: 0.006,
            accel_bias: 0.002,
            gyro_bias: 2.0e-4,
        }
    }
}

pub struct Eskf {
    pub nom: Nominal,
    pub p: Mat<N, N>,
    pub noise: Noise,
}

impl Eskf {
    /// Builds a filter from an initial nominal state and the standard deviations of the initial
    /// error on each block (position, velocity, orientation, accel bias, gyro bias).
    pub fn new(nom: Nominal, sigma: InitialSigma, noise: Noise) -> Self {
        let mut p = Mat::<N, N>::zeros();
        let set = |p: &mut Mat<N, N>, i: usize, s: f64| {
            for k in 0..3 {
                p.m[i + k][i + k] = s * s;
            }
        };
        set(&mut p, IP, sigma.position);
        set(&mut p, IV, sigma.velocity);
        set(&mut p, ITH, sigma.orientation);
        set(&mut p, IAB, sigma.accel_bias);
        set(&mut p, IWB, sigma.gyro_bias);
        Eskf { nom, p, noise }
    }

    /// Propagate with one IMU sample: integrate the nominal state and push the covariance
    /// through the error-state transition.
    pub fn predict(&mut self, accel_meas: V3, gyro_meas: V3, dt: f64) {
        let a = v3::sub(accel_meas, self.nom.accel_bias); // specific force, body
        let w = v3::sub(gyro_meas, self.nom.gyro_bias); // angular rate, body
        let r = self.nom.q.to_matrix(); // body → world

        // --- Nominal integration ---
        let a_world = v3::add(mat_vec(&r, a), self.noise.gravity);
        self.nom.p = v3::add(self.nom.p, v3::add(v3::scale(self.nom.v, dt), v3::scale(a_world, 0.5 * dt * dt)));
        self.nom.v = v3::add(self.nom.v, v3::scale(a_world, dt));
        self.nom.q = self.nom.q.mul(Quat::from_rotation_vector(v3::scale(w, dt))).normalized();

        // --- Error-state transition F ---
        let mut f = Mat::<N, N>::identity();
        f.set_block(IP, IV, &Mat::<3, 3>::identity().scale(dt)); // δp ← δv
        f.set_block(IV, ITH, &r.mul(&skew(a)).scale(-dt)); // δv ← δθ  (−R[a]× dt)
        f.set_block(IV, IAB, &r.scale(-dt)); // δv ← δa_b  (−R dt)
        // δθ ← δθ  is exp(−[w]× dt); δθ ← δω_b is −I dt.
        let phi_theta = Quat::from_rotation_vector(v3::scale(w, -dt)).to_matrix();
        f.set_block(ITH, ITH, &phi_theta);
        f.set_block(ITH, IWB, &Mat::<3, 3>::identity().scale(-dt));

        // --- Process noise Q (block-diagonal) ---
        // The IMU white noise is a per-sample standard deviation: a noisy reading `σ` perturbs
        // the integrated velocity/orientation by `σ·dt` over the step, so the injected variance
        // is `σ²·dt²` (also the only dimensionally correct choice — `σ²·dt` would be m²/s³, not
        // (m/s)²). The biases are a random walk with rate density `σ_bias`, whose per-step
        // variance is `σ_bias²·dt`.
        let mut q = Mat::<N, N>::zeros();
        let diag = |q: &mut Mat<N, N>, i: usize, var: f64| {
            for k in 0..3 {
                q.m[i + k][i + k] = var;
            }
        };
        diag(&mut q, IV, self.noise.accel * self.noise.accel * dt * dt);
        diag(&mut q, ITH, self.noise.gyro * self.noise.gyro * dt * dt);
        diag(&mut q, IAB, self.noise.accel_bias * self.noise.accel_bias * dt);
        diag(&mut q, IWB, self.noise.gyro_bias * self.noise.gyro_bias * dt);

        // P ← F P Fᵀ + Q
        self.p = f.mul(&self.p).mul(&f.transpose()).add(&q);
        self.symmetrize();
    }

    /// GPS: a direct position measurement, world frame.
    pub fn update_gps(&mut self, pos: V3, sigma: f64) {
        let mut h = Mat::<3, N>::zeros();
        h.set_block(0, IP, &Mat::<3, 3>::identity());
        let residual = col3(v3::sub(pos, self.nom.p));
        let r = Mat::<3, 3>::identity().scale(sigma * sigma);
        self.update(&h, &residual, &r);
    }

    /// Barometer: altitude only (world up = +z).
    pub fn update_baro(&mut self, altitude: f64, sigma: f64) {
        let mut h = Mat::<1, N>::zeros();
        h.m[0][IP + 2] = 1.0;
        let residual = Mat::<1, 1>::from_rows([[altitude - self.nom.p[2]]]);
        let r = Mat::<1, 1>::from_rows([[sigma * sigma]]);
        self.update(&h, &residual, &r);
    }

    /// Magnetometer: the full field vector in the body frame, against a known world reference
    /// direction. Observes heading (and, weakly, tilt) — the vector form, not a yaw shortcut.
    pub fn update_mag(&mut self, field_body: V3, reference_world: V3, sigma: f64) {
        let h0 = self.nom.q.rotate_inv(reference_world); // predicted body measurement Rᵀ m
        let mut h = Mat::<3, N>::zeros();
        h.set_block(0, ITH, &skew(h0)); // ∂(Rᵀ m)/∂δθ = [Rᵀ m]×
        let residual = col3(v3::sub(field_body, h0));
        let r = Mat::<3, 3>::identity().scale(sigma * sigma);
        self.update(&h, &residual, &r);
    }

    /// Radio ranging (UWB / RF beacon) to a fixed transmitter at a known world position: a scalar
    /// range measurement. It sees only position, so a handful of beacons localise the vehicle with
    /// no GPS at all — the classic GPS-denied fallback.
    pub fn update_range(&mut self, beacon: V3, range: f64, sigma: f64) {
        let (dist, h) = match self.range_predict(beacon) {
            Some(v) => v,
            None => return,
        };
        let residual = Mat::<1, 1>::from_rows([[range - dist]]);
        let r = Mat::<1, 1>::from_rows([[sigma * sigma]]);
        self.update(&h, &residual, &r);
    }

    pub(crate) fn range_predict(&self, beacon: V3) -> Option<(f64, Mat<1, N>)> {
        let d = v3::sub(self.nom.p, beacon);
        let dist = v3::norm(d);
        if dist < 1e-3 {
            return None;
        }
        let u = v3::scale(d, 1.0 / dist); // unit line of sight
        let mut h = Mat::<1, N>::zeros();
        h.m[0][IP] = u[0];
        h.m[0][IP + 1] = u[1];
        h.m[0][IP + 2] = u[2];
        Some((dist, h))
    }

    /// Downward laser altimeter over flat ground (`z = 0`): the slant range along the body-down
    /// axis. Because the beam tilts with the vehicle, the measurement senses both height and
    /// attitude — `range = p_z / R₂₂`, so its Jacobian touches δp and δθ.
    pub fn update_lidar_altimeter(&mut self, range: f64, sigma: f64) {
        let (h_pred, h) = match self.lidar_predict() {
            Some(v) => v,
            None => return,
        };
        let residual = Mat::<1, 1>::from_rows([[range - h_pred]]);
        let r = Mat::<1, 1>::from_rows([[sigma * sigma]]);
        self.update(&h, &residual, &r);
    }

    pub(crate) fn lidar_predict(&self) -> Option<(f64, Mat<1, N>)> {
        let rm = self.nom.q.to_matrix();
        let r22 = rm.m[2][2];
        // Skip when the beam is not pointing at the ground below (steep bank, or below the plane).
        if r22 < 0.2 || self.nom.p[2] <= 0.0 {
            return None;
        }
        let pz = self.nom.p[2];
        let mut h = Mat::<1, N>::zeros();
        h.m[0][IP + 2] = 1.0 / r22; // ∂h/∂p_z
        // ∂h/∂δθ = −p_z/R₂₂² · ∂R₂₂/∂δθ,  with ∂R₂₂/∂δθ = [−R₂₁, R₂₀, 0].
        let c = pz / (r22 * r22);
        h.m[0][ITH] = c * rm.m[2][1];
        h.m[0][ITH + 1] = -c * rm.m[2][0];
        Some((pz / r22, h))
    }

    /// Downward optical flow: the vehicle's horizontal velocity resolved in the body frame, as a
    /// metric-rate camera would report it. `h = (Rᵀ v)_{xy}`, observing velocity and attitude —
    /// which is what keeps velocity from drifting when GPS is gone.
    pub fn update_optical_flow(&mut self, flow_body_xy: [f64; 2], sigma: f64) {
        let (pred, h) = self.flow_predict();
        let residual = Mat::<2, 1>::from_rows([[flow_body_xy[0] - pred[0]], [flow_body_xy[1] - pred[1]]]);
        let r = Mat::<2, 2>::from_rows([[sigma * sigma, 0.0], [0.0, sigma * sigma]]);
        self.update(&h, &residual, &r);
    }

    pub(crate) fn flow_predict(&self) -> ([f64; 2], Mat<2, N>) {
        let rt = self.nom.q.to_matrix().transpose(); // world → body
        let vb = mat_vec(&rt, self.nom.v);
        let sk = skew(vb); // ∂(Rᵀv)/∂δθ = [Rᵀv]×
        let mut h = Mat::<2, N>::zeros();
        for row in 0..2 {
            for k in 0..3 {
                h.m[row][IV + k] = rt.m[row][k];
                h.m[row][ITH + k] = sk.m[row][k];
            }
        }
        ([vb[0], vb[1]], h)
    }

    /// GPS Doppler velocity: a direct world-frame velocity measurement, `h = v`.
    pub fn update_gps_velocity(&mut self, vel: V3, sigma: f64) {
        let mut h = Mat::<3, N>::zeros();
        h.set_block(0, IV, &Mat::<3, 3>::identity());
        let residual = col3(v3::sub(vel, self.nom.v));
        let r = Mat::<3, 3>::identity().scale(sigma * sigma);
        self.update(&h, &residual, &r);
    }

    /// A body-frame 3-D velocity sensor (Doppler velocity log / Doppler radar): the full velocity
    /// resolved in the body frame, `h = Rᵀ v`. Like optical flow but all three axes, so it also
    /// bounds vertical velocity.
    pub fn update_body_velocity(&mut self, vel_body: V3, sigma: f64) {
        let (pred, h) = self.body_velocity_predict();
        let residual = col3(v3::sub(vel_body, pred));
        let r = Mat::<3, 3>::identity().scale(sigma * sigma);
        self.update(&h, &residual, &r);
    }

    pub(crate) fn body_velocity_predict(&self) -> (V3, Mat<3, N>) {
        let rt = self.nom.q.to_matrix().transpose();
        let vb = mat_vec(&rt, self.nom.v);
        let mut h = Mat::<3, N>::zeros();
        h.set_block(0, IV, &rt); // ∂(Rᵀv)/∂δv = Rᵀ
        h.set_block(0, ITH, &skew(vb)); // ∂(Rᵀv)/∂δθ = [Rᵀv]×
        (vb, h)
    }

    /// A direct attitude fix (star tracker, or a visual-inertial / AHRS orientation output). The
    /// measurement is an orientation `z_q`; the innovation is the body-frame rotation carrying the
    /// nominal onto it, and the Jacobian is identity on the orientation error — the standard
    /// multiplicative attitude update.
    pub fn update_attitude(&mut self, z_q: Quat, sigma: f64) {
        let residual = col3(boxminus(z_q, self.nom.q));
        let mut h = Mat::<3, N>::zeros();
        h.set_block(0, ITH, &Mat::<3, 3>::identity());
        let r = Mat::<3, 3>::identity().scale(sigma * sigma);
        self.update(&h, &residual, &r);
    }

    /// The generic Kalman correction for a measurement of dimension `M`: gain, inject, and a
    /// Joseph-form covariance update (which stays symmetric positive-definite where the short
    /// `(I − KH)P` form drifts).
    fn update<const M: usize>(&mut self, h: &Mat<M, N>, residual: &Mat<M, 1>, r: &Mat<M, M>) {
        let pht = self.p.mul(&h.transpose()); // N×M
        let s = h.mul(&pht).add(r); // M×M innovation covariance
        let s_inv = match s.inverse() {
            Some(inv) => inv,
            None => return, // singular innovation — skip rather than corrupt the state
        };
        let k = pht.mul(&s_inv); // N×M Kalman gain
        let dx = k.mul(residual); // N×1 error-state correction
        self.inject(&dx);

        // Joseph: P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ
        let ikh = Mat::<N, N>::identity().sub(&k.mul(h));
        self.p = ikh.mul(&self.p).mul(&ikh.transpose()).add(&k.mul(r).mul(&k.transpose()));
        self.symmetrize();
    }

    /// Fold the error-state correction into the nominal state and reset the error to zero,
    /// including the covariance reset Jacobian for the orientation block.
    fn inject(&mut self, dx: &Mat<N, 1>) {
        let block = |i: usize| [dx.m[i][0], dx.m[i + 1][0], dx.m[i + 2][0]];
        self.nom.p = v3::add(self.nom.p, block(IP));
        self.nom.v = v3::add(self.nom.v, block(IV));
        let dth = block(ITH);
        self.nom.q = self.nom.q.mul(Quat::from_rotation_vector(dth)).normalized();
        self.nom.accel_bias = v3::add(self.nom.accel_bias, block(IAB));
        self.nom.gyro_bias = v3::add(self.nom.gyro_bias, block(IWB));

        // Covariance reset: G = I except the θ block, G_θ = I − ½[δθ]×.
        let mut g = Mat::<N, N>::identity();
        g.set_block(ITH, ITH, &Mat::<3, 3>::identity().sub(&skew(v3::scale(dth, 0.5))));
        self.p = g.mul(&self.p).mul(&g.transpose());
    }

    fn symmetrize(&mut self) {
        for i in 0..N {
            for j in (i + 1)..N {
                let avg = 0.5 * (self.p.m[i][j] + self.p.m[j][i]);
                self.p.m[i][j] = avg;
                self.p.m[j][i] = avg;
            }
        }
    }

    /// The 3×3 position covariance block, for drawing the uncertainty ellipsoid.
    pub fn position_covariance(&self) -> Mat<3, 3> {
        let mut out = Mat::<3, 3>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = self.p.m[IP + i][IP + j];
            }
        }
        out
    }

    pub fn covariance(&self) -> &Mat<N, N> {
        &self.p
    }
}

/// Standard deviations of the initial error on each block.
#[derive(Clone, Copy, Debug)]
pub struct InitialSigma {
    pub position: f64,
    pub velocity: f64,
    pub orientation: f64,
    pub accel_bias: f64,
    pub gyro_bias: f64,
}

impl Default for InitialSigma {
    fn default() -> Self {
        InitialSigma {
            position: 3.0,
            velocity: 0.3,
            orientation: 0.1,
            accel_bias: 0.1,
            gyro_bias: 0.01,
        }
    }
}

fn mat_vec(m: &Mat<3, 3>, v: V3) -> V3 {
    [
        m.m[0][0] * v[0] + m.m[0][1] * v[1] + m.m[0][2] * v[2],
        m.m[1][0] * v[0] + m.m[1][1] * v[1] + m.m[1][2] * v[2],
        m.m[2][0] * v[0] + m.m[2][1] * v[1] + m.m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_static_imu_stays_put() {
        // Level, at rest: the accelerometer reads +g up in body, gyro reads zero. The nominal
        // state must not drift.
        let nom = Nominal::at_rest([0.0; 3], Quat::IDENTITY);
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        let accel = [0.0, 0.0, 9.80665]; // specific force of gravity, level
        for _ in 0..1000 {
            f.predict(accel, [0.0; 3], 0.01);
        }
        assert!(v3::norm(f.nom.p) < 1e-6, "drifted to {:?}", f.nom.p);
        assert!(v3::norm(f.nom.v) < 1e-6, "velocity {:?}", f.nom.v);
    }

    #[test]
    fn covariance_grows_without_measurements() {
        let nom = Nominal::at_rest([0.0; 3], Quat::IDENTITY);
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        let before = f.p.m[IP][IP];
        for _ in 0..500 {
            f.predict([0.0, 0.0, 9.80665], [0.0; 3], 0.01);
        }
        assert!(f.p.m[IP][IP] > before, "position variance should grow when coasting");
    }

    #[test]
    fn a_gps_fix_shrinks_position_covariance() {
        let nom = Nominal::at_rest([0.0; 3], Quat::IDENTITY);
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        for _ in 0..200 {
            f.predict([0.0, 0.0, 9.80665], [0.0; 3], 0.01);
        }
        let before = f.p.m[IP][IP];
        f.update_gps([0.1, -0.2, 0.05], 0.5);
        assert!(f.p.m[IP][IP] < before, "a fix should reduce position variance");
    }

    #[test]
    fn gps_pulls_the_estimate_toward_the_measurement() {
        let nom = Nominal::at_rest([5.0, 0.0, 0.0], Quat::IDENTITY);
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        // Truth is at the origin; the fix should pull the 5 m error down.
        for _ in 0..50 {
            f.update_gps([0.0, 0.0, 0.0], 0.5);
        }
        assert!(v3::norm(f.nom.p) < 0.5, "estimate did not converge: {:?}", f.nom.p);
    }

    #[test]
    fn covariance_stays_symmetric_after_updates() {
        let nom = Nominal::at_rest([0.0; 3], Quat::IDENTITY);
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        f.predict([0.1, 0.0, 9.9], [0.01, -0.02, 0.03], 0.01);
        f.update_gps([0.2, 0.1, -0.1], 0.5);
        f.update_mag([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05);
        for i in 0..N {
            for j in 0..N {
                assert!((f.p.m[i][j] - f.p.m[j][i]).abs() < 1e-12, "asymmetric at {i},{j}");
            }
        }
    }

    /// A filter at a non-trivial pose, so a Jacobian bug in any block actually shows.
    fn sample_filter() -> Eskf {
        let nom = Nominal {
            p: [12.0, -4.0, 30.0],
            v: [3.0, 8.0, -1.0],
            q: Quat::from_rotation_vector([0.2, -0.35, 0.6]).normalized(),
            accel_bias: [0.03, -0.02, 0.01],
            gyro_bias: [0.001, 0.002, -0.0015],
        };
        Eskf::new(nom, InitialSigma::default(), Noise::default())
    }

    /// Apply an error-state increment to a nominal state — the `⊞` used to perturb it numerically.
    fn apply_error(nom: Nominal, dx: [f64; N]) -> Nominal {
        let mut n = nom;
        n.p = v3::add(n.p, [dx[0], dx[1], dx[2]]);
        n.v = v3::add(n.v, [dx[3], dx[4], dx[5]]);
        n.q = n.q.mul(Quat::from_rotation_vector([dx[6], dx[7], dx[8]])).normalized();
        n.accel_bias = v3::add(n.accel_bias, [dx[9], dx[10], dx[11]]);
        n.gyro_bias = v3::add(n.gyro_bias, [dx[12], dx[13], dx[14]]);
        n
    }

    /// Compare a scalar measurement's analytic Jacobian to a central finite difference across all
    /// 15 error directions.
    fn check_scalar_jacobian(f: &Eskf, jac: &Mat<1, N>, h_of: impl Fn(&Eskf) -> f64) {
        let eps = 1e-6;
        for i in 0..N {
            let mut dp = [0.0; N];
            let mut dm = [0.0; N];
            dp[i] = eps;
            dm[i] = -eps;
            let fp = Eskf::new(apply_error(f.nom, dp), InitialSigma::default(), Noise::default());
            let fm = Eskf::new(apply_error(f.nom, dm), InitialSigma::default(), Noise::default());
            let numeric = (h_of(&fp) - h_of(&fm)) / (2.0 * eps);
            assert!(
                (numeric - jac.m[0][i]).abs() < 1e-4,
                "jacobian[{i}]: finite-diff {numeric:.6} vs analytic {:.6}",
                jac.m[0][i]
            );
        }
    }

    #[test]
    fn range_jacobian_matches_finite_difference() {
        let f = sample_filter();
        let beacon = [-20.0, 10.0, 2.0];
        let (_, jac) = f.range_predict(beacon).unwrap();
        check_scalar_jacobian(&f, &jac, |g| g.range_predict(beacon).unwrap().0);
    }

    #[test]
    fn lidar_jacobian_matches_finite_difference() {
        let f = sample_filter();
        let (_, jac) = f.lidar_predict().unwrap();
        check_scalar_jacobian(&f, &jac, |g| g.lidar_predict().unwrap().0);
    }

    #[test]
    fn optical_flow_jacobian_matches_finite_difference() {
        let f = sample_filter();
        let (h0, jac) = f.flow_predict();
        let eps = 1e-6;
        for i in 0..N {
            let mut dp = [0.0; N];
            let mut dm = [0.0; N];
            dp[i] = eps;
            dm[i] = -eps;
            let fp = Eskf::new(apply_error(f.nom, dp), InitialSigma::default(), Noise::default());
            let fm = Eskf::new(apply_error(f.nom, dm), InitialSigma::default(), Noise::default());
            let (hp, _) = fp.flow_predict();
            let (hm, _) = fm.flow_predict();
            for row in 0..2 {
                let numeric = (hp[row] - hm[row]) / (2.0 * eps);
                assert!(
                    (numeric - jac.m[row][i]).abs() < 1e-4,
                    "flow jacobian[{row}][{i}]: {numeric:.6} vs {:.6}",
                    jac.m[row][i]
                );
            }
            let _ = h0;
        }
    }

    #[test]
    fn range_beacons_localize_without_gps() {
        // The estimate starts 8 m off; ranges from four fixed beacons pull it home with no GPS.
        let truth = [5.0, 3.0, 20.0];
        let beacons = [
            [-40.0, -40.0, 0.0],
            [40.0, -40.0, 5.0],
            [40.0, 40.0, 30.0],
            [-40.0, 40.0, 15.0],
        ];
        let nom = Nominal::at_rest([13.0, -2.0, 26.0], Quat::IDENTITY);
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        for _ in 0..100 {
            for b in beacons {
                let true_range = v3::norm(v3::sub(truth, b));
                f.update_range(b, true_range, 0.3);
            }
        }
        assert!(
            v3::norm(v3::sub(f.nom.p, truth)) < 0.6,
            "range-only localization failed: {:?}",
            f.nom.p
        );
    }

    #[test]
    fn body_velocity_jacobian_matches_finite_difference() {
        let f = sample_filter();
        let (_, jac) = f.body_velocity_predict();
        let eps = 1e-6;
        for i in 0..N {
            let mut dp = [0.0; N];
            let mut dm = [0.0; N];
            dp[i] = eps;
            dm[i] = -eps;
            let fp = Eskf::new(apply_error(f.nom, dp), InitialSigma::default(), Noise::default());
            let fm = Eskf::new(apply_error(f.nom, dm), InitialSigma::default(), Noise::default());
            let (hp, _) = fp.body_velocity_predict();
            let (hm, _) = fm.body_velocity_predict();
            for row in 0..3 {
                let numeric = (hp[row] - hm[row]) / (2.0 * eps);
                assert!(
                    (numeric - jac.m[row][i]).abs() < 1e-4,
                    "body-velocity jacobian[{row}][{i}]: {numeric:.6} vs {:.6}",
                    jac.m[row][i]
                );
            }
        }
    }

    #[test]
    fn gps_velocity_pulls_velocity_to_the_measurement() {
        let nom = Nominal { v: [2.0, -1.0, 0.5], ..Nominal::at_rest([0.0; 3], Quat::IDENTITY) };
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        for _ in 0..80 {
            f.update_gps_velocity([0.0, 0.0, 0.0], 0.1);
        }
        assert!(v3::norm(f.nom.v) < 0.2, "velocity did not converge: {:?}", f.nom.v);
    }

    #[test]
    fn attitude_fix_converges_to_the_measurement() {
        // Seed a ~20° orientation error; attitude fixes at truth must drive it down (which only
        // happens if the update's sign is right — a flipped sign diverges).
        let truth = Quat::from_rotation_vector([0.1, -0.2, 0.15]);
        let nom = Nominal {
            q: truth.mul(Quat::from_rotation_vector([0.35, 0.0, 0.0])),
            ..Nominal::at_rest([0.0; 3], Quat::IDENTITY)
        };
        let mut f = Eskf::new(nom, InitialSigma::default(), Noise::default());
        for _ in 0..80 {
            f.update_attitude(truth, 0.02);
        }
        let err = v3::norm(boxminus(truth, f.nom.q)).to_degrees();
        assert!(err < 1.0, "attitude did not converge: {err}°");
    }
}
