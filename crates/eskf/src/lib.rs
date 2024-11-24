//! A quaternion error-state Kalman filter (ESKF) fusing a strapdown IMU with GPS, barometer and
//! magnetometer, plus the simulator and the consistency metrics that prove it is done right.
//!
//! The layers, native-testable from the bottom up:
//! - [`linalg`] — a tiny fixed-size matrix kit, so dimensions are checked at compile time.
//! - [`quat`] — unit quaternions with a body-frame minimal error (the ESKF's reason to exist).
//! - [`filter`] — the ESKF: nominal integration, error-state covariance, GPS/baro/mag updates.
//! - [`sim`] — an analytic banking flight with synthesised, tunable sensors and ground truth.
//!
//! The correctness gate is *statistical consistency*: over many Monte-Carlo flights the filter's
//! estimation error, weighed by its own covariance (the NEES), must match the state dimension.
//! A filter that is merely "close" but reports the wrong uncertainty fails this — which is how a
//! subtly wrong Jacobian or noise model is caught. See [`nees`] and the `eskf` CLI.

// Matrix code reads more clearly with explicit index loops than with zipped iterators, and the
// dimensions are const-generic anyway, so the usual bounds-elision argument does not apply.
#![allow(clippy::needless_range_loop)]

pub mod filter;
pub mod linalg;
pub mod quat;
pub mod sim;

pub use filter::{Eskf, InitialSigma, Nominal, Noise, N};
pub use quat::Quat;
pub use sim::{SimConfig, Simulator, Tick, TrueState, GRAVITY, MAG_REFERENCE};

use linalg::{v3, Mat};
use quat::boxminus;

/// The estimation error in the filter's own 15-dimensional tangent space,
/// `[δp, δv, δθ, δa_b, δω_b]`, each component `truth − nominal` (orientation via the body-frame
/// log map). This is exactly the vector the covariance `P` claims to describe, so it is what any
/// consistency check must use — comparing a quaternion to a quaternion here would be a category
/// error.
pub fn error_state(nom: &Nominal, truth: &TrueState) -> Mat<N, 1> {
    let mut e = Mat::<N, 1>::zeros();
    let blocks = [
        v3::sub(truth.p, nom.p),
        v3::sub(truth.v, nom.v),
        boxminus(truth.q, nom.q),
        v3::sub(truth.accel_bias, nom.accel_bias),
        v3::sub(truth.gyro_bias, nom.gyro_bias),
    ];
    for (b, block) in blocks.iter().enumerate() {
        for k in 0..3 {
            e.m[b * 3 + k][0] = block[k];
        }
    }
    e
}

/// Full-state NEES, `eᵀ P⁻¹ e`. For a consistent filter its expectation is the state dimension,
/// 15. Persistently far above means the filter is overconfident (covariance too small); far below
/// means it is conservative.
pub fn nees(nom: &Nominal, truth: &TrueState, p: &Mat<N, N>) -> Option<f64> {
    let e = error_state(nom, truth);
    let pinv = p.inverse()?;
    Some(e.transpose().mul(&pinv).mul(&e).m[0][0])
}

/// Position-only NEES (3 degrees of freedom, expectation 3). The most legible consistency signal
/// for the playground, since position is what the ellipsoid draws.
pub fn position_nees(nom: &Nominal, truth: &TrueState, p: &Mat<N, N>) -> Option<f64> {
    let e = linalg::col3(v3::sub(truth.p, nom.p));
    let mut pp = Mat::<3, 3>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            pp.m[i][j] = p.m[i][j];
        }
    }
    let pinv = pp.inverse()?;
    Some(e.transpose().mul(&pinv).mul(&e).m[0][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A filter seeded with the truth and fed all sensors should track a full flight to well
    /// under a metre — a coarse but decisive "it actually works" check.
    #[test]
    fn filter_tracks_a_full_flight() {
        let cfg = SimConfig::default();
        let mut sim = Simulator::new(cfg, 7);
        let mut f = Eskf::new(sim.truth_nominal(), InitialSigma::default(), Noise::default());

        let mut last_err = 0.0;
        for _ in 0..(cfg.imu_rate as usize * 40) {
            let tick = sim.step();
            f.predict(tick.accel, tick.gyro, tick.dt);
            if let Some(z) = tick.gps {
                f.update_gps(z, cfg.gps_noise);
            }
            if let Some(z) = tick.baro {
                f.update_baro(z, cfg.baro_noise);
            }
            if let Some(z) = tick.mag {
                f.update_mag(z, MAG_REFERENCE, cfg.mag_noise);
            }
            last_err = v3::norm(v3::sub(f.nom.p, tick.truth.p));
        }
        assert!(last_err < 1.5, "final position error {last_err:.2} m too large");
    }
}
