//! Unit quaternions, Hamilton convention, `[w, x, y, z]`, representing a body→world rotation.
//!
//! The filter carries orientation as a unit quaternion and its error as a 3-vector rotation in
//! the **body** frame (the local / multiplicative error, `q_true = q_nominal ⊗ exp(½ δθ)`).
//! Keeping the error minimal — three numbers, not four — is the whole reason an error-state
//! filter exists: the covariance stays 3×3 per rotation block and never has to pretend a
//! four-component unit quaternion has four independent degrees of freedom.

use crate::linalg::{v3, Mat, V3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quat { w, x, y, z }
    }

    /// The exponential map from a rotation vector (axis·angle, radians) to a unit quaternion.
    /// This is how an error rotation `δθ` becomes the correction quaternion `exp(½ δθ)`.
    pub fn from_rotation_vector(phi: V3) -> Self {
        let angle = v3::norm(phi);
        if angle < 1e-9 {
            // Small angle: cos ≈ 1, sin(θ/2)/θ ≈ 1/2. Normalising cleans up the truncation.
            Quat::new(1.0, phi[0] * 0.5, phi[1] * 0.5, phi[2] * 0.5).normalized()
        } else {
            let half = angle * 0.5;
            let s = half.sin() / angle;
            Quat::new(half.cos(), phi[0] * s, phi[1] * s, phi[2] * s)
        }
    }

    /// The log map: a unit quaternion back to its rotation vector. Inverse of
    /// [`Quat::from_rotation_vector`], used to measure the angular error between two orientations.
    pub fn to_rotation_vector(self) -> V3 {
        let q = self.canonical(); // w >= 0, so the angle is the short way round
        let v = [q.x, q.y, q.z];
        let vn = v3::norm(v);
        if vn < 1e-9 {
            // cos ≈ 1: angle ≈ 2·|v|, direction v/|v| — the limit is just 2v.
            v3::scale(v, 2.0)
        } else {
            let angle = 2.0 * vn.atan2(q.w);
            v3::scale(v, angle / vn)
        }
    }

    /// Hamilton product `self ⊗ rhs`.
    #[allow(clippy::should_implement_trait)] // quaternion product is not commutative scalar Mul
    pub fn mul(self, r: Quat) -> Quat {
        Quat {
            w: self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            x: self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            y: self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            z: self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        }
    }

    pub fn conjugate(self) -> Quat {
        Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    pub fn norm(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Quat {
        let n = self.norm();
        if n > 0.0 {
            Quat { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
        } else {
            Quat::IDENTITY
        }
    }

    /// Flips sign so `w ≥ 0`. `q` and `-q` are the same rotation; this picks one representative
    /// so the log map returns the shorter of the two equivalent rotations.
    pub fn canonical(self) -> Quat {
        if self.w < 0.0 {
            Quat { w: -self.w, x: -self.x, y: -self.y, z: -self.z }
        } else {
            self
        }
    }

    /// The body→world rotation matrix.
    pub fn to_matrix(self) -> Mat<3, 3> {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        Mat::from_rows([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ])
    }

    /// Rotate a body-frame vector into the world frame (`R · v`).
    pub fn rotate(self, v: V3) -> V3 {
        mat_vec(&self.to_matrix(), v)
    }

    /// Rotate a world-frame vector into the body frame (`Rᵀ · v`).
    pub fn rotate_inv(self, v: V3) -> V3 {
        self.conjugate().rotate(v)
    }

    /// A quaternion from a body→world rotation matrix (Shepperd's method — pick the largest
    /// diagonal term to stay away from the degenerate cases). Used to seed ground truth.
    pub fn from_matrix(m: &Mat<3, 3>) -> Quat {
        let r = &m.m;
        let trace = r[0][0] + r[1][1] + r[2][2];
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat::new(
                0.25 * s,
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s,
            )
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
            Quat::new(
                (r[2][1] - r[1][2]) / s,
                0.25 * s,
                (r[0][1] + r[1][0]) / s,
                (r[0][2] + r[2][0]) / s,
            )
        } else if r[1][1] > r[2][2] {
            let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
            Quat::new(
                (r[0][2] - r[2][0]) / s,
                (r[0][1] + r[1][0]) / s,
                0.25 * s,
                (r[1][2] + r[2][1]) / s,
            )
        } else {
            let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
            Quat::new(
                (r[1][0] - r[0][1]) / s,
                (r[0][2] + r[2][0]) / s,
                (r[1][2] + r[2][1]) / s,
                0.25 * s,
            )
        };
        q.normalized()
    }

    /// Roll, pitch, yaw (radians), the aerospace Z-Y-X sequence, for display.
    pub fn to_euler(self) -> V3 {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            core::f64::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        [roll, pitch, yaw]
    }
}

/// The body-frame angular error carrying `a` onto `b`: the `δθ` with `a = b ⊗ exp(½ δθ)`.
/// This is the filter's own error definition, so it is what the consistency check must use.
pub fn boxminus(a: Quat, b: Quat) -> V3 {
    b.conjugate().mul(a).to_rotation_vector()
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
    use crate::linalg::v3;

    #[test]
    fn exp_log_round_trip() {
        for phi in [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [1.4, 0.0, 0.0], [0.7, 0.7, -0.7]] {
            let back = Quat::from_rotation_vector(phi).to_rotation_vector();
            for i in 0..3 {
                assert!((back[i] - phi[i]).abs() < 1e-9, "phi {phi:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn rotation_is_orthonormal_and_matches_inverse() {
        let q = Quat::from_rotation_vector([0.3, -0.9, 0.5]);
        let v = [1.2, -0.4, 2.0];
        let back = q.rotate_inv(q.rotate(v));
        for i in 0..3 {
            assert!((back[i] - v[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn known_90_deg_yaw_rotates_x_to_y() {
        // 90° about world/body z takes body +x to world +y.
        let q = Quat::from_rotation_vector([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
        let r = q.rotate([1.0, 0.0, 0.0]);
        assert!((r[0]).abs() < 1e-12 && (r[1] - 1.0).abs() < 1e-12 && r[2].abs() < 1e-12);
    }

    #[test]
    fn matrix_round_trips_through_quaternion() {
        let q = Quat::from_rotation_vector([0.2, 1.1, -0.6]).normalized();
        let back = Quat::from_matrix(&q.to_matrix());
        // Same rotation up to sign.
        let err = v3::norm(boxminus(q, back));
        assert!(err < 1e-9, "err {err}");
    }

    #[test]
    fn boxminus_of_a_known_perturbation_recovers_it() {
        let base = Quat::from_rotation_vector([0.4, -0.1, 0.8]);
        let delta = [0.02, -0.03, 0.01];
        let perturbed = base.mul(Quat::from_rotation_vector(delta));
        let recovered = boxminus(perturbed, base);
        for i in 0..3 {
            assert!((recovered[i] - delta[i]).abs() < 1e-6, "{recovered:?} vs {delta:?}");
        }
    }
}
