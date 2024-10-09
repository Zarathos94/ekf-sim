//! A small, dependency-free linear-algebra kit sized for an error-state Kalman filter.
//!
//! Fixed-size, stack-allocated matrices with const-generic dimensions: the compiler checks
//! that a 15×15 covariance never gets multiplied by a 3×15 Jacobian the wrong way round, and
//! there is no heap traffic in the filter loop. Only what the filter actually needs — multiply,
//! transpose, add, a block copy for assembling the transition matrix, a skew-symmetric operator,
//! and a Gauss–Jordan inverse for the (at most 3×3) innovation covariance.

/// A plain 3-vector. The filter's 3-blocks (position, velocity, biases, axes) are these.
pub type V3 = [f64; 3];

pub mod v3 {
    use super::V3;

    pub fn add(a: V3, b: V3) -> V3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    pub fn sub(a: V3, b: V3) -> V3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    pub fn scale(a: V3, s: f64) -> V3 {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    pub fn dot(a: V3, b: V3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    pub fn cross(a: V3, b: V3) -> V3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    pub fn norm(a: V3) -> f64 {
        dot(a, a).sqrt()
    }
    pub fn normalize(a: V3) -> V3 {
        let n = norm(a);
        if n > 0.0 {
            scale(a, 1.0 / n)
        } else {
            a
        }
    }
}

/// A dense `R×C` matrix in row-major order.
#[derive(Clone, Copy, Debug)]
pub struct Mat<const R: usize, const C: usize> {
    pub m: [[f64; C]; R],
}

impl<const R: usize, const C: usize> Mat<R, C> {
    pub fn zeros() -> Self {
        Self { m: [[0.0; C]; R] }
    }

    pub fn from_rows(m: [[f64; C]; R]) -> Self {
        Self { m }
    }

    pub fn transpose(&self) -> Mat<C, R> {
        let mut out = Mat::<C, R>::zeros();
        for r in 0..R {
            for c in 0..C {
                out.m[c][r] = self.m[r][c];
            }
        }
        out
    }

    /// Matrix product `self · rhs`.
    #[allow(clippy::should_implement_trait)] // dimension-changing product; a plain method reads clearer than a generic Mul impl
    pub fn mul<const K: usize>(&self, rhs: &Mat<C, K>) -> Mat<R, K> {
        let mut out = Mat::<R, K>::zeros();
        for r in 0..R {
            for k in 0..K {
                let mut s = 0.0;
                for c in 0..C {
                    s += self.m[r][c] * rhs.m[c][k];
                }
                out.m[r][k] = s;
            }
        }
        out
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = *self;
        for r in 0..R {
            for c in 0..C {
                out.m[r][c] += rhs.m[r][c];
            }
        }
        out
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = *self;
        for r in 0..R {
            for c in 0..C {
                out.m[r][c] -= rhs.m[r][c];
            }
        }
        out
    }

    pub fn scale(&self, s: f64) -> Self {
        let mut out = *self;
        for r in 0..R {
            for c in 0..C {
                out.m[r][c] *= s;
            }
        }
        out
    }

    /// Copies a smaller matrix into this one with its top-left corner at `(r0, c0)`. Used to
    /// assemble the block-structured transition and noise matrices without spelling out zeros.
    pub fn set_block<const BR: usize, const BC: usize>(
        &mut self,
        r0: usize,
        c0: usize,
        b: &Mat<BR, BC>,
    ) {
        for r in 0..BR {
            for c in 0..BC {
                self.m[r0 + r][c0 + c] = b.m[r][c];
            }
        }
    }
}

impl<const N: usize> Mat<N, N> {
    pub fn identity() -> Self {
        let mut out = Self::zeros();
        for i in 0..N {
            out.m[i][i] = 1.0;
        }
        out
    }

    /// Adds `s` to the leading diagonal — `self + sI`, the workhorse for `(I - KH)` and for
    /// nudging a covariance back to symmetry-positive-definite.
    pub fn add_diag(&self, s: f64) -> Self {
        let mut out = *self;
        for i in 0..N {
            out.m[i][i] += s;
        }
        out
    }

    /// Gauss–Jordan inverse with partial pivoting. `None` if singular. Only ever called on the
    /// innovation covariance, which is 1×1 to 3×3, so the cubic cost is nothing.
    pub fn inverse(&self) -> Option<Self> {
        let mut a = self.m;
        let mut inv = Self::identity().m;
        for col in 0..N {
            // Partial pivot: swap in the row with the largest magnitude in this column.
            let mut pivot = col;
            let mut best = a[col][col].abs();
            for r in (col + 1)..N {
                let v = a[r][col].abs();
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if best < 1e-18 {
                return None;
            }
            if pivot != col {
                a.swap(col, pivot);
                inv.swap(col, pivot);
            }
            let d = a[col][col];
            for c in 0..N {
                a[col][c] /= d;
                inv[col][c] /= d;
            }
            for r in 0..N {
                if r == col {
                    continue;
                }
                let f = a[r][col];
                if f == 0.0 {
                    continue;
                }
                for c in 0..N {
                    a[r][c] -= f * a[col][c];
                    inv[r][c] -= f * inv[col][c];
                }
            }
        }
        Some(Self { m: inv })
    }
}

/// A column vector as an `N×1` matrix.
pub type Vec<const N: usize> = Mat<N, 1>;

/// Packs a 3-vector into a column.
pub fn col3(v: V3) -> Vec<3> {
    Mat::from_rows([[v[0]], [v[1]], [v[2]]])
}

/// The skew-symmetric matrix `[v]×`, so that `[v]× w == v × w`.
pub fn skew(v: V3) -> Mat<3, 3> {
    Mat::from_rows([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_and_transpose_agree_with_hand_calc() {
        let a = Mat::<2, 3>::from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b = Mat::<3, 2>::from_rows([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
        let c = a.mul(&b);
        assert_eq!(c.m, [[58.0, 64.0], [139.0, 154.0]]);
        assert_eq!(a.transpose().m, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn inverse_recovers_identity() {
        let a = Mat::<3, 3>::from_rows([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 2.0]]);
        let inv = a.inverse().expect("nonsingular");
        let prod = a.mul(&inv);
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((prod.m[r][c] - want).abs() < 1e-12, "at {r},{c}: {}", prod.m[r][c]);
            }
        }
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let a = Mat::<2, 2>::from_rows([[1.0, 2.0], [2.0, 4.0]]);
        assert!(a.inverse().is_none());
    }

    #[test]
    fn skew_matches_cross_product() {
        let a = [0.3, -1.1, 2.0];
        let b = [4.0, 0.5, -0.7];
        let via_skew = skew(a).mul(&col3(b));
        let via_cross = v3::cross(a, b);
        for i in 0..3 {
            assert!((via_skew.m[i][0] - via_cross[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn set_block_places_submatrix() {
        let mut big = Mat::<4, 4>::zeros();
        let blk = Mat::<2, 2>::from_rows([[1.0, 2.0], [3.0, 4.0]]);
        big.set_block(1, 2, &blk);
        assert_eq!(big.m[1][2], 1.0);
        assert_eq!(big.m[2][3], 4.0);
        assert_eq!(big.m[0][0], 0.0);
    }
}
